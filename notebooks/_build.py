"""코랩 노트북 생성기. 노트북을 고칠 때는 이 파일을 고치고 `python notebooks/_build.py` 실행."""

import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r'''# 저장소 받기 + 패키지 설치
BRANCH = "ccr-a463bea6-bpbyrh"   # main에 합친 뒤에는 "main"
REPO = "github.com/b1bibig/Cfinetune"
import os
try:  # 저장소가 비공개면 코랩 왼쪽 열쇠(보안 비밀)에 GH_TOKEN 을 넣어 두세요
    from google.colab import userdata
    _tok = userdata.get("GH_TOKEN")
    REPO_URL = f"https://{_tok}@{REPO}"
except Exception:
    REPO_URL = f"https://{REPO}"
if not os.path.exists("/content/Cfinetune"):
    !git clone -q -b {BRANCH} {REPO_URL} /content/Cfinetune
%cd /content/Cfinetune
!git pull -q
%pip install -q -r requirements-colab.txt'''

DRIVE = r'''# 구글 드라이브: 결과(results/)를 세션이 끊겨도 남도록 저장
from google.colab import drive
drive.mount("/content/drive")
import os, shutil, sys
DRIVE = "/content/drive/MyDrive/cfinetune"
os.makedirs(DRIVE, exist_ok=True)

def pull_results():
    if os.path.exists(f"{DRIVE}/results"):
        shutil.copytree(f"{DRIVE}/results", "results", dirs_exist_ok=True)

def push_results():
    shutil.copytree("results", f"{DRIVE}/results", dirs_exist_ok=True)
    print("드라이브에 결과 저장함")

pull_results()
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")'''

GPU_CHECK = r'''!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU 메모리 {mem:.0f}GB")
assert mem >= {need}, "런타임 > 런타임 유형 변경에서 {gpu} 를 고르세요"'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells, gpu=None):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": gpu}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def write(name, nb):
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", name)


# ------------------------------------------------------------------ 00
write("00_smoke_test.ipynb", notebook([
    md("""
# 00. 환경 점검 (무료, GPU 필요 없음)
작은 무작위 gpt-oss 모델로 파이프라인 코드가 이 환경에서 잘 도는지 확인합니다.
전부 `passed`가 나오면 다음 노트북으로 넘어가세요. 코랩 유닛을 쓰지 않도록 **CPU 런타임**으로 실행하세요.
"""),
    code(SETUP),
    code("!python -m pytest tests -q"),
]))

# ------------------------------------------------------------------ 01
write("01_calibrate.ipynb", notebook([
    md("""
# 01. 전문가 중요도 측정 (A100 80GB 또는 G4, 약 1시간 / 약 8~10유닛)
한국어 위키와 영어 위키를 각각 모델에 흘려 보내며, 층마다 32개 전문가가
- 얼마나 자주 뽑히는지 (빈도, `freq`)
- 뽑혔을 때 출력에 얼마나 크게 기여하는지 (REAP 점수, `reap`)

를 기록합니다. 결과는 작은 JSON 파일(`results/calib/stats-ko.json`, `stats-en.json`)이고 드라이브에 저장됩니다.

원본 MXFP4를 불러오는 데 문제가 생기면 `--model unsloth/gpt-oss-20b-BF16` 을 붙이세요. 값이 같은 BF16 버전입니다.
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "45").replace("{gpu}", "A100 (80GB) 또는 G4")),
    code("!python scripts/01_calibrate.py --langs ko en\npush_results()"),
    md("## 한국어와 영어는 같은 전문가를 쓸까? (보고서용 그림)"),
    code(r'''import json, matplotlib.pyplot as plt
from cfinetune.observe import select_experts
ko = json.load(open("results/calib/stats-ko.json")); en = json.load(open("results/calib/stats-en.json"))
for k in (24, 16):
    sk, se = select_experts(ko, "reap", k), select_experts(en, "reap", k)
    overlap = [len(set(sk[i]) & set(se[i])) / k for i in sk]
    plt.plot(overlap, marker="o", label=f"keep {k}")
plt.xlabel("layer"); plt.ylabel("ko/en overlap of kept experts"); plt.ylim(0, 1.05); plt.grid(alpha=.3); plt.legend()
plt.savefig("results/fig_expert_overlap.png", dpi=150); plt.show()
push_results()'''),
], gpu="A100"))

# ------------------------------------------------------------------ 02
write("02_prune_quantize_eval.ipynb", notebook([
    md("""
# 02. 프루닝 + 양자화 + 평가 (L4, 약 3시간 / 약 5~6유닛)
변형마다 **프루닝(CPU) → GGUF 변환 → 객관식·퍼플렉시티 평가 → 속도·전력 측정 → 파일 삭제** 순서로 진행합니다.
결과 JSON은 변형이 끝날 때마다 드라이브에 저장되므로, 세션이 끊겨도 다시 실행하면 끝난 변형은 건너뜁니다.

| 단계 | 변형 |
|---|---|
| 기준 | `original` (전문가 32개) |
| 프루닝 | config의 `pruning.variants` (한국어/영어 보정 × 24/16개, 빈도 기준 비교) |
| 양자화 | `ko-reap-k16`을 MXFP4_MOE / Q8_0 / Q4_0 로 |
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "20").replace("{gpu}", "L4")),
    code("# llama.cpp 빌드 (처음 약 15분, 이후에는 드라이브 캐시에서 복사)\n!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
SRC = resolve_model_dir(cfg["model"]["id"])   # 원본 MXFP4 (약 13GB)
GGUF, PRUNED = "/content/gguf", "/content/models/pruned"
os.makedirs(GGUF, exist_ok=True)

def done(name):
    return os.path.exists(f"results/eval/{name}.json") and os.path.exists(f"results/bench/{name}.json")

def evaluate(name, gguf):
    !python scripts/05_eval.py --gguf {gguf} --name {name}
    !python scripts/06_bench.py --gguf {gguf} --name {name}
    push_results()'''),
    md("## 기준: 원본"),
    code(r'''if not done("original"):
    !python scripts/03_make_gguf.py --hf-dir {SRC} --out {GGUF}/original.gguf
    evaluate("original", f"{GGUF}/original.gguf")
    os.remove(f"{GGUF}/original.gguf")'''),
    md("## 프루닝 변형"),
    code(r'''for v in cfg["pruning"]["variants"]:
    name = v["name"]
    if done(name):
        print("건너뜀:", name); continue
    !python scripts/02_prune.py --variant {name} --src {SRC} --out-dir {PRUNED}
    !python scripts/03_make_gguf.py --hf-dir {PRUNED}/{name} --out {GGUF}/{name}.gguf
    evaluate(name, f"{GGUF}/{name}.gguf")
    os.remove(f"{GGUF}/{name}.gguf")
    if name != cfg["quantization"]["base_variant"]:
        shutil.rmtree(f"{PRUNED}/{name}")'''),
    md("## 양자화 비교"),
    code(r'''base = cfg["quantization"]["base_variant"]
todo = [q for q in cfg["quantization"]["types"] if not done(f"{base}-{q}")]
if todo:
    if not os.path.exists(f"{PRUNED}/{base}"):
        !python scripts/02_prune.py --variant {base} --src {SRC} --out-dir {PRUNED}
    QS = " ".join(todo)
    !python scripts/03_make_gguf.py --hf-dir {PRUNED}/{base} --out {GGUF}/{base}.gguf --quant {QS}
    for q in todo:
        evaluate(f"{base}-{q}", f"{GGUF}/{base}-{q}.gguf")
        os.remove(f"{GGUF}/{base}-{q}.gguf")'''),
    md("## 중간 결과 표"),
    code('!python scripts/07_econ.py --measured-gpu "L4 (Colab)"\npush_results()'),
], gpu="L4"))

# ------------------------------------------------------------------ 03
write("03_lora.ipynb", notebook([
    md("""
# 03. 저랭크(LoRA) 한국어 추가학습 (A100 40GB, 약 1~2시간 / 약 6~11유닛)
`ko-reap-k16`(전문가 16개) 모델에 한국어 지시문(KULLM-v2 6천 개)으로 LoRA를 학습합니다.
- 어텐션 q/k/v/o + 남은 전문가 가중치에 LoRA, 라우터는 고정
- 학습 후 합친 모델을 GGUF로 바꾸고 다시 4비트(MXFP4_MOE)로 양자화 → 드라이브에 저장
- 평가는 04 노트북(L4)에서 합니다 (더 싸서)
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "38").replace("{gpu}", "A100")),
    code("# 변환/양자화만 하면 되므로 CPU 빌드 (약 5분)\n!LLAMA_BACKEND=cpu bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
base = cfg["lora"]["base_variant"]
SRC = resolve_model_dir(cfg["model"]["id"])
!python scripts/02_prune.py --variant {base} --src {SRC} --out-dir /content/models/pruned'''),
    code(r'''OUT = f"/content/lora/{base}-lora"
!python scripts/04_train_lora.py --model /content/models/pruned/{base} --out {OUT}
shutil.copytree(f"{OUT}/adapter", f"{DRIVE}/lora/{base}-lora/adapter", dirs_exist_ok=True)
push_results()'''),
    code(r'''!python scripts/03_make_gguf.py --hf-dir {OUT}/merged --out /content/gguf/{base}-lora.gguf --quant MXFP4_MOE Q8_0
os.makedirs(f"{DRIVE}/gguf", exist_ok=True)
for q in ("MXFP4_MOE", "Q8_0"):   # 드라이브 용량이 부족하면 Q8_0 은 빼세요 (약 12GB)
    shutil.copy(f"/content/gguf/{base}-lora-{q}.gguf", f"{DRIVE}/gguf/")
!ls -la {DRIVE}/gguf'''),
], gpu="A100"))

# ------------------------------------------------------------------ 04
write("04_eval_lora_and_report.ipynb", notebook([
    md("""
# 04. LoRA 모델 평가 + 최종 경제성 보고서 (L4, 약 1시간 / 약 2유닛)
드라이브 `cfinetune/gguf/`에 있는 GGUF를 02와 똑같은 방법으로 평가하고, 모든 결과로 표와 그래프를 만듭니다.
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "20").replace("{gpu}", "L4")),
    code("!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''import glob
for g in sorted(glob.glob(f"{DRIVE}/gguf/*.gguf")):
    name = os.path.basename(g)[:-5]
    if os.path.exists(f"results/eval/{name}.json") and os.path.exists(f"results/bench/{name}.json"):
        print("건너뜀:", name); continue
    local = f"/content/{name}.gguf"
    shutil.copy(g, local)            # 드라이브에서 바로 읽으면 느려서 로컬로 복사
    !python scripts/05_eval.py --gguf {local} --name {name}
    !python scripts/06_bench.py --gguf {local} --name {name}
    os.remove(local)
    push_results()'''),
    code(r'''!python scripts/07_econ.py --measured-gpu "L4 (Colab)"
push_results()
from IPython.display import Markdown, Image, display
display(Markdown(open("results/report.md").read()))
for f in ("fig_size_vs_korean.png", "fig_energy.png", "fig_expert_overlap.png"):
    if os.path.exists(f"results/{f}"): display(Image(f"results/{f}"))'''),
], gpu="L4"))
