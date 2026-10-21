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
ERNIE-4.5-21B-A3B(층마다 전문가 64개)에 **한국어 데이터**(위키 50% + 지시문 대화 50%, 약 26만 토큰)를 흘려 보내며,
층마다 전문가가
- 얼마나 자주 뽑히는지 (빈도, `freq`)
- 뽑혔을 때 출력에 얼마나 크게 기여하는지 (REAP 점수, `reap`)

를 기록합니다. 결과는 작은 JSON 파일(`results/calib/stats-ko.json`)이고 드라이브에 저장됩니다.
원본 BF16이 약 44GB라 A100 80GB 또는 G4가 필요합니다.
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "50").replace("{gpu}", "A100 (80GB) 또는 G4")),
    code("!python scripts/01_calibrate.py --sets ko\npush_results()"),
    md("## 한국어는 소수의 전문가만 쓸까? (보고서용 그림)\n층마다 REAP 점수가 높은 순으로 전문가를 늘어놓고, 상위 k개가 전체 점수의 몇 %를 차지하는지 봅니다."),
    code(r'''import json, numpy as np, matplotlib.pyplot as plt
st = json.load(open("results/calib/stats-ko.json"))
shares = []
for layer in st["layers"].values():
    s = np.sort(np.array(layer["reap"]) * np.array(layer["freq"]))[::-1]   # 기여도 = 점수 x 빈도
    shares.append(np.cumsum(s) / s.sum())
shares = np.array(shares)
plt.plot(range(1, shares.shape[1] + 1), shares.mean(0), marker=".")
for k in (32, 48): plt.axvline(k, ls="--", c="gray"); print(f"상위 {k}개가 평균 {shares.mean(0)[k-1]*100:.1f}% 차지")
plt.xlabel("kept experts (sorted)"); plt.ylabel("cumulative share of Korean contribution"); plt.grid(alpha=.3)
plt.savefig("results/fig_korean_expert_share.png", dpi=150); plt.show()
push_results()'''),
], gpu="A100"))

# ------------------------------------------------------------------ 02
write("02_prune_quantize_eval.ipynb", notebook([
    md("""
# 02. 칼질 + 양자화 + 평가 (L4, 약 3시간 / 약 5~6유닛)
| 단계 | 만드는 GGUF | 확인하는 것 |
|---|---|---|
| 기준 | `original-Q6_K`, `original-Q4_K_M` | 출발점 (Q6_K는 거의 무손실, Q4_K_M은 배포 형식 비교용) |
| 칼질 | `ko-k48-Q6_K`, `ko-k32-Q6_K` | 전문가를 줄이면 한국어가 얼마나 떨어지나 |
| 양자화 | `ko-k32-Q4_K_M`, `-Q3_K_M`, `-Q2_K` | 비트를 줄이면 얼마나 떨어지나 |

변형마다 GGUF 만들기 → 객관식·퍼플렉시티 평가 → 속도·전력 측정 → 파일 삭제 순서로 진행합니다.
결과는 끝날 때마다 드라이브에 저장되므로, 세션이 끊기면 다시 실행하세요. 끝난 것은 건너뜁니다.
**디스크가 약 100GB 필요합니다.** 아래 `df` 결과를 확인하세요.
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "20").replace("{gpu}", "L4") + "\n!df -h /content | tail -1"),
    code("# llama.cpp 빌드 (처음 약 15분, 이후에는 드라이브 캐시에서 복사)\n!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
SRC = resolve_model_dir(cfg["model"]["id"])   # 원본 BF16 (약 44GB)
GGUF, PRUNED = "/content/gguf", "/content/models/pruned"
REF, DEPLOY = cfg["reference_quant"], cfg["deploy_quant"]

def done(name):
    return os.path.exists(f"results/eval/{name}.json") and os.path.exists(f"results/bench/{name}.json")

def run_variant(hf_dir, stem, quants):
    todo = [q for q in quants if not done(f"{stem}-{q}")]
    if not todo:
        print("건너뜀:", stem); return
    QS = " ".join(todo)
    !python scripts/03_make_gguf.py --hf-dir {hf_dir} --name {stem} --out-dir {GGUF} --quant {QS}
    for q in todo:
        g = f"{GGUF}/{stem}-{q}.gguf"
        !python scripts/05_eval.py --gguf {g} --name {stem}-{q}
        !python scripts/06_bench.py --gguf {g} --name {stem}-{q}
        os.remove(g)
        push_results()'''),
    md("## 기준: 원본"),
    code("run_variant(SRC, \"original\", [REF, DEPLOY])"),
    md("## 칼질 (+ 기준 변형은 양자화 비교까지)"),
    code(r'''base = cfg["quantization"]["base_variant"]
for v in cfg["pruning"]["variants"]:
    name = v["name"]
    quants = [REF] + (cfg["quantization"]["types"] if name == base else [])
    if all(done(f"{name}-{q}") for q in quants):
        print("건너뜀:", name); continue
    !python scripts/02_prune.py --variant {name} --src {SRC} --out-dir {PRUNED}
    run_variant(f"{PRUNED}/{name}", name, quants)
    shutil.rmtree(f"{PRUNED}/{name}")'''),
    md("## 중간 결과 표"),
    code('!python scripts/07_econ.py --measured-gpu "L4 (Colab)"\npush_results()'),
], gpu="L4"))

# ------------------------------------------------------------------ 03
write("03_lora.ipynb", notebook([
    md("""
# 03. 한국어 저랭크(LoRA) 추가학습 + 평가 (A100 40GB, 약 2시간 / 약 10~12유닛)
`ko-k32`(전문가 32개)에 한국어 지시문(KULLM-v2, 칼질 보정에 안 쓴 6천 개)으로 LoRA를 학습합니다.
- 어텐션 + 공유 전문가 + 남은 전문가에 LoRA, 라우터는 고정
- 학습 후 합쳐서 GGUF(Q6_K, Q4_K_M)로 만들고 **이 노트북에서 바로 평가**합니다
- 속도·전력은 같은 크기의 `ko-k32` 값을 그대로 씁니다 (구조와 크기가 같음)
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "38").replace("{gpu}", "A100")),
    code("!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
base = cfg["lora"]["base_variant"]
SRC = resolve_model_dir(cfg["model"]["id"])
!python scripts/02_prune.py --variant {base} --src {SRC} --out-dir /content/models/pruned
shutil.rmtree(SRC)   # 디스크 확보 (원본은 더 이상 필요 없음)'''),
    code(r'''OUT = f"/content/lora/{base}-lora"
!python scripts/04_train_lora.py --model /content/models/pruned/{base} --out {OUT}
shutil.copytree(f"{OUT}/adapter", f"{DRIVE}/lora/{base}-lora/adapter", dirs_exist_ok=True)
push_results()'''),
    code(r'''QS = " ".join(cfg["lora"]["quant_after"])
!python scripts/03_make_gguf.py --hf-dir {OUT}/merged --name {base}-lora --out-dir /content/gguf --quant {QS}
for q in cfg["lora"]["quant_after"]:
    g = f"/content/gguf/{base}-lora-{q}.gguf"
    !python scripts/05_eval.py --gguf {g} --name {base}-lora-{q}
    push_results()'''),
], gpu="A100"))

# ------------------------------------------------------------------ 04
write("04_report.ipynb", notebook([
    md("""
# 04. 최종 경제성 보고서 (CPU, 무료)
01~03의 결과(드라이브 `cfinetune/results/`)를 모아 표와 그래프를 만들고 Qwen3.8-Flash-Next로 확장 계산합니다.
"""),
    code(SETUP),
    code(DRIVE),
    code(r'''!python scripts/07_econ.py --measured-gpu "L4 (Colab)"
push_results()
from IPython.display import Markdown, Image, display
display(Markdown(open("results/report.md").read()))
for f in ("fig_size_vs_korean.png", "fig_energy.png", "fig_korean_expert_share.png"):
    if os.path.exists(f"results/{f}"): display(Image(f"results/{f}"))'''),
]))
