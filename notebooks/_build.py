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
%pip install -q -r requirements-colab.txt
%pip uninstall -y -q torchao   # 코랩의 옛 torchao가 있으면 PEFT가 LoRA 합치기·불러오기에서 멈춤 (이 실험은 안 씀)'''

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


AUTO_OFF = r'''# 다 끝나면 런타임을 자동으로 끈다 (켜 둔 채 두면 아무것도 안 해도 유닛이 나감). 결과는 이미 드라이브에 있음
# 끄지 않고 더 볼 게 있으면 아래를 False로
AUTO_OFF = True
if AUTO_OFF:
    push_results()
    drive.flush_and_unmount()   # 드라이브 업로드가 끝날 때까지 기다린다 (안 하면 방금 복사한 큰 파일이 사라질 수 있음)
    from google.colab import runtime
    runtime.unassign()'''


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
작은 무작위 MoE 모델(ERNIE·gpt-oss 구조)로 파이프라인 코드가 이 환경에서 잘 도는지 확인합니다.
llama.cpp를 빌드해서 GGUF 변환과 어휘 확장 토크나이저 테스트까지 돌립니다 (CPU 빌드 약 10분).
전부 `passed`가 나오면 다음 노트북으로 넘어가세요. 코랩 유닛을 쓰지 않도록 **CPU 런타임**으로 실행하세요.
"""),
    code(SETUP),
    code("!LLAMA_BACKEND=cpu bash scripts/build_llama_cpp.sh /content/llama.cpp"),
    code("!LLAMA_CPP_DIR=/content/llama.cpp python -m pytest tests -q"),
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
    md("""
## (선택) 원본 BF16의 속도·전력 측정 (약 45분 / 약 6유닛)
원본을 그대로(BF16, 약 44GB) 돌리려면 48GB 이상 GPU가 필요합니다. 원본을 이미 받아 둔 이 세션에서
속도·전력을 재 두면 "큰 GPU의 원본 vs 작은 GPU의 최종 모델"을 직접 비교할 수 있습니다.
"""),
    code("# llama.cpp 빌드 (약 15분, 드라이브에 캐시)\n!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''!mkdir -p /content/gguf && df -h /content | tail -1
!python /content/llama.cpp/convert_hf_to_gguf.py /content/models/baidu__ERNIE-4.5-21B-A3B-PT --outfile /content/gguf/original-BF16.gguf --outtype bf16
!python scripts/06_bench.py --gguf /content/gguf/original-BF16.gguf --name original-BF16
push_results()'''),
    md("## 런타임 끄기"),
    code(AUTO_OFF),
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
    import glob
    return os.path.exists(f"results/eval/{name}.json") and bool(glob.glob(f"results/bench/{name}@*.json"))

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
        if stem == "original" and q == cfg["pc"]["copy_original"] and os.path.exists(g):   # 내 PC 측정용 (docs/pc.md)
            try:
                os.makedirs(f"{DRIVE}/gguf", exist_ok=True)
                shutil.copy(g, f"{DRIVE}/gguf/")
            except OSError as e:
                print("드라이브에 복사 못 함 (공간 부족?):", e)
        if os.path.exists(g):
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
    shutil.rmtree(f"{PRUNED}/{name}", ignore_errors=True)'''),
    md("## 중간 결과 표"),
    code('!python scripts/07_econ.py\npush_results()'),
    md("## 런타임 끄기"),
    code(AUTO_OFF),
], gpu="L4"))

# ------------------------------------------------------------------ 03
write("03_lora.ipynb", notebook([
    md("""
# 03. 한국어 어휘 확장 + 추가학습 + 평가 (A100, 시간은 config의 학습량에 따라)
`ko-k32`(전문가 32개)를 한국어로 개조합니다.
1. **어휘 확장 (CPU, 수 분)**: 한국어 BPE를 학습해 한국어 토큰 `vocab.n_new`개를 덧붙입니다. 같은 한국어를 약 40% 적은 토큰으로 처리합니다.
2. **1단계 학습**: 모델은 고정하고 새 토큰 임베딩만 한국어 위키로 학습합니다.
3. **2단계 학습**: 한국어 대화(smol-koreantalk 6천 개)로 LoRA (어텐션 + 공유 전문가 + 남은 전문가, 라우터 고정) + 새 토큰 임베딩.
4. 합쳐서 GGUF(Q6_K, Q4_K_M, Q3_K_M)로 만들고 바로 평가합니다. Q4_K_M·Q3_K_M은 드라이브에 저장해 04 노트북(T4·L4)과 내 PC에서 잽니다.

학습 중간 저장은 드라이브(`cfinetune/ckpt/`)에 하므로, 세션이 끊기면 **처음부터 다시 실행**하면 이어서 학습합니다.
드라이브에 약 15GB 여유가 필요합니다 (학습 중간 저장 + 최종 모델 Q4_K_M·Q3_K_M).
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "38").replace("{gpu}", "A100")),
    code("!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
base, VX = cfg["lora"]["base_variant"], cfg["vocab"]["enabled"]
NAME = f"{base}{'-vx' if VX else ''}-lora"
TRAIN_IN = f"/content/models/pruned/{base}{'-vx' if VX else ''}"
OUT, CKPT = f"/content/lora/{NAME}", f"{DRIVE}/ckpt/{NAME}"
QUANTS, DEPLOY = cfg["lora"]["quant_after"], cfg["deploy_quant"]

def done():
    return all(os.path.exists(f"results/eval/{NAME}-{q}.json") for q in QUANTS)

def merged():
    return os.path.exists(f"{OUT}/merged/config.json")

ADAPTER = f"{DRIVE}/lora/{NAME}/adapter"   # 2단계까지 끝난 어댑터 (있으면 다시 학습하지 않고 합치기만)
need_gguf = [q for q in cfg["lora"]["to_drive"] if not os.path.exists(f"{DRIVE}/gguf/{NAME}-{q}.gguf")]
rebuild = bool(need_gguf) and os.path.exists(f"{ADAPTER}/adapter_config.json")   # 학습은 끝났는데 GGUF가 없음
import json
print(NAME, "- 평가까지 끝남" if done() else "", f"- 드라이브에 없는 GGUF {need_gguf}를 어댑터로 다시 만듦" if rebuild else "")'''),
    md("## 1. 칼질 + 어휘 확장 (CPU)"),
    code(r'''if (not done() or rebuild) and not merged() and not os.path.exists(f"{TRAIN_IN}/config.json"):
    SRC = resolve_model_dir(cfg["model"]["id"])
    !python scripts/02_prune.py --variant {base} --src {SRC} --out-dir /content/models/pruned
    shutil.rmtree(SRC, ignore_errors=True)   # 디스크 확보 (원본은 더 이상 필요 없음)
    if VX:
        vjson = f"results/vocab/{os.path.basename(TRAIN_IN)}.json"
        before = json.load(open(vjson)) if os.path.exists(vjson) else None
        !python scripts/02b_extend_vocab.py --src /content/models/pruned/{base} --dst {TRAIN_IN}
        shutil.rmtree(f"/content/models/pruned/{base}", ignore_errors=True)
        if before is not None:   # 학습해 둔 임베딩은 토큰 번호에 묶여 있으므로, 다시 만든 어휘가 그때와 같아야 한다
            after = json.load(open(vjson))
            same = all(before.get(k) == after.get(k) for k in ("first_pieces", "tok_per_char", "sample"))
            print("어휘가 이전과 같음" if same else "주의: 어휘가 이전과 다름 (저장된 학습 결과가 맞지 않음)")
            assert same or not rebuild, "어휘가 달라서 저장된 어댑터를 쓸 수 없음"
    push_results()'''),
    md("""
## 시범 학습: 속도 재고 1단계 학습량 정하기 (약 10분)
각 단계를 5스텝만 돌려 속도와 최대 VRAM을 잽니다. 1단계(새 토큰 임베딩) 학습량은
`vocab.stage1.hours`(기본 3시간) 안에 끝나도록 자동으로 정하고, 이어 학습할 때 같은 값을 쓰도록 드라이브에 저장합니다.
"""),
    code(r'''os.makedirs(CKPT, exist_ok=True)
plan = f"{CKPT}/stage1_tokens.json"
if os.path.exists(plan):
    STAGE1_TOKENS = json.load(open(plan))["n_tokens"]
    print("이전에 정한 1단계 학습량:", f"{STAGE1_TOKENS:,} 토큰")
elif not done() and not merged():
    !nohup nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > /content/mem.log 2>&1 &
    !python scripts/04_train_lora.py --model {TRAIN_IN} --out /content/pilot --ckpt-dir /content/pilot/ckpt --stage1-tokens 300000 --stage1-max-steps 5 --max-steps 5 --no-merge
    !pkill nvidia-smi
    s1 = cfg["vocab"]["stage1"]
    try:
        peak = max(float(x) for x in open("/content/mem.log") if x.strip())
        log = json.load(open("/content/pilot/train_log.json"))
        tps1 = log["stage1"]["metrics"]["train_samples_per_second"] * s1["seq_len"]   # 1단계 초당 토큰
        sps2 = log["metrics"]["train_samples_per_second"]                             # 2단계 초당 예제
        STAGE1_TOKENS = int(min(s1["n_tokens"], tps1 * 3600 * s1["hours"]) // 1_000_000 * 1_000_000) or 1_000_000
        json.dump({"n_tokens": STAGE1_TOKENS, "stage1_tokens_per_s": tps1, "stage2_examples_per_s": sps2,
                   "peak_vram_mib": peak}, open(plan, "w"))
        print(f"최대 VRAM {peak / 1024:.1f} GB / 1단계 초당 {tps1:,.0f} 토큰 -> {STAGE1_TOKENS:,} 토큰 "
              f"(약 {STAGE1_TOKENS / tps1 / 3600:.1f}시간) / 2단계 약 {cfg['lora']['n_examples'] / sps2 / 3600:.1f}시간")
    except Exception as e:   # 시범 학습이 실패해도 멈추지 않게 (멈추면 마지막 자동 종료가 안 됨)
        STAGE1_TOKENS = s1["n_tokens"]
        print("시범 학습 결과를 읽지 못함 (위 출력 확인):", e, "-> 기본값", STAGE1_TOKENS)
    shutil.rmtree("/content/pilot", ignore_errors=True)
    push_results()'''),
    md("## 2. 학습 (1단계 + 2단계, 끊기면 이 셀까지 다시 실행)"),
    code(r'''if rebuild and not merged():
    !python scripts/04_train_lora.py --model {TRAIN_IN} --out {OUT} --ckpt-dir {CKPT} --stage1-tokens {STAGE1_TOKENS} --adapter {ADAPTER}
elif not done() and not merged():
    # 2단계 중간 저장은 런타임 디스크에 (드라이브에 두면 갈아 끼운 저장분이 휴지통에 쌓여 드라이브가 꽉 참). 끊기면 2단계는 처음부터
    !python scripts/04_train_lora.py --model {TRAIN_IN} --out {OUT} --ckpt-dir {CKPT} --stage2-ckpt-dir /content/ckpt-stage2 --stage1-tokens {STAGE1_TOKENS}
    if merged():
        shutil.copytree(f"{OUT}/adapter", ADAPTER, dirs_exist_ok=True)
        print("드라이브에 어댑터 저장:", ADAPTER)
    push_results()'''),
    md("## 3. GGUF 변환 + 평가"),
    code(r'''todo = [q for q in QUANTS if not os.path.exists(f"results/eval/{NAME}-{q}.json")]
make = [q for q in QUANTS if q in todo or (rebuild and q in need_gguf)]
if make:
    QS = " ".join(make)
    !python scripts/03_make_gguf.py --hf-dir {OUT}/merged --name {NAME} --out-dir /content/gguf --quant {QS}
    for q in todo:
        g = f"/content/gguf/{NAME}-{q}.gguf"
        !python scripts/05_eval.py --gguf {g} --name {NAME}-{q} --tokenizer {OUT}/merged
        push_results()
os.makedirs(f"{DRIVE}/gguf", exist_ok=True)
for q in cfg["lora"]["to_drive"]:   # 04 노트북(작은 GPU)과 내 PC 측정용
    g = f"/content/gguf/{NAME}-{q}.gguf"
    if os.path.exists(g) and not os.path.exists(f"{DRIVE}/gguf/{NAME}-{q}.gguf"):
        free = shutil.disk_usage(DRIVE).free   # 드라이브 남은 용량 (휴지통도 용량을 차지함)
        if os.path.getsize(g) > free - 200e6:
            print(f"드라이브 공간 부족으로 {q} 저장 안 함: 필요 {os.path.getsize(g) / 1e9:.2f}GB, 남음 {free / 1e9:.2f}GB (휴지통 비우기)")
            continue
        try:
            shutil.copy(g, f"{DRIVE}/gguf/")
            print("드라이브에 저장:", f"{DRIVE}/gguf/{NAME}-{q}.gguf")
        except OSError as e:
            print("드라이브에 복사 못 함 (공간 부족?):", e)
print("드라이브 gguf:", {f: f"{os.path.getsize(f'{DRIVE}/gguf/{f}') / 1e9:.2f}GB" for f in sorted(os.listdir(f"{DRIVE}/gguf"))})
# 실제 업로드는 마지막 셀의 drive.flush_and_unmount()가 끝날 때까지 기다린다'''),
    md("## 런타임 끄기"),
    code(AUTO_OFF),
], gpu="A100"))

# ------------------------------------------------------------------ 04
write("04_small_gpu.ipynb", notebook([
    md("""
# 04. 최종 모델을 작은 GPU에서 재기 (T4 또는 L4, 약 30분 / 1유닛 이하)
03에서 드라이브에 저장한 최종 모델(배포 형식 Q4_K_M)의 속도·전력·메모리를 잽니다.
**T4 런타임에서 한 번, L4 런타임에서 한 번** 실행하면 두 GPU 결과가 모두 남습니다.
"""),
    code(SETUP),
    code(DRIVE),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code("!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import load_config
cfg = load_config()
NAME = f"{cfg['lora']['base_variant']}{'-vx' if cfg['vocab']['enabled'] else ''}-lora-{cfg['deploy_quant']}"
!cp {DRIVE}/gguf/{NAME}.gguf /content/
!python scripts/06_bench.py --gguf /content/{NAME}.gguf --name {NAME}
push_results()'''),
    md("## 런타임 끄기"),
    code(AUTO_OFF),
], gpu="T4"))

# ------------------------------------------------------------------ 06
write("06_lingbench.ipynb", notebook([
    md("""
# 06. 한국어 언어력 벤치 (L4, 약 2시간 / 약 3~4유닛)
점수 하나로 안 보이는 '한국어를 한국어답게 하는가'를 잽니다 (`scripts/08_lingbench.py`, 설명은 `src/cfinetune/lingbench.py`).

| 부분 | 내용 |
|---|---|
| 객관식 | KoBALT-700(언어학 24현상, 보기 10개), KoGEM(문법 1524), CLIcK 언어 영역(650) |
| 번역투 최소쌍 | 같은 뜻의 자연스러운 문장과 번역투 문장 210쌍(15유형). 자연스러운 쪽에 확률을 더 주는 비율 (SLOR) |
| 생성 통계 | 말투 지시 24개 + K2-Eval 90개에 답하게 하고 번역투 표지 빈도, 반말·존댓말 지시 준수, 반복, 한자 섞임, 끝맺음 |

비교 대상: 최종 모델(드라이브 GGUF), 원본 ERNIE, 학습 전 ko-k32, 한국어 모델 EXAONE 4.0 1.2B·32B.
끝난 모델은 건너뛰므로 세션이 끊기면 처음부터 다시 실행하세요.
"""),
    code(SETUP),
    code(DRIVE),
    code(GPU_CHECK.replace("{need}", "20").replace("{gpu}", "L4") + "\n!df -h /content | tail -1"),
    code("!bash scripts/build_llama_cpp.sh /content/llama.cpp {DRIVE}/cache"),
    code(r'''from _common import resolve_model_dir, load_config
cfg = load_config()
L = cfg["lingbench"]
GGUF = "/content/gguf"
os.makedirs(GGUF, exist_ok=True)

def done(name):
    return os.path.exists(f"results/lingbench/{name}.json")

def bench(g, name, ernie=True):
    eos = f"--eos-token-id {L['ernie_eos_token_id']}" if ernie else ""
    !python scripts/08_lingbench.py --gguf {g} --name {name} {eos}
    push_results()
    os.remove(g)'''),
    md("## 최종 모델 (드라이브)"),
    code(r'''NAME = f"{cfg['lora']['base_variant']}{'-vx' if cfg['vocab']['enabled'] else ''}-lora"
for q in cfg["lora"]["to_drive"]:
    if done(f"{NAME}-{q}"):
        print("건너뜀:", q); continue
    shutil.copy(f"{DRIVE}/gguf/{NAME}-{q}.gguf", GGUF)
    bench(f"{GGUF}/{NAME}-{q}.gguf", f"{NAME}-{q}")'''),
    md("## 한국어 기준 모델 (EXAONE 공식 GGUF)"),
    code(r'''from huggingface_hub import hf_hub_download
for r in L["reference"]:
    if done(r["name"]):
        print("건너뜀:", r["name"]); continue
    bench(hf_hub_download(r["repo"], r["file"], local_dir=GGUF), r["name"], ernie=False)'''),
    md("## 원본 ERNIE와 학습 전 ko-k32 (원본 BF16 약 44GB를 받아 만듦)"),
    code(r'''OQ, PQ, base = L["original_quant"], L["pruned_quant"], cfg["lora"]["base_variant"]
if not (done(f"original-{OQ}") and done(f"{base}-{PQ}")):
    SRC = resolve_model_dir(cfg["model"]["id"])
    if not done(f"original-{OQ}"):
        !python scripts/03_make_gguf.py --hf-dir {SRC} --name original --out-dir {GGUF} --quant {OQ}
        bench(f"{GGUF}/original-{OQ}.gguf", f"original-{OQ}")
    if not done(f"{base}-{PQ}"):
        !python scripts/02_prune.py --variant {base} --src {SRC} --out-dir /content/models/pruned
        !python scripts/03_make_gguf.py --hf-dir /content/models/pruned/{base} --name {base} --out-dir {GGUF} --quant {PQ}
        shutil.rmtree(f"/content/models/pruned/{base}", ignore_errors=True)
        bench(f"{GGUF}/{base}-{PQ}.gguf", f"{base}-{PQ}")
    shutil.rmtree(SRC, ignore_errors=True)'''),
    md("## 결과 표"),
    code(r'''import glob, json
from cfinetune import lingbench as lb
from IPython.display import Markdown, display
res = [json.load(open(f)) for f in sorted(glob.glob("results/lingbench/*.json"))]
md_table = lb.report(res)
open("results/lingbench/report.md", "w").write(md_table + "\n")
push_results()
display(Markdown(md_table))'''),
    md("## 런타임 끄기"),
    code(AUTO_OFF),
], gpu="L4"))

# ------------------------------------------------------------------ 05
write("05_report.ipynb", notebook([
    md("""
# 05. 최종 경제성 보고서 (CPU, 무료)
01~04의 결과(드라이브 `cfinetune/results/`)를 모아 표와 그래프를 만들고 Qwen3.8-Flash-Next로 확장 계산합니다.
"""),
    code(SETUP),
    code(DRIVE),
    code(r'''!python scripts/07_econ.py
push_results()
from IPython.display import Markdown, Image, display
display(Markdown(open("results/report.md").read()))
for f in ("fig_size_vs_korean.png", "fig_energy.png", "fig_energy_per_day.png", "fig_korean_expert_share.png"):
    if os.path.exists(f"results/{f}"): display(Image(f"results/{f}"))'''),
]))
