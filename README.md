# Cfinetune: MoE 언어모델 한국어 특화 경량화

한국어를 공식 지원하지 않는 ERNIE-4.5-21B-A3B(MoE, 층마다 전문가 64개)는 BF16 약 44GB라 48GB 이상 GPU가 있어야 돌아간다.
**전문가 프루닝(칼질) → 한국어 어휘 확장 → 한국어 추가학습(LoRA) → 양자화**로
"원본보다 한국어를 잘하면서, 작은 GPU(T4·L4)에서 더 빠르고 싸게 도는" 모델을 만들고 경제성(비용·에너지)을 측정한다.
결과는 Qwen3.8-Flash-Next에 계산으로 확장한다. (gpt-oss-20b도 같은 코드로 지원)

- 방법론: [`docs/methodology.md`](docs/methodology.md)
- 실험 설정: [`configs/experiment.yaml`](configs/experiment.yaml)

## 실행 순서 (코랩)

| 노트북 | GPU | 하는 일 |
|---|---|---|
| `notebooks/00_smoke_test.ipynb` | CPU | 작은 무작위 모델로 코드 점검 (llama.cpp 변환·토크나이저 일치 포함) |
| `notebooks/01_calibrate.ipynb` | A100 80GB / G4 | 한국어 데이터로 전문가 중요도 측정 (+ 선택: 원본 BF16 속도·전력) |
| `notebooks/02_prune_quantize_eval.ipynb` | L4 | 칼질·양자화 변형 만들고 평가·측정 |
| `notebooks/03_lora.ipynb` | A100 | 한국어 어휘 확장 + 2단계 추가학습 + 평가 |
| `notebooks/04_small_gpu.ipynb` | T4, L4 | 최종 모델을 작은 GPU에서 속도·전력 측정 |
| `notebooks/05_report.ipynb` | CPU | 경제성 보고서 |
| (선택) [`docs/pc.md`](docs/pc.md) | 내 PC (예: RTX 3070 8GB) | 가정용 GPU에서 원본 vs 최종 측정 |

결과는 구글 드라이브 `MyDrive/cfinetune/results/`에 저장된다. 03의 학습 중간 저장도 드라이브(`cfinetune/ckpt/`)에 해서 끊겨도 이어서 학습한다.

## 로컬 테스트

```bash
pip install torch transformers peft datasets safetensors accelerate pyyaml requests pytest sentencepiece
python -m pytest tests -q
# llama.cpp 변환 테스트까지: LLAMA_CPP_DIR=/path/to/llama.cpp python -m pytest tests -q
```
