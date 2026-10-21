# Cfinetune: MoE 언어모델 한국어 특화 경량화

한국어를 공식 지원하지 않는 ERNIE-4.5-21B-A3B(MoE, 층마다 전문가 64개)를
**전문가 프루닝(칼질) → 양자화 → 한국어 저랭크 추가학습(LoRA)** 순서로 경량화해서
"원본보다 작은데 한국어는 더 잘하는" 모델을 만들고, 경제성(비용·에너지)을 측정한다.
결과는 Qwen3.8-Flash-Next에 계산으로 확장한다. (gpt-oss-20b도 같은 코드로 지원)

- 방법론: [`docs/methodology.md`](docs/methodology.md)
- 실험 설정: [`configs/experiment.yaml`](configs/experiment.yaml)

## 실행 순서 (코랩)

| 노트북 | GPU | 하는 일 |
|---|---|---|
| `notebooks/00_smoke_test.ipynb` | CPU | 작은 무작위 모델로 코드 점검 |
| `notebooks/01_calibrate.ipynb` | A100 80GB / G4 | 한국어 데이터로 전문가 중요도 측정 |
| `notebooks/02_prune_quantize_eval.ipynb` | L4 | 칼질·양자화 변형 만들고 평가·측정 |
| `notebooks/03_lora.ipynb` | A100 40GB | 한국어 LoRA 추가학습 + 평가 |
| `notebooks/04_report.ipynb` | CPU | 경제성 보고서 |

결과는 구글 드라이브 `MyDrive/cfinetune/results/`에 저장된다.

## 로컬 테스트

```bash
pip install torch transformers peft datasets safetensors accelerate pyyaml requests pytest sentencepiece
python -m pytest tests -q
# llama.cpp 변환 테스트까지: LLAMA_CPP_DIR=/path/to/llama.cpp python -m pytest tests -q
```
