# 내 PC(가정용 GPU)에서 측정하기

8GB 게이밍 GPU(예: RTX 3070)에서 **원본은 통째로 안 들어가서 CPU와 나눠 돌고, 최종 모델은 통째로 들어가서 빠르게 도는지** 잰다.
측정만 하므로 학습·평가는 코랩에서 끝난 뒤에 한다.

| 파일 (드라이브 `cfinetune/gguf/`) | 크기 | 만드는 곳 |
|---|---|---|
| `original-Q4_K_M.gguf` | 약 13GB | 02 노트북 (`pc.copy_original`) |
| `ko-k32-vx-lora-Q4_K_M.gguf` | 약 7GB | 03 노트북 (`lora.to_drive`) |
| `ko-k32-vx-lora-Q3_K_M.gguf` | 약 6GB | 03 노트북 |

드라이브 공간이 부족하면 원본 파일은 PC로 받은 뒤 드라이브에서 지운다.

## 1. 준비 (윈도우, 한 번만)

1. 파이썬 3.11 이상 설치 (python.org, 설치할 때 "Add to PATH" 체크)
2. 저장소 받기 (PowerShell)
   ```
   git clone -b ccr-a463bea6-bpbyrh https://github.com/b1bibig/Cfinetune
   cd Cfinetune
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements-colab.txt
   ```
3. llama.cpp: 코랩과 같은 버전 [b11784](https://github.com/ggml-org/llama.cpp/releases/tag/b11784)에서
   `llama-b11784-bin-win-cuda-12.4-x64.zip`과 `cudart-llama-bin-win-cuda-12.4-x64.zip`을 받아 **같은 폴더**(예: `C:\llama`)에 푼다.
4. 위 GGUF 파일을 드라이브에서 받아 한 폴더(예: `D:\gguf`)에 둔다.

## 2. 측정

게임·브라우저 등 GPU를 쓰는 프로그램을 끄고 저장소 폴더에서 실행한다. 하나에 10분 안팎.

```
python scripts\06_bench.py --bin-dir C:\llama --gguf D:\gguf\ko-k32-vx-lora-Q3_K_M.gguf --name ko-k32-vx-lora-Q3_K_M
python scripts\06_bench.py --bin-dir C:\llama --gguf D:\gguf\ko-k32-vx-lora-Q4_K_M.gguf --name ko-k32-vx-lora-Q4_K_M
python scripts\06_bench.py --bin-dir C:\llama --gguf D:\gguf\original-Q4_K_M.gguf --name original-Q4_K_M --ngl 14
```

- `--ngl`은 GPU에 올릴 층 수(전체 28층). 원본은 8GB에 다 안 들어가므로 14부터 시작하고, `out of memory`가 나면 12, 10으로 낮춘다.
- 최종 Q4_K_M이 `out of memory`로 실패하면 `--ngl 26`처럼 조금 낮춰서 다시 재고, 그 사실을 보고서에 적는다.
- 전력은 GPU만 잰다. 원본은 CPU도 같이 일하므로 실제 에너지는 측정값보다 크다 (원본에 유리한 쪽이라 비교는 보수적).

## 3. 결과 올리기

`results\bench\`에 생긴 `*@NVIDIA_GeForce_RTX_3070.json` 파일들을 드라이브 `cfinetune/results/bench/`에 올린 뒤
05 노트북(보고서)을 실행하면 표에 함께 나온다.
