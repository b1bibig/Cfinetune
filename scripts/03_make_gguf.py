"""3단계: HF 체크포인트 -> GGUF 변환, 필요하면 다른 양자화 형식도 만든다.

- 원본 MXFP4 체크포인트(프루닝만 한 것)는 전문가가 MXFP4 그대로 옮겨진다 (손실 없음).
  나머지(어텐션 등)는 --outtype 으로 정한 형식(기본 BF16).
- LoRA를 합친 BF16 체크포인트는 전문가도 BF16으로 변환되므로 --quant MXFP4_MOE 로 다시 4비트로 줄인다.
- gpt-oss는 행렬 크기 2880이 256으로 나눠지지 않아 Q2_K/Q3_K/Q4_K 등을 요청해도 Q4_0/Q5_0 등으로 대체된다.

예) python scripts/03_make_gguf.py --hf-dir /content/models/pruned/ko-reap-k16 --out /content/gguf/ko-reap-k16.gguf --quant Q8_0 Q4_0
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True, help="기본 GGUF 경로 (예: .../ko-reap-k16.gguf)")
    ap.add_argument("--llama-dir", default="/content/llama.cpp")
    ap.add_argument("--outtype", default="bf16")
    ap.add_argument("--quant", nargs="*", default=[], help="추가로 만들 형식 (MXFP4_MOE, Q8_0, Q4_0, BF16 ...)")
    ap.add_argument("--keep-base", action="store_true", help="--quant 후에도 기본 GGUF를 지우지 않음")
    args = ap.parse_args()

    llama = Path(args.llama_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        run([sys.executable, llama / "convert_hf_to_gguf.py", args.hf_dir, "--outfile", out, "--outtype", args.outtype])
    for q in args.quant:
        dst = out.with_name(f"{out.stem}-{q}.gguf")
        run([llama / "build/bin/llama-quantize", "--allow-requantize", out, dst, q])
    if args.quant and not args.keep_base:
        out.unlink()
        print(f"기본 GGUF 삭제: {out}")


if __name__ == "__main__":
    main()
