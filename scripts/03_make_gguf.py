"""3단계: HF 체크포인트 -> GGUF 변환 -> 원하는 형식으로 양자화.

변환은 Q8_0으로 먼저 하고(디스크 절약), 거기서 각 형식으로 다시 양자화한다. 모든 변형에 같은 절차를 쓴다.
출력: {out_dir}/{name}-{형식}.gguf

예) python scripts/03_make_gguf.py --hf-dir /content/models/pruned/ko-k32 --name ko-k32 --quant Q6_K Q4_K_M Q3_K_M Q2_K
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="/content/gguf")
    ap.add_argument("--llama-dir", default="/content/llama.cpp")
    ap.add_argument("--quant", nargs="+", required=True, help="Q8_0 Q6_K Q4_K_M Q3_K_M Q2_K ...")
    args = ap.parse_args()

    llama, out = Path(args.llama_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{args.name}-Q8_0.gguf"
    if not base.exists():
        run([sys.executable, llama / "convert_hf_to_gguf.py", args.hf_dir, "--outfile", base, "--outtype", "q8_0"])
    for q in args.quant:
        if q.upper() == "Q8_0":
            continue
        dst = out / f"{args.name}-{q}.gguf"
        if not dst.exists():
            run([llama / "build/bin/llama-quantize", "--allow-requantize", base, dst, q])
    if "Q8_0" not in [q.upper() for q in args.quant]:
        base.unlink()
    for q in args.quant:
        p = out / f"{args.name}-{q}.gguf"
        print(f"{p.name}: {p.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
