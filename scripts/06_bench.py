"""6단계: 같은 GPU에서 속도(토큰/초), 전력(W), VRAM을 잰다.

다른 프로그램이 GPU를 쓰지 않는 상태에서 실행할 것 (llama-server가 떠 있으면 안 됨).
출력: results/bench/{name}.json
"""

import argparse
import json

from _common import RESULTS

from cfinetune.llama import llama_bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin-dir", default="/content/llama.cpp/build/bin")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    r = llama_bench(args.bin_dir, args.gguf, reps=args.reps)
    r["name"] = args.name
    dst = RESULTS / "bench" / f"{args.name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
