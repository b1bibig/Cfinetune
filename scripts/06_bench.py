"""6단계: 속도(토큰/초), 전력(W), VRAM을 잰다.

다른 프로그램이 GPU를 쓰지 않는 상태에서 실행할 것 (llama-server가 떠 있으면 안 됨).
같은 변형을 여러 GPU에서 잴 수 있도록 결과 파일 이름에 GPU를 붙인다.
출력: results/bench/{name}@{GPU}.json  예) original-BF16@A100-80GB.json, ko-k32-vx-lora-Q4_K_M@T4.json
"""

import argparse
import json

from _common import RESULTS, load_config

from cfinetune.econ import gpu_catalog, gpu_label, gpu_tag
from cfinetune.llama import llama_bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin-dir", default="/content/llama.cpp/build/bin")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    r = llama_bench(args.bin_dir, args.gguf, reps=args.reps)
    r["name"] = args.name
    r["gpu_label"] = gpu_label(r["gpu"], gpu_catalog(load_config(args.config)))
    dst = RESULTS / "bench" / f"{args.name}@{gpu_tag(r['gpu_label'])}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
