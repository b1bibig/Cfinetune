"""2단계: 측정한 중요도로 전문가를 잘라낸 체크포인트를 만든다 (CPU로 충분, 수 분).

원본 가중치를 그대로 잘라 옮기므로 남은 전문가의 가중치는 원본과 비트 단위로 같다.
예) python scripts/02_prune.py --variant ko-k32
    python scripts/02_prune.py --all
"""

import argparse
import json
from pathlib import Path

from _common import RESULTS, load_config, resolve_model_dir

from cfinetune.observe import select_experts
from cfinetune.prune import prune_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None, help="원본 HF 체크포인트 폴더 (기본: config의 model.id 다운로드)")
    ap.add_argument("--stats-dir", default=str(RESULTS / "calib"))
    ap.add_argument("--out-dir", default="/content/models/pruned")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--variant", nargs="+")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    variants = {v["name"]: v for v in cfg["pruning"]["variants"]}
    names = list(variants) if args.all else args.variant
    src = resolve_model_dir(args.src or cfg["model"]["id"])

    for name in names:
        v = variants[name]
        stats = json.loads((Path(args.stats_dir) / f"stats-{v['calib']}.json").read_text())
        keep = select_experts(stats, v["criterion"], v["keep"])
        dst = prune_checkpoint(src, Path(args.out_dir) / name, keep, info={"variant": v})
        (RESULTS / "keep").mkdir(parents=True, exist_ok=True)
        (RESULTS / "keep" / f"{name}.json").write_text(json.dumps({str(k): x for k, x in keep.items()}))
        print(f"{name}: {stats['num_experts']} -> {v['keep']} 전문가, 저장 {dst}")


if __name__ == "__main__":
    main()
