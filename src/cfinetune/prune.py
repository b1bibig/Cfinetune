"""체크포인트에서 전문가를 잘라낸다 (GPU 필요 없음).

gpt-oss의 전문가 관련 텐서는 전부 0번 축이 전문가 축이다.
  원본 MXFP4: experts.{gate_up_proj,down_proj}_{blocks,scales}, experts.*_bias, router.{weight,bias}
  BF16     : experts.{gate_up_proj,down_proj}, experts.*_bias, router.{weight,bias}
그래서 0번 축에서 남길 전문가만 index_select 하면 된다. MXFP4 블록/스케일을 그대로 옮기므로
프루닝 자체는 수치 손실이 전혀 없다(남은 전문가의 가중치는 비트 단위로 동일).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.(experts\.[A-Za-z_]+|router\.(?:weight|bias))$")
WEIGHT_FILES = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _shard_map(src: Path) -> dict[str, list[str]]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            shards.setdefault(shard, []).append(key)
        return shards
    single = src / "model.safetensors"
    with safe_open(single, framework="pt") as f:
        return {single.name: list(f.keys())}


def prune_checkpoint(src: str | Path, dst: str | Path, keep: dict[int, list[int]], info: dict | None = None) -> Path:
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    n_orig, top_k = cfg["num_local_experts"], cfg["num_experts_per_tok"]
    n_keep = {len(v) for v in keep.values()}
    if len(n_keep) != 1:
        raise ValueError("모든 층에서 남길 전문가 수가 같아야 함 (config의 num_local_experts는 하나)")
    n_keep = n_keep.pop()
    if n_keep < top_k:
        raise ValueError(f"남길 전문가 수 {n_keep} < top_k {top_k}")
    if set(keep) != set(range(cfg["num_hidden_layers"])):
        raise ValueError("keep에 모든 층 번호가 있어야 함")

    dst.mkdir(parents=True, exist_ok=True)
    keep_idx = {int(k): torch.tensor(sorted(v), dtype=torch.long) for k, v in keep.items()}
    weight_map, total = {}, 0
    for shard, keys in _shard_map(src).items():
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            meta = f.metadata() or {}
            for key in keys:
                t = f.get_tensor(key)
                m = EXPERT_KEY.match(key)
                if m:
                    if t.shape[0] != n_orig:
                        raise ValueError(f"{key}: 0번 축 {t.shape[0]} != 전문가 수 {n_orig}")
                    t = t.index_select(0, keep_idx[int(m.group(1))]).contiguous()
                tensors[key] = t
                weight_map[key] = shard
                total += t.numel() * t.element_size()
        save_file(tensors, dst / shard, metadata={"format": "pt", **meta})
        del tensors

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    cfg["num_local_experts"] = n_keep
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for p in src.iterdir():
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json") \
                and not any(p.match(g) for g in WEIGHT_FILES):
            shutil.copy2(p, dst / p.name)

    (dst / "prune_info.json").write_text(json.dumps({
        "source": str(src), "num_experts_before": n_orig, "num_experts_after": n_keep,
        "keep": {str(k): v for k, v in sorted(keep.items())}, **(info or {})}, indent=1))
    return dst
