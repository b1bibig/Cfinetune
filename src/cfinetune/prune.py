"""체크포인트 파일에서 전문가를 잘라낸다 (GPU 필요 없음, 수 분).

지원하는 저장 방식
  - 전문가별 키   (ERNIE-4.5 등): model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight
                    -> 남길 j만 골라 0..K-1로 번호를 다시 붙인다
  - 묶음 텐서     (gpt-oss)      : model.layers.{i}.mlp.experts.{이름}  (0번 축이 전문가 축, MXFP4 블록 포함)
  - 라우터 가중치                 : mlp.gate.weight / mlp.router.{weight,bias}  (0번 축)
  - 라우터 보정 bias (ERNIE)      : mlp.moe_statics.e_score_correction_bias  ([1, E], 마지막 축)
남은 전문가의 가중치는 원본과 비트 단위로 같다 (프루닝 자체에는 수치 손실이 없음).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_P = r"^model\.layers\.(\d+)\.(?:mlp|feed_forward)\."
PER_EXPERT = re.compile(_P + r"experts\.(\d+)\.(.+)$")
FUSED = re.compile(_P + r"experts\.([A-Za-z_]+)$")
ROUTER = re.compile(_P + r"(?:gate\.weight|router\.weight|router\.bias)$")
LAST_AXIS = re.compile(_P + r"(?:moe_statics\.e_score_correction_bias|expert_bias)$")

EXPERT_KEYS = ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")
TOPK_KEYS = ("num_experts_per_tok", "moe_k", "top_k_experts", "experts_per_token")
WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _first(cfg: dict, keys) -> str:
    for k in keys:
        if k in cfg:
            return k
    raise ValueError(f"config.json에 {keys} 중 하나가 없음")


def _shard_map(src: Path) -> dict[str, list[str]]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        shards: dict[str, list[str]] = {}
        for key, shard in json.loads(index.read_text())["weight_map"].items():
            shards.setdefault(shard, []).append(key)
        return shards
    with safe_open(src / "model.safetensors", framework="pt") as f:
        return {"model.safetensors": list(f.keys())}


def moe_layer_ids(src: str | Path) -> list[int]:
    """라우터 가중치가 있는 층 = MoE 층."""
    ids = set()
    for keys in _shard_map(Path(src)).values():
        for k in keys:
            m = ROUTER.match(k)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def prune_checkpoint(src: str | Path, dst: str | Path, keep: dict[int, list[int]], info: dict | None = None) -> Path:
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    n_key, k_key = _first(cfg, EXPERT_KEYS), _first(cfg, TOPK_KEYS)
    n_orig, top_k = cfg[n_key], cfg[k_key]

    keep = {int(i): sorted(v) for i, v in keep.items()}
    sizes = {len(v) for v in keep.values()}
    if len(sizes) != 1:
        raise ValueError("모든 층에서 남길 전문가 수가 같아야 함 (config의 전문가 수는 하나)")
    n_keep = sizes.pop()
    if n_keep < top_k:
        raise ValueError(f"남길 전문가 수 {n_keep} < top_k {top_k}")
    layers = moe_layer_ids(src)
    if sorted(keep) != layers:
        raise ValueError(f"keep의 층 번호 {sorted(keep)} 가 MoE 층 {layers} 와 다름")
    remap = {i: {old: new for new, old in enumerate(v)} for i, v in keep.items()}
    idx = {i: torch.tensor(v, dtype=torch.long) for i, v in keep.items()}

    dst.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, 0
    for shard, keys in _shard_map(src).items():
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            meta = f.metadata() or {}
            for key in keys:
                if m := PER_EXPERT.match(key):
                    layer, j = int(m.group(1)), int(m.group(2))
                    if j not in remap[layer]:
                        continue
                    new_key = key.replace(f".experts.{j}.", f".experts.{remap[layer][j]}.", 1)
                    t = f.get_tensor(key)
                elif (m := FUSED.match(key)) or (m := ROUTER.match(key)):
                    new_key, t = key, f.get_tensor(key)
                    if t.shape[0] != n_orig:
                        raise ValueError(f"{key}: 0번 축 {t.shape[0]} != 전문가 수 {n_orig}")
                    t = t.index_select(0, idx[int(m.group(1))])
                elif m := LAST_AXIS.match(key):
                    new_key, t = key, f.get_tensor(key)
                    t = t.index_select(t.dim() - 1, idx[int(m.group(1))])
                else:
                    new_key, t = key, f.get_tensor(key)
                tensors[new_key] = t.contiguous()
                weight_map[new_key] = shard
                total += t.numel() * t.element_size()
        save_file(tensors, dst / shard, metadata={"format": "pt", **meta})
        del tensors

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    cfg[n_key] = n_keep
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for p in src.iterdir():
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json") \
                and not any(p.match(g) for g in WEIGHT_GLOBS):
            shutil.copy2(p, dst / p.name)
    (dst / "prune_info.json").write_text(json.dumps({
        "source": str(src), "num_experts_before": n_orig, "num_experts_after": n_keep,
        "keep": {str(k): v for k, v in sorted(keep.items())}, **(info or {})}, indent=1))
    return dst
