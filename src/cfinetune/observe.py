"""전문가 중요도(saliency) 관측기.

각 MoE 층에 forward hook을 걸어, 보정 데이터가 지나갈 때 전문가마다
  - count : 이 전문가가 top-k에 뽑힌 토큰 수 (빈도)
  - reap  : 뽑힌 토큰에서 (라우터 가중치 g) x (전문가 출력 크기 ||f(x)||) 의 평균
을 모은다. reap 점수는 REAP(Cerebras, 2025)의 router-weighted expert activation 기준이다.

gpt-oss 구조(transformers GptOssMLP):
  router logits = x W^T + b -> top-k -> softmax(top-k 값)
  expert e: gate_up = x @ gate_up_proj[e] + bias, (gate, up) 교차 분리, clamp, glu, @ down_proj[e] + bias
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


def _apply_gate(experts, gate_up: torch.Tensor) -> torch.Tensor:
    if hasattr(experts, "_apply_gate"):
        return experts._apply_gate(gate_up)
    limit, alpha = experts.limit, experts.alpha
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


def expert_forward(experts, e: int, x: torch.Tensor) -> torch.Tensor:
    gate_up = x @ experts.gate_up_proj[e] + experts.gate_up_proj_bias[e]
    return _apply_gate(experts, gate_up) @ experts.down_proj[e] + experts.down_proj_bias[e]


@dataclass
class LayerStats:
    num_experts: int
    count: torch.Tensor = field(init=False)
    reap_sum: torch.Tensor = field(init=False)
    gate_sum: torch.Tensor = field(init=False)
    norm_sum: torch.Tensor = field(init=False)
    tokens: int = 0
    max_rel_err: float = 0.0  # 재구성 검사: 우리가 계산한 MoE 출력 vs 실제 모듈 출력

    def __post_init__(self):
        z = lambda: torch.zeros(self.num_experts, dtype=torch.float64)  # noqa: E731
        self.count, self.reap_sum, self.gate_sum, self.norm_sum = z(), z(), z(), z()


class MoEObserver:
    """model.model.layers[i].mlp 에 hook을 걸어 통계를 모은다. with 문으로 사용."""

    def __init__(self, model, check_every: int = 50):
        self.layers = [layer.mlp for layer in model.model.layers]
        cfg = model.config
        self.top_k = cfg.num_experts_per_tok
        self.stats = [LayerStats(cfg.num_local_experts) for _ in self.layers]
        self.check_every = check_every
        self._calls = [0] * len(self.layers)
        self._handles = []

    def __enter__(self):
        for i, mlp in enumerate(self.layers):
            self._handles.append(mlp.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, i: int):
        @torch.no_grad()
        def hook(module, args, output):
            x = args[0]
            x = x.reshape(-1, x.shape[-1])
            router, experts = module.router, module.experts
            logits = F.linear(x, router.weight, router.bias)
            top_v, top_i = torch.topk(logits, self.top_k, dim=-1)
            gates = torch.softmax(top_v.float(), dim=-1)

            st = self.stats[i]
            check = self._calls[i] % self.check_every == 0
            self._calls[i] += 1
            recon = torch.zeros(x.shape, dtype=torch.float32, device=x.device) if check else None

            for e in range(st.num_experts):
                tok, slot = torch.where(top_i == e)
                if tok.numel() == 0:
                    continue
                out = expert_forward(experts, e, x[tok]).float()
                g = gates[tok, slot]
                n = out.norm(dim=-1)
                st.count[e] += tok.numel()
                st.reap_sum[e] += (g * n).sum().item()
                st.gate_sum[e] += g.sum().item()
                st.norm_sum[e] += n.sum().item()
                if check:
                    recon.index_add_(0, tok, out * g[:, None])
            st.tokens += x.shape[0]

            if check:
                y = output[0] if isinstance(output, tuple) else output
                y = y.reshape(-1, y.shape[-1]).float()
                err = ((recon - y).norm() / y.norm().clamp_min(1e-6)).item()
                st.max_rel_err = max(st.max_rel_err, err)

        return hook

    def to_dict(self) -> dict:
        layers = []
        for st in self.stats:
            c = st.count.clamp_min(1)
            layers.append({
                "tokens": st.tokens,
                "count": st.count.tolist(),
                "freq": (st.count / max(st.tokens, 1)).tolist(),
                "reap": torch.where(st.count > 0, st.reap_sum / c, torch.zeros_like(c)).tolist(),
                "mean_gate": (st.gate_sum / c).tolist(),
                "mean_norm": (st.norm_sum / c).tolist(),
                "max_rel_err": st.max_rel_err,
            })
        return {"top_k": self.top_k, "num_experts": self.stats[0].num_experts, "layers": layers}

    def save(self, path: str | Path, **meta):
        d = self.to_dict()
        d["meta"] = meta
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def select_experts(stats: dict, criterion: str, keep: int) -> dict[int, list[int]]:
    """층마다 점수 상위 keep개 전문가 번호(오름차순)를 고른다."""
    if keep < stats["top_k"]:
        raise ValueError(f"keep={keep} 는 top_k={stats['top_k']} 이상이어야 함")
    out = {}
    for i, layer in enumerate(stats["layers"]):
        score = torch.tensor(layer[criterion], dtype=torch.float64)
        # 점수가 같으면 빈도로, 그래도 같으면 번호로 정렬해 결과가 항상 같게
        freq = torch.tensor(layer["count"], dtype=torch.float64)
        order = sorted(range(len(score)), key=lambda e: (-score[e].item(), -freq[e].item(), e))
        out[i] = sorted(order[:keep])
    return out
