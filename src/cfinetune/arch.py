"""모델 구조별 차이를 한 곳에 모은 어댑터 (ERNIE-4.5 MoE, gpt-oss).

각 어댑터는
  - moe_layers : {층 번호: MoE 모듈}
  - route      : 라우터가 고른 전문가 번호(top-k)와 최종 가중치
  - expert     : 전문가 하나의 출력 f_e(x)
  - shared     : 공유 전문가 출력 (없으면 None)
  - prompt     : 채팅 형식
  - lora       : LoRA 대상
를 제공한다. 계산식은 transformers 5.x 구현과 같게 맞췄고, observe.py의 재구성 검사로 확인한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ErnieMoe:
    """baidu/ERNIE-4.5-21B-A3B: 층마다 전문가 64개 중 6개 + 공유 전문가 2개, 0번 층은 일반 MLP.

    라우터: p = softmax(xWᵀ) (전체 64개) -> (p + 보정 bias)로 상위 6개 선택 -> 고른 p를 합이 1이 되게 재정규화.
    재정규화 덕분에 전문가를 지워도 남은 전문가의 가중치 계산 방식은 그대로다.
    """

    model_type = "ernie4_5_moe"

    @staticmethod
    def moe_layers(model):
        return {i: l.mlp for i, l in enumerate(model.model.layers) if hasattr(l.mlp, "experts")}

    @staticmethod
    def route(mlp, x):
        gate = mlp.gate
        logits = F.linear(x.float(), gate.weight.float())
        p = torch.softmax(logits, dim=-1)
        bias = gate.moe_statics.e_score_correction_bias.squeeze().float()
        _, top_i = torch.topk(p + bias, gate.top_k, dim=-1)
        w = torch.gather(p, -1, top_i)
        return top_i, w / w.sum(-1, keepdim=True).clamp_min(gate.norm_min)

    @staticmethod
    def expert(mlp, e, x):
        ex = mlp.experts
        gate, up = F.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
        return F.linear(ex.act_fn(gate) * up, ex.down_proj[e])

    @staticmethod
    def shared(mlp, x):
        return mlp.shared_experts(x) if mlp.shared_experts is not None else None

    # ---- 채팅 형식: <|begin_of_sentence|>User: ...\nAssistant: ...<|end_of_sentence|>
    @staticmethod
    def chat_prefix(user: str) -> str:
        return f"<|begin_of_sentence|>User: {user}\nAssistant: "

    END = "<|end_of_sentence|>"

    @staticmethod
    def lora_targets(model):
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # + 공유 전문가, 0번 층 MLP
        params = [f"{i}.mlp.experts.{p}" for i in ErnieMoe.moe_layers(model) for p in ("gate_up_proj", "down_proj")]
        return modules, params


class GptOss:
    """openai/gpt-oss-20b: 층마다 전문가 32개 중 4개, 공유 전문가 없음. 라우터는 top-4 로짓에 softmax."""

    model_type = "gpt_oss"

    @staticmethod
    def moe_layers(model):
        return {i: l.mlp for i, l in enumerate(model.model.layers)}

    @staticmethod
    def route(mlp, x):
        r = mlp.router
        logits = F.linear(x, r.weight, r.bias)
        top_v, top_i = torch.topk(logits, r.top_k, dim=-1)
        return top_i, torch.softmax(top_v.float(), dim=-1)

    @staticmethod
    def expert(mlp, e, x):
        ex = mlp.experts
        gate_up = x @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
        return ex._apply_gate(gate_up) @ ex.down_proj[e] + ex.down_proj_bias[e]

    @staticmethod
    def shared(mlp, x):
        return None

    FIXED_DATE = "2026-01-01"  # transformers 템플릿은 오늘 날짜를 넣어 날마다 프롬프트가 달라지므로 고정
    _SYSTEM = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
               "Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: low\n\n"
               "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")

    @staticmethod
    def chat_prefix(user: str) -> str:
        # 분석 채널을 건너뛰고 최종 답 채널을 바로 연다
        return (GptOss._SYSTEM.format(date=GptOss.FIXED_DATE)
                + f"<|start|>user<|message|>{user}<|end|><|start|>assistant<|channel|>final<|message|>")

    END = "<|return|>"

    @staticmethod
    def lora_targets(model):
        params = [f"{i}.mlp.experts.{p}" for i in range(model.config.num_hidden_layers) for p in ("gate_up_proj", "down_proj")]
        return ["q_proj", "k_proj", "v_proj", "o_proj"], params


ARCHS = {a.model_type: a for a in (ErnieMoe, GptOss)}


def get_arch(model_type: str):
    if model_type not in ARCHS:
        raise ValueError(f"지원하지 않는 구조: {model_type} (지원: {list(ARCHS)})")
    return ARCHS[model_type]


def arch_of(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    return get_arch(cfg.model_type)
