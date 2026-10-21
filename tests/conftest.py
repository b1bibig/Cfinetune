"""CPU에서 도는 작은 무작위 MoE 모델(ERNIE-4.5, gpt-oss 구조)로 파이프라인 코드를 검증한다."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TOKENIZERS = {"ernie4_5_moe": "baidu/ERNIE-4.5-21B-A3B-PT", "gpt_oss": "openai/gpt-oss-20b"}


def tiny_model(kind: str):
    if kind == "ernie4_5_moe":
        from transformers import Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM

        cfg = Ernie4_5_MoeConfig(vocab_size=103424, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                                 num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, moe_num_experts=8,
                                 moe_k=2, moe_num_shared_experts=2, moe_layer_start_index=1, moe_layer_end_index=2,
                                 max_position_embeddings=4096, tie_word_embeddings=True)
        return Ernie4_5_MoeForCausalLM(cfg)
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(vocab_size=201088, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_local_experts=8,
                       num_experts_per_tok=2, layer_types=["sliding_attention", "full_attention"], sliding_window=8,
                       max_position_embeddings=4096, pad_token_id=199999, eos_token_id=200002)
    return GptOssForCausalLM(cfg)


def build_tiny(dst: Path, kind: str = "ernie4_5_moe", dtype=torch.float32, seed=0, dead_experts=()):
    """dead_experts: 라우터가 절대 선택하지 않게 만든 전문가 번호."""
    from cfinetune.arch import arch_of

    torch.manual_seed(seed)
    model = tiny_model(kind).to(dtype)
    with torch.no_grad():
        for mlp in arch_of(model).moe_layers(model).values():
            if kind == "ernie4_5_moe":
                mlp.gate.weight.normal_(0, 0.5)
                for e in dead_experts:
                    mlp.gate.moe_statics.e_score_correction_bias[0, e] = -1e4
            else:
                mlp.router.weight.normal_(0, 0.5)
                mlp.router.bias.normal_(0, 0.1)
                for e in dead_experts:
                    mlp.router.bias[e] = -1e4
                mlp.experts.gate_up_proj_bias.normal_(0, 0.02)
                mlp.experts.down_proj_bias.normal_(0, 0.02)
    model.save_pretrained(dst)
    # 원본 저장소의 토크나이저 파일을 그대로 받아 둔다 (llama.cpp 변환기는 ERNIE의 tokenizer.model이 필요)
    from huggingface_hub import snapshot_download
    snapshot_download(TOKENIZERS[kind], local_dir=dst, allow_patterns=[
        "tokenizer*", "*.model", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"])
    return model.eval()


@pytest.fixture(scope="session", params=["ernie4_5_moe", "gpt_oss"])
def kind(request):
    return request.param


@pytest.fixture(scope="session")
def tiny_dirs(tmp_path_factory):
    cache = {}

    def get(kind, dead=()):
        key = (kind, tuple(dead))
        if key not in cache:
            d = tmp_path_factory.mktemp(f"{kind}{'_dead' if dead else ''}")
            build_tiny(d, kind, dead_experts=dead)
            cache[key] = d
        return cache[key]

    return get
