"""CPU에서 도는 작은 무작위 gpt-oss 모델로 파이프라인 코드를 검증한다 (실제 가중치 불필요)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TOKENIZER_ID = "openai/gpt-oss-20b"


def tiny_config(**kw):
    from transformers import GptOssConfig

    base = dict(vocab_size=201088, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_local_experts=8,
                num_experts_per_tok=2, layer_types=["sliding_attention", "full_attention"], sliding_window=8,
                max_position_embeddings=4096, pad_token_id=199999, eos_token_id=200002)
    base.update(kw)
    return GptOssConfig(**base)


def build_tiny(dst: Path, dtype=torch.float32, seed=0, dead_experts=()):
    from transformers import AutoTokenizer, GptOssForCausalLM

    torch.manual_seed(seed)
    model = GptOssForCausalLM(tiny_config()).to(dtype)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.weight.normal_(0, 0.5)
            layer.mlp.router.bias.normal_(0, 0.1)
            for e in dead_experts:  # 절대 선택되지 않는 전문가
                layer.mlp.router.bias[e] = -1e4
            for p in (layer.mlp.experts.gate_up_proj_bias, layer.mlp.experts.down_proj_bias):
                p.normal_(0, 0.02)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(TOKENIZER_ID).save_pretrained(dst)
    return model.eval()


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    build_tiny(d)
    return d


@pytest.fixture(scope="session")
def tiny_dead_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_dead")
    build_tiny(d, dead_experts=(1, 5))
    return d
