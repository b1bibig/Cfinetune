import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfinetune.arch import arch_of
from cfinetune.observe import MoEObserver, select_experts
from cfinetune.prune import moe_layer_ids, prune_checkpoint

from conftest import ROOT, TOKENIZERS


def _ids(n=2, s=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 50000, (n, s), generator=g)


def _load(d):
    return AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()


def test_observer_matches_module_output(kind, tiny_dirs):
    model = _load(tiny_dirs(kind))
    with MoEObserver(model, check_every=1) as obs, torch.no_grad():
        model.model(input_ids=_ids(), use_cache=False)
    d = obs.to_dict()
    assert d["model_type"] == kind and d["top_k"] == 2 and d["num_experts"] == 8
    assert [int(k) for k in d["layers"]] == list(arch_of(model).moe_layers(model))
    for layer in d["layers"].values():
        assert layer["max_rel_err"] < 1e-5, layer["max_rel_err"]
        assert layer["tokens"] == 48 and sum(layer["count"]) == 48 * 2


def test_dead_experts_pruned_exactly(kind, tiny_dirs, tmp_path):
    src = tiny_dirs(kind, dead=(1, 5))
    model = _load(src)
    ids = _ids(seed=3)
    with MoEObserver(model) as obs, torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
    stats = obs.to_dict()
    for crit in ("reap", "freq"):
        keep = select_experts(stats, crit, 6)
        assert all(1 not in v and 5 not in v for v in keep.values()), (crit, keep)

    dst = prune_checkpoint(src, tmp_path / "pruned", keep)
    pruned = _load(dst)
    assert arch_of(pruned).moe_layers(pruned).keys() == arch_of(model).moe_layers(model).keys()
    for mlp in arch_of(pruned).moe_layers(pruned).values():
        assert mlp.experts.gate_up_proj.shape[0] == 6
    with torch.no_grad():
        out = pruned(input_ids=ids, use_cache=False).logits
    # 한 번도 선택되지 않던 전문가만 지웠으므로 출력이 같아야 함
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_kept_experts_are_bit_identical(kind, tiny_dirs, tmp_path):
    src = tiny_dirs(kind)
    model = _load(src)
    layers = moe_layer_ids(src)
    keep = {i: [0, 2, 3, 7] for i in layers}
    pruned = _load(prune_checkpoint(src, tmp_path / "p", keep))
    a, b = arch_of(model).moe_layers(model), arch_of(pruned).moe_layers(pruned)
    for i in layers:
        torch.testing.assert_close(b[i].experts.down_proj, a[i].experts.down_proj[keep[i]], atol=0, rtol=0)
        torch.testing.assert_close(b[i].experts.gate_up_proj, a[i].experts.gate_up_proj[keep[i]], atol=0, rtol=0)
        if kind == "ernie4_5_moe":
            torch.testing.assert_close(b[i].gate.weight, a[i].gate.weight[keep[i]], atol=0, rtol=0)
            assert b[i].gate.moe_statics.e_score_correction_bias.shape == (1, 4)
    cfg = json.loads((tmp_path / "p" / "config.json").read_text())
    assert cfg.get("moe_num_experts", cfg.get("num_local_experts")) == 4


def test_prune_mxfp4_layout(tmp_path):
    """gpt-oss 원본 MXFP4 체크포인트 키(_blocks/_scales, uint8)도 0번 축으로 잘리는지."""
    src = tmp_path / "mx"
    src.mkdir()
    E, L = 8, 2
    rng = np.random.default_rng(0)
    t = {"model.embed_tokens.weight": torch.randn(10, 4)}
    for i in range(L):
        p = f"model.layers.{i}.mlp."
        t[p + "experts.gate_up_proj_blocks"] = torch.from_numpy(rng.integers(0, 255, (E, 6, 2, 16), dtype=np.uint8))
        t[p + "experts.gate_up_proj_scales"] = torch.from_numpy(rng.integers(0, 255, (E, 6, 2), dtype=np.uint8))
        t[p + "experts.gate_up_proj_bias"] = torch.randn(E, 6).bfloat16()
        t[p + "experts.down_proj_blocks"] = torch.from_numpy(rng.integers(0, 255, (E, 3, 2, 16), dtype=np.uint8))
        t[p + "experts.down_proj_scales"] = torch.from_numpy(rng.integers(0, 255, (E, 3, 2), dtype=np.uint8))
        t[p + "experts.down_proj_bias"] = torch.randn(E, 3).bfloat16()
        t[p + "router.weight"] = torch.randn(E, 4).bfloat16()
        t[p + "router.bias"] = torch.randn(E).bfloat16()
    save_file(t, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps({"num_local_experts": E, "num_experts_per_tok": 4,
                                                 "num_hidden_layers": L, "quantization_config": {"quant_method": "mxfp4"}}))
    keep = {0: [0, 2, 4, 6, 7], 1: [1, 2, 3, 4, 5]}
    dst = prune_checkpoint(src, tmp_path / "out", keep)
    out = load_file(dst / "model.safetensors")
    for i in range(L):
        for k, v in t.items():
            if k.startswith(f"model.layers.{i}.mlp."):
                assert torch.equal(out[k], v.index_select(0, torch.tensor(keep[i]))), k
    assert json.loads((dst / "config.json").read_text())["num_local_experts"] == 5


def test_prune_rejects_bad_keep(tiny_dirs, tmp_path):
    src = tiny_dirs("ernie4_5_moe")
    with pytest.raises(ValueError):  # top_k(2)보다 적음
        prune_checkpoint(src, tmp_path / "a", {1: [0], 2: [1]})
    with pytest.raises(ValueError):  # 0번 층은 MoE가 아님
        prune_checkpoint(src, tmp_path / "b", {0: [0, 1], 1: [0, 1], 2: [0, 1]})


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lora_train_merge_reload(kind, tiny_dirs, tmp_path):
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    m = _load_script("04_train_lora.py")
    src = tiny_dirs(kind)
    tok = AutoTokenizer.from_pretrained(src)
    model = m.load_for_training(src, device_map="cpu").float()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, m.lora_config({"r": 4, "alpha": 8, "lora_experts": True}, model))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("experts" in n for n in names) and any("q_proj" in n for n in names)
    assert not any(".gate.weight" in n or "router" in n for n in names)  # 라우터는 고정

    ex = [{"prompt": "태양광 발전이란?", "response": "햇빛을 전기로 바꾸는 발전 방식입니다."}] * 8
    rows, dropped = m.tokenize_examples(tok, arch_of(model), ex, 256)
    assert dropped == 0 and rows[0]["labels"][0] == -100 and rows[0]["labels"][-1] != -100
    targs = TrainingArguments(output_dir=str(tmp_path / "ck"), per_device_train_batch_size=4, max_steps=3,
                              learning_rate=1e-2, report_to="none", remove_unused_columns=False, use_cpu=True,
                              save_strategy="no", logging_steps=1)
    Trainer(model=model, args=targs, train_dataset=rows, data_collator=m.Collator(tok.pad_token_id)).train()

    ids = _ids(1, 16, seed=7)
    model.eval()
    with torch.no_grad():
        peft_logits = model(input_ids=ids).logits
    m.save_merged(model, tok, src, tmp_path / "merged")
    assert "quantization_config" not in json.loads((tmp_path / "merged" / "config.json").read_text())
    merged = _load(tmp_path / "merged")
    with torch.no_grad():
        merged_logits = merged(input_ids=ids).logits
        base_logits = _load(src)(input_ids=ids).logits
    torch.testing.assert_close(merged_logits, peft_logits, atol=1e-3, rtol=1e-3)
    assert not torch.allclose(base_logits, merged_logits, atol=1e-4)  # 학습이 실제로 반영됨
    if kind == "ernie4_5_moe":  # llama.cpp 변환기가 읽는 전문가별 키로 저장돼야 함
        idx = json.loads((tmp_path / "merged" / "model.safetensors.index.json").read_text()) \
            if (tmp_path / "merged" / "model.safetensors.index.json").exists() else None
        keys = idx["weight_map"] if idx else load_file(tmp_path / "merged" / "model.safetensors").keys()
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" in keys


def test_prompts_match_chat_templates():
    from cfinetune.arch import ErnieMoe, GptOss
    from cfinetune.prompts import sft_text

    msgs = [{"role": "user", "content": "안녕?"}]
    full = msgs + [{"role": "assistant", "content": "반가워요."}]

    tok = AutoTokenizer.from_pretrained(TOKENIZERS["ernie4_5_moe"])
    assert tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) == ErnieMoe.chat_prefix("안녕?")
    assert tok.apply_chat_template(full, tokenize=False) == "".join(sft_text(ErnieMoe, "안녕?", "반가워요."))

    tok = AutoTokenizer.from_pretrained(TOKENIZERS["gpt_oss"])
    strip = lambda s: "\n".join(l for l in s.split("\n") if not l.startswith("Current date"))  # noqa: E731
    ref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="low")
    assert strip(ref + "<|channel|>final<|message|>") == strip(GptOss.chat_prefix("안녕?"))
    ref_full = tok.apply_chat_template(full, tokenize=False, reasoning_effort="low")
    assert strip(ref_full) == strip("".join(sft_text(GptOss, "안녕?", "반가워요.")))


def test_econ_and_flash_next():
    from cfinetune import load_config
    from cfinetune.econ import cheapest_fit, flash_next_memory, gpu_catalog, variant_economics

    cfg = load_config()
    cat = gpu_catalog(cfg)
    assert cheapest_fit(cat, 10)["name"] == "T4 (Colab)"
    assert cheapest_fit(cat, 30)["vram_gb"] >= 30
    bench = {"tg_tps": 50.0, "tg_power_w": 70.0, "idle_w": 20.0, "pp_tps": 1000.0, "file_gb": 7.0, "peak_vram_gb": 8}
    r = variant_economics(bench, cat, "L4 (Colab)", 1.5)
    assert abs(r["wh_per_1k_tok"] - 70 / 50 * 1000 / 3600) < 1e-3
    l4 = next(g for g in cat if g["name"] == "L4 (Colab)")
    assert abs(r["krw_per_1m_tok_measured_gpu"] - l4["krw_per_hour"] / (50 * 3600) * 1e6) < 0.1
    full, half = flash_next_memory(1.0), flash_next_memory(0.5)
    assert full["keep_experts"] == 512 and half["keep_experts"] == 256
    assert 30 < half["gpu_gb"] < 40 < full["gpu_gb"]


@pytest.mark.skipif(not os.environ.get("LLAMA_CPP_DIR"), reason="LLAMA_CPP_DIR 미설정 (llama.cpp 빌드 필요)")
def test_gguf_conversion_and_quantization(tiny_dirs, tmp_path):
    """칼질한 ERNIE를 실제 llama.cpp로 변환·양자화하고 전문가 수가 반영되는지."""
    llama = Path(os.environ["LLAMA_CPP_DIR"])
    src = tiny_dirs("ernie4_5_moe")
    pruned = prune_checkpoint(src, tmp_path / "p", {i: [0, 1, 3, 4, 6, 7] for i in moe_layer_ids(src)})
    subprocess.run([sys.executable, ROOT / "scripts" / "03_make_gguf.py", "--hf-dir", pruned, "--name", "t",
                    "--out-dir", tmp_path / "g", "--llama-dir", llama, "--quant", "Q8_0", "Q4_K_M"],
                   check=True, capture_output=True)
    sys.path.insert(0, str(llama / "gguf-py"))
    from gguf import GGUFReader

    for q in ("Q8_0", "Q4_K_M"):
        r = GGUFReader(tmp_path / "g" / f"t-{q}.gguf")
        f = r.fields["ernie4_5-moe.expert_count"]
        assert int(f.parts[f.data[0]][0]) == 6
