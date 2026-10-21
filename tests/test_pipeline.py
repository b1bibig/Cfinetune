import importlib.util
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from cfinetune.observe import MoEObserver, select_experts
from cfinetune.prune import prune_checkpoint

from conftest import ROOT


def _ids(n=2, s=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 50000, (n, s), generator=g)


def test_observer_matches_module_output(tiny_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    with MoEObserver(model, check_every=1) as obs, torch.no_grad():
        model.model(input_ids=_ids(), use_cache=False)
    d = obs.to_dict()
    for layer in d["layers"]:
        assert layer["max_rel_err"] < 1e-5, layer["max_rel_err"]
        assert layer["tokens"] == 48
        assert sum(layer["count"]) == 48 * d["top_k"]
        assert all(r >= 0 for r in layer["reap"])


def test_dead_experts_ranked_last_and_pruning_is_exact(tiny_dead_dir, tmp_path):
    model = AutoModelForCausalLM.from_pretrained(tiny_dead_dir, dtype=torch.float32).eval()
    ids = _ids(seed=3)
    with MoEObserver(model) as obs, torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
    stats = obs.to_dict()
    for crit in ("reap", "freq"):
        keep = select_experts(stats, crit, 6)
        assert all(1 not in v and 5 not in v for v in keep.values()), (crit, keep)

    dst = prune_checkpoint(tiny_dead_dir, tmp_path / "pruned", keep)
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["num_local_experts"] == 6
    pruned = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float32).eval()
    assert pruned.model.layers[0].mlp.experts.gate_up_proj.shape[0] == 6
    assert pruned.model.layers[0].mlp.router.weight.shape[0] == 6
    with torch.no_grad():
        out = pruned(input_ids=ids, use_cache=False).logits
    # 한 번도 선택되지 않던 전문가만 지웠으므로 출력이 같아야 함
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_pruning_changes_output_when_used_experts_removed(tiny_dir, tmp_path):
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    dst = prune_checkpoint(tiny_dir, tmp_path / "p4", {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})
    pruned = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float32).eval()
    # 남긴 전문가 가중치는 원본의 해당 행과 정확히 같아야 함
    torch.testing.assert_close(pruned.model.layers[1].mlp.experts.down_proj,
                               model.model.layers[1].mlp.experts.down_proj[4:8])
    torch.testing.assert_close(pruned.model.layers[0].mlp.router.bias, model.model.layers[0].mlp.router.bias[:4])


def test_prune_mxfp4_layout(tmp_path):
    """원본 MXFP4 체크포인트 키(_blocks/_scales, uint8)도 0번 축으로 잘리는지."""
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
        t[f"model.layers.{i}.self_attn.sinks"] = torch.randn(2)
    save_file(t, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps({"num_local_experts": E, "num_experts_per_tok": 4,
                                                 "num_hidden_layers": L, "quantization_config": {"quant_method": "mxfp4"}}))
    (src / "tokenizer.json").write_text("{}")
    keep = {0: [0, 2, 4, 6, 7], 1: [1, 2, 3, 4, 5]}
    dst = prune_checkpoint(src, tmp_path / "out", keep)
    out = {}
    for shard in set(json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"].values()):
        out.update(load_file(dst / shard))
    for i in range(L):
        idx = torch.tensor(keep[i])
        for k, v in t.items():
            if k.startswith(f"model.layers.{i}.mlp."):
                assert torch.equal(out[k], v.index_select(0, idx)), k
    assert torch.equal(out["model.embed_tokens.weight"], t["model.embed_tokens.weight"])
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["num_local_experts"] == 5 and cfg["quantization_config"]["quant_method"] == "mxfp4"
    assert (dst / "tokenizer.json").exists() and (dst / "prune_info.json").exists()


def test_prune_rejects_keep_below_topk(tiny_dir, tmp_path):
    with pytest.raises(ValueError):
        prune_checkpoint(tiny_dir, tmp_path / "bad", {0: [0], 1: [1]})


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lora_train_merge_reload(tiny_dir, tmp_path):
    from peft import get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    m = _load_script("04_train_lora.py")
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = m.load_for_training(tiny_dir, device_map="cpu").float()
    model.config.use_cache = False
    c = {"r": 4, "alpha": 8, "lora_experts": True}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, m.lora_config(c, model.config.num_hidden_layers))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("experts" in n for n in names) and any("q_proj" in n for n in names)
    assert not any("router" in n for n in names)

    ex = [{"prompt": "태양광 발전이란?", "response": "햇빛을 전기로 바꾸는 발전 방식입니다."}] * 8
    rows, dropped = m.tokenize_examples(tok, ex, 256)
    assert dropped == 0 and rows[0]["labels"][0] == -100 and rows[0]["labels"][-1] != -100
    targs = TrainingArguments(output_dir=str(tmp_path / "ck"), per_device_train_batch_size=4, max_steps=3,
                              learning_rate=1e-2, report_to="none", remove_unused_columns=False, use_cpu=True,
                              save_strategy="no", logging_steps=1)
    Trainer(model=model, args=targs, train_dataset=rows, data_collator=m.Collator(tok.pad_token_id)).train()

    ids = _ids(1, 16, seed=7)
    model.eval()
    with torch.no_grad():
        peft_logits = model(input_ids=ids).logits
    m.save_merged(model, tok, tiny_dir, tmp_path / "merged")
    cfg = json.loads((tmp_path / "merged" / "config.json").read_text())
    assert "quantization_config" not in cfg and cfg["num_local_experts"] == 8
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32).eval()
    with torch.no_grad():
        merged_logits = merged(input_ids=ids).logits
    torch.testing.assert_close(merged_logits, peft_logits, atol=1e-3, rtol=1e-3)
    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        assert not torch.allclose(base(input_ids=ids).logits, merged_logits, atol=1e-4)  # 학습이 실제로 반영됨


def test_prompts_match_chat_template():
    from transformers import AutoTokenizer

    from cfinetune.prompts import harmony_prefix, sft_text

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    msgs = [{"role": "user", "content": "안녕?"}]
    ref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="low")
    ours = harmony_prefix("안녕?")
    # 날짜 줄만 다르고 나머지는 같아야 함
    strip = lambda s: "\n".join(l for l in s.split("\n") if not l.startswith("Current date"))  # noqa: E731
    assert strip(ref) == strip(ours)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "반가워요."}], tokenize=False,
                                   reasoning_effort="low")
    head, tail = sft_text("안녕?", "반가워요.")
    assert strip(full) == strip(head + tail)


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
