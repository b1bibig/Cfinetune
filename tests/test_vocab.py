"""한국어 어휘 확장: 토크나이저 일치(허깅페이스 = SentencePiece = llama.cpp), 임베딩 늘리기, 2단계 학습."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfinetune.arch import arch_of
from cfinetune.vocab import (extend_checkpoint, extend_tokenizer_files, select_new_pieces, tokens_per_char,
                             train_korean_bpe)

from conftest import ROOT, TOKENIZERS

KO = """대한민국은 동아시아의 한반도 남부에 있는 민주공화국이다. 수도는 서울특별시이며 공용어는 한국어이다.
태양광 발전은 햇빛을 전기로 바꾸는 발전 방식이다. 풍력 발전은 바람의 힘으로 전기를 만든다.
에너지 저장 장치는 전력망에서 남는 전기를 저장했다가 필요할 때 공급해 전력망을 안정시킨다.
세종대왕은 백성이 쉽게 글을 읽고 쓸 수 있도록 훈민정음을 만들었다. 한글은 과학적인 문자로 평가받는다.
전기차 배터리를 오래 쓰려면 완전 방전을 피하고, 너무 덥거나 추운 곳에 오래 두지 않는 것이 좋다."""
CORPUS = [KO] * 30 + (ROOT / "configs" / "gen_prompts.txt").read_text(encoding="utf-8").splitlines() * 30
TEXTS = [KO, "에너지 절약을 실천하는 방법 3가지: 1) 대기전력 차단 2) LED 조명 3) 대중교통 이용.",
         "Mixed 한국어 and English, 숫자 12345, 기호 ~!@#  두 칸 띄어쓰기\n줄바꿈\t탭", "뷁 똠방각하 쀍쀍 🙂 漢字",
         "  앞 공백과 뒤 공백  "]
CHAT = "<|begin_of_sentence|>User: 안녕?\nAssistant: 반가워요.<|end_of_sentence|>"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ernie_tok_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ernie_tok")
    snapshot_download(TOKENIZERS["ernie4_5_moe"], local_dir=d, allow_patterns=[
        "tokenizer*", "*.model", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"])
    return d


@pytest.fixture(scope="module")
def new_pieces(ernie_tok_dir):
    base = AutoTokenizer.from_pretrained(ernie_tok_dir)
    bpe = train_korean_bpe(CORPUS, 600, limit_alphabet=300)
    pieces = select_new_pieces(bpe, base.get_vocab(), 200)
    assert len(pieces) > 50
    vocab = set(base.get_vocab())
    for i, p in enumerate(pieces):  # 부분 조각이 항상 먼저 들어가서 합치는 경로가 이어짐
        assert len(p) == 1 or any(p[:k] in vocab and p[k:] in vocab for k in range(1, len(p))), p
        vocab.add(p)
    return pieces


def test_merges_regeneration_matches_original(ernie_tok_dir):
    """원본 tokenizer.json의 merges를 SentencePiece 점수로 똑같이 다시 만들 수 있어야 같은 방식으로 늘릴 수 있다."""
    from transformers.tokenization_utils_base import generate_merges

    from cfinetune.vocab import _load_proto

    proto = _load_proto(ernie_tok_dir / "tokenizer.model")
    tj = json.loads((ernie_tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = {p.piece: i for i, p in enumerate(proto.pieces)}
    assert vocab == tj["model"]["vocab"]
    merges = generate_merges(vocab, {p.piece: p.score for p in proto.pieces})
    assert [list(m) for m in merges] == [list(m) if isinstance(m, list) else m.split(" ") for m in tj["model"]["merges"]]


def test_extended_tokenizer_matches_sentencepiece(ernie_tok_dir, new_pieces, tmp_path):
    import sentencepiece as spm

    info = extend_tokenizer_files(ernie_tok_dir, tmp_path, new_pieces)
    base, new = AutoTokenizer.from_pretrained(ernie_tok_dir), AutoTokenizer.from_pretrained(tmp_path)
    sp = spm.SentencePieceProcessor(model_file=str(tmp_path / "tokenizer.model"))
    assert info["first_id"] == 101304 and sp.get_piece_size() == info["vocab_size"] == len(new)
    for text in TEXTS:
        ids = new(text, add_special_tokens=False).input_ids
        assert ids == sp.encode(text), text
        assert new.decode(ids) == text
    assert any(i >= info["first_id"] for i in new(KO, add_special_tokens=False).input_ids)
    assert tokens_per_char(new, KO) < 0.8 * tokens_per_char(base, KO)
    # 특수 토큰 번호와 채팅 형식은 원본 그대로
    added = json.loads((ernie_tok_dir / "added_tokens.json").read_text())
    assert all(new.convert_tokens_to_ids(t) == i for t, i in added.items())
    assert new(CHAT, add_special_tokens=False).input_ids[:3] == base(CHAT, add_special_tokens=False).input_ids[:3]
    assert new.decode(new(CHAT, add_special_tokens=False).input_ids) == CHAT
    for p, i in zip(new_pieces, info["ids"]):
        assert new.convert_ids_to_tokens(i) == p


def test_extend_checkpoint_grows_embeddings(tiny_dirs, ernie_tok_dir, tmp_path):
    """새 토큰이 남는 칸(101304~103423)보다 많으면 임베딩을 늘리고, 새 행은 원래 조각 평균에서 시작."""
    src = tiny_dirs("ernie4_5_moe")
    base = AutoTokenizer.from_pretrained(src)
    vocab = base.get_vocab()
    pieces = [chr(c) for c in range(0xAC00, 0xD7A4) if chr(c) not in vocab and "▁" + chr(c) not in vocab][:2500]
    info = extend_checkpoint(src, tmp_path / "x", pieces, base)
    assert info["vocab_size_after"] == 103808 and json.loads((tmp_path / "x" / "config.json").read_text())["vocab_size"] == 103808

    old = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).get_input_embeddings().weight
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "x", dtype=torch.float32).eval()
    new = model.get_input_embeddings().weight
    torch.testing.assert_close(new[:101304], old[:101304])
    for p, i in list(zip(pieces, info["new_token_ids"]))[::250]:
        torch.testing.assert_close(new[i], old[base(p, add_special_tokens=False).input_ids].mean(0), atol=1e-6, rtol=0)
    tok = AutoTokenizer.from_pretrained(tmp_path / "x")
    ids = torch.tensor([tok(pieces[-1] + pieces[0] + "입니다", add_special_tokens=False).input_ids])
    assert ids.max() >= 103000
    with torch.no_grad():
        assert model(input_ids=ids).logits.shape[-1] == 103808


def test_two_stage_training(tiny_dirs, new_pieces, tmp_path):
    """1단계(새 임베딩만) -> 2단계(LoRA + 새 임베딩) -> 합치기. 기존 토큰 임베딩과 묶인 lm_head가 그대로 유지되는지."""
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    m = _load_script("04_train_lora.py")
    src = tiny_dirs("ernie4_5_moe")
    dst = tmp_path / "vx"
    info = extend_checkpoint(src, dst, new_pieces, AutoTokenizer.from_pretrained(src))
    ids = m.new_token_ids(dst)
    assert ids == info["new_token_ids"]
    tok = AutoTokenizer.from_pretrained(dst)

    def load():
        model = m.load_for_training(dst, device_map="cpu").float()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        return model

    model = load()
    emb0 = model.get_input_embeddings().weight.detach().clone()
    chunks = torch.tensor([tok(KO, add_special_tokens=False).input_ids[:32]] * 8, dtype=torch.int32)
    s = {"batch_size": 4, "grad_accum": 1, "lr": 5e-2, "seq_len": 32}
    model, log = m.train_new_embeddings(model, ids, lambda: chunks, s, tmp_path / "ck1", max_steps=3, cpu=True)
    emb1 = model.get_input_embeddings().weight.detach().clone()
    used = sorted(set(chunks.flatten().tolist()) & set(ids))
    old = [i for i in range(len(emb0)) if i not in set(ids)]
    assert used and not torch.allclose(emb1[used], emb0[used])
    torch.testing.assert_close(emb1[old], emb0[old])
    torch.testing.assert_close(model.lm_head.weight, model.get_input_embeddings().weight)  # 입력·출력 공유 유지

    # 다시 실행하면 1단계 결과를 불러와서 건너뜀 (세션이 끊긴 경우)
    again, log2 = m.train_new_embeddings(load(), ids, lambda: pytest.fail("다시 학습하면 안 됨"), s, tmp_path / "ck1")
    assert log2 is None
    torch.testing.assert_close(again.get_input_embeddings().weight, emb1)

    model = get_peft_model(model, m.lora_config({"r": 4, "alpha": 8, "lora_experts": True}, model, ids))
    ex = [{"prompt": "태양광 발전이란?", "response": "햇빛을 전기로 바꾸는 발전 방식이다."}] * 8
    rows, _ = m.tokenize_examples(tok, arch_of(model), ex, 256)
    targs = TrainingArguments(output_dir=str(tmp_path / "ck2"), per_device_train_batch_size=4, max_steps=3,
                              learning_rate=5e-2, report_to="none", remove_unused_columns=False, use_cpu=True,
                              save_strategy="no", logging_steps=1)
    Trainer(model=model, args=targs, train_dataset=rows, data_collator=m.Collator(tok.pad_token_id)).train()
    x = torch.tensor([tok("햇빛을 전기로 바꾸는 발전 방식", add_special_tokens=False).input_ids])
    model.eval()
    with torch.no_grad():
        peft_logits = model(input_ids=x).logits
    m.save_merged(model, dst, tmp_path / "merged")
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32).eval()
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=x).logits, peft_logits, atol=1e-3, rtol=1e-3)
    emb2 = merged.get_input_embeddings().weight
    torch.testing.assert_close(emb2[old], emb0[old])
    assert not torch.allclose(emb2[used], emb1[used])
    for f in ("tokenizer.model", "tokenizer.json", "vocab_info.json"):  # 토크나이저는 학습 전 그대로
        assert (tmp_path / "merged" / f).read_bytes() == (dst / f).read_bytes()


@pytest.mark.skipif(not os.environ.get("LLAMA_CPP_DIR"), reason="LLAMA_CPP_DIR 미설정 (llama.cpp 빌드 필요)")
def test_llama_cpp_tokenizes_like_hf(tiny_dirs, new_pieces, tmp_path):
    """GGUF로 바꾼 뒤 llama.cpp가 허깅페이스와 같은 토큰으로 자르는지 (원본·어휘 확장 둘 다)."""
    import requests

    from cfinetune.llama import LlamaServer

    llama = Path(os.environ["LLAMA_CPP_DIR"])
    src = tiny_dirs("ernie4_5_moe")
    extend_checkpoint(src, tmp_path / "vx", new_pieces, AutoTokenizer.from_pretrained(src))
    for name, d in (("orig", src), ("vx", tmp_path / "vx")):
        subprocess.run([sys.executable, ROOT / "scripts" / "03_make_gguf.py", "--hf-dir", d, "--name", name,
                        "--out-dir", tmp_path / "g", "--llama-dir", llama, "--quant", "Q8_0"], check=True, capture_output=True)
        tok = AutoTokenizer.from_pretrained(d)
        with LlamaServer(llama / "build" / "bin", tmp_path / "g" / f"{name}-Q8_0.gguf", port=8093, parallel=1,
                         ctx_per_slot=2048, log=tmp_path / f"{name}.log") as srv:
            for text in TEXTS:
                r = requests.post(srv.url + "/tokenize", json={"content": text, "add_special": False}, timeout=60)
                assert r.json()["tokens"] == tok(text, add_special_tokens=False).input_ids, (name, text)
