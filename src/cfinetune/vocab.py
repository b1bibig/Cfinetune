"""한국어 어휘 확장: 한국어 BPE를 새로 학습해 ERNIE 토크나이저에 토큰을 덧붙이고, 임베딩을 늘린다 (GPU 필요 없음).

ERNIE 토크나이저는 SentencePiece BPE(바이트 대체 사용)이고 어휘에 한글이 거의 없어서,
흔한 글자도 UTF-8 바이트 3개로 쪼개 읽는다. 새 토큰을 넣으면 같은 한국어 문장을 훨씬 적은 토큰으로 처리한다.

두 엔진이 같은 문장을 똑같이 자르도록 토크나이저 파일 두 개를 함께 고친다.
  - tokenizer.model (SentencePiece) : llama.cpp 변환기가 읽음. 새 조각은 기존 조각보다 낮은 점수(=나중에 합쳐짐)로 덧붙인다.
  - tokenizer.json  (허깅페이스)    : 같은 어휘·점수로 merges를 다시 만든다. 원본 merges도 이 방식으로 똑같이 재현된다.
새 토큰 번호는 기존 추가 토큰(<|IMAGE_PLACEHOLDER|> 등) 다음부터 붙인다. ERNIE는 101304번부터 빈 자리다.
새 임베딩은 그 토큰을 원래 토크나이저로 쪼갠 조각들의 임베딩 평균으로 시작한다.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HANGUL = re.compile("[가-힣]")
SPACE = "▁"
TOKENIZER_FILES = ("tokenizer.model", "tokenizer.json")


# ---------------------------------------------------------------- 1. 한국어 BPE 학습

def train_korean_bpe(texts, vocab_size: int, limit_alphabet: int = 3000) -> dict:
    """SentencePiece와 같은 방식(공백 -> ▁, 단어 앞에 붙음)으로 BPE를 학습한다. 반환: tokenizers의 BPE 모델 JSON."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement=SPACE, prepend_scheme="never")
    tk.decoder = decoders.Metaspace(replacement=SPACE, prepend_scheme="never")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, limit_alphabet=limit_alphabet,
                                  show_progress=False)
    tk.train_from_iterator(texts, trainer=trainer)
    return json.loads(tk.to_str())["model"]


def select_new_pieces(bpe: dict, base_vocab: dict[str, int], n_new: int) -> list[str]:
    """학습한 BPE에서 기존 어휘에 없는 한글 조각을 만들어진 순서(=자주 나오는 순서)대로 n_new개 고른다.

    어떤 조각을 고르면 그 조각을 만드는 두 부분도 (기존 어휘에 없으면) 먼저 넣어서, 합치는 경로가 끊기지 않게 한다.
    부분이 항상 전체보다 앞에 오므로 앞에서부터 잘라도 경로가 유지된다.
    """
    parts = {}
    for m in bpe["merges"]:
        l, r = m if isinstance(m, list) else m.split(" ")
        parts.setdefault(l + r, (l, r))
    chosen: list[str] = []
    have = set(base_vocab)

    def add(p):
        if p in have:
            return
        for q in parts.get(p, ()):
            add(q)
        have.add(p)
        chosen.append(p)

    for piece, _ in sorted(bpe["vocab"].items(), key=lambda kv: kv[1]):
        if len(chosen) >= n_new:
            break
        if piece not in have and HANGUL.search(piece):
            add(piece)
    return chosen[:n_new]


# ---------------------------------------------------------------- 2. 토크나이저 파일 고치기

def _load_proto(path):
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(Path(path).read_bytes())
    return m


def extend_tokenizer_files(src: str | Path, dst: str | Path, new_pieces: list[str]) -> dict:
    """src의 토크나이저 파일에 new_pieces를 덧붙여 dst에 쓴다. 반환: {"first_id", "ids", "vocab_size"}."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    from transformers.tokenization_utils_base import generate_merges

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    proto = _load_proto(src / "tokenizer.model")
    if proto.trainer_spec.model_type != pb.TrainerSpec.BPE:
        raise ValueError("SentencePiece BPE 토크나이저만 지원")
    n_base = len(proto.pieces)
    added = json.loads((src / "added_tokens.json").read_text()) if (src / "added_tokens.json").exists() else {}
    first = max([n_base] + [i + 1 for i in added.values()])
    existing = {p.piece for p in proto.pieces} | set(added)
    if dup := [p for p in new_pieces if p in existing]:
        raise ValueError(f"이미 있는 토큰: {dup[:5]}")

    # SentencePiece는 번호 = 순서라서, 추가 토큰 자리(n_base..first-1)를 제어 토큰으로 채운다.
    # llama.cpp 변환기는 이 자리를 added_tokens.json 내용으로 다시 덮어쓰므로 원본과 같아진다.
    for i in range(n_base, first):
        proto.pieces.add(piece=f"<|cfinetune_reserved_{i}|>", score=0.0, type=pb.ModelProto.SentencePiece.CONTROL)
    low = min(p.score for p in proto.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL)
    for k, piece in enumerate(new_pieces):
        proto.pieces.add(piece=piece, score=low - 1.0 - k, type=pb.ModelProto.SentencePiece.NORMAL)
    (dst / "tokenizer.model").write_bytes(proto.SerializeToString())

    # 허깅페이스: 같은 어휘와 점수로 merges를 다시 만든다 (SentencePiece와 같은 규칙).
    # tokenizers 라이브러리는 기본 어휘에 없는 추가 토큰의 번호를 (파일에 적힌 값과 상관없이) 기본 어휘 크기 다음부터
    # 다시 매긴다. 새 토큰이 추가 토큰보다 뒤 번호를 쓰므로, 추가 토큰도 원래 번호로 기본 어휘에 넣어 번호를 고정한다.
    tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = dict(tj["model"]["vocab"])
    special = {t["content"]: t["id"] for t in tj["added_tokens"] if t["content"] not in vocab}
    vocab.update(special)
    scores = {p.piece: p.score for i, p in enumerate(proto.pieces) if i < n_base or i >= first}
    ids = list(range(first, first + len(new_pieces)))
    vocab.update(zip(new_pieces, ids))
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("토큰 번호가 0부터 연속이 아님")
    merges = generate_merges(vocab, scores, skip_tokens=set(special))
    tj["model"]["vocab"] = vocab
    tj["model"]["merges"] = [list(m) for m in merges]
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")

    for p in src.iterdir():
        if p.is_file() and p.name not in TOKENIZER_FILES and (p.name.startswith(("tokenizer", "special_tokens"))
                                                              or p.name in ("added_tokens.json", "chat_template.jinja")):
            shutil.copy2(p, dst / p.name)
    return {"first_id": first, "ids": ids, "vocab_size": first + len(new_pieces)}


# ---------------------------------------------------------------- 3. 체크포인트 임베딩 늘리기

def _round_up(n: int, k: int = 64) -> int:
    return (n + k - 1) // k * k


def extend_checkpoint(src: str | Path, dst: str | Path, new_pieces: list[str], base_tokenizer, info: dict | None = None) -> dict:
    """HF 체크포인트(src)를 복사하면서 토크나이저와 임베딩(+ 묶이지 않은 lm_head)을 늘린다.

    base_tokenizer: 원래 토크나이저 (새 토큰을 원래 방식으로 쪼개 평균 임베딩을 구할 때 사용)
    dst/vocab_info.json 에 새 토큰 번호를 기록한다 (학습 단계가 이 번호만 학습).
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    tinfo = extend_tokenizer_files(src, dst, new_pieces)
    cfg = json.loads((src / "config.json").read_text())
    old_n = cfg["vocab_size"]
    new_n = max(old_n, _round_up(tinfo["vocab_size"]))

    pieces_ids = [base_tokenizer(p.replace(SPACE, " "), add_special_tokens=False).input_ids for p in new_pieces]
    index_path = src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() \
        else {k: "model.safetensors" for k in safe_open(src / "model.safetensors", "pt").keys()}
    targets = [k for k in weight_map if k.endswith("embed_tokens.weight") or k == "lm_head.weight"]

    for shard in sorted(set(weight_map.values())):
        if not any(weight_map[k] == shard for k in targets):
            shutil.copy2(src / shard, dst / shard)
            continue
        with safe_open(src / shard, framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for k in targets:
            if weight_map[k] != shard:
                continue
            old = tensors[k]
            # 원본에서 쓰지 않던 칸(패딩)의 평균. 새로 생긴 칸 중 토큰이 없는 칸에 쓴다.
            unused = old[tinfo["first_id"]:] if old_n > tinfo["first_id"] else old
            new = unused.float().mean(0).to(old.dtype).repeat(new_n, 1)
            new[:old_n] = old
            # 새 토큰은 원래 조각들의 평균에서 시작
            for i, ids in zip(tinfo["ids"], pieces_ids):
                new[i] = old[ids].float().mean(0).to(old.dtype)
            tensors[k] = new
        save_file(tensors, dst / shard, metadata={"format": "pt", **meta})
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        (dst / index_path.name).write_text(json.dumps(idx, indent=2))

    cfg["vocab_size"] = new_n
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for p in src.iterdir():
        if p.is_file() and not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)
    vinfo = {"new_token_ids": tinfo["ids"], "n_new": len(new_pieces), "vocab_size_before": old_n,
             "vocab_size_after": new_n, "pieces": new_pieces, **(info or {})}
    (dst / "vocab_info.json").write_text(json.dumps(vinfo, ensure_ascii=False, indent=1), encoding="utf-8")
    return vinfo


def tokens_per_char(tokenizer, text: str) -> float:
    return len(tokenizer(text, add_special_tokens=False).input_ids) / max(len(text), 1)
