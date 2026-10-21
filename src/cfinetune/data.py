"""데이터 불러오기: 보정용 텍스트, 퍼플렉시티용 텍스트, 추가학습용 지시문, 평가 문항."""

from __future__ import annotations

import random
from collections import defaultdict

from datasets import get_dataset_config_names, load_dataset

LETTERS = "ABCDE"


def _wiki_articles(src: dict, min_chars: int = 400):
    ds = load_dataset(src["dataset"], data_files=src["file"], split="train", streaming=True)
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_chars:
            yield text


def calibration_chunks(src: dict, tokenizer, n_chunks: int, seq_len: int) -> list[list[int]]:
    """위키 문서를 이어 붙여 seq_len 토큰짜리 조각 n_chunks개를 만든다 (패딩 없음)."""
    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    chunks, buf = [], []
    for text in _wiki_articles(src):
        buf += tokenizer(text, add_special_tokens=False).input_ids + sep
        while len(buf) >= seq_len and len(chunks) < n_chunks:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(chunks) >= n_chunks:
            return chunks
    raise RuntimeError(f"조각이 부족함: {len(chunks)}/{n_chunks}")


def heldout_text(src: dict) -> str:
    parts, total = [], 0
    for text in _wiki_articles(src, min_chars=1000):
        parts.append(text)
        total += len(text)
        if total >= src["n_chars"]:
            break
    return "\n\n".join(parts)


def sft_examples(name: str, n: int, seed: int = 0) -> list[dict]:
    """[{"prompt", "response"}] 형태로 반환. 기본은 nlpai-lab/kullm-v2 (instruction/input/output)."""
    ds = load_dataset(name, split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        prompt = row["instruction"].strip()
        if row.get("input"):
            prompt += "\n\n" + row["input"].strip()
        resp = row["output"].strip()
        if prompt and resp:
            out.append({"prompt": prompt, "response": resp})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 평가 문항
# 모든 문항을 {task, subject, question, choices, answer(0부터)} 로 통일

def load_kmmlu(n_per_subject: int, seed: int) -> list[dict]:
    items = []
    for subject in sorted(get_dataset_config_names("HAERAE-HUB/KMMLU")):
        ds = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        idx = random.Random(f"{seed}-{subject}").sample(range(len(ds)), min(n_per_subject, len(ds)))
        for i in sorted(idx):
            r = ds[i]
            items.append({"task": "kmmlu", "subject": subject, "question": r["question"],
                          "choices": [r[c] for c in "ABCD"], "answer": int(r["answer"]) - 1, "lang": "ko"})
    return items


def load_haerae() -> list[dict]:
    items = []
    for subject in sorted(get_dataset_config_names("HAERAE-HUB/HAE_RAE_BENCH_1.0")):
        for r in load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", subject, split="test"):
            keys = [k for k in "abcde" if (r.get(k) or "").strip()]
            ans = r["answer"].strip().lower().strip("()")
            if ans not in keys:
                continue
            items.append({"task": "haerae", "subject": subject, "question": r["question"].strip(),
                          "choices": [r[k].strip() for k in keys], "answer": keys.index(ans), "lang": "ko"})
    return items


def load_mmlu(n_per_subject: int, seed: int) -> list[dict]:
    by_subject = defaultdict(list)
    for r in load_dataset("cais/mmlu", "all", split="test"):
        by_subject[r["subject"]].append(r)
    items = []
    for subject in sorted(by_subject):
        rows = by_subject[subject]
        for i in sorted(random.Random(f"{seed}-{subject}").sample(range(len(rows)), min(n_per_subject, len(rows)))):
            r = rows[i]
            items.append({"task": "mmlu", "subject": subject, "question": r["question"],
                          "choices": list(r["choices"]), "answer": int(r["answer"]), "lang": "en"})
    return items


def load_eval_items(cfg: dict) -> list[dict]:
    e = cfg["eval"]
    items = load_kmmlu(e["kmmlu_per_subject"], e["seed"]) + load_haerae()
    if e.get("mmlu_per_subject"):
        items += load_mmlu(e["mmlu_per_subject"], e["seed"])
    return items
