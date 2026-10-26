"""데이터 불러오기: 보정용 텍스트, 퍼플렉시티용 텍스트, 추가학습용 지시문, 평가 문항."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from datasets import get_dataset_config_names, load_dataset

LETTERS = "ABCDE"


def _wiki_articles(src: dict, min_chars: int = 400):
    ds = load_dataset(src["dataset"], data_files=src["file"], split="train", streaming=True)
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_chars:
            yield text


def _chunk_stream(token_lists, seq_len: int, n_chunks: int, sep: list[int]) -> list[list[int]]:
    chunks, buf = [], []
    for ids in token_lists:
        buf += ids + sep
        while len(buf) >= seq_len and len(chunks) < n_chunks:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(chunks) >= n_chunks:
            return chunks
    raise RuntimeError(f"조각이 부족함: {len(chunks)}/{n_chunks}")


def calibration_chunks(sources: list[dict], tokenizer, arch, n_chunks: int, seq_len: int) -> list[list[int]]:
    """여러 출처를 weight 비율대로 섞어 seq_len 토큰짜리 조각 n_chunks개를 만든다 (패딩 없음).

    type: wiki     -> 위키 문서를 그대로 이어 붙임
    type: instruct -> 지시문을 모델의 채팅 형식(질문 + 답변)으로 바꿔 이어 붙임
    """
    from .prompts import sft_text

    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    total_w = sum(s["weight"] for s in sources)
    chunks = []
    for i, src in enumerate(sources):
        n = round(n_chunks * src["weight"] / total_w) if i < len(sources) - 1 else n_chunks - len(chunks)
        if src["type"] == "wiki":
            texts = _wiki_articles(src)
        elif src["type"] == "instruct":
            texts = ("".join(sft_text(arch, ex["prompt"], ex["response"]))
                     for ex in sft_examples(src["dataset"], 10 ** 9, offset=src.get("offset", 0), file=src.get("file")))
        else:
            raise ValueError(src["type"])
        token_lists = (tokenizer(t, add_special_tokens=False).input_ids for t in texts)
        chunks += _chunk_stream(token_lists, seq_len, n, sep)
    return chunks


def packed_chunks(src: dict, tokenizer, n_tokens: int, seq_len: int):
    """위키 문서를 이어 붙여 seq_len 토큰짜리 조각으로 자른다 (어휘 확장 1단계 학습용). 반환: [조각 수, seq_len] int32 텐서."""
    import torch

    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    n = max(n_tokens // seq_len, 1)
    out = torch.empty((n, seq_len), dtype=torch.int32)
    buf, k, batch = [], 0, []

    def flush():
        nonlocal buf, k
        for ids in tokenizer(batch, add_special_tokens=False).input_ids:
            buf += ids + sep
            while len(buf) >= seq_len and k < n:
                out[k] = torch.tensor(buf[:seq_len], dtype=torch.int32)
                buf = buf[seq_len:]
                k += 1
        batch.clear()

    for text in _wiki_articles(src):
        batch.append(text)
        if len(batch) == 256:
            flush()
            if k >= n:
                return out
    flush()
    if k < n:
        raise RuntimeError(f"조각이 부족함: {k}/{n}")
    return out


def heldout_text(src: dict) -> str:
    parts, total = [], 0
    for text in _wiki_articles(src, min_chars=1000):
        parts.append(text)
        total += len(text)
        if total >= src["n_chars"]:
            break
    return "\n\n".join(parts)


def sft_pair(row: dict) -> tuple[str, str] | None:
    """여러 형식의 지시문 데이터를 (질문, 답변) 하나로.

    - messages (대화, 예: lemon-mint/smol-koreantalk): 첫 질문과 그 답. 시스템 메시지는 질문 앞에 붙임
    - instruction/input/output (예: nlpai-lab/kullm-v2)
    - question/answer
    """
    if "messages" in row:
        msgs = row["messages"]
        u = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
        if u is None or u + 1 >= len(msgs) or msgs[u + 1]["role"] != "assistant":
            return None
        system = [m["content"].strip() for m in msgs[:u] if m["role"] == "system" and m["content"].strip()]
        prompt, resp = "\n\n".join(system + [msgs[u]["content"].strip()]), msgs[u + 1]["content"]
    elif "instruction" in row:
        prompt = row["instruction"].strip() + ("\n\n" + row["input"].strip() if row.get("input") else "")
        resp = row["output"]
    else:
        prompt, resp = row["question"].strip(), row["answer"]
    resp = resp.strip()
    return (prompt, resp) if prompt and resp else None


_CODE = re.compile(r"```.*?(```|$)|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]*\$", re.S)


def is_korean_answer(text: str, min_ratio: float = 0.5, min_chars: int = 10) -> bool:
    """코드·수식을 뺀 나머지에서 한글이 (한글+영문 글자의) min_ratio 이상이고 min_chars자 이상인지.

    번역 데이터에는 질문만 번역되고 답은 영어로 남은 예제(예: 영어 문장 고치기)가 섞여 있어서 거른다.
    """
    t = _CODE.sub(" ", text)
    h = len(re.findall("[\uac00-\ud7a3]", t))
    return h >= min_chars and h / max(h + len(re.findall("[A-Za-z]", t)), 1) >= min_ratio


def sft_examples(name: str, n: int, seed: int = 0, offset: int = 0, file: str | None = None):
    """{"prompt", "response"} 를 차례로 내준다 (형식은 sft_pair 참고). file을 주면 그 파일만 받는다.

    같은 seed로 섞은 순서에서 offset부터 가져오므로, 보정용(offset 큼)과 LoRA 학습용(offset 0)이 겹치지 않는다.
    답이 한국어가 아닌 예제는 건너뛴다 (is_korean_answer).
    """
    ds = load_dataset(name, data_files=file, split="train") if file else load_dataset(name, split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    count = 0
    for i in idx[offset:]:
        pair = sft_pair(ds[i])
        if pair and is_korean_answer(pair[1]):
            yield {"prompt": pair[0], "response": pair[1]}
            count += 1
        if count >= n:
            return


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
