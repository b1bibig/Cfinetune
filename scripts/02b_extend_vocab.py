"""2b단계: 프루닝한 체크포인트에 한국어 어휘를 덧붙인다 (CPU로 충분, 수 분).

한국어 위키로 BPE를 학습해서 원래 어휘에 없는 한국어 조각을 vocab.n_new개 고르고,
토크나이저 파일(tokenizer.model, tokenizer.json)과 임베딩을 늘린 체크포인트를 만든다.
새 임베딩은 원래 토크나이저로 쪼갠 조각들의 평균에서 시작하고, 04_train_lora.py 1단계에서 학습한다.
예) python scripts/02b_extend_vocab.py --src /content/models/pruned/ko-k32 --dst /content/models/pruned/ko-k32-vx
출력: {dst} 체크포인트, results/vocab/{dst 이름}.json (토큰 수 변화)
"""

import argparse
import json
import time
from pathlib import Path

from _common import RESULTS, load_config
from datasets import load_dataset
from transformers import AutoTokenizer

from cfinetune.data import heldout_text
from cfinetune.vocab import extend_checkpoint, select_new_pieces, tokens_per_char, train_korean_bpe

SAMPLE = "대한민국의 수도는 서울이다. 태양광 발전과 풍력 발전의 장단점을 비교해 줘."


def wiki_texts(src: dict):
    n = 0
    for row in load_dataset(src["dataset"], data_files=src["file"], split="train", streaming=True):
        yield row["text"]
        n += len(row["text"])
        if n >= src["n_chars"]:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", required=True, help="프루닝한 체크포인트 폴더")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-new", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    v = cfg["vocab"]
    n_new = args.n_new or v["n_new"]
    base = AutoTokenizer.from_pretrained(args.src)

    t0 = time.time()
    bpe = train_korean_bpe(wiki_texts(v["train"]), v["bpe_vocab_size"])
    pieces = select_new_pieces(bpe, base.get_vocab(), n_new)
    print(f"한국어 BPE 학습 {time.time() - t0:.0f}초, 새 토큰 {len(pieces)}개")
    if len(pieces) < n_new:
        print(f"주의: 후보가 부족해 {len(pieces)}개만 추가 (vocab.bpe_vocab_size를 늘리세요)")
    info = extend_checkpoint(args.src, args.dst, pieces, base, info={"bpe_vocab_size": v["bpe_vocab_size"]})

    new = AutoTokenizer.from_pretrained(args.dst)
    stats = {"n_new": len(pieces), "vocab_size_after": info["vocab_size_after"], "tok_per_char": {}}
    for lang, src in cfg["heldout"].items():
        text = heldout_text(src)
        stats["tok_per_char"][lang] = {"before": round(tokens_per_char(base, text), 4),
                                       "after": round(tokens_per_char(new, text), 4)}
    stats["sample"] = {"text": SAMPLE,
                       "before": base.convert_ids_to_tokens(base(SAMPLE, add_special_tokens=False).input_ids),
                       "after": new.convert_ids_to_tokens(new(SAMPLE, add_special_tokens=False).input_ids)}
    stats["first_pieces"] = pieces[:50]
    (RESULTS / "vocab").mkdir(parents=True, exist_ok=True)
    (RESULTS / "vocab" / f"{Path(args.dst).name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                                                   encoding="utf-8")
    ko = stats["tok_per_char"]["ko"]
    print(f"한국어 글자당 토큰 {ko['before']} -> {ko['after']} ({(ko['after'] / ko['before'] - 1) * 100:+.0f}%)")
    print("예시:", " | ".join(stats["sample"]["after"]))


if __name__ == "__main__":
    main()
