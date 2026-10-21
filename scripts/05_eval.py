"""5단계: GGUF 한 개를 평가한다 (모든 변형을 같은 방식으로).

- 객관식: KMMLU(일부), HAE-RAE(전체), MMLU(일부, 영어 능력 확인용)
- 퍼플렉시티: 한국어/영어 위키 (보정에 안 쓴 파일)
- 생성 예시: configs/gen_prompts.txt
필요 GPU: L4 (ERNIE 원본 Q6_K 약 18GB까지 들어감)
출력: results/eval/{name}.json
"""

import argparse
import json
import time
from pathlib import Path

from _common import RESULTS, ROOT, load_config
from transformers import AutoConfig, AutoTokenizer

from cfinetune import data, llama
from cfinetune.arch import arch_of


def cached(path: Path, make):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    obj = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin-dir", default="/content/llama.cpp/build/bin")
    ap.add_argument("--tokenizer", default=None, help="기본: config의 model.id")
    ap.add_argument("--limit", type=int, default=None, help="디버그용 문항 수 제한")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()

    cfg = load_config(args.config)
    e = cfg["eval"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or cfg["model"]["id"])
    arch = arch_of(AutoConfig.from_pretrained(args.tokenizer or cfg["model"]["id"]))
    items = cached(RESULTS / "cache" / f"eval_items-s{e['seed']}.json", lambda: data.load_eval_items(cfg))
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    out = {"name": args.name, "gguf": args.gguf, "prompt_format": e["prompt_format"],
           "file_gb": Path(args.gguf).stat().st_size / 1e9}

    t0 = time.time()
    with llama.LlamaServer(args.bin_dir, args.gguf, ngl=args.ngl, log=RESULTS / "cache" / f"server-{args.name}.log") as srv:
        res = llama.score_mcq(srv, tok, arch, items, e["prompt_format"],
                              progress=lambda i, n: print(f"  객관식 {i}/{n}", flush=True))
        out["mcq"] = llama.summarize_mcq(res)
        out["mcq_items"] = res
        if not args.skip_gen:
            prompts = [l.strip() for l in (ROOT / "configs" / "gen_prompts.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
            out["generations"] = [{"prompt": p, "output": o} for p, o in zip(prompts, llama.generate(srv, tok, arch, prompts))]
    out["mcq_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out["mcq"], ensure_ascii=False, indent=1))

    if not args.skip_ppl:
        out["ppl"] = {}
        for lang, src in cfg["heldout"].items():
            txt = RESULTS / "cache" / f"heldout-{lang}.txt"
            if not txt.exists():
                txt.write_text(data.heldout_text(src), encoding="utf-8")
            out["ppl"][lang] = llama.perplexity(args.bin_dir, args.gguf, txt, e["ppl_ctx"], e["ppl_chunks"], args.ngl)
            print(f"  PPL[{lang}] = {out['ppl'][lang]['ppl']:.3f}")

    dst = RESULTS / "eval" / f"{args.name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
