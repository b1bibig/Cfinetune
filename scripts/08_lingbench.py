"""8단계: 한국어 언어력 벤치 (GGUF 한 개). 자세한 내용은 src/cfinetune/lingbench.py.

- 객관식: KoBALT-700, KoGEM, CLIcK 언어 영역
- 번역투 최소쌍 210개: 자연스러운 문장을 더 그럴듯하다고 보는 비율 (글자당 로그확률)
- 생성 114개(말투 지시 24 + K2-Eval 90): 번역투 표지 빈도, 반말/존댓말 지시 준수율, 반복, 한자·로마자 섞임, 끝맺음
필요 GPU: 모델이 다 올라가면 됨 (내 PC 3070에서 Q3_K_M 약 15분)
출력: results/lingbench/{name}.json

ERNIE 계열 GGUF는 끝 토큰이 </s>로 적혀 있지만 답을 <|end_of_sentence|>로 끝내므로 --eos-token-id 100272를 준다.

예) python scripts/08_lingbench.py --gguf D:/models/gguf/ko-k32-vx-lora-Q3_K_M.gguf --name ko-k32-vx-lora-Q3_K_M \
        --bin-dir D:/models/llama-b11784 --eos-token-id 100272
"""

import argparse
import json
import time

from _common import RESULTS

from cfinetune import lingbench as lb
from cfinetune.llama import LlamaServer


def cached(name: str, make):
    p = RESULTS / "cache" / f"lingbench-{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    obj = make()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin-dir", default="/content/llama.cpp/build/bin")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ctx-per-slot", type=int, default=4096)
    ap.add_argument("--eos-token-id", type=int, default=None)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--limit", type=int, default=None, help="디버그용: 부분마다 앞에서 이만큼만")
    ap.add_argument("--skip", nargs="*", default=[], choices=["mcq", "pairs", "gen"])
    args = ap.parse_args()

    mcq = cached("mcq", lb.load_mcq)
    pairs = lb.load_pairs()
    prompts = cached("gen_prompts", lb.load_gen_prompts)
    if args.limit:
        mcq, pairs, prompts = mcq[::max(1, len(mcq) // args.limit)][:args.limit], pairs[:args.limit], prompts[:args.limit]

    extra = ["--override-kv", f"tokenizer.ggml.eos_token_id=int:{args.eos_token_id}"] if args.eos_token_id else []
    out = {"name": args.name, "gguf": args.gguf, "eos_token_id": args.eos_token_id}
    log = RESULTS / "cache" / f"server-lingbench-{args.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with LlamaServer(args.bin_dir, args.gguf, port=args.port, ngl=args.ngl, parallel=args.parallel,
                     ctx_per_slot=args.ctx_per_slot, log=log, extra=extra) as srv:
        cl = lb.Client(srv.url)
        if "mcq" not in args.skip:
            t0 = time.time()
            res = lb.score_mcq(cl, mcq, args.parallel, progress=lambda i, n: print(f"  객관식 {i}/{n}", flush=True))
            out["mcq"], out["mcq_items"] = lb.summarize_mcq(res), res
            out["mcq_seconds"] = round(time.time() - t0, 1)
            for t, s in out["mcq"].items():
                print(f"{t}: 정답률 {s['acc']*100:.1f}% (찍기 {s['chance']*100:.1f}%, {s['n']}문항)")
        if "pairs" not in args.skip:
            corpus = [m["question"] + "\n" + "\n".join(m["choices"]) for m in cached("mcq", lb.load_mcq)]  # 토큰 빈도용 한국어 글
            rows = lb.score_pairs(cl, pairs, corpus, args.parallel)
            out["pairs"], out["pair_items"] = lb.summarize_pairs(rows), rows
            p = out["pairs"]
            print(f"번역투 최소쌍: 자연스러운 쪽 선호 SLOR {p['acc_slor']*100:.1f}% / 문장 전체 {p['acc_sum']*100:.1f}% / "
                  f"글자당 {p['acc_char']*100:.1f}%")
        if "gen" not in args.skip:
            t0 = time.time()
            rows = lb.generate(cl, prompts, parallel=args.parallel)
            out["gen"], out["gen_items"] = lb.summarize_gen(rows), rows
            out["gen_seconds"] = round(time.time() - t0, 1)
            print(json.dumps(out["gen"], ensure_ascii=False, indent=1))

    dst = RESULTS / "lingbench" / f"{args.name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
