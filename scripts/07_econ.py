"""7단계: 평가/벤치 결과를 모아 경제성 표와 그래프를 만들고, Qwen3.8-Flash-Next로 확장 계산한다.

입력: results/eval/*.json, results/bench/*@{GPU}.json, results/vocab/*.json
출력: results/summary.csv, results/report.md, results/fig_*.png
변형 이름 규칙: {칼질변형 또는 original}[-vx][-lora]-{양자화형식}
  예) original-BF16, original-Q6_K, ko-k32-Q4_K_M, ko-k32-vx-lora-Q4_K_M  (vx = 한국어 어휘 확장)
속도·에너지·비용은 한국어 글자 기준으로도 계산한다 (어휘가 다르면 같은 글도 토큰 수가 달라서).
"""

import argparse
import csv
import json
import re

from _common import RESULTS, load_config

from cfinetune.econ import flash_next_memory, gpu_catalog, gpu_label, to_markdown, variant_economics

QUANT = re.compile(r"-((?:BF|F)16|Q\d\w*)$")


def load_dir(name):
    d = RESULTS / name
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def stem_quant(name):
    m = QUANT.search(name)
    return (name[:m.start()], m.group(1)) if m else (name, "")


def family(name):
    return "vx" if "-vx" in name else "base"


def quality(evals, name, ref_q):
    """정답률 등. 그 변형을 직접 평가하지 않았으면 같은 모델의 기준 정밀도(Q6_K) 값을 쓴다 (예: 원본 BF16)."""
    stem, _ = stem_quant(name)
    src = name if name in evals else f"{stem}-{ref_q}" if f"{stem}-{ref_q}" in evals else None
    if not src:
        return {}
    ev, row = evals[src], {}
    for task, m in ev["mcq"].items():
        row[f"{task}_acc"] = round(m["acc"] * 100, 1)
    ko = [ev["mcq"][t]["acc"] for t in ("kmmlu", "haerae") if t in ev["mcq"]]
    if ko:
        row["ko_avg"] = round(sum(ko) / len(ko) * 100, 1)
    for lang, p in ev.get("ppl", {}).items():
        row[f"ppl_{lang}"] = round(p["ppl"], 3)
        if "bpc" in p:
            row[f"bpc_{lang}"] = round(p["bpc"], 4)
    if src != name:
        row["quality_from"] = src
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--baseline", default="original-Q6_K", help="한국어 성능 비교 기준")
    args = ap.parse_args()

    cfg = load_config(args.config)
    catalog = gpu_catalog(cfg)
    ref_q, deploy_q = cfg["reference_quant"], cfg["deploy_quant"]
    overhead = cfg["economics"]["vram_overhead_gb"]
    evals, benches = load_dir("eval"), load_dir("bench")

    tpc = {}  # 어휘 종류별 한국어 글자당 토큰 수
    for n, ev in evals.items():
        if "ko" in ev.get("tok_per_char", {}):
            tpc.setdefault(family(n), ev["tok_per_char"]["ko"])

    rows, benched = [], set()
    for key, b in benches.items():
        name = b.get("name") or key.split("@")[0]
        gpu = b.get("gpu_label") or gpu_label(b["gpu"], catalog)
        row = {"variant": name, **variant_economics(b, catalog, gpu, overhead, tpc.get(family(name)))}
        row.update(quality(evals, name, ref_q))
        rows.append(row)
        benched.add(name)
    for name in evals:
        if name not in benched:
            rows.append({"variant": name, "gpu": "-", "file_gb": round(evals[name]["file_gb"], 2),
                         **quality(evals, name, ref_q)})
    rows.sort(key=lambda r: (not r["variant"].startswith("original"), r["variant"], r.get("gpu", "")))

    base = quality(evals, args.baseline, ref_q)
    orig = next((r for r in rows if r["variant"] == "original-BF16" and "tg_tps" in r), None)
    for r in rows:
        if "ko_avg" in r and "ko_avg" in base:
            r["ko_vs_base_%p"] = round(r["ko_avg"] - base["ko_avg"], 1)
        if orig:
            for k, out in (("file_gb", "size_vs_orig_%"), ("wh_per_1k_ko_chars", "energy_vs_orig_%"),
                           ("krw_per_1m_ko_chars", "cost_vs_orig_%")):
                if k in r and orig.get(k):
                    r[out] = round(r[k] / orig[k] * 100, 1)

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k != "gpu", k))
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- 핵심 표: 원본 vs 최종
    final = f"{cfg['lora']['base_variant']}{'-vx' if cfg['vocab']['enabled'] else ''}-lora-{deploy_q}"
    head_cols = ["variant", "gpu", "file_gb", "cheapest_gpu", "ko_avg", "kmmlu_acc", "haerae_acc", "bpc_ko",
                 "ko_chars_per_s", "tg_power_w", "wh_per_1k_ko_chars", "krw_per_1m_ko_chars",
                 "size_vs_orig_%", "energy_vs_orig_%", "cost_vs_orig_%"]
    head = [r for r in rows if "tg_tps" in r and (r["variant"].startswith("original") or r["variant"] == final)]
    main_cols = [c for c in ["variant", "gpu", "kmmlu_acc", "haerae_acc", "mmlu_acc", "ko_avg", "ko_vs_base_%p",
                             "bpc_ko", "bpc_en", "file_gb", "tg_tps", "ko_chars_per_s", "tg_power_w",
                             "wh_per_1k_ko_chars", "krw_per_1m_ko_chars", "cheapest_gpu", "quality_from"] if c in cols]

    # ---- Flash-Next 확장: ERNIE에서 측정한 '남긴 비율 -> 한국어 성능 변화'를 그대로 적용
    n_exp = cfg["model"]["num_experts"]
    fn_rows = []
    for name in sorted(evals):
        stem, q = stem_quant(name)
        if q != ref_q:
            continue  # 기준 정밀도로 잰 칼질 효과만 사용
        core = stem.replace("-lora", "").replace("-vx", "")
        v = next((v for v in cfg["pruning"]["variants"] if v["name"] == core), None)
        ratio = v["keep"] / n_exp if v else (1.0 if core == "original" else None)
        if ratio is None:
            continue
        mem = flash_next_memory(ratio)
        fit = next((g for g in sorted(catalog, key=lambda g: g["krw_per_hour"]) if g["vram_gb"] >= mem["gpu_gb"] + 4), None)
        q_row = quality(evals, name, ref_q)
        fn_rows.append({"실험 변형": name, "남긴 비율": f"{ratio:.2f}", "FlashNext 전문가": mem["keep_experts"],
                        "GPU 메모리 GB": mem["gpu_gb"], "전체 GB(n-gram 포함)": mem["total_gb"],
                        "48GB GPU 1장": "들어감" if mem["gpu_gb"] + 4 <= 48 else "안 들어감",
                        "가장 싼 GPU": fit["name"] if fit else "여러 장 필요",
                        "시간당 원": round(fit["krw_per_hour"]) if fit else "",
                        "한국어 점수 (원본 대비 %p)": round(q_row["ko_avg"] - base["ko_avg"], 1)
                        if "ko_avg" in q_row and "ko_avg" in base else ""})

    # 한 줄 결론: 최종 모델(작은 GPU 중 가장 싼 측정) vs 원본 BF16
    verdict = []
    fin = sorted((r for r in head if r["variant"] == final), key=lambda r: r.get("gpu_krw_per_hour", 1e9))
    if fin and orig:
        f = fin[0]
        if "ko_avg" in f and "ko_avg" in orig:
            verdict.append(f"한국어 점수 {orig['ko_avg']} -> {f['ko_avg']} ({f['ko_avg'] - orig['ko_avg']:+.1f}%p)")
        if f.get("ko_chars_per_s") and orig.get("ko_chars_per_s"):
            verdict.append(f"한국어 생성 속도 {f['ko_chars_per_s'] / orig['ko_chars_per_s']:.1f}배")
        for k, label in (("size_vs_orig_%", "크기"), ("energy_vs_orig_%", "1000글자당 에너지"),
                         ("cost_vs_orig_%", "100만 글자당 비용")):
            if k in f:
                verdict.append(f"{label} {f[k]:.0f}%")
        verdict = [f"**최종 모델({f['gpu']}) vs 원본 BF16({orig['gpu']})**: " + ", ".join(verdict), ""]

    vocab = load_dir("vocab")
    md = ["# 결과 요약", "", *verdict, f"최종 모델: `{final}` / 한국어 성능 기준: `{args.baseline}`", "",
          "## 원본 vs 최종", "",
          "속도·에너지·비용은 **한국어 글자** 기준 (어휘를 늘린 모델은 같은 글을 더 적은 토큰으로 처리). "
          "`*_vs_orig_%`는 원본 BF16 대비 비율.", "",
          to_markdown(head, [c for c in head_cols if any(c in r for r in head)]) if head else "(측정 결과 없음)", ""]
    for n, v in vocab.items():
        ko = v["tok_per_char"]["ko"]
        md += [f"어휘 확장 `{n}`: 새 토큰 {v['n_new']}개, 한국어 글자당 토큰 {ko['before']} -> {ko['after']} "
               f"({(ko['after'] / ko['before'] - 1) * 100:+.0f}%)", ""]
    md += ["## 전체 결과", "", to_markdown(rows, main_cols), "", "## Qwen3.8-Flash-Next 확장 계산", "",
           "전문가 MXFP4(4.25비트), 나머지 8.5비트, n-gram 임베딩은 CPU 메모리, MTP 제외 가정.", "",
           to_markdown(fn_rows, list(fn_rows[0])) if fn_rows else "(프루닝 결과 없음)"]
    (RESULTS / "report.md").write_text("\n".join(md), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = {r["variant"]: r for r in rows if "ko_avg" in r and "file_gb" in r}.values()
        if pts:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter([r["file_gb"] for r in pts], [r["ko_avg"] for r in pts])
            for r in pts:
                ax.annotate(r["variant"], (r["file_gb"], r["ko_avg"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("model file (GB)")
            ax.set_ylabel("Korean MCQ avg (%)")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(RESULTS / "fig_size_vs_korean.png", dpi=150)
        pts = [r for r in rows if "wh_per_1k_ko_chars" in r]
        if pts:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([f"{r['variant']}\n{r['gpu']}" for r in pts], [r["wh_per_1k_ko_chars"] for r in pts])
            ax.set_ylabel("Wh per 1k generated Korean characters")
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            fig.tight_layout()
            fig.savefig(RESULTS / "fig_energy.png", dpi=150)
    except ImportError:
        pass

    print("\n".join(md))


if __name__ == "__main__":
    main()
