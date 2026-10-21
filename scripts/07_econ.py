"""7단계: 평가/벤치 결과를 모아 경제성 표와 그래프를 만들고, Qwen3.8-Flash-Next로 확장 계산한다.

입력: results/eval/*.json, results/bench/*.json
출력: results/summary.csv, results/report.md, results/fig_*.png
예) python scripts/07_econ.py --measured-gpu "L4 (Colab)"
변형 이름 규칙: {칼질변형 또는 original}[-lora]-{양자화형식}  예) original-Q6_K, ko-k32-Q4_K_M, ko-k32-lora-Q4_K_M
"""

import argparse
import csv
import json

from _common import RESULTS, load_config

from cfinetune.econ import flash_next_memory, gpu_catalog, to_markdown, variant_economics


def load_dir(name):
    d = RESULTS / name
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--measured-gpu", default="L4 (Colab)", help="벤치를 돌린 GPU (config economics.gpus 이름)")
    ap.add_argument("--baseline", default="original-Q6_K", help="한국어 성능 유지율의 기준")
    ap.add_argument("--cost-baseline", default="original-Q4_K_M", help="크기·에너지 비율의 기준 (같은 배포 형식)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    catalog = gpu_catalog(cfg)
    evals, benches = load_dir("eval"), load_dir("bench")
    names = sorted(set(evals) | set(benches), key=lambda n: (n != args.baseline, n))

    rows = []
    for n in names:
        row = {"variant": n}
        if n in evals:
            ev = evals[n]
            for task, m in ev["mcq"].items():
                row[f"{task}_acc"] = round(m["acc"] * 100, 1)
            for lang, p in ev.get("ppl", {}).items():
                row[f"ppl_{lang}"] = round(p["ppl"], 3)
            ko = [ev["mcq"][t]["acc"] for t in ("kmmlu", "haerae") if t in ev["mcq"]]
            if ko:
                row["ko_avg"] = round(sum(ko) / len(ko) * 100, 1)
        # LoRA를 합친 모델은 크기·구조가 원래 모델과 같아서 속도·전력이 같다 -> 벤치가 없으면 원래 모델 값을 쓴다
        b = benches.get(n) or benches.get(n.replace("-lora-", "-"))
        if b:
            row.update(variant_economics(b, catalog, args.measured_gpu, cfg["economics"]["vram_overhead_gb"]))
        rows.append(row)

    base = next((r for r in rows if r["variant"] == args.baseline), None)
    cbase = next((r for r in rows if r["variant"] == args.cost_baseline), base)
    for r in rows:
        if base and "ko_avg" in r and "ko_avg" in base:
            r["ko_vs_base_%"] = round(r["ko_avg"] / base["ko_avg"] * 100, 1)
        if cbase and "wh_per_1k_tok" in r and "wh_per_1k_tok" in cbase:
            r["energy_vs_base_%"] = round(r["wh_per_1k_tok"] / cbase["wh_per_1k_tok"] * 100, 1)
        if cbase and "file_gb" in r and "file_gb" in cbase:
            r["size_vs_base_%"] = round(r["file_gb"] / cbase["file_gb"] * 100, 1)

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    main_cols = [c for c in ["variant", "kmmlu_acc", "haerae_acc", "mmlu_acc", "ko_avg", "ko_vs_base_%", "ppl_ko",
                             "ppl_en", "file_gb", "size_vs_base_%", "tg_tps", "tg_power_w", "wh_per_1k_tok",
                             "energy_vs_base_%", "krw_per_1m_tok_measured_gpu", "cheapest_gpu",
                             "cheapest_gpu_krw_per_hour"] if c in cols]

    # Flash-Next 확장: gpt-oss에서 측정한 '남긴 비율 -> 한국어 성능 유지율'을 그대로 적용
    n_exp = cfg["model"]["num_experts"]
    ref_q = cfg["reference_quant"]
    fn_rows = []
    for r in rows:
        if not r["variant"].endswith(f"-{ref_q}"):
            continue  # 기준 정밀도로 잰 칼질 효과만 사용
        stem = r["variant"][: -len(ref_q) - 1]
        v = next((v for v in cfg["pruning"]["variants"] if v["name"] == stem.removesuffix("-lora")), None)
        ratio = v["keep"] / n_exp if v else (1.0 if stem.startswith("original") else None)
        if ratio is None:
            continue
        mem = flash_next_memory(ratio)
        fit = next((g for g in sorted(catalog, key=lambda g: g["krw_per_hour"]) if g["vram_gb"] >= mem["gpu_gb"] + 4), None)
        fn_rows.append({"실험 변형": r["variant"], "남긴 비율": f"{ratio:.2f}", "FlashNext 전문가": mem["keep_experts"],
                        "GPU 메모리 GB": mem["gpu_gb"], "전체 GB(n-gram 포함)": mem["total_gb"],
                        "48GB GPU 1장": "들어감" if mem["gpu_gb"] + 4 <= 48 else "안 들어감",
                        "가장 싼 GPU": fit["name"] if fit else "여러 장 필요",
                        "시간당 원": round(fit["krw_per_hour"]) if fit else "",
                        "한국어 점수 (원본 대비 %)": r.get("ko_vs_base_%", "")})

    md = ["# 결과 요약", "", f"벤치 GPU: {args.measured_gpu} / 기준 변형: {args.baseline}", "",
          to_markdown(rows, main_cols), "", "## Qwen3.8-Flash-Next 확장 계산", "",
          "전문가 MXFP4(4.25비트), 나머지 8.5비트, n-gram 임베딩은 CPU 메모리, MTP 제외 가정.", "",
          to_markdown(fn_rows, list(fn_rows[0])) if fn_rows else "(프루닝 결과 없음)"]
    (RESULTS / "report.md").write_text("\n".join(md), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [r for r in rows if "ko_avg" in r and "file_gb" in r]
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
        pts = [r for r in rows if "wh_per_1k_tok" in r]
        if pts:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([r["variant"] for r in pts], [r["wh_per_1k_tok"] for r in pts])
            ax.set_ylabel("Wh per 1k generated tokens")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            fig.tight_layout()
            fig.savefig(RESULTS / "fig_energy.png", dpi=150)
    except ImportError:
        pass

    print("\n".join(md))


if __name__ == "__main__":
    main()
