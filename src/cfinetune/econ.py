"""경제성 분석: 측정값(속도, 전력, 크기)을 비용과 에너지로 바꾸고, Qwen3.8-Flash-Next로 확장 계산."""

from __future__ import annotations


def gpu_catalog(cfg: dict) -> list[dict]:
    e = cfg["economics"]
    won_per_cu = e["colab_usd_per_cu"] * e["krw_per_usd"]
    out = []
    for name, g in e["gpus"].items():
        krw = g.get("krw_per_hour", g.get("cu_per_hour", 0) * won_per_cu)
        out.append({"name": name, "vram_gb": g["vram_gb"], "krw_per_hour": krw})
    return sorted(out, key=lambda g: (g["vram_gb"], g["krw_per_hour"]))


def cheapest_fit(catalog: list[dict], need_gb: float) -> dict | None:
    fits = [g for g in catalog if g["vram_gb"] >= need_gb]
    return min(fits, key=lambda g: g["krw_per_hour"]) if fits else None


def variant_economics(bench: dict, catalog: list[dict], measured_gpu: str, overhead_gb: float) -> dict:
    """한 변형의 경제성 지표.

    - 1000토큰 생성 에너지(Wh) = 생성 중 평균 전력(W) / 생성 속도(토큰/s) * 1000 / 3600
    - 100만 토큰 비용(원)   = GPU 시간당 가격 / (생성 속도 * 3600) * 1e6   (요청 1개씩 처리하는 보수적 기준)
    - 최소 GPU            = (모델 파일 + 여유분)이 들어가는 가장 싼 GPU
    """
    tg, w = bench["tg_tps"], bench["tg_power_w"]
    j_per_tok = w / tg
    measured = next((g for g in catalog if g["name"] == measured_gpu), None)
    need = bench["file_gb"] + overhead_gb
    fit = cheapest_fit(catalog, need)
    out = {
        "file_gb": round(bench["file_gb"], 2),
        "need_vram_gb": round(need, 1),
        "peak_vram_gb": round(bench.get("peak_vram_gb", 0), 1),
        "pp_tps": round(bench["pp_tps"], 1),
        "tg_tps": round(tg, 1),
        "tg_power_w": round(w, 1),
        "wh_per_1k_tok": round(j_per_tok * 1000 / 3600, 4),
        "wh_per_1k_tok_above_idle": round(max(w - bench["idle_w"], 0) / tg * 1000 / 3600, 4),
        "cheapest_gpu": fit["name"] if fit else "없음",
        "cheapest_gpu_krw_per_hour": round(fit["krw_per_hour"]) if fit else None,
    }
    if measured:
        out["krw_per_1m_tok_measured_gpu"] = round(measured["krw_per_hour"] / (tg * 3600) * 1e6, 1)
    return out


# ---------------------------------------------------------------- Qwen3.8-Flash-Next 확장

FLASH_NEXT = {
    # 모델 카드 기준: MoE 본체 125B(이 중 라우팅 전문가 약 121B), n-gram 임베딩 51B, MTP 4B
    "routed_expert_params": 121e9,
    "other_core_params": 4e9,
    "ngram_params": 51e9,
    "mtp_params": 4e9,
    "num_experts": 512,
}


def flash_next_memory(keep_ratio: float, expert_bits: float = 4.25, other_bits: float = 8.5,
                      ngram_bits: float = 4.5, ngram_on_cpu: bool = True, drop_mtp: bool = True) -> dict:
    """gpt-oss에서 측정한 프루닝 비율을 Flash-Next에 적용했을 때 필요한 GPU 메모리(GB) 추정."""
    f = FLASH_NEXT
    gb = lambda p, b: p * b / 8 / 1e9  # noqa: E731
    experts = gb(f["routed_expert_params"] * keep_ratio, expert_bits)
    other = gb(f["other_core_params"], other_bits)
    ngram = gb(f["ngram_params"], ngram_bits)
    mtp = 0 if drop_mtp else gb(f["mtp_params"], other_bits)
    gpu = experts + other + mtp + (0 if ngram_on_cpu else ngram)
    return {"keep_experts": round(f["num_experts"] * keep_ratio), "experts_gb": round(experts, 1),
            "ngram_gb": round(ngram, 1), "ngram_on_cpu": ngram_on_cpu, "gpu_gb": round(gpu, 1),
            "total_gb": round(experts + other + mtp + ngram, 1)}


def to_markdown(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
