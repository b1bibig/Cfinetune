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


def gpu_label(nvidia_name: str, catalog: list[dict]) -> str:
    """nvidia-smi 이름 -> config의 GPU 이름 (예: 'NVIDIA A100-SXM4-80GB' -> 'A100-80GB (Colab)'). 없으면 그대로."""
    n = nvidia_name.upper()
    key = ("T4" if "T4" in n else "L4" if "L4" in n else
           ("A100-80GB" if "80GB" in n else "A100-40GB") if "A100" in n else
           "H100-80GB" if "H100" in n else "B200" if "B200" in n else None)
    hits = [g["name"] for g in catalog if key and g["name"].startswith(key)]
    colab = [h for h in hits if "Colab" in h]
    return (colab or hits or [nvidia_name])[0]


def gpu_tag(label: str) -> str:
    """파일 이름용 짧은 이름: 'A100-80GB (Colab)' -> 'A100-80GB'."""
    return label.split(" (")[0].replace(" ", "_")


def variant_economics(bench: dict, catalog: list[dict], gpu: str, overhead_gb: float,
                      ko_tok_per_char: float | None = None) -> dict:
    """한 변형을 한 GPU에서 잰 결과의 경제성 지표.

    - 1000토큰 생성 에너지(Wh) = 생성 중 평균 전력(W) / 생성 속도(토큰/s) * 1000 / 3600
    - 100만 토큰 비용(원)   = GPU 시간당 가격 / (생성 속도 * 3600) * 1e6   (요청 1개씩 처리하는 보수적 기준)
    - 한국어 글자 기준       = 위 식에서 토큰/s 대신 글자/s (= 토큰/s / 글자당 토큰). 어휘가 다른 모델끼리 공정한 비교
    - 최소 GPU            = (모델 파일 + 여유분)이 들어가는 가장 싼 GPU
    """
    tg, w = bench["tg_tps"], bench["tg_power_w"]
    measured = next((g for g in catalog if g["name"] == gpu), None)
    need = bench["file_gb"] + overhead_gb
    fit = cheapest_fit(catalog, need)
    out = {
        "gpu": gpu,
        "file_gb": round(bench["file_gb"], 2),
        "need_vram_gb": round(need, 1),
        "peak_vram_gb": round(bench.get("peak_vram_gb", 0), 1),
        "pp_tps": round(bench["pp_tps"], 1),
        "tg_tps": round(tg, 1),
        "tg_power_w": round(w, 1),
        "wh_per_1k_tok": round(w / tg * 1000 / 3600, 4),
        "wh_per_1k_tok_above_idle": round(max(w - bench["idle_w"], 0) / tg * 1000 / 3600, 4),
        "cheapest_gpu": fit["name"] if fit else "없음",
        "cheapest_gpu_krw_per_hour": round(fit["krw_per_hour"]) if fit else None,
    }
    if measured:
        out["gpu_krw_per_hour"] = round(measured["krw_per_hour"])
        out["krw_per_1m_tok"] = round(measured["krw_per_hour"] / (tg * 3600) * 1e6, 1)
    out["idle_w"] = round(bench.get("loaded_idle_w", bench["idle_w"]), 1)
    if a := bench.get("answer"):
        out["answer_wh"] = round(a["wh_per_answer"], 4)
        out["answer_wh_per_1k_chars"] = round(a["wh_per_1k_chars"], 4)
        out["answer_chars_per_s"] = round(a["chars_per_s"], 1)
        out["answer_mean_w"] = round(a["mean_w"], 1)
    if ko_tok_per_char:
        cps = tg / ko_tok_per_char
        out["ko_tok_per_char"] = round(ko_tok_per_char, 4)
        out["ko_chars_per_s"] = round(cps, 1)
        out["wh_per_1k_ko_chars"] = round(w / cps * 1000 / 3600, 4)
        if measured:
            out["krw_per_1m_ko_chars"] = round(measured["krw_per_hour"] / (cps * 3600) * 1e6, 1)
    return out


def service_day(row: dict, econ: dict, basis: str = "answer") -> dict:
    """하루 운영 시나리오: 하루 econ.service.ko_chars_per_day 글자를 생성하는 한국어 챗봇을 24시간 켜 둔다.

    basis="answer": 실제 답변 측정값(프롬프트 처리 포함), "bench": llama-bench 생성 속도·전력 (한국어 글자 기준)
    - 일하는 시간 = 하루 글자 수 / 초당 글자 수, 그 동안의 에너지 = 글자당 에너지 x 글자 수
    - 나머지 시간은 대기 전력(모델을 올려 둔 채 쉬는 GPU)으로 계산. 24시간에 다 못 하면 GPU를 여러 장으로
    """
    import math

    if basis == "answer" and "answer_wh_per_1k_chars" in row:
        wh_per_char, cps = row["answer_wh_per_1k_chars"] / 1000, row["answer_chars_per_s"]
    elif "wh_per_1k_ko_chars" in row:
        wh_per_char, cps = row["wh_per_1k_ko_chars"] / 1000, row["ko_chars_per_s"]
    else:
        return {}
    demand = econ["service"]["ko_chars_per_day"]
    busy_h = demand / cps / 3600
    n = max(1, math.ceil(busy_h / 24))
    active_kwh = wh_per_char * demand / 1000
    idle_kwh = row["idle_w"] * (24 * n - busy_h) / 1000
    kwh = active_kwh + idle_kwh
    out = {"n_gpu": n, "busy_h_per_day": round(busy_h, 2), "kwh_per_day": round(kwh, 3),
           "kwh_per_day_active": round(active_kwh, 3), "kwh_per_day_idle": round(idle_kwh, 3),
           "kwh_per_year": round(kwh * 365, 1), "krw_elec_per_year": round(kwh * 365 * econ["krw_per_kwh"]),
           "kg_co2_per_year": round(kwh * 365 * econ["kg_co2_per_kwh"], 1), "service_basis": basis}
    if "gpu_krw_per_hour" in row:
        out["krw_gpu_rent_per_day"] = round(row["gpu_krw_per_hour"] * 24 * n)
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
    """ERNIE에서 측정한 프루닝 비율(남긴 전문가 비율)을 Flash-Next에 적용했을 때 필요한 GPU 메모리(GB) 추정."""
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
