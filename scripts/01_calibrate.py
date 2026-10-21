"""1단계: 보정 데이터를 모델에 흘려 보내며 전문가 중요도를 측정한다.

필요 GPU: A100 80GB 또는 G4 (ERNIE-4.5-21B BF16 약 44GB)
출력: results/calib/stats-{세트이름}.json  (작은 파일. 이후 단계는 이것만 있으면 됨)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from _common import RESULTS, load_config, resolve_model_dir
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cfinetune.arch import arch_of
from cfinetune.data import calibration_chunks
from cfinetune.observe import MoEObserver


def load_model(path, device, dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(path)
    kw = {}
    qc = getattr(cfg, "quantization_config", None)
    if qc and (qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)) == "mxfp4":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)  # 전문가 MXFP4 -> BF16 으로 풀어서 계산
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device, **kw)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="기본값: config의 model.id")
    ap.add_argument("--sets", nargs="+", default=["ko"], help="config calibration.sets 이름")
    ap.add_argument("--out", default=str(RESULTS / "calib"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-chunks", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["calibration"]
    n_chunks = args.n_chunks or c["n_chunks"]
    path = resolve_model_dir(args.model or cfg["model"]["id"])
    tok = AutoTokenizer.from_pretrained(path)
    model = load_model(path, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arch = arch_of(model)
    for lang in args.sets:
        t0 = time.time()
        chunks = calibration_chunks(c["sets"][lang], tok, arch, n_chunks, c["seq_len"])
        with MoEObserver(model) as obs, torch.inference_mode():
            for i in range(0, len(chunks), c["batch_size"]):
                batch = torch.tensor(chunks[i:i + c["batch_size"]], device=model.device)
                model.model(input_ids=batch, use_cache=False)  # lm_head는 필요 없음
                if i % (c["batch_size"] * 16) == 0:
                    print(f"[{lang}] {i}/{len(chunks)}", flush=True)
        errs = [l["max_rel_err"] for l in obs.to_dict()["layers"].values()]
        print(f"[{lang}] 재구성 오차 최대 {max(errs):.4f} (0.05보다 크면 전문가 계산식이 모델과 다름)")
        if max(errs) > 0.05:
            raise RuntimeError("관측기 계산이 실제 모델 출력과 맞지 않음. transformers 버전을 확인하세요.")
        obs.save(out / f"stats-{lang}.json", lang=lang, model=str(args.model or cfg["model"]["id"]),
                 n_chunks=len(chunks), seq_len=c["seq_len"], sources=c["sets"][lang],
                 seconds=round(time.time() - t0, 1))
        print(f"[{lang}] 저장: {out / f'stats-{lang}.json'} ({time.time() - t0:.0f}초)")

    print(json.dumps({"saved": [str(out / f"stats-{l}.json") for l in args.sets]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
