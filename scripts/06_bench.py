"""6단계: 속도(토큰/초), 전력(W), VRAM, 그리고 질문에 실제로 답하는 에너지(Wh)를 잰다.

- llama-bench: 프롬프트 처리·생성 속도와 그동안의 평균 전력
- 답변 에너지: configs/gen_prompts.txt 의 한국어 질문 8개에 차례로 답하게 하고 GPU 에너지 계량기로 잰다
  (프롬프트 처리 + 생성 전체, 사용자 1명 기준). 답변 1개당 Wh, 답변 1000글자당 Wh
- 대기 전력: 모델을 올려 두고 아무것도 안 할 때의 전력 (하루 운영 에너지 계산에 사용)

다른 프로그램이 GPU를 쓰지 않는 상태에서 실행할 것 (llama-server가 떠 있으면 안 됨).
같은 변형을 여러 GPU에서 잴 수 있도록 결과 파일 이름에 GPU를 붙인다.
출력: results/bench/{name}@{GPU}.json  예) original-BF16@A100-80GB.json, ko-k32-vx-lora-Q4_K_M@T4.json
"""

import argparse
import json

from _common import RESULTS, ROOT, load_config
from transformers import AutoConfig

from cfinetune.arch import arch_of
from cfinetune.econ import gpu_catalog, gpu_label, gpu_tag
from cfinetune.llama import LlamaServer, answer_energy, idle_power, llama_bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin-dir", default="/content/llama.cpp/build/bin")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--skip-answers", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = llama_bench(args.bin_dir, args.gguf, reps=args.reps)
    r["name"] = args.name
    r["gpu_label"] = gpu_label(r["gpu"], gpu_catalog(cfg))
    if not args.skip_answers:
        arch = arch_of(AutoConfig.from_pretrained(cfg["model"]["id"]))
        (RESULTS / "cache").mkdir(parents=True, exist_ok=True)
        prompts = [l.strip() for l in (ROOT / "configs" / "gen_prompts.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        with LlamaServer(args.bin_dir, args.gguf, parallel=1, log=RESULTS / "cache" / f"bench-server-{args.name}.log") as srv:
            r["loaded_idle_w"] = idle_power(10)  # 모델을 올려 둔 채 대기
            r["answer"] = answer_energy(srv, arch, prompts)
        a = r["answer"]
        print(f"답변 {a['n_answers']}개: 1개당 {a['wh_per_answer']:.3f} Wh, 1000글자당 {a['wh_per_1k_chars']:.3f} Wh, "
              f"초당 {a['chars_per_s']:.1f}글자, 평균 {a['mean_w']:.0f} W, 대기 {r['loaded_idle_w']:.0f} W")
    dst = RESULTS / "bench" / f"{args.name}@{gpu_tag(r['gpu_label'])}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
