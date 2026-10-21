"""4단계: 프루닝한 모델에 한국어 지시문으로 저랭크(LoRA) 추가학습.

필요 GPU: 전문가 16개 모델은 A100 40GB (BF16 약 23GB + 학습 메모리), 24개 모델은 A100 80GB
- 원본 형식(MXFP4) 체크포인트를 불러올 때 BF16으로 풀어서 학습한다.
- 라우터는 학습하지 않는다 (프루닝으로 정한 전문가 선택 구조를 유지).
- 어텐션 q/k/v/o + (선택) 전문가 3D 가중치(gate_up_proj, down_proj)에 LoRA.
- 손실은 답변(assistant final) 부분에만 건다.
출력: {out}/adapter (LoRA만), {out}/merged (합친 BF16 체크포인트, GGUF 변환용)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from _common import RESULTS, load_config
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from cfinetune.data import sft_examples
from cfinetune.prompts import sft_text


def tokenize_examples(tok, examples, max_len):
    rows, dropped = [], 0
    for ex in examples:
        head, tail = sft_text(ex["prompt"], ex["response"])
        h = tok(head, add_special_tokens=False).input_ids
        t = tok(tail, add_special_tokens=False).input_ids
        if len(h) + 16 > max_len:
            dropped += 1
            continue
        ids = (h + t)[:max_len]
        labels = ([-100] * len(h) + t)[:max_len]
        rows.append({"input_ids": ids, "labels": labels})
    return rows, dropped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def load_for_training(path, device_map="auto"):
    cfg = AutoConfig.from_pretrained(path)
    kw = {}
    qc = getattr(cfg, "quantization_config", None)
    if qc and (qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)) == "mxfp4":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map,
                                                attn_implementation="eager", **kw)


def lora_config(c, n_layers):
    target_params = None
    if c["lora_experts"]:
        target_params = [f"{i}.mlp.experts.{p}" for i in range(n_layers) for p in ("gate_up_proj", "down_proj")]
    return LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], target_parameters=target_params)


def save_merged(model, tok, src_dir, dst):
    merged = model.merge_and_unload()
    merged.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    cfg_path = Path(dst) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("quantization_config", None)  # BF16로 저장됐으므로 MXFP4 표시 제거 (GGUF 변환이 헷갈리지 않게)
    cfg["dtype"] = cfg["torch_dtype"] = "bfloat16"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    for name in ("chat_template.jinja", "generation_config.json", "prune_info.json"):
        if (Path(src_dir) / name).exists():
            shutil.copy2(Path(src_dir) / name, Path(dst) / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True, help="프루닝한 체크포인트 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1, help="디버그용")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="작은 테스트 모델용")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["lora"]
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_for_training(args.model, device_map="cpu" if args.cpu else "auto")
    model.config.use_cache = False

    examples = sft_examples(c["dataset"], args.n_examples or c["n_examples"])
    rows, dropped = tokenize_examples(tok, examples, c["max_len"])
    print(f"학습 예제 {len(rows)}개 (너무 길어 제외 {dropped}개)")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(c, model.config.num_hidden_layers))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=c["batch_size"],
        gradient_accumulation_steps=c["grad_accum"], num_train_epochs=c["epochs"], max_steps=args.max_steps,
        learning_rate=c["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        bf16=not args.cpu, logging_steps=10, save_strategy="steps", save_steps=200, save_total_limit=1,
        report_to="none", remove_unused_columns=False, use_cpu=args.cpu, group_by_length=True)
    trainer = Trainer(model=model, args=targs, train_dataset=rows, data_collator=Collator(tok.pad_token_id))
    result = trainer.train()

    model.save_pretrained(out / "adapter")
    log = {"train_loss": result.training_loss, "steps": result.global_step, "n_examples": len(rows),
           "lora": c, "log_history": trainer.state.log_history}
    (out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    (RESULTS / "lora").mkdir(parents=True, exist_ok=True)
    (RESULTS / "lora" / f"{out.name}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))

    if not args.no_merge:
        save_merged(model, tok, args.model, out / "merged")
        print(f"합친 모델 저장: {out / 'merged'}")


if __name__ == "__main__":
    main()
