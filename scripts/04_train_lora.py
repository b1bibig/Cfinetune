"""4단계: 프루닝한 모델을 한국어로 추가학습한다.

필요 GPU: ERNIE 전문가 32개 모델은 A100 40GB (BF16 약 23GB + 학습 메모리), 48개 모델은 A100 80GB
- (어휘 확장한 모델이면) 1단계: 모델은 고정하고 새 토큰 임베딩만 한국어 위키로 학습한다.
- 2단계: 한국어 지시문으로 LoRA. 어텐션 q/k/v/o + 공유 전문가 + (선택) 남은 전문가 3D 가중치에 LoRA,
  새 토큰 임베딩도 계속 학습한다. 라우터는 학습하지 않는다 (프루닝으로 정한 전문가 선택 구조를 유지).
- 2단계 손실은 답변 부분에만 건다.
- 중간 저장(--ckpt-dir)을 구글 드라이브에 두면 세션이 끊겨도 다시 실행했을 때 이어서 학습한다.
출력: {out}/adapter (LoRA만), {out}/merged (합친 BF16 체크포인트, GGUF 변환용)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from _common import RESULTS, load_config
from peft import LoraConfig, PeftModel, TrainableTokensConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from cfinetune.arch import arch_of
from cfinetune.data import packed_chunks, sft_examples
from cfinetune.prompts import sft_text

TOKENIZER_FILES = ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja")


def tokenize_examples(tok, arch, examples, max_len):
    rows, dropped = [], 0
    for ex in examples:
        head, tail = sft_text(arch, ex["prompt"], ex["response"])
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
            ids[i, :k] = torch.as_tensor(b["input_ids"])
            labels[i, :k] = torch.as_tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class PackedDataset(torch.utils.data.Dataset):
    """[조각 수, seq_len] 텐서. 모든 토큰에 손실 (일반 언어모델 학습)."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        ids = self.chunks[i].long()
        return {"input_ids": ids, "labels": ids}


def load_for_training(path, device_map="auto"):
    cfg = AutoConfig.from_pretrained(path)
    kw = {}
    qc = getattr(cfg, "quantization_config", None)
    if qc and (qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)) == "mxfp4":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map,
                                                attn_implementation="eager", **kw)


def lora_config(c, model, new_token_ids=()):
    modules, params = arch_of(model).lora_targets(model)
    extra = {"trainable_token_indices": {"embed_tokens": list(new_token_ids)}} if new_token_ids else {}
    return LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=modules, target_parameters=params if c["lora_experts"] else None, **extra)


def new_token_ids(model_dir) -> list[int]:
    p = Path(model_dir) / "vocab_info.json"
    return json.loads(p.read_text(encoding="utf-8"))["new_token_ids"] if p.exists() else []


def train_args(out_dir, s, max_steps, cpu, save_steps, **kw):
    return TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=s["batch_size"], gradient_accumulation_steps=s["grad_accum"],
        num_train_epochs=s.get("epochs", 1), max_steps=max_steps, learning_rate=s["lr"], lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.0, bf16=not cpu, logging_steps=10, save_strategy="steps",
        save_steps=save_steps, save_total_limit=1, report_to="none", remove_unused_columns=False, use_cpu=cpu, **kw)


def run(trainer, ckpt_dir):
    last = get_last_checkpoint(str(ckpt_dir)) if Path(ckpt_dir).is_dir() else None
    if last:
        print(f"이어서 학습: {last}")
    return trainer.train(resume_from_checkpoint=last)


def train_new_embeddings(model, ids, make_chunks, s, ckpt, max_steps=-1, cpu=False):
    """1단계: 모델은 고정하고 새 토큰 임베딩(입력·출력 공유)만 학습한 뒤 합친다. 결과는 ckpt/final 에 저장.

    make_chunks: 학습 데이터([조각 수, seq_len] 텐서)를 만드는 함수. 이미 끝난 경우엔 부르지 않는다.
    """
    final = Path(ckpt) / "final"
    if (final / "adapter_config.json").exists():
        print(f"1단계 결과 불러옴: {final}")
        return PeftModel.from_pretrained(model, final).merge_and_unload(), None
    model = get_peft_model(model, TrainableTokensConfig(token_indices=list(ids), target_modules=["embed_tokens"]))
    model.print_trainable_parameters()
    chunks = make_chunks()
    print(f"1단계 학습 토큰 {chunks.numel():,}개 ({len(chunks)}조각)")
    trainer = Trainer(model=model, args=train_args(Path(ckpt) / "steps", s, max_steps, cpu, s.get("save_steps", 200)),
                      train_dataset=PackedDataset(chunks))
    result = run(trainer, Path(ckpt) / "steps")
    model.save_pretrained(final)
    log = {"train_loss": result.training_loss, "steps": result.global_step, "tokens": chunks.numel(),
           "metrics": result.metrics, "log_history": trainer.state.log_history}
    return model.merge_and_unload(), log


def save_merged(model, src_dir, dst):
    merged = model.merge_and_unload()
    merged.save_pretrained(dst, safe_serialization=True)
    cfg_path = Path(dst) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("quantization_config", None)  # (gpt-oss) BF16로 저장됐으므로 MXFP4 표시 제거
    cfg["dtype"] = cfg["torch_dtype"] = "bfloat16"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    # 토크나이저는 학습 전 폴더의 파일을 그대로 쓴다 (llama.cpp 변환기가 읽는 tokenizer.model 등이 바뀌지 않게)
    for p in Path(src_dir).iterdir():
        if p.is_file() and (p.name in TOKENIZER_FILES or not (Path(dst) / p.name).exists()) \
                and p.suffix not in (".safetensors", ".bin") and p.name != "model.safetensors.index.json":
            shutil.copy2(p, Path(dst) / p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True, help="프루닝(+어휘 확장)한 체크포인트 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None, help="중간 저장 폴더 (기본: {out}/ckpt). 드라이브에 두면 끊겨도 이어서 학습")
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--stage1-tokens", type=int, default=None, help="1단계 학습 토큰 수 (기본: config vocab.stage1.n_tokens)")
    ap.add_argument("--max-steps", type=int, default=-1, help="2단계 스텝 제한 (시범 측정용)")
    ap.add_argument("--stage1-max-steps", type=int, default=-1, help="1단계 스텝 제한 (시범 측정용)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="작은 테스트 모델용")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["lora"]
    out = Path(args.out)
    ckpt = Path(args.ckpt_dir) if args.ckpt_dir else out / "ckpt"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_for_training(args.model, device_map="cpu" if args.cpu else "auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    ids = new_token_ids(args.model)
    log = {"lora": c, "n_new_tokens": len(ids)}

    if ids:
        s = cfg["vocab"]["stage1"]
        n_tok = args.stage1_tokens if args.stage1_tokens is not None else s["n_tokens"]
        if n_tok > 0:
            model, log["stage1"] = train_new_embeddings(
                model, ids, lambda: packed_chunks(s, tok, n_tok, s["seq_len"]), s, ckpt / "stage1",
                args.stage1_max_steps, args.cpu)

    examples = list(sft_examples(c["dataset"], args.n_examples or c["n_examples"], file=c.get("dataset_file")))
    rows, dropped = tokenize_examples(tok, arch_of(model), examples, c["max_len"])
    print(f"2단계 학습 예제 {len(rows)}개 (너무 길어 제외 {dropped}개)")
    model = get_peft_model(model, lora_config(c, model, ids))
    model.print_trainable_parameters()
    trainer = Trainer(model=model, args=train_args(ckpt / "stage2", c, args.max_steps, args.cpu, c.get("save_steps", 50),
                                                   train_sampling_strategy="group_by_length"),
                      train_dataset=rows, data_collator=Collator(tok.pad_token_id))
    result = run(trainer, ckpt / "stage2")

    model.save_pretrained(out / "adapter")
    log.update({"train_loss": result.training_loss, "steps": result.global_step, "n_examples": len(rows),
                "metrics": result.metrics, "log_history": trainer.state.log_history})
    (out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    (RESULTS / "lora").mkdir(parents=True, exist_ok=True)
    (RESULTS / "lora" / f"{out.name}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))

    if not args.no_merge:
        save_merged(model, args.model, out / "merged")
        print(f"합친 모델 저장: {out / 'merged'}")


if __name__ == "__main__":
    main()
