#!/usr/bin/env python3
"""Causal LoRA SFT for Ling-3.0-tiny word-memory records.

Use eager for training and inference unless a fixed-right-padded SDPA run has
separately passed reload-and-generate evaluation. Ling's custom MLA can lack a
causal mask for unpadded SDPA inputs. Prompt labels are masked with -100.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
    "f_proj", "b_proj", "g_proj", "dense",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-length", type=int, default=896)
    p.add_argument("--padding", choices=["dynamic", "fixed"], default="dynamic",
        help="dynamic reproduces the validated pilot; fixed is an unvalidated speed experiment")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.output_dir)
    if out.joinpath("final").exists() and not args.resume_from_checkpoint:
        raise FileExistsError(f"Refusing to overwrite final adapter: {out / 'final'}")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(line) for line in Path(args.train_file).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    records, truncated = [], 0
    for row in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": row["instruction"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(prompt + row["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        if len(f_ids) > args.max_length:
            truncated += 1
            f_ids, labels = f_ids[:args.max_length], labels[:args.max_length]
        n = args.max_length - len(f_ids) if args.padding == "fixed" else 0
        records.append({"input_ids": f_ids + [tok.pad_token_id] * n,
            "labels": labels + [-100] * n,
            "attention_mask": [1] * len(f_ids) + [0] * n})
    config = vars(args).copy()
    config.update({"records": len(records), "truncated": truncated,
        "target_modules": TARGET_MODULES,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps})
    out.joinpath("run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_CONFIG=" + json.dumps(config, ensure_ascii=False), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
        target_modules=TARGET_MODULES, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    targs = TrainingArguments(output_dir=str(out), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, logging_first_step=True, save_strategy="steps" if args.max_steps < 0 else "no",
        save_steps=args.save_steps, save_total_limit=2, bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=0, dataloader_pin_memory=True, report_to=[],
        remove_unused_columns=False, seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(records),
        data_collator=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id))
    started = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    summary = {"adapter": str(final), "seconds": time.time()-started,
        "global_step": trainer.state.global_step,
        "train_loss": trainer.state.log_history[-1].get("train_loss"),
        "rank": args.rank, "attn_implementation": args.attn_implementation,
        "padding": args.padding, "max_length": args.max_length}
    out.joinpath("run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
