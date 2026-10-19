#!/usr/bin/env python3
"""
train/train.py
===============
基于 HF PEFT + TRL 的 QLoRA 微调脚本 (4-bit NF4), 面向中文 NL2SQL 任务。

依赖: pip install -U transformers peft trl accelerate bitsandbytes datasets pyyaml

用法:
    python3 train/train.py --config train/config.yaml
    python3 train/train.py --base-model inclusionAI/Ling-3.0-tiny --epochs 2 --lr 5e-5

产物:
    {output_dir}/adapter_model.safetensors 等 LoRA 权重 (可直接 push 到 Hugging Face 托管)。
"""

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(base: str, rel: str) -> str:
    """相对路径以 config.yaml 所在目录为基准解析。"""
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str((Path(base).resolve().parent / p).resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.base_model:
        cfg["base_model"] = args.base_model
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.max_seq_length is not None:
        cfg["training"]["max_seq_length"] = args.max_seq_length
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    base_dir = Path(args.config).resolve().parent
    train_file = resolve(args.config, cfg["train_file"])
    val_file = resolve(args.config, cfg["val_file"])
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] base_model={cfg['base_model']}")
    print(f"[train] train_file={train_file}")
    print(f"[train] val_file={val_file}")
    print(f"[train] output_dir={output_dir}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    q = cfg["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16")),
    )

    print("[train] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora.get("target_modules") or "all-linear",
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
    )

    print("[train] loading datasets...")
    dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})

    def _manual_format(messages) -> str:
        parts = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        return "".join(parts)

    use_chat_template = bool(getattr(tokenizer, "chat_template", None))

    def formatting_func(example):
        messages = example["messages"]
        if use_chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        return {"text": _manual_format(messages)}

    dataset = dataset.map(formatting_func)
    if "val" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=cfg.get("seed", 42))
        train_ds, val_ds = dataset["train"], dataset["test"]
    else:
        train_ds, val_ds = dataset["train"], dataset["val"]

    tr = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_ratio=tr["warmup_ratio"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        fp16=tr.get("fp16", False),
        bf16=tr.get("bf16", False),
        logging_steps=tr["logging_steps"],
        save_strategy=tr["save_strategy"],
        save_steps=tr["save_steps"],
        eval_strategy=tr.get("eval_strategy", tr["save_strategy"]),
        eval_steps=tr.get("eval_steps", tr["save_steps"]),
        save_total_limit=tr["save_total_limit"],
        load_best_model_at_end=tr.get("load_best_model_at_end", False),
        report_to=tr.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=tr["max_seq_length"],
    )

    print("[train] training...")
    trainer.train()

    print(f"[train] saving LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[train] done. 可通过 `trainer.model.push_to_hub(...)` 或手动上传 LoRA 权重到 Hugging Face。")


if __name__ == "__main__":
    main()