#!/usr/bin/env python3
"""Train a 4-bit NF4 QLoRA adapter on Ling-3.0-tiny with prompt loss masking."""

import argparse
import json
import os
import time
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

# Mitigate CUDA allocator fragmentation for Ling-3.0-tiny's 157,184-token vocabulary logits
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="4-bit NF4 QLoRA fine-tuning for Ling-3.0-tiny tool calling.")
    parser.add_argument("--model-id", default="inclusionAI/Ling-3.0-tiny", help="Base model ID or local path")
    parser.add_argument("--train-file", default="data/train_480.jsonl", help="Training JSONL file")
    parser.add_argument("--output-dir", default="runs/ling/adapter", help="Directory to save PEFT adapter")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device micro-batch size (8 for L4 22GB, 4 for T4 15GB)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (keep batch_size * grad_accum = 16)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(args.train_file)

    # Mask prompt tokens with -100 so cross-entropy loss is computed strictly on the assistant JSON + <|role_end|>
    def tokenize_and_mask(sample: dict) -> dict:
        prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(sample["text"], truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: len(full_ids)]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    train_dataset = Dataset.from_list([tokenize_and_mask(row) for row in train_rows])

    def collate_pad(batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {args.model_id} in 4-bit NF4 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collate_pad,
        args=training_args,
    )

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    # Note: Step 1 takes ~3.5 minutes while flash-linear-attention JIT-compiles Triton kernels.
    trainer.train()
    elapsed = round(time.time() - t0, 2)
    peak_alloc = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
    peak_resv = round(torch.cuda.max_memory_reserved(0) / (1024**3), 2)
    print(f"Training finished in {elapsed}s | Peak VRAM allocated={peak_alloc} GB | reserved={peak_resv} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved QLoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
