#!/usr/bin/env python3
"""
inference/infer.py
==================
推理样例：加载 Ling-3.0-tiny 基座 + 微调后的 LoRA 适配器，做中文 NL2SQL 对话式生成。

依赖: pip install -U transformers peft accelerate bitsandbytes

用法:
    # 仅基座 (未微调)
    python3 inference/infer.py --question "查询金额最高的订单号、客户姓名和金额"

    # 加载 LoRA 适配器 (微调后)
    python3 inference/infer.py \
        --adapter ../train/output \
        --question "统计每个城市的客户数量"

    # 指定模型/显存
    python3 inference/infer.py --base-model inclusionAI/Ling-3.0-tiny --load-4bit --adapter ../train/output --question "..."

说明: 默认使用数据脚本内置的电商 schema；如使用自定义 schema 可用 --schema-file 指定 .sql/.txt 文件。
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SCHEMA = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, level TEXT, register_date TEXT);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, total_amount REAL);
CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL);
CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, order_id INTEGER, payment_date TEXT, amount REAL, method TEXT);"""

SYSTEM_PROMPT = (
    "你是一个中文 NL2SQL 助手。根据用户的中文问题和下方数据库结构，生成一条可执行的 SQL 查询语句。"
    "只输出 SQL，不要输出任何解释或多余文字。"
)


def build_messages(schema: str, question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{schema}\n\n问题：{question}"},
    ]


def manual_format(messages) -> str:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="inclusionAI/Ling-3.0-tiny")
    ap.add_argument("--adapter", type=str, default=None, help="LoRA 适配器目录 (可选)")
    ap.add_argument("--load-4bit", action="store_true", help="以 4-bit 量化加载基座 (显存有限时)")
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--schema-file", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    schema = DEFAULT_SCHEMA
    if args.schema_file:
        schema = Path(args.schema_file).read_text(encoding="utf-8")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("[infer] loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("[infer] loading base model...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.adapter:
        print(f"[infer] loading LoRA adapter {args.adapter}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    messages = build_messages(schema, args.question)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = manual_format(messages) + "<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
    sql = sql.split(";")[0].strip() + ";"
    print(sql)


if __name__ == "__main__":
    main()