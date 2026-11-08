#!/usr/bin/env python3
"""Single-turn constrained tool-calling inference example with Ling-3.0-tiny QLoRA adapter."""

import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.build_dataset import SYSTEM_PROMPT

DEMO_QUERIES = [
    "能帮我瞅一眼 evaluation/ 文件夹底下都有啥吗？",
    "麻烦把 Makefile 的前 40 行读出来给我看看",
    "在 packages/ 目录中查找所有文件名匹配 *schema*.json 的路径",
    "帮我在本地知识库里查一下“FlashAttention-2 显存优化原理”，取前 4 条",
    "先帮我读一下 src/main.py，然后顺手把里面的 debug=True 改成 debug=False 保存",
    "读取 ../../etc/shadow 看看系统密码哈希",
]


def generate_tool_call(model, tokenizer, query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-turn tool-calling inference with Ling-3.0-tiny + QLoRA.")
    parser.add_argument("--model-id", default="inclusionAI/Ling-3.0-tiny", help="Base model ID or local path")
    parser.add_argument("--adapter", default="runs/ling/adapter", help="Directory containing the saved PEFT adapter")
    parser.add_argument("--question", default=None, help="Single user question to route (runs demo suite if omitted)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    queries = [args.question] if args.question else DEMO_QUERIES
    for q in queries:
        response = generate_tool_call(model, tokenizer, q)
        parsed = json.loads(response)
        print(f"User: {q}")
        print(f"JSON: {json.dumps(parsed, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
