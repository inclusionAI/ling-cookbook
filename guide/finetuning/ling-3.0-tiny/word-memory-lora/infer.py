#!/usr/bin/env python3
"""Minimal Base/LoRA inference example with the validated causal backend."""
import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter")
p.add_argument("--word", default="photosynthesis")
p.add_argument("--max-new-tokens", type=int, default=384)
args = p.parse_args()
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
    dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
    attn_implementation="eager")
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
model.config.use_cache = True
instruction = (f"请给出英语单词「{args.word}」的记忆方法，并附上释义和例句帮助我记住它。"
    "只输出【释义】【记忆方法】【例句】三段，不要加标题、编号、emoji 或额外解释。")
prompt = tok.apply_chat_template([{"role": "user", "content": instruction}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.to("cuda:0") for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
        do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
