#!/usr/bin/env python3
"""Deterministic generation for Base or PEFT LoRA on independent cases."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "lora"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="eager")
    args = parser.parse_args()
    if args.mode == "lora" and not args.adapter:
        parser.error("--adapter is required in lora mode")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one idle GPU")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation)
    if args.mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    records, started = [], time.time()
    for index, case in enumerate(cases, 1):
        prompt = tok.apply_chat_template([{"role": "user", "content": case["instruction"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to("cuda:0") for k, v in encoded.items()
                  if k in ("input_ids", "attention_mask")}
        t0 = time.time()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(generated, skip_special_tokens=True)
        record = {**case, "mode": args.mode, "response": text,
            "generated_tokens": len(generated), "elapsed_seconds": round(time.time()-t0, 3),
            "stopped_at_eos": bool(len(generated) and generated[-1].item() == tok.eos_token_id)}
        records.append(record)
        print(f"GENERATED mode={args.mode} {index}/{len(cases)} id={case['id']} "
              f"tokens={len(generated)} eos={record['stopped_at_eos']}", flush=True)
        Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_SUMMARY=" + json.dumps({"mode": args.mode, "model": args.model,
        "adapter": args.adapter, "cases": len(records), "total_seconds": round(time.time()-started, 3),
        "gpu": torch.cuda.get_device_name(0), "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy", "attn_implementation": args.attn_implementation}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
