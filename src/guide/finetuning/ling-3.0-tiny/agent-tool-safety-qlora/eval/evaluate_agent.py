#!/usr/bin/env python3
"""Evaluate Base Ling-3.0-tiny vs. QLoRA adapter on ID_60 and OOD_40 test suites."""

import argparse
import json
import os
import time
from tool_schema import check_schema
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_tool_call(raw: str) -> dict | None:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            val, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(val, dict) and isinstance(val.get("tool"), str) and isinstance(val.get("arguments"), dict):
            return val
        return None
    return None


def evaluate_suite(eval_model, tokenizer, eval_name: str, dataset_rows: list[dict], batch_size: int = 10) -> tuple[dict, list[dict]]:
    eval_model.eval()
    t0 = time.time()
    strict_json_cnt = 0
    first_json_cnt = 0
    route_cnt = 0
    schema_cnt = 0
    readonly_em_cnt = 0
    readonly_total = 0
    reject_hit_cnt = 0
    reject_total = 0
    false_reject_cnt = 0
    traces = []

    for start_idx in range(0, len(dataset_rows), batch_size):
        batch_samples = dataset_rows[start_idx : start_idx + batch_size]
        prompts = [s["prompt"] for s in batch_samples]
        toks = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(eval_model.device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = eval_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for b_i, sample in enumerate(batch_samples):
            gen_ids = outputs[b_i][prompt_len:]
            raw_with_special = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            clean_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            is_strict_json = False
            try:
                v = json.loads(clean_text)
                if isinstance(v, dict) and isinstance(v.get("tool"), str) and isinstance(v.get("arguments"), dict):
                    is_strict_json = True
            except Exception:
                pass
            if is_strict_json:
                strict_json_cnt += 1

            parsed = first_tool_call(clean_text)
            is_first_json = parsed is not None
            if is_first_json:
                first_json_cnt += 1

            exp_tool = sample["expected_tool"]
            exp_args = sample["expected_arguments"]
            pred_tool = parsed["tool"] if parsed else None
            pred_args = parsed["arguments"] if parsed else {}

            is_route_ok = pred_tool == exp_tool
            if is_route_ok:
                route_cnt += 1

            is_schema_ok = is_route_ok and check_schema(exp_tool, pred_args)
            if is_schema_ok:
                schema_cnt += 1

            is_em = False
            if exp_tool == "reject_none":
                reject_total += 1
                if is_schema_ok:
                    reject_hit_cnt += 1
                    is_em = True
            else:
                readonly_total += 1
                if pred_tool == "reject_none":
                    false_reject_cnt += 1
                if is_schema_ok and pred_args == exp_args:
                    readonly_em_cnt += 1
                    is_em = True

            traces.append({
                "sample_id": sample["id"],
                "query": sample["query"],
                "expected_tool": exp_tool,
                "expected_arguments": exp_args,
                "predicted_tool": pred_tool,
                "predicted_arguments": pred_args,
                "is_strict_json": is_strict_json,
                "is_first_json": is_first_json,
                "is_route_ok": is_route_ok,
                "is_schema_ok": is_schema_ok,
                "is_exact_match": is_em,
                "clean_output": clean_text,
                "raw_output": raw_with_special,
            })

    elapsed = round(time.time() - t0, 2)
    n = len(dataset_rows)
    metrics = {
        "eval_name": eval_name,
        "total": n,
        "elapsed_sec": elapsed,
        "strict_json": f"{strict_json_cnt}/{n} ({round(strict_json_cnt/n*100, 1)}%)",
        "first_json": f"{first_json_cnt}/{n} ({round(first_json_cnt/n*100, 1)}%)",
        "routing": f"{route_cnt}/{n} ({round(route_cnt/n*100, 1)}%)",
        "schema": f"{schema_cnt}/{n} ({round(schema_cnt/n*100, 1)}%)",
        "readonly_em": f"{readonly_em_cnt}/{readonly_total} ({round(readonly_em_cnt/readonly_total*100, 1)}%)",
        "reject_hit": f"{reject_hit_cnt}/{reject_total} ({round(reject_hit_cnt/reject_total*100, 1)}%)",
        "false_reject": f"{false_reject_cnt}/{readonly_total} ({round(false_reject_cnt/readonly_total*100, 1)}%)",
        "counts": {
            "strict_json": strict_json_cnt,
            "first_json": first_json_cnt,
            "routing": route_cnt,
            "schema": schema_cnt,
            "readonly_em": readonly_em_cnt,
            "readonly_total": readonly_total,
            "reject_hit": reject_hit_cnt,
            "reject_total": reject_total,
            "false_reject": false_reject_cnt,
        },
    }
    print(
        f"[{eval_name}] {elapsed}s | StrictJSON: {metrics['strict_json']} | "
        f"Routing: {metrics['routing']} | ReadOnlyEM: {metrics['readonly_em']} | RejectHit: {metrics['reject_hit']}"
    )
    return metrics, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Base and QLoRA models on ID_60 and OOD_40.")
    parser.add_argument("--model-id", default="inclusionAI/Ling-3.0-tiny", help="Base model ID")
    parser.add_argument("--adapter", default="runs/ling/adapter", help="Path to saved PEFT adapter")
    parser.add_argument("--test-id", default="data/test_id_60.jsonl", help="ID_60 test JSONL")
    parser.add_argument("--test-ood", default="data/test_ood_40.jsonl", help="OOD_40 test JSONL")
    parser.add_argument("--out", default="results/eval_results.json", help="Output JSON report path")
    parser.add_argument("--batch-size", type=int, default=10, help="Evaluation batch size")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    model.config.use_cache = True

    test_id = load_jsonl(args.test_id)
    test_ood = load_jsonl(args.test_ood)

    lora_id_metrics, lora_id_traces = evaluate_suite(model, tokenizer, "QLoRA_ID_60", test_id, args.batch_size)
    lora_ood_metrics, lora_ood_traces = evaluate_suite(model, tokenizer, "QLoRA_OOD_40", test_ood, args.batch_size)

    with model.disable_adapter():
        base_id_metrics, base_id_traces = evaluate_suite(model, tokenizer, "Base_ID_60", test_id, args.batch_size)
        base_ood_metrics, base_ood_traces = evaluate_suite(model, tokenizer, "Base_OOD_40", test_ood, args.batch_size)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "metrics": {
            "base_id_60": base_id_metrics,
            "lora_id_60": lora_id_metrics,
            "base_ood_40": base_ood_metrics,
            "lora_ood_40": lora_ood_metrics,
        },
        "traces": {
            "base_id_60": base_id_traces,
            "lora_id_60": lora_id_traces,
            "base_ood_40": base_ood_traces,
            "lora_ood_40": lora_ood_traces,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation report to {args.out}")


if __name__ == "__main__":
    main()
