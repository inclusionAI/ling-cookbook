#!/usr/bin/env python3
"""Recalculate archived 60-case Base/LoRA results without a GPU.

Metric definitions:
  - first_object_json: first decodable JSON object has tool:str, arguments:dict
  - routing: parsed tool equals recorded expected_tool
  - reject_hit: among expected_tool=reject_none cases, predicted tool is reject_none
No parameter-level exact match can be reconstructed without original expected_arguments.
"""
import argparse
import json
from pathlib import Path


def first_tool_call(raw: str):
    decoder = json.JSONDecoder()
    for idx, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("tool"), str) and isinstance(value.get("arguments"), dict):
            return value
        # A nested object without a tool header is not the intended outer call.
        return None
    return None


def score(traces):
    total = len(traces)
    valid = routed = reject_cases = reject_hits = strict_full_response = 0
    errors=[]
    for rec in traces:
        raw = rec["raw_output"]
        try:
            v = json.loads(raw)
            strict_full_response += isinstance(v, dict) and isinstance(v.get("tool"), str) and isinstance(v.get("arguments"), dict)
        except json.JSONDecodeError:
            pass
        tool = first_tool_call(raw)
        valid += tool is not None
        correct = tool is not None and tool["tool"] == rec["expected_tool"]
        routed += correct
        if rec["expected_tool"] == "reject_none":
            reject_cases += 1
            reject_hits += correct
        if not correct:
            errors.append(rec["sample_id"])
    return {
        "total": total,
        "first_object_json": valid,
        "first_object_json_pct": round(valid / total * 100, 1) if total else 0,
        "routing_correct": routed,
        "routing_pct": round(routed / total * 100, 1) if total else 0,
        "reject_case_hits": reject_hits,
        "reject_case_total": reject_cases,
        "reject_case_hit_rate_pct": round(reject_hits / reject_cases * 100, 1) if reject_cases else None,
        "strict_full_response_json": strict_full_response,
        "routing_failures": errors,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--traces",type=Path,default=Path(__file__).parent/"evidence"/"recorded-traces.json")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    payload=json.loads(args.traces.read_text(encoding="utf-8"))
    result={
        "base": score(payload["base_traces"]),
        "lora": score(payload["lora_traces"]),
        "metric_definitions": payload["score_contract"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=="__main__":
    main()
