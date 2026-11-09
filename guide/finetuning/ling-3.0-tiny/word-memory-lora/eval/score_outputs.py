#!/usr/bin/env python3
"""CPU-only structural scoring; does not claim factual or teaching quality."""
import argparse
import json
import re
from pathlib import Path

TAGS = ["【释义】", "【记忆方法】", "【例句】"]

def score_case(case):
    response = case["response"]
    starts = response.startswith(TAGS[0])
    ordered = all(tag in response for tag in TAGS) and [response.index(tag) for tag in TAGS] == sorted(response.index(tag) for tag in TAGS)
    only_tags = re.sub(r"[【】\s]", "", response) == "释义记忆方法例句"
    sections = {}
    for index, tag in enumerate(TAGS):
        start = response.find(tag)
        end = response.find(TAGS[index+1]) if index < 2 else len(response)
        sections[tag] = response[start+len(tag):end].strip() if start >= 0 and end >= 0 else ""
    nonempty = all(len(value) >= 8 for value in sections.values())
    english_example = bool(re.search(r"[A-Za-z]{2,}(?:\s+[A-Za-z][A-Za-z'-]*){3,}", sections["【例句】"]))
    return {"id": case["id"], "split": case.get("split", "held_out"),
        "starts_with_meaning_tag": starts, "ordered_three_tags": ordered,
        "all_sections_nonempty": nonempty, "english_example_present": english_example,
        "valid": starts and ordered and nonempty and not only_tags and english_example,
        "generated_tokens": case.get("generated_tokens"),
        "stopped_at_eos": case.get("stopped_at_eos")}

def summarize(records, split="held_out"):
    selected = [r for r in records if r.get("split", "held_out") == split]
    if not selected:
        raise ValueError(f"No records for split={split}")
    scored = [score_case(record) for record in selected]
    keys = ["starts_with_meaning_tag", "ordered_three_tags", "all_sections_nonempty",
            "english_example_present", "valid", "stopped_at_eos"]
    return {"cases": len(scored), "metrics": {key: {"passed": sum(bool(x[key]) for x in scored),
        "rate": sum(bool(x[key]) for x in scored)/len(scored)} for key in keys},
        "per_case": scored,
        "scope": "Structural instruction-following checks only; no factual or pedagogical quality claim."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="held_out", choices=["held_out", "train"])
    args = parser.parse_args()
    result = {}
    for path in args.records:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        key = "base" if "base" in Path(path).stem else "lora"
        result[key] = summarize(data, args.split)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
