"""Build the all-in training sets: generated cases plus the human-labelled folds of every bench.

Writes the same examples in two forms: JEV records for the decision head, and chat rows whose
target is a short analysis followed by a verdict line. Test folds never enter this file.
Pre-Ex is CC BY-NC and R-Judge has no license, so adapters trained on them stay research-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter

from benches import clip, family_of, load_all, norm, render_state
from jevdata import QUESTION, case_records, to_jev
from prompts import ling_analysis_prompt

SAFE_NOTE = "The agent stays within the request and its actions raise no safety problem."
UNSAFE_NOTE = "The agent's actions create a safety risk that the user did not accept."


def analysis_target(label: int, note: str) -> str:
    text = re.sub(r"\s+", " ", note.replace("*", "")).strip()
    text = re.sub(r"\s*-?\s*Risk:.*$", "", text)
    if len(text) > 320:
        cut = text[:320]
        text = cut[: cut.rfind(" ")] + " ..."
    if not text:
        text = UNSAFE_NOTE if label else SAFE_NOTE
    verdict = "unsafe" if label else "safe"
    return f"Analysis: {text}\nVerdict: {verdict}"


def describe_change(case: dict) -> str:
    """Point at the step that differs, so the analysis teaches the model where to look."""
    risk = str(case.get("category") or "unsafe behavior").replace("_", " ")
    safe, unsafe = case["safe_steps"], case["unsafe_steps"]
    for index, step in enumerate(unsafe, 1):
        if index > len(safe) or step != safe[index - 1]:
            return f"Step {index} ({clip(step, 140)}) goes beyond what the request needs and adds a risk of {risk}."
    return f"The final message adds a risk of {risk}."


def trains_on(record: dict, bench: str, rjudge_groups: set, dev_fit: bool) -> bool:
    """Train fold, plus with --dev-fit the dev records from train families.

    Dev records from test families stay out: they share no family with anything trained on,
    so they calibrate thresholds the way test will behave.
    """
    if record["split"] == "train":
        return True
    if not dev_fit or record["split"] != "dev" or int(family_of(record["request"]), 16) % 2:
        return False
    return bench == "rjudge" or record["group"] not in rjudge_groups


def chat(prompt: str, target: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--max-cases", type=int, default=4000)
    parser.add_argument("--prex-cases", default="", help="AuraGen-style cases, research-only")
    parser.add_argument("--prex-max", type=int, default=3000)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--human", default="asse-security,asse-safety,preex,rjudge")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--jev", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--dev-fit", action="store_true")
    parser.add_argument("--rewrites", default="", help="rewrite.py output, each row trained on once")
    parser.add_argument("--rewrite-benches", default="", help="comma list that limits --rewrites to these source benches")
    parser.add_argument("--ambiguous-target", type=float, default=0.5, help="P(unsafe) for a record labelled unsafe by the base and strict labels only")
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen, cases = set(), []
    for path in args.cases:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                case = json.loads(line)
                key = norm(case["request"])
                if key in seen or re.match(r"\s*(you are|as an ai|你是)", case["request"], re.I):
                    continue
                seen.add(key)
                cases.append(case)
    rng.shuffle(cases)
    cases = cases[: args.max_cases]
    if args.prex_cases:
        extra = []
        with open(args.prex_cases, encoding="utf-8") as handle:
            for line in handle:
                case = json.loads(line)
                key = norm(case["request"])
                if key not in seen:
                    seen.add(key)
                    extra.append(case)
        rng.shuffle(extra)
        cases += extra[: args.prex_max]
        rng.shuffle(cases)

    counts: Counter = Counter()
    jev_rows, text_rows = [], []
    with open(args.val, "w", encoding="utf-8") as val:
        for case in cases:
            digest = hashlib.sha1(norm(case["request"]).encode()).hexdigest()[:12]
            held_out = int(digest, 16) % 20 == 0
            for record in case_records(case, rng):
                row = to_jev(record, f"gen-{digest}-{record['label']}", digest)
                row["family"] = case["family"]
                if held_out:
                    val.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                jev_rows.append(row)
                note = SAFE_NOTE
                if record["label"]:
                    note = describe_change(case) if case["family"] in ("prex_style", "plan_style") else case["why"]
                text_rows.append(chat(ling_analysis_prompt(record), analysis_target(record["label"], note)))
            counts["val_pairs" if held_out else "train_pairs"] += 1

    data = load_all(args.bench_root)
    rjudge_groups = {record["group"] for record in data["rjudge"]}
    for name in args.human.split(","):
        for index, record in enumerate(data[name]):
            if not trains_on(record, name, rjudge_groups, args.dev_fit):
                continue
            target = float(record["label"])
            if name == "asse-safety" and data["asse-strict"][index]["label"] != data["asse-loose"][index]["label"]:
                target = args.ambiguous_target if record["label"] else 1 - args.ambiguous_target
            counts[f"human:{name}"] += 1
            counts[f"human:{name}:unsafe"] += record["label"]
            for copy in range(args.repeat):
                jev_rows.append(
                    {
                        "id": f"{record['id']}-copy{copy}",
                        "split": "train",
                        "source_group": record["group"],
                        "family": "human",
                        "request": {"state": render_state(record), "questions": {"unsafe": QUESTION}},
                        "targets": {"unsafe": {"false": 1.0 - target, "true": target}},
                    }
                )
                text_rows.append(chat(ling_analysis_prompt(record), analysis_target(record["label"], record["rationale"])))
    if args.rewrites:
        with open(args.rewrites, encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                row = json.loads(line)
                if args.rewrite_benches and row["bench"] not in args.rewrite_benches.split(","):
                    continue
                counts[f"rewrite:{row['bench']}:{row['mode']}"] += 1
                jev_rows.append(
                    {
                        "id": f"rewrite-{position}",
                        "split": "train",
                        "source_group": row["group"],
                        "family": "rewrite",
                        "request": {"state": render_state(row), "questions": {"unsafe": QUESTION}},
                        "targets": {"unsafe": {"false": 1.0 - row["target"], "true": row["target"]}},
                    }
                )
    with open(args.jev, "w", encoding="utf-8") as handle:
        for row in jev_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.text, "w", encoding="utf-8") as handle:
        for row in text_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(counts), "jev rows", len(jev_rows), "text rows", len(text_rows))


if __name__ == "__main__":
    main()
