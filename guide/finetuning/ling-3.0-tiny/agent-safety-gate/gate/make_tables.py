"""Markdown tables for the final test results: our frozen arms next to every baseline, same test slices.

Our arms come from test_report.py JSON files. Baselines are hard predictions, scored here with the
same metric code. Run only after every arm's test scoring has finished.
"""

from __future__ import annotations

import argparse
import json
import os

from benches import load_all
from metrics import binary_metrics
from score import RELABELLED, key_of, load_predictions

BENCHES = ("preex", "asse-security", "asse-safety", "asse-strict", "asse-loose", "rjudge")
MAIN = ("preex", "asse-security", "asse-safety", "rjudge")
NAMES = {
    "rules": "Keyword rules", "ling_nothink": "Ling-3.0-tiny base (thinking off)", "safiron": "Safiron",
    "shield": "ShieldAgent", "c4b": "11-05 clean adapter (GPU)", "c7j_s130": "Published gate c7j (clean)",
    "allin5": "Research gate, 5-model mean (all folds)",
}


def baseline_row(data, pred_dir, arm):
    row = {}
    for bench in BENCHES:
        path = os.path.join(pred_dir, f"{arm}.{RELABELLED.get(bench, bench)}.jsonl")
        if not os.path.exists(path):
            return None
        preds = load_predictions(path)
        labels, guesses = [], []
        for record in data[bench]:
            if record["split"] != "test":
                continue
            item = preds.get(key_of(record["id"]))
            labels.append(record["label"])
            # a missing or unparsed answer counts as wrong
            guesses.append(1 - record["label"] if item is None or item.get("prediction") is None else int(item["prediction"]))
        row[bench] = binary_metrics(labels, guesses)
    return row


def report_row(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["benches"]


def fmt(row, stat):
    cells = [f"{row[b][stat]:.3f}" for b in BENCHES]
    mean = sum(row[b][stat] for b in MAIN) / len(MAIN)
    return cells, mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--report-glob", required=True, help="prefix of test_report JSON files, e.g. runs/cpu_report_")
    parser.add_argument("--arms", default="allin5,a7j_s110,a11j_s110,b2j_s195,b3j_s110,b4j_s110,c7j_s130")
    args = parser.parse_args()
    data = load_all(args.bench_root)
    rows = []
    for arm in ("rules", "ling_nothink", "safiron", "shield", "c4b"):
        row = baseline_row(data, args.baseline_dir, arm)
        if row:
            rows.append((NAMES.get(arm, arm), row, False))
    for arm in args.arms.split(","):
        path = f"{args.report_glob}{arm}.json"
        if os.path.exists(path):
            rows.append((NAMES.get(arm, arm), report_row(path), True))
    for stat in ("acc", "f1", "mcc"):
        print(f"\n{stat}\n")
        print("| Model | Pre-Ex | ASSE-Security | ASSE-Safety | Strict | Lenient | R-Judge | Mean (4) |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, row, _ in rows:
            cells, mean = fmt(row, stat)
            print(f"| {name} | " + " | ".join(cells) + f" | {mean:.3f} |")
    print("\nAUROC, ECE and the three-way gate (ours only)\n")
    print("| Model | AUROC Pre-Ex / Sec / Safety / R-Judge | ECE mean | leak | friction | ask |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for name, row, ours in rows:
        if not ours:
            continue
        aurocs = " / ".join(f"{row[b]['auroc']:.3f}" for b in MAIN)
        mean = lambda key: sum(row[b][key] for b in MAIN) / len(MAIN)
        print(f"| {name} | {aurocs} | {mean('ece'):.3f} | {mean('leak'):.3f} | {mean('friction'):.3f} | {mean('ask'):.3f} |")


if __name__ == "__main__":
    main()
