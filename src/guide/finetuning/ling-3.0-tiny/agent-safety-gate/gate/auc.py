"""Threshold-free discrimination per bench, so a moved decision boundary is not mistaken for progress."""

from __future__ import annotations

import argparse
import os

from benches import load_all
from score import key_of, load_predictions


def auroc(labels, scores) -> float:
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    rank, total, index = 1, 0.0, 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = rank + (end - index) / 2
        total += average * sum(y for _, y in pairs[index : end + 1])
        rank += end - index + 1
        index = end + 1
    return (total - positives * (positives + 1) / 2) / (positives * negatives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--slice", choices=["all", "dev", "test"], default="dev")
    args = parser.parse_args()

    data = load_all(args.bench_root)
    benches = ["preex", "asse-security", "asse-safety", "rjudge"]
    print(f"{'AUROC':16s}" + "".join(f"{bench:>15s}" for bench in benches) + f"{'mean':>8s}")
    for arm in args.arms.split(","):
        row, kept = f"{arm:16s}", []
        for bench in benches:
            path = os.path.join(args.pred_dir, f"{arm}.{bench}.jsonl")
            if not os.path.exists(path):
                row += f"{'-':>15s}"
                continue
            rows = load_predictions(path)
            labels, scores = [], []
            for record in data[bench]:
                if args.slice != "all" and record["split"] != args.slice:
                    continue
                item = rows.get(key_of(record["id"]))
                if item is not None and "prob" in item:
                    labels.append(record["label"])
                    scores.append(item["prob"])
            value = auroc(labels, scores)
            kept.append(value)
            row += f"{value:15.3f}"
        if kept:
            row += f"{sum(kept) / len(kept):8.3f}"
        print(row)


if __name__ == "__main__":
    main()
