"""Blend arms by averaging logits, and report dev scores so a blend can be chosen before any test scoring."""

from __future__ import annotations

import argparse
import math
import os

from auc import auroc
from benches import load_all
from metrics import binary_metrics
from score import key_of, load_predictions

BENCHES = ("preex", "asse-security", "asse-safety", "rjudge")


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load(pred_dirs, arm, bench):
    for folder in pred_dirs:
        path = os.path.join(folder, f"{arm}.{bench}.jsonl")
        if os.path.exists(path):
            return load_predictions(path)
    raise FileNotFoundError(f"{arm}.{bench}")


def scores_for(data, pred_dirs, arms, weights, bench, split):
    tables = [load(pred_dirs, arm, bench) for arm in arms]
    labels, values = [], []
    for record in data[bench]:
        if record["split"] != split:
            continue
        key = key_of(record["id"])
        if not all(key in table for table in tables):
            continue
        labels.append(record["label"])
        values.append(sum(w * logit(t[key]["prob"]) for w, t in zip(weights, tables)) / sum(weights))
    return labels, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--pred-dir", nargs="+", required=True)
    parser.add_argument("--blends", nargs="+", required=True, help="arm:weight+arm:weight ...")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    data = load_all(args.bench_root)
    print(f"{'blend':46s}" + "".join(f"{b:>22s}" for b in BENCHES) + f"{'mean acc':>10s}{'mean mcc':>10s}{'thr':>6s}")
    for blend in args.blends:
        parts = [item.split(":") for item in blend.split("+")]
        arms, weights = [p[0] for p in parts], [float(p[1]) if len(p) > 1 else 1.0 for p in parts]
        per = {b: scores_for(data, args.pred_dir, arms, weights, b, args.split) for b in BENCHES}

        def score(threshold):
            return [binary_metrics(y, [int(v >= threshold) for v in s]) for y, s in per.values()]

        grid = [i * 0.1 for i in range(-30, 31)]
        best = max(grid, key=lambda t: sum(m["mcc"] for m in score(t)) / len(BENCHES))
        results = score(best)
        row = f"{blend[:46]:46s}"
        for (y, s), m in zip(per.values(), results):
            row += f"  acc {m['acc']:.3f} auc {auroc(y, s):.3f}  "
        row += f"{sum(m['acc'] for m in results) / 4:10.3f}{sum(m['mcc'] for m in results) / 4:10.3f}{best:6.1f}"
        print(row)


if __name__ == "__main__":
    main()
