"""Apply a frozen file to test predictions. Run once per frozen arm, after freeze.py wrote its file."""

from __future__ import annotations

import argparse
import json
import math
import os

from auc import auroc
from benches import load_all
from metrics import binary_metrics, ece, group_bootstrap
from score import RELABELLED, key_of, load_predictions


def scale(probability: float, temperature: float) -> float:
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-math.log(probability / (1 - probability)) / temperature))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    with open(args.frozen, encoding="utf-8") as handle:
        frozen = json.load(handle)
    arm, temperature, threshold = frozen["arm"], frozen["temperature"], frozen["threshold"]
    low, high = frozen["low"], frozen["high"]
    data = load_all(args.bench_root)
    report = {"frozen": frozen, "benches": {}}
    print(f"arm={arm} temperature={temperature} threshold={threshold} low={low:.3f} high={high:.3f}")
    print(f"{'bench':14s} {'n':>5s} {'acc':>6s} {'F1':>6s} {'bal':>6s} {'MCC':>6s} {'AUROC':>6s} {'miss':>6s} {'alarm':>6s} {'ECE':>6s} {'leak':>6s} {'frict':>6s} {'ask':>6s}  acc 95% CI")
    for bench in ("preex", "asse-security", "asse-safety", "asse-strict", "asse-loose", "rjudge"):
        path = os.path.join(args.pred_dir, f"{arm}.{RELABELLED.get(bench, bench)}.jsonl")
        rows = load_predictions(path)
        records = [r for r in data[bench] if r["split"] == "test"]
        labels = [r["label"] for r in records]
        scores = [scale(rows[key_of(r["id"])]["prob"], temperature) for r in records]
        predictions = [int(p >= threshold) for p in scores]
        metrics = binary_metrics(labels, predictions)
        unsafe = [p for y, p in zip(labels, scores) if y == 1]
        safe = [p for y, p in zip(labels, scores) if y == 0]
        leak = sum(p < low for p in unsafe) / len(unsafe)
        friction = sum(p >= low for p in safe) / len(safe)
        ask = sum(low <= p < high for p in scores) / len(scores)
        ci_low, ci_high = group_bootstrap(labels, predictions, [r["group"] for r in records])
        roc, calibration = auroc(labels, scores), ece(labels, scores)
        print(
            f"{bench:14s} {metrics['n']:5d} {metrics['acc']:6.3f} {metrics['f1']:6.3f} {metrics['balanced_acc']:6.3f} {metrics['mcc']:6.3f} "
            f"{roc:6.3f} {metrics['miss_rate']:6.3f} {metrics['false_alarm']:6.3f} {calibration:6.3f} {leak:6.3f} {friction:6.3f} {ask:6.3f}  [{ci_low:.3f}, {ci_high:.3f}]"
        )
        report["benches"][bench] = {**metrics, "auroc": roc, "ece": calibration, "leak": leak, "friction": friction, "ask": ask, "acc_ci": [ci_low, ci_high]}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
