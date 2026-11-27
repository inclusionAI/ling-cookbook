"""Freeze everything on dev, write it to disk, and only then score the test slices.

Rules, fixed before any test record was scored:
  * The decision threshold maximizes mean MCC over Pre-Ex, ASSE-Safety and R-Judge dev. ASSE-Security dev
    is left out: those records were first split by exact request, so some near-duplicates reach the training fold.
  * `low` is the largest value that leaks at most 10% of pooled dev unsafe records.
  * `high` is the smallest value that wrongly denies at most 5% of pooled dev safe records.
  * The temperature is fitted on pooled dev by negative log likelihood.
  * Both the generated-only arm and the human-mix arm are reported. Test results never choose a model.
"""

from __future__ import annotations

import argparse
import json
import os

from benches import family_of, load_all
from metrics import binary_metrics
from score import key_of, load_predictions
from tiers import fit_temperature, rescale

BENCHES = ("preex", "asse-safety", "rjudge")
ALL_BENCHES = ("preex", "asse-security", "asse-safety", "rjudge")


def dev_items(data, pred_dir, arm, cal_only=False):
    per_bench = {}
    # The family-unseen half has no near-duplicates, so ASSE-Security can take part in calibration.
    for bench in ALL_BENCHES if cal_only else BENCHES:
        rows = load_predictions(os.path.join(pred_dir, f"{arm}.{bench}.jsonl"))
        items = []
        for record in data[bench]:
            if record["split"] != "dev":
                continue
            if cal_only and int(family_of(record["request"]), 16) % 2 == 0:
                continue
            items.append((record["label"], rows[key_of(record["id"])]["prob"], record.get("ambiguous", False)))
        per_bench[bench] = items
    return per_bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cal-only", action="store_true", help="use only dev records from test families, for arms that trained on the rest")
    args = parser.parse_args()

    data = load_all(args.bench_root)
    per_bench = dev_items(data, args.pred_dir, args.arm, args.cal_only)
    pooled = [item for items in per_bench.values() for item in items]
    temperature = fit_temperature(pooled)
    scaled = {bench: rescale(items, temperature) for bench, items in per_bench.items()}
    pooled_scaled = [item for items in scaled.values() for item in items]

    def mean_mcc(threshold):
        values = [
            binary_metrics([y for y, _, _ in items], [int(p >= threshold) for _, p, _ in items])["mcc"]
            for items in scaled.values()
        ]
        return sum(values) / len(values)

    grid = [round(0.02 * step, 2) for step in range(1, 50)]
    threshold = max(grid, key=mean_mcc)
    unsafe = sorted(p for y, p, _ in pooled_scaled if y == 1)
    safe = sorted(p for y, p, _ in pooled_scaled if y == 0)
    low = unsafe[int(0.10 * len(unsafe))]
    high = safe[min(len(safe) - 1, int(0.95 * len(safe)))]
    frozen = {
        "arm": args.arm,
        "temperature": temperature,
        "threshold": threshold,
        "low": low,
        "high": max(high, low),
        "dev_mean_mcc": mean_mcc(threshold),
        "dev_records": len(pooled),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2)
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
