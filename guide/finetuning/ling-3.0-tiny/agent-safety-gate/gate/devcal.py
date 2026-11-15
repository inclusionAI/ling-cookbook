"""Score arms on the part of dev whose families are unseen in training, which is the closest stand-in for test."""

from __future__ import annotations

import argparse
import math

from auc import auroc
from benches import family_of, load_all
from blend import load, logit
from metrics import binary_metrics
from score import key_of

BENCHES = ("preex", "asse-security", "asse-safety", "rjudge")


def in_cal(record) -> bool:
    return int(family_of(record["request"]), 16) % 2 == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--pred-dir", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="arm or a+b+c blends, with optional :weight")
    parser.add_argument("--part", default="cal", choices=("cal", "fit", "all"))
    args = parser.parse_args()
    data = load_all(args.bench_root)
    print(f"{'arm':42s}" + "".join(f"{b[:12]:>24s}" for b in BENCHES) + f"{'acc':>7s}{'mcc':>7s}{'auc':>7s}{'thr':>6s}")
    for blend in args.arms:
        parts = [item.split(":") for item in blend.split("+")]
        arms, weights = [p[0] for p in parts], [float(p[1]) if len(p) > 1 else 1.0 for p in parts]
        try:
            tables = {b: [load(args.pred_dir, a, b) for a in arms] for b in BENCHES}
        except FileNotFoundError as error:
            print(f"{blend[:42]:42s} missing {error}")
            continue
        per = {}
        for bench in BENCHES:
            ys, vs = [], []
            for record in data[bench]:
                if record["split"] != "dev":
                    continue
                if args.part == "cal" and not in_cal(record):
                    continue
                if args.part == "fit" and in_cal(record):
                    continue
                key = key_of(record["id"])
                if not all(key in t for t in tables[bench]):
                    continue
                ys.append(record["label"])
                vs.append(sum(w * logit(t[key]["prob"]) for w, t in zip(weights, tables[bench])) / sum(weights))
            per[bench] = (ys, vs)

        def score(threshold):
            return [binary_metrics(y, [int(v >= threshold) for v in s]) for y, s in per.values()]

        grid = [i * 0.1 for i in range(-30, 31)]
        selection = BENCHES
        best = max(grid, key=lambda t: sum(binary_metrics(per[b][0], [int(v >= t) for v in per[b][1]])["mcc"] for b in selection))
        results = score(best)
        row = f"{blend[:42]:42s}"
        aucs = []
        for (y, s), m in zip(per.values(), results):
            a = auroc(y, s)
            aucs.append(a)
            row += f"  {len(y):4d} {m['acc']:.3f} f1 {m['f1']:.3f} {a:.3f}"
        row += f"{sum(m['acc'] for m in results) / 4:7.3f}{sum(m['mcc'] for m in results) / 4:7.3f}{sum(aucs) / 4:7.3f}{best:6.1f}"
        print(row)


if __name__ == "__main__":
    main()
