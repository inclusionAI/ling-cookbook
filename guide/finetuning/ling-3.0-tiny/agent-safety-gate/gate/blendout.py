"""Write a blend of arms as a new arm, so freeze.py and test_report.py treat it like one model."""

from __future__ import annotations

import argparse
import json
import math
import os

from blend import load, logit

BENCHES = ("preex", "asse-security", "asse-safety", "rjudge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--blend", required=True, help="arm:weight+arm:weight")
    parser.add_argument("--pred-dir", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    parts = [item.split(":") for item in args.blend.split("+")]
    arms, weights = [p[0] for p in parts], [float(p[1]) if len(p) > 1 else 1.0 for p in parts]
    for bench in BENCHES:
        tables = [load(args.pred_dir, arm, bench) for arm in arms]
        keys = [key for key in tables[0] if all(key in table for table in tables)]
        with open(os.path.join(args.out_dir, f"{args.arm}.{bench}.jsonl"), "w", encoding="utf-8") as handle:
            for key in keys:
                value = sum(w * logit(t[key]["prob"]) for w, t in zip(weights, tables)) / sum(weights)
                prob = 1 / (1 + math.exp(-value))
                row = {"id": tables[0][key]["id"], "prediction": int(prob >= 0.5), "prob": prob}
                handle.write(json.dumps(row) + "\n")
        print(args.arm, bench, len(keys), flush=True)


if __name__ == "__main__":
    main()
