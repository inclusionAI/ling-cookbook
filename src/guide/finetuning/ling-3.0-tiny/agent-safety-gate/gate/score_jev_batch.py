"""Score benches in-process through AReno's JEV scorer, many records per forward pass.

The HTTP endpoint runs one model forward per request. For offline evaluation we batch the
same encoded rows into one forward, which keeps scoring identical and makes it much faster.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from areno.api.jev.scorer import DecisionScorer, answer_from_logits

from benches import load_all, render_state
from jevdata import QUESTIONS
from metrics import binary_metrics, ece


def score_states(scorer, states: list[str], batch: int, question: dict = QUESTIONS["default"]) -> list[float]:
    probabilities: list[float] = []
    for start in range(0, len(states), batch):
        rows, sizes = [], []
        for state in states[start : start + batch]:
            prepared, _ = scorer.prepare(state, {"unsafe": question})
            encoded = prepared[0][2]
            rows.extend(encoded)
            sizes.append(len(encoded))
        logits = scorer.trainer._backend.decision_scores(scorer.trainer._ctx, rows)
        offset = 0
        for size in sizes:
            answer = answer_from_logits(question, logits[offset : offset + size], scorer.temperature)
            probabilities.append(float(answer["noul"]))
            offset += size
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benches", default="preex,asse-security,asse-safety,rjudge")
    parser.add_argument("--slice", choices=["all", "dev", "test"], default="dev")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32, help="records per forward pass")
    parser.add_argument("--val", default="")
    parser.add_argument("--question", choices=sorted(QUESTIONS), default="default")
    args = parser.parse_args()

    scorer = DecisionScorer(args.model, args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        if args.val:
            with open(args.val, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            started = time.time()
            probabilities = score_states(scorer, [row["request"]["state"] for row in rows], args.batch)
            labels = [int(row["targets"]["unsafe"]["true"]) for row in rows]
            overall = binary_metrics(labels, [int(p >= 0.5) for p in probabilities])
            print(
                f"generated val n={len(rows)} acc={overall['acc']:.3f} f1={overall['f1']:.3f} "
                f"ece={ece(labels, probabilities):.3f} in {time.time() - started:.1f}s",
                flush=True,
            )
            families: dict = {}
            for row, label, probability in zip(rows, labels, probabilities):
                for key in (row.get("family", "?"), "lang:" + row.get("lang", "?")):
                    families.setdefault(key, []).append((label, int(probability >= 0.5)))
            for family, items in sorted(families.items()):
                result = binary_metrics([y for y, _ in items], [p for _, p in items])
                print(
                    f"  {family:18s} n={len(items):4d} acc={result['acc']:.3f} miss={result['miss_rate']:.3f} alarm={result['false_alarm']:.3f}",
                    flush=True,
                )
        data = load_all(args.bench_root)
        for bench in filter(None, args.benches.split(",")):
            records = data[bench]
            if args.slice != "all":
                records = [r for r in records if r["split"] == args.slice]
            started = time.time()
            probabilities = score_states(scorer, [render_state(r) for r in records], args.batch, QUESTIONS[args.question])
            elapsed = time.time() - started
            path = os.path.join(args.out_dir, f"{args.arm}.{bench}.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for record, probability in zip(records, probabilities):
                    row = {"id": record["id"], "prediction": int(probability >= args.threshold), "prob": probability}
                    handle.write(json.dumps(row) + "\n")
            print(f"{args.arm} {bench}: {len(records)} in {elapsed:.1f}s", flush=True)
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
