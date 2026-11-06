"""Query an OpenAI-compatible endpoint on FDB decision points and score the actions.

Example:
    python eval/run_eval.py --base-url http://127.0.0.1:8100 --data data/fdb_eval.jsonl \
        --out results/baseline_nothink.jsonl --max-tokens 128
    python eval/run_eval.py --score-only --out results/baseline_nothink.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdb_brain.prompt import ACTIONS, build_messages, parse_action  # noqa: E402

FLOOR_TAKING = {"SPEAK", "STOP"}


async def query(args) -> None:
    points = [json.loads(line) for line in open(args.data)]
    if args.limit:
        points = points[: args.limit]
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path)}
    todo = [p for p in points if p["id"] not in done]
    print(f"{len(points)} points, {len(done)} cached, {len(todo)} to query", flush=True)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=900) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        out = open(out_path, "a")

        async def one(p: dict) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/v1/chat/completions", json={
                    "model": model,
                    "messages": build_messages(p),
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                })
                r.raise_for_status()
                data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            pred, reply = parse_action(text)
            rec = {**p, "pred": pred, "reply": reply, "raw": text,
                   "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                   "latency_s": round(time.perf_counter() - t0, 3)}
            async with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

        await asyncio.gather(*(one(p) for p in todo))
        out.close()


def score(path: str) -> dict:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    correct = [r["pred"] in r["accept"] for r in rows]
    report = {"n": n, "accuracy": sum(correct) / n,
              "parse_fail_rate": sum(r["pred"] is None for r in rows) / n,
              "mean_completion_tokens": sum(r["completion_tokens"] or 0 for r in rows) / n,
              "mean_latency_s": sum(r["latency_s"] for r in rows) / n}

    # Floor-taking errors (the FDB "takeover rate" analogue): the user is mid-turn
    # (gold LISTEN/BACKCHANNEL) or the speech is not for us (gold IGNORE), yet the
    # model speaks over it.
    def rate(golds: set[str], preds: set[str]) -> float | None:
        sel = [r for r in rows if r["gold"] in golds]
        return sum(r["pred"] in preds for r in sel) / len(sel) if sel else None

    report["takeover_rate_midturn"] = rate({"LISTEN", "BACKCHANNEL"}, FLOOR_TAKING)
    report["false_response_rate_ignore"] = rate({"IGNORE"}, FLOOR_TAKING)
    report["respond_rate_end_of_turn"] = rate({"SPEAK"}, {"SPEAK"})
    report["stop_rate_interruption"] = rate({"STOP"}, {"STOP"})

    per_label = {}
    for a in ACTIONS:
        tp = sum(r["pred"] == a and a in r["accept"] for r in rows)
        fp = sum(r["pred"] == a and a not in r["accept"] for r in rows)
        fn = sum(r["gold"] == a and r["pred"] not in r["accept"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per_label[a] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                        "support": sum(r["gold"] == a for r in rows)}
    report["per_label"] = per_label
    report["macro_f1"] = sum(v["f1"] for v in per_label.values() if v["support"]) / sum(
        1 for v in per_label.values() if v["support"])

    by_src = defaultdict(list)
    for r, c in zip(rows, correct):
        by_src[f'{r["source"]}:{r["gold"]}'].append(c)
    report["per_source_accuracy"] = {k: sum(v) / len(v) for k, v in sorted(by_src.items())}
    report["confusion"] = {g: dict(Counter(str(r["pred"]) for r in rows if r["gold"] == g)) for g in ACTIONS}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--data", default="data/fdb_eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    if not args.score_only:
        asyncio.run(query(args))
    report = score(args.out)
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_label", "confusion")}, indent=2))
    print("macro_f1", round(report["macro_f1"], 4))
    print("confusion (gold -> pred):")
    for g, c in report["confusion"].items():
        print(f"  {g:12s} {c}")


if __name__ == "__main__":
    main()
