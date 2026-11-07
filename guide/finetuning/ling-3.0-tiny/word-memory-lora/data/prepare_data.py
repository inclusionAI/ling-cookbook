#!/usr/bin/env python3
"""Validate user-owned JSONL records and split by a stable word hash.

The source books and their derived train/test files are not redistributed.
Each input row needs word, instruction and output; phonetic/source are optional.
"""
import argparse
import hashlib
import json
from pathlib import Path

TAGS = ["【释义】", "【记忆方法】", "【例句】"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.08)
    args = p.parse_args()
    if not 0 < args.test_ratio < 1:
        p.error("--test-ratio must be between 0 and 1")
    rows, seen, skipped = [], set(), 0
    for path in args.inputs:
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            word = str(row.get("word", "")).strip()
            output = str(row.get("output", "")).strip()
            if not word or word in seen or not row.get("instruction") or not all(tag in output for tag in TAGS):
                skipped += 1
                continue
            seen.add(word)
            rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = [], []
    for row in rows:
        bucket = int(hashlib.md5(row["word"].encode("utf-8")).hexdigest()[:8], 16) % 100
        (test if bucket < args.test_ratio * 100 else train).append(row)
    for name, records in [("train.jsonl", train), ("test.jsonl", test)]:
        out.joinpath(name).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    meta = {"input_records": len(rows), "train": len(train), "test": len(test),
        "skipped": skipped, "test_ratio": args.test_ratio, "split": "md5(word) first 8 hex digits modulo 100"}
    out.joinpath("meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
