#!/usr/bin/env python3
"""Static validator for the single public 600-record corpus (no GPU required)."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re

ACTIONS = {"list_files", "read_file", "search_files", "text_retrieval", "reject_none"}
SPLITS = {"train": 480, "validation": 60, "test": 60}
MODES = {"zh", "en", "mixed", "machine"}


def validate(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 600, f"Expected 600 rows, got {len(rows)}"
    assert len({row["id"] for row in rows}) == 600, "Duplicate IDs"
    assert all(row["id"].startswith("ling-") for row in rows), "Invalid ID prefix"
    assert all(re.fullmatch(r"ling-(train|validation|test)-[a-z_]+-[0-9]{3}-(zh|en|mixed|machine)", row["id"]) for row in rows), "Invalid final ID format"
    assert all("dataset_version" not in row["metadata"] for row in rows), "Unexpected release-specific metadata"
    assert Counter(row["split"] for row in rows) == SPLITS, "Incorrect split counts"
    assert Counter(row["category"] for row in rows) == {a: 120 for a in ACTIONS}, "Incorrect action counts"

    for split, n in SPLITS.items():
        subset = [r for r in rows if r["split"] == split]
        expected_action = n // len(ACTIONS)
        expected_mode = n // len(MODES)
        assert Counter(r["category"] for r in subset) == {a: expected_action for a in ACTIONS}
        assert Counter(r["metadata"]["input_mode"] for r in subset) == {m: expected_mode for m in MODES}
        assert len({r["messages"][1]["content"] for r in subset}) == n, "Duplicate prompts within a split"

    for row in rows:
        messages = row["messages"]
        assert len(messages) == 3 and [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert row["target"]["name"] == row["category"] and isinstance(row["target"]["arguments"], dict)
        canonical = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        assert messages[2]["content"] == canonical, f"Target mismatch: {row['id']}"
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        assert digest == row["metadata"]["sample_sha256"], f"Message digest mismatch: {row['id']}"
    print("PASS: 600 rows; 480/60/60 split; 4+1 balanced actions; 4 input modes; unique IDs/prompts; target/message hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path(__file__).parent / "data" / "ling-dataset-all-600.jsonl")
    validate(parser.parse_args().file)
