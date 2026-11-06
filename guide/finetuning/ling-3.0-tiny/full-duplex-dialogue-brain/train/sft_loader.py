"""AReno SFT dataset loader: rows are pre-tokenized `tokens` + `prompt_mask` (see data/build_train.py)."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    return [{"tokens": list(r["tokens"]), "prompt_mask": list(r["prompt_mask"])} for r in default_loader(dataset_path)]
