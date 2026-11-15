"""AReno SFT loader for the text-LoRA arm: same prompt as the base model, thinking off, one-word target."""

from __future__ import annotations

import os

from areno.api.data_utils import apply_chat_template, encode_prompt_value
from areno.api.tokenizer import configure_chat_template_enable_thinking, load_tokenizer


def load_training_dataset(dataset_path: str, *, default_loader, model_path: str | None = None, **_: object) -> list[dict]:
    model_dir = model_path or os.environ.get("LING_MODEL_DIR")
    if not model_dir:
        raise ValueError("Pass model_path from areno train or set LING_MODEL_DIR")
    tokenizer = load_tokenizer(model_dir)
    configure_chat_template_enable_thinking(tokenizer, False)
    rows = []
    for row in default_loader(dataset_path):
        messages = row["messages"]
        tokens = apply_chat_template(tokenizer, messages)
        prompt = encode_prompt_value(tokenizer, messages[:-1])
        if tokens[: len(prompt)] != prompt or len(prompt) >= len(tokens):
            raise ValueError("chat template prompt does not match the training sequence")
        rows.append({"tokens": tokens, "prompt_mask": [index < len(prompt) for index in range(len(tokens))]})
    return rows
