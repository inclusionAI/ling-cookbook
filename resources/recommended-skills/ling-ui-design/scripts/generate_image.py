#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from _common import (
    DEFAULT_API_BASE,
    artifact_path,
    env,
    image_data_url,
    image_items,
    item_bytes,
    parse_size,
    post_json,
    require_env,
    validate_image_bytes,
)


DEFAULT_MODEL = "inclusionai/ming-image-0.1-design"
DEFAULT_SIZE = "2048x2048"


def build_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    parse_size(args.size)
    output_format = "jpeg" if args.format == "jpg" else args.format
    if args.image:
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": args.prompt,
            "size": args.size,
            "images": [{"image_url": image_data_url(args.image, args.resize)}],
            "output_format": output_format,
            "response_format": "b64_json",
            "use_pe": args.use_pe,
            "use_sr": False,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "stream": False,
        }
        return "images/edits", payload
    return "images/generations", {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "output_format": output_format,
        "response_format": "b64_json",
        "enable_thinking": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate or edit a raster image.")
    result.add_argument("--prompt", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--image", help="source image; switches to edit mode")
    result.add_argument("--size", default=env("LING_UI_DESIGN_IMAGE_SIZE", DEFAULT_SIZE))
    result.add_argument("--format", choices=("jpg", "jpeg", "png", "webp"), default="jpeg")
    result.add_argument("--resize", type=int, help="optional edit-input downscale")
    result.add_argument("--seed", type=int, default=248)
    result.add_argument("--steps", type=int, default=30)
    result.add_argument("--use-pe", action="store_true", help="enable upstream PE for edits")
    result.add_argument("--api-base", default=env("LING_UI_DESIGN_API_BASE", DEFAULT_API_BASE))
    result.add_argument("--model", default=env("LING_UI_DESIGN_IMAGE_MODEL", DEFAULT_MODEL))
    return result


def main() -> None:
    args = parser().parse_args()
    output = artifact_path(args.out)
    api_key = require_env("LING_UI_DESIGN_API_KEY")
    route, payload = build_payload(args)
    timeout = float(env("LING_UI_DESIGN_IMAGE_TIMEOUT", "600"))
    started = time.perf_counter()
    body = post_json(
        f"{args.api_base.rstrip('/')}/{route}",
        payload,
        timeout=timeout,
        api_key=api_key,
        multipart=bool(args.image),
    )
    raw = item_bytes(image_items(body)[0], timeout)
    validate_image_bytes(raw, label="generated image")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    elapsed = time.perf_counter() - started
    print(f"[generate_image] OK in {elapsed:.1f}s -> {output} ({len(raw)} bytes)")
    if args.image:
        print("[generate_image] Inspect the edited asset; revise the edit prompt if it is unusable.")
    else:
        print("[generate_image] Inspect the page reference; do not rerun it to fix copy.")


if __name__ == "__main__":
    main()
