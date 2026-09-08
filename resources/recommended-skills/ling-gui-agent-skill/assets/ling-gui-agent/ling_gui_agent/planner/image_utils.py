"""Image input helpers."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image


def resolve_image_url(payload: dict[str, Any], max_pixels: int = 2_621_440) -> str:
    direct = payload.get("image_url")
    if isinstance(direct, str) and _is_remote_or_data_url(direct):
        return direct

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        meta_url = meta.get("image_url")
        if isinstance(meta_url, str) and _is_remote_or_data_url(meta_url):
            return meta_url

    screenshot_path = payload.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path:
        if _is_remote_or_data_url(screenshot_path):
            return screenshot_path
        return _local_file_to_data_url(screenshot_path, max_pixels=max_pixels)
    return ""


def _is_remote_or_data_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://") or value.startswith("data:")


def _local_file_to_data_url(path: str, max_pixels: int) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""

    suffix = p.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "application/octet-stream")

    try:
        with Image.open(p) as img:
            width, height = img.size
            new_width, new_height = _bounded_dimensions(width, height, max_pixels)
            if (new_width, new_height) != (width, height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                fmt = "PNG" if suffix in (".png", ".webp") else "JPEG"
                buf = io.BytesIO()
                img.save(buf, format=fmt, quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            else:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")

    return f"data:{mime};base64,{b64}"


def _bounded_dimensions(
    width: int, height: int, max_pixels: int, multiple: int = 32,
) -> tuple[int, int]:
    """Downscale proportionally within the pixel budget and vision patch grid."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_pixels < multiple * multiple:
        raise ValueError(f"max_pixels must be at least {multiple * multiple}")

    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    scaled_width = max(multiple, int(width * scale) // multiple * multiple)
    scaled_height = max(multiple, int(height * scale) // multiple * multiple)
    return scaled_width, scaled_height
