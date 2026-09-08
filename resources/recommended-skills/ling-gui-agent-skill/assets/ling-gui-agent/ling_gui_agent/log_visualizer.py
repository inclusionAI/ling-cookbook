"""Regenerate annotated frames, an HTML timeline, and an optional MP4 from a log directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from .executor import GUIExecutor
from .planner.action_result import ActionResult


def _logical_screen_size(log_dir: Path, first_image: Path) -> tuple[int, int]:
    """Infer the logical action coordinate space for common desktop Retina logs."""
    with Image.open(first_image) as image:
        if image.size == (3024, 1964):
            return image.width // 2, image.height // 2
        return image.size


def visualize_log_dir(
    log_dir: str | Path,
    output_dir: str | Path | None = None,
    logical_screen_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build visual artifacts from existing ``step_*`` log files without an API call.

    ``output_dir`` defaults to ``log_dir``. If a separate output directory is
    supplied, action JSON and source screenshots are copied there while the
    original log remains untouched.
    """
    source_dir = Path(log_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"log directory not found: {source_dir}")
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    first_image = next(iter(sorted(source_dir.glob("step_*_in.png"))), None)
    inferred_size = logical_screen_size
    if inferred_size is None and first_image is not None:
        inferred_size = _logical_screen_size(source_dir, first_image)

    executor = object.__new__(GUIExecutor)
    annotated = 0
    for action_path in sorted(source_dir.glob("step_*_action.json")):
        stem = action_path.name.removesuffix("_action.json")
        input_path = source_dir / f"{stem}_in.png"
        if not input_path.is_file():
            continue
        action_data = json.loads(action_path.read_text(encoding="utf-8"))
        extras = dict(action_data.get("extras") or {})
        extras["step_index"] = action_data.get("step_index", annotated + 1)
        result = ActionResult(
            str(action_data.get("action", "")),
            dict(action_data.get("params") or {}),
            extras,
        )
        if target_dir != source_dir:
            shutil.copy2(action_path, target_dir / action_path.name)
            shutil.copy2(input_path, target_dir / input_path.name)
        executor._annotate_screenshot(
            target_dir / input_path.name if target_dir != source_dir else input_path,
            result,
            target_dir / f"{stem}_out.png",
            inferred_size,
        )
        annotated += 1

    executor._finalize_visualization(target_dir)
    return {
        "log_dir": str(source_dir),
        "output_dir": str(target_dir),
        "actions": annotated,
        "html": str(target_dir / "visualization.html"),
        "video": str(target_dir / "sequence.mp4") if (target_dir / "sequence.mp4").is_file() else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="existing run directory containing step_* files")
    parser.add_argument("--output-dir", help="optional separate output directory")
    parser.add_argument("--logical-width", type=int)
    parser.add_argument("--logical-height", type=int)
    args = parser.parse_args()
    size = None
    if args.logical_width and args.logical_height:
        size = (args.logical_width, args.logical_height)
    result = visualize_log_dir(args.log_dir, args.output_dir, size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
