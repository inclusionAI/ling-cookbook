"""Abstract prompt template and coordinate transformation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .action_result import ActionResult


class PromptTemplate(ABC):
    def __init__(self, coord_transform: str = "none") -> None:
        self.coord_transform = (coord_transform or "none").lower()

    @abstractmethod
    def build_messages(
        self,
        user_goal: str,
        plan_instruction: str,
        gui_state: dict[str, Any],
        history_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def parse_model_output(self, output_text: str, gui_state: dict[str, Any]) -> ActionResult:
        raise NotImplementedError

    def apply_coordinate_transform(
        self,
        action: str,
        params: dict[str, Any],
        gui_state: dict[str, Any],
    ) -> dict[str, Any]:
        width, height = self._get_screen_size(gui_state)
        if not width or not height or self.coord_transform == "none":
            return params

        out = dict(params or {})
        for key in ("x", "y"):
            if key in out:
                out[key] = self._scale_xy(out[key], width if key == "x" else height)
        for key in ("x1", "y1", "x2", "y2"):
            if key in out:
                dim = width if key in ("x1", "x2") else height
                out[key] = self._scale_xy(out[key], dim)
        if isinstance(out.get("box"), (list, tuple)):
            box = list(out["box"])
            if len(box) >= 2:
                box[0] = self._scale_xy(box[0], width)
                box[1] = self._scale_xy(box[1], height)
                if len(box) >= 4:
                    box[2] = self._scale_xy(box[2], width)
                    box[3] = self._scale_xy(box[3], height)
                out["box"] = box
        return out

    def _get_screen_size(self, gui_state: dict[str, Any]) -> tuple[int, int]:
        metadata = gui_state.get("metadata", {}) if isinstance(gui_state, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        width = metadata.get("screen_width")
        height = metadata.get("screen_height")
        try:
            return int(width), int(height)
        except Exception:
            return 0, 0

    def _scale_xy(self, value: Any, screen_size: int) -> int:
        try:
            fv = float(value)
        except Exception:
            return int(value)

        t = self.coord_transform
        if t == "thousand":
            return int(fv / 1000.0 * screen_size)
        if t == "ratio":
            return int(fv * screen_size)
        if t == "auto":
            if 0.0 <= fv <= 1.0:
                return int(fv * screen_size)
            if 0.0 <= fv <= 1000.0:
                return int(fv / 1000.0 * screen_size)
        return int(fv)
