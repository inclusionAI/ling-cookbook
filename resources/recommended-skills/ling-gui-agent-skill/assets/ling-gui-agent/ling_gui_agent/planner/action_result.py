"""Normalized action structure produced by planner templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ActionParseError(ValueError):
    """Raised when model output cannot be parsed as a valid action."""

    pass


@dataclass
class ActionResult:
    action: str
    params: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
