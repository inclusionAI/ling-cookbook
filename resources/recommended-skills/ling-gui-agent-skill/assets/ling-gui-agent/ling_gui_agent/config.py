"""Environment configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = str(Path(__file__).parent.parent / "logs")


def _load_local_env() -> None:
    """Load local .env; Ling settings override inherited values, runtime flags do not."""
    configured_path = os.environ.get("LING_CONFIG_FILE", "").strip()
    env_path = Path(configured_path).expanduser() if configured_path else Path(__file__).parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            value = value.strip().strip("'\"")
            if key.startswith("LING_"):
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)


_load_local_env()


def _pixel_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "") or str(default)
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1024:
        raise ValueError(f"{name} must be at least 1024 pixels, got {value}")
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class AgentConfig:
    appium_server_url: str
    appium_udid: Optional[str]
    max_steps: int
    screenshot_dir: str
    platform: str
    step_delay_seconds: float
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            appium_server_url=os.environ.get("APPIUM_SERVER_URL", "http://127.0.0.1:4723"),
            appium_udid=os.environ.get("APPIUM_UDID") or None,
            max_steps=int(os.environ.get("MAX_STEPS", "50") or "50"),
            screenshot_dir=os.environ.get(
                "SCREENSHOT_DIR",
                os.environ.get("LING_SKILL_LOG_DIR", DEFAULT_LOG_DIR),
            ),
            platform=os.environ.get("PLATFORM", "mobile").lower() or "mobile",
            step_delay_seconds=float(os.environ.get("STEP_DELAY_SECONDS", "2.0") or "2.0"),
            debug=_bool_env("LING_DEBUG"),
        )


@dataclass(frozen=True)
class LingConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_tokens: int
    temperature: float
    min_pixels: int
    max_pixels: int
    max_visible_screenshots: int

    @classmethod
    def from_env(cls) -> "LingConfig":
        min_pixels = _pixel_env("LING_MIN_PIXELS", 102_400)
        max_pixels = _pixel_env("LING_MAX_PIXELS", 2_621_440)
        if min_pixels > max_pixels:
            raise ValueError(
                f"LING_MIN_PIXELS ({min_pixels}) cannot exceed LING_MAX_PIXELS ({max_pixels})"
            )
        return cls(
            base_url=os.environ.get("LING_BASE_URL", "").strip(),
            api_key=os.environ.get("LING_API_KEY", ""),
            model=os.environ.get("LING_MODEL", "").strip(),
            timeout_seconds=float(os.environ.get("LING_TIMEOUT_SECONDS", "180") or "180"),
            max_tokens=int(os.environ.get("LING_MAX_TOKENS", "4096") or "4096"),
            temperature=float(os.environ.get("LING_TEMPERATURE", "0.3") or "0.3"),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_visible_screenshots=int(os.environ.get("LING_MAX_VISIBLE_SCREENSHOTS", "3") or "3"),
        )
