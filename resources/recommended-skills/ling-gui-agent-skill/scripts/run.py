#!/usr/bin/env python3
"""Run the bundled Ling GUI Agent without copying it to a runtime directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

SKILL_DIR = Path(__file__).resolve().parent.parent
PROJECT = SKILL_DIR / "assets" / "ling-gui-agent"
ENV_FILE = SKILL_DIR / ".env"
ENV_EXAMPLE = SKILL_DIR / ".env.example"
VENV_DIR = SKILL_DIR / ".venv"
DEP_CHECK = "import pyautogui, pyperclip, PIL, httpx"
REQUIRED_CONFIG = ("LING_BASE_URL", "LING_MODEL", "LING_API_KEY")
DEFAULT_TASK_TIMEOUT_SECONDS = 600.0


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_config_file(explicit: str | None = None) -> Path:
    """Resolve an explicit config, then LING_CONFIG_FILE, then the local default."""
    configured = explicit or os.environ.get("LING_CONFIG_FILE", "").strip()
    return Path(configured).expanduser().resolve() if configured else ENV_FILE


def _valid_base_url(value: str) -> bool:
    """Accept a real OpenAI-compatible HTTP(S) base URL ending in /v1."""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower()
    placeholder = hostname == "example.com" or hostname.endswith(".example.com")
    return (
        parsed.scheme in {"http", "https"}
        and bool(hostname)
        and not placeholder
        and parsed.path.rstrip("/").endswith("/v1")
        and not parsed.query
        and not parsed.fragment
    )


def validate_env(config_file: Path | None = None) -> dict[str, str]:
    """Stop before dependency checks or GUI access when local config is incomplete."""
    path = config_file or ENV_FILE
    if not path.is_file():
        default_hint = (
            f"Copy {ENV_EXAMPLE.name} to .env, fill in "
            if path == ENV_FILE
            else "Create the selected config file and fill in "
        )
        sys.exit(
            f"[run.py] Missing required configuration: {path}\n"
            f"  {default_hint}"
            "LING_BASE_URL, LING_MODEL, and LING_API_KEY, then retry."
        )
    values = _read_env_file(path)
    missing = [name for name in REQUIRED_CONFIG if not values.get(name, "").strip()]
    if missing:
        sys.exit(
            f"[run.py] Incomplete {path}: missing {', '.join(missing)}.\n"
            "  Fill in these values in the selected config file, then retry. "
            "Do not pass credentials on the command line."
        )
    if not _valid_base_url(values["LING_BASE_URL"]):
        sys.exit(
            f"[run.py] Invalid LING_BASE_URL in {path}.\n"
            "  Ask the user or service provider for the exact OpenAI-compatible API "
            "base URL, usually in the form https://<host>/v1. Do not use "
            "https://example.com/v1 or guess an endpoint."
        )
    return values


def _python_candidates(force: str | None) -> list[str]:
    candidates = [force or ""]
    if os.name == "nt":
        candidates.extend(
            [
                str(VENV_DIR / "Scripts" / "python.exe"),
                sys.executable,
                shutil.which("python") or "",
                shutil.which("python3") or "",
            ]
        )
    else:
        home = Path.home()
        candidates.extend(
            [
                str(VENV_DIR / "bin" / "python"),
                sys.executable,
                shutil.which("python3") or "",
                shutil.which("python") or "",
                str(home / "miniforge3" / "bin" / "python"),
                str(home / "miniforge" / "bin" / "python"),
                str(home / "anaconda3" / "bin" / "python"),
                "/opt/homebrew/bin/python3",
                "/usr/local/bin/python3",
            ]
        )
    return candidates


def find_python(force: str | None) -> str:
    """Use the first interpreter that passes one lightweight import probe."""
    seen: set[str] = set()
    for candidate in _python_candidates(force):
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        if resolved in seen or not Path(resolved).is_file():
            continue
        seen.add(resolved)
        try:
            result = subprocess.run(
                [resolved, "-c", DEP_CHECK], capture_output=True, timeout=20, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return resolved

    if os.name == "nt":
        create = f'py -3 -m venv "{VENV_DIR}"'
        install = f'"{VENV_DIR / "Scripts" / "python.exe"}" -m pip install -e "{PROJECT}"'
    else:
        create = f'python3 -m venv "{VENV_DIR}"'
        install = f'"{VENV_DIR / "bin" / "python"}" -m pip install -e "{PROJECT}"'
    sys.exit(
        "[run.py] No Python interpreter with the required dependencies was found.\n"
        f"  Run once: {create}\n"
        f"  Then run: {install}\n"
        "  Retry the original command after setup completes."
    )


def _effective_value(name: str, config: dict[str, str], default: str = "") -> str:
    """Match the package's precedence for non-secret runtime settings."""
    return os.environ.get(name, "").strip() or config.get(name, "").strip() or default


def effective_platform(cli_value: str | None, config: dict[str, str]) -> str:
    return (cli_value or _effective_value("PLATFORM", config, "mobile")).lower()


def check_runtime(python: str, platform_name: str, config_file: Path) -> None:
    """Check device prerequisites without starting a GUI-agent task."""
    if not PROJECT.is_dir():
        sys.exit(f"[run.py] Bundled project not found: {PROJECT}")

    env = os.environ.copy()
    for key in REQUIRED_CONFIG:
        env.pop(key, None)
    env["LING_CONFIG_FILE"] = str(config_file)
    env["PLATFORM"] = platform_name
    if platform_name in ("desktop", "web"):
        probe = (
            "import pyautogui; "
            "from ling_gui_agent.device.desktop_client import require_macos_permissions; "
            "require_macos_permissions(); "
            "size = pyautogui.size(); "
            "assert size.width > 0 and size.height > 0"
        )
    else:
        probe = (
            "from ling_gui_agent.config import AgentConfig; "
            "from ling_gui_agent.device.appium_client import AppiumDeviceClient; "
            "cfg = AgentConfig.from_env(); "
            "client = AppiumDeviceClient(cfg.appium_server_url, cfg.appium_udid); "
            "client.discover_device(); client._request_json('GET', '/status', None, timeout=5.0)"
        )
    try:
        result = subprocess.run(
            [python, "-c", probe], cwd=PROJECT, env=env, capture_output=True,
            text=True, timeout=20, check=False,
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"[run.py] {platform_name} runtime check timed out after 20 seconds.")
    except OSError as exc:
        sys.exit(f"[run.py] Could not start the {platform_name} runtime check: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        sys.exit(f"[run.py] {platform_name} runtime check failed{suffix}")


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _lock_path(platform_name: str, config: dict[str, str]) -> Path:
    device = (
        "desktop" if platform_name in ("desktop", "web")
        else _effective_value("APPIUM_UDID", config, "default-android-device")
    )
    target_type = "desktop" if platform_name in ("desktop", "web") else "mobile"
    digest = hashlib.sha256(f"{target_type}:{device}".encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"ling-gui-agent-{digest}.lock"


@contextmanager
def device_lock(platform_name: str, config: dict[str, str]) -> Iterator[None]:
    """Allow only one active controller per desktop or configured Android device."""
    path = _lock_path(platform_name, config)
    for _attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                owner_pid = int(path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                owner_pid = -1
            if _process_exists(owner_pid):
                sys.exit(
                    f"[run.py] Device is busy: another Ling GUI Agent runner "
                    f"is active (PID {owner_pid})."
                )
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            finally:
                os.close(descriptor)
            break
    else:
        sys.exit("[run.py] Could not acquire the device lock; retry shortly.")

    try:
        yield
    finally:
        try:
            if path.read_text(encoding="ascii").strip() == str(os.getpid()):
                path.unlink()
        except FileNotFoundError:
            pass


def _stop_child(child: subprocess.Popen[bytes]) -> None:
    """Forward cancellation and stop the child process group on every platform."""
    if child.poll() is not None:
        return
    try:
        if os.name == "nt":
            child.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        if child.poll() is None:
            child.kill()


def launch(
    goal: str, python: str, platform_name: str, max_steps: int | None,
    config_file: Path, timeout_seconds: float | None,
) -> int:
    if not PROJECT.is_dir():
        sys.exit(f"[run.py] Bundled project not found: {PROJECT}")

    env = os.environ.copy()
    for key in REQUIRED_CONFIG:
        env.pop(key, None)
    env["LING_CONFIG_FILE"] = str(config_file)
    env["LING_SKILL_LOG_DIR"] = str(SKILL_DIR / "logs")
    env["LING_PARENT_PID"] = str(os.getpid())
    if platform_name:
        env["PLATFORM"] = platform_name

    command = [python, "-m", "ling_gui_agent", goal]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])

    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    child = subprocess.Popen(
        command,
        cwd=PROJECT,
        env=env,
        creationflags=creation_flags,
        start_new_session=os.name != "nt",
    )
    previous_handlers: dict[int, object] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        _stop_child(child)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    try:
        return child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"[run.py] Task timed out after {timeout_seconds:g} seconds; stopping it.",
            file=sys.stderr,
        )
        _stop_child(child)
        return 124
    except KeyboardInterrupt:
        _stop_child(child)
        return 130
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _stop_child(child)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the skill's bundled Ling GUI Agent.")
    parser.add_argument("goal", nargs="?", help="User goal, for example: 'Open the Pictures folder'")
    parser.add_argument("--platform", choices=("desktop", "web", "mobile", "auto"))
    parser.add_argument("--max-steps", type=int, default=None, help="Override MAX_STEPS")
    parser.add_argument("--python", default=None, help="Use a specific Python interpreter")
    parser.add_argument("--config", default=None, help="Read configuration from this file")
    parser.add_argument(
        "--check", action="store_true",
        help="Validate configuration, dependencies, and device prerequisites, then exit",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help=f"Stop the complete task after this many seconds (default: {DEFAULT_TASK_TIMEOUT_SECONDS:g})",
    )
    args = parser.parse_args()

    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not args.check and not args.goal:
        parser.error("goal is required unless --check is used")

    config_file = resolve_config_file(args.config)
    config = validate_env(config_file)
    python = find_python(args.python)
    platform_name = effective_platform(args.platform, config)
    if args.check:
        check_runtime(python, platform_name, config_file)
        print(f"[run.py] Preflight passed for {platform_name}.")
        return 0
    with device_lock(platform_name, config):
        return launch(
            args.goal, python, platform_name, args.max_steps, config_file, args.timeout,
        )


if __name__ == "__main__":
    raise SystemExit(main())
