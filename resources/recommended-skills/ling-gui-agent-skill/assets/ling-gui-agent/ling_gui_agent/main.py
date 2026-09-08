"""CLI entry: python -m ling_gui_agent"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from .config import AgentConfig
from .device.appium_client import AppiumDeviceClient
from .executor import GUIExecutor


def _parent_exists(parent_pid: int) -> bool:
    """Return whether the runner process still exists on POSIX or Windows."""
    if os.name != "nt":
        try:
            os.kill(parent_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    import ctypes

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, parent_pid)
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _start_parent_watchdog() -> None:
    """Exit if the skill runner is killed before it can forward cancellation."""
    raw_parent_pid = os.environ.get("LING_PARENT_PID", "").strip()
    if not raw_parent_pid:
        return
    try:
        parent_pid = int(raw_parent_pid)
    except ValueError:
        return

    def watch() -> None:
        while _parent_exists(parent_pid):
            time.sleep(0.5)
        os._exit(130)

    threading.Thread(target=watch, name="runner-watchdog", daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ling GUI agent with Appium or PyAutoGUI")
    parser.add_argument("goal", help="User goal, for example: 'Open the photo gallery'")
    parser.add_argument("--list-devices", action="store_true", help="List connected Android devices and exit")
    parser.add_argument("--max-steps", type=int, default=None, help="Override MAX_STEPS")
    args = parser.parse_args()

    _start_parent_watchdog()

    config = AgentConfig.from_env()
    if args.max_steps is not None:
        object.__setattr__(config, "max_steps", args.max_steps)

    if args.list_devices:
        client = AppiumDeviceClient(config.appium_server_url, udid=config.appium_udid)
        try:
            udid = client.discover_device()
            print(udid)
            return 0
        except Exception as exc:
            print(f"No device found: {exc}", file=sys.stderr)
            return 1

    executor = GUIExecutor(config)
    try:
        result = executor.run(args.goal)
        print("\n[result]", result)
        return 0 if result.get("success") else 1
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1
    finally:
        executor.device.close()


if __name__ == "__main__":
    raise SystemExit(main())
