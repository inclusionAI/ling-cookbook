"""Minimal Appium client for a local Android device."""

from __future__ import annotations

import base64
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import httpx


class AppiumDeviceError(RuntimeError):
    pass


class AppiumDeviceClient:
    """Wrap device discovery, sessions, screenshots, and Appium actions."""

    def __init__(self, server_url: str = "http://127.0.0.1:4723", udid: Optional[str] = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.udid = udid
        self.session_id: Optional[str] = None

    # ---------- discovery ----------

    def discover_device(self) -> str:
        if self.udid:
            return self.udid
        try:
            result = subprocess.run(
                ["adb", "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AppiumDeviceError(f"adb discovery failed: {exc}") from exc

        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]
        raise AppiumDeviceError("No connected Android device found via adb")

    # ---------- session ----------

    def ensure_session(self, udid: Optional[str] = None) -> str:
        if self.session_id and self._is_alive():
            return self.session_id
        if udid:
            self.udid = udid
        if not self.udid:
            self.udid = self.discover_device()

        body = {
            "capabilities": {
                "alwaysMatch": {
                    "platformName": "Android",
                    "appium:automationName": "UiAutomator2",
                    "appium:udid": self.udid,
                    "appium:noReset": True,
                    "appium:newCommandTimeout": 300,
                    "appium:skipDeviceInitialization": True,
                    "appium:adbExecTimeout": 60000,
                    "appium:uiautomator2ServerLaunchTimeout": 60000,
                },
                "firstMatch": [{}],
            }
        }
        response = self._request_json("POST", "/session", body, timeout=120.0)
        value = response.get("value", {}) if isinstance(response, dict) else {}
        session_id = value.get("sessionId") if isinstance(value, dict) else None
        if not session_id and isinstance(response, dict):
            session_id = response.get("sessionId")
        if not session_id:
            raise AppiumDeviceError(f"Failed to create Appium session: {response}")
        self.session_id = session_id
        return session_id

    def close(self) -> None:
        if self.session_id:
            try:
                self._request_json("DELETE", f"/session/{self.session_id}", None, timeout=5.0)
            except Exception:
                pass
            self.session_id = None

    def _is_alive(self) -> bool:
        if not self.session_id:
            return False
        try:
            self._request_json("GET", f"/session/{self.session_id}/window/rect", None, timeout=5.0)
            return True
        except Exception:
            return False

    # ---------- screenshot ----------

    def screenshot(self, out_path: Path) -> tuple[int, int]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        session_id = self.ensure_session()
        response = self._request_json("GET", f"/session/{session_id}/screenshot", None)
        value = response.get("value")
        if not isinstance(value, str) or not value.strip():
            raise AppiumDeviceError(f"Invalid screenshot payload: {response}")
        image_data = base64.b64decode(value.encode("ascii"))
        out_path.write_bytes(image_data)
        return self._png_dimensions(image_data)

    # ---------- actions ----------

    def tap(self, x: int, y: int, holding: Any = ()) -> None:
        self._pointer_action([(x, y, 0), (x, y, 50)])

    def double_click(self, x: int, y: int) -> None:
        self._pointer_action([
            (x, y, 0), (x, y, 50),
            (x, y, 80), (x, y, 130),
        ])

    def long_press(self, x: int, y: int, duration_ms: int = 700) -> None:
        self._pointer_action([(x, y, 0), (x, y, duration_ms)])

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, holding: Any = (),
    ) -> None:
        self._pointer_action([(x1, y1, 0), (x2, y2, duration_ms)])

    def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 700, holding: Any = (),
    ) -> None:
        self._pointer_action([(x1, y1, 0), (x2, y2, duration_ms)])

    def _pointer_action(self, points: list[tuple[int, int, int]]) -> None:
        session_id = self.ensure_session()
        pointer_actions = []
        for i, (x, y, duration) in enumerate(points):
            pointer_actions.append({
                "type": "pointerMove",
                "duration": duration if i > 0 else 0,
                "x": x,
                "y": y,
            })
            if i == 0:
                pointer_actions.append({"type": "pointerDown"})
        pointer_actions.append({"type": "pointerUp"})
        body = {
            "actions": [
                {
                    "type": "pointer",
                    "id": "finger1",
                    "parameters": {"pointerType": "touch"},
                    "actions": pointer_actions,
                }
            ]
        }
        self._request_json("POST", f"/session/{session_id}/actions", body)

    def input_text(self, text: str) -> None:
        session_id = self.ensure_session()
        payload = {"text": text, "value": list(text)}
        self._request_json("POST", f"/session/{session_id}/keys", payload)

    def press_key(self, keycode: int) -> None:
        session_id = self.ensure_session()
        self._execute_mobile_script(session_id, "mobile: pressKey", {"keycode": keycode})

    def back(self) -> None:
        self.press_key(4)

    def home(self) -> None:
        self.press_key(3)

    def enter(self) -> None:
        self.press_key(66)

    def menu(self) -> None:
        self.press_key(187)

    def launch_app(self, app_id: str) -> None:
        session_id = self.ensure_session()
        self._execute_mobile_script(session_id, "mobile: activateApp", {"appId": app_id})

    # ---------- helpers ----------

    def _execute_mobile_script(self, session_id: str, script: str, args: Any) -> Any:
        body = {"script": script, "args": [args] if args is not None else []}
        response = self._request_json("POST", f"/session/{session_id}/execute/sync", body)
        return response.get("value")

    def _request_json(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]],
        timeout: float = 20.0,
        retries: int = 1,
    ) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout, trust_env=False) as client:
                    response = client.request(
                        method,
                        url,
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5)
                    continue
                raise AppiumDeviceError(f"Appium request failed: {exc}") from exc

            if response.status_code >= 400:
                text = response.text
                if "invalid session id" in text.lower() or "session is either terminated" in text.lower():
                    self.session_id = None
                raise AppiumDeviceError(f"Appium error {response.status_code}: {text}")

            try:
                return response.json()
            except Exception as exc:
                raise AppiumDeviceError(f"Invalid JSON from Appium: {response.text[:200]}") from exc

        raise AppiumDeviceError(f"Appium request failed after retries: {last_error}")

    def _png_dimensions(self, data: bytes) -> tuple[int, int]:
        try:
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                width = int.from_bytes(data[16:20], "big")
                height = int.from_bytes(data[20:24], "big")
                return width, height
        except Exception:
            pass
        return 0, 0
