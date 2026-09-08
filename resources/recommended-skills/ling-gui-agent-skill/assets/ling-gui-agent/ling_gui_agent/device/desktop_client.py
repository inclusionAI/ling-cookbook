"""Minimal desktop mouse, keyboard, and screenshot client."""

from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


class DesktopDeviceError(RuntimeError):
    pass


MACOS_PERMISSION_GUIDANCE = (
    "Open System Settings > Privacy & Security, enable the missing permission "
    "for the application running this command (Codex, Claude Code, or Terminal), "
    "then quit and reopen that application before retrying."
)


def macos_permission_status() -> tuple[bool, bool]:
    """Return macOS GUI permission status without requesting or changing access."""
    if platform.system() != "Darwin":
        return True, True
    try:
        import ApplicationServices
        import Quartz
    except ImportError as exc:
        raise DesktopDeviceError(
            "macOS permission checks require PyObjC. Reinstall the desktop dependencies."
        ) from exc

    accessibility = bool(ApplicationServices.AXIsProcessTrusted())
    preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
    screen_recording = bool(preflight()) if preflight is not None else True
    return accessibility, screen_recording


def require_macos_permissions() -> None:
    """Raise an actionable error without requesting or changing macOS access."""
    accessibility, screen_recording = macos_permission_status()
    missing = []
    if not accessibility:
        missing.append("Accessibility (allows clicking, typing, and scrolling)")
    if not screen_recording:
        missing.append("Screen Recording (allows reading the screen)")
    if missing:
        raise DesktopDeviceError(
            f"Missing macOS permission: {', '.join(missing)}. {MACOS_PERMISSION_GUIDANCE}"
        )


class DesktopDeviceClient:
    """Control the current desktop while preserving its logical coordinates."""

    def __init__(self) -> None:
        try:
            import pyautogui
        except ImportError as exc:
            raise DesktopDeviceError(
                "Desktop mode requires PyAutoGUI. Run: pip install -e ."
            ) from exc
        self.gui = pyautogui
        self.gui.PAUSE = 0.05
        self.is_macos = platform.system() == "Darwin"
        if self.is_macos:
            require_macos_permissions()
        self.quartz = self._load_quartz() if self.is_macos else None

    def discover_device(self) -> str:
        return "desktop"

    def ensure_session(self, _device: str | None = None) -> str:
        return "desktop"

    def close(self) -> None:
        pass

    def screenshot(self, out_path: Path) -> tuple[int, int]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = self.gui.screenshot()
        except Exception as exc:
            if self.is_macos:
                require_macos_permissions()
            raise DesktopDeviceError(f"Desktop screenshot failed: {exc}") from exc
        size = self.gui.size()
        screen_size = (int(size.width), int(size.height))
        # Keep the native screenshot pixels for vision. On Retina displays this
        # is larger than the logical PyAutoGUI coordinate space; callers still
        # receive the latter for correct mouse-coordinate conversion.
        image.save(out_path)
        return screen_size

    def tap(self, x: int, y: int, holding: Sequence[str] = ()) -> None:
        with self._hold(holding):
            self.gui.click(x, y)

    def double_click(self, x: int, y: int) -> None:
        if self.quartz is not None:
            # Finder requires the click-state metadata, not merely two clicks.
            self._quartz_click(x, y, click_state=1)
            time.sleep(0.05)
            self._quartz_click(x, y, click_state=2)
            return
        if self.is_macos:
            # Fallback for macOS installations where PyObjC/Quartz is absent.
            self.gui.click(x, y)
            time.sleep(0.05)
            self.gui.click(x, y)
            return
        self.gui.doubleClick(x, y, interval=0.05)

    @staticmethod
    def _load_quartz() -> object | None:
        try:
            import Quartz
        except ImportError:
            return None
        return Quartz

    def _quartz_click(self, x: int, y: int, click_state: int) -> None:
        """Post one click with macOS's explicit multi-click state."""
        assert self.quartz is not None
        for event_type in (self.quartz.kCGEventLeftMouseDown, self.quartz.kCGEventLeftMouseUp):
            event = self.quartz.CGEventCreateMouseEvent(
                None, event_type, (x, y), self.quartz.kCGMouseButtonLeft,
            )
            self.quartz.CGEventSetIntegerValueField(
                event, self.quartz.kCGMouseEventClickState, click_state,
            )
            self.quartz.CGEventPost(self.quartz.kCGHIDEventTap, event)

    def triple_click(self, x: int, y: int) -> None:
        self.gui.click(x, y, clicks=3, interval=0.05)

    def click(self, x: int, y: int, button: str) -> None:
        self.gui.click(x, y, button=button)

    def hover(self, x: int, y: int) -> None:
        self.gui.moveTo(x, y)

    def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 700,
        holding: Sequence[str] = (),
    ) -> None:
        """Move an item or select content with the primary mouse button held."""
        with self._hold(holding):
            self.gui.moveTo(x1, y1)
            self.gui.dragTo(x2, y2, duration=duration_ms / 1000, button="left")

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300,
        holding: Sequence[str] = (),
    ) -> None:
        """Scroll in the direction of a touch-style swipe.

        A desktop ``Drag`` needs a held primary button, whereas a ``Swipe`` in
        the desktop prompt means scrolling.  Using ``dragTo`` for both made a
        model-requested scroll select text or move an item instead.
        """
        # Do not hold modifiers while scrolling: Ctrl/Command + wheel is often
        # interpreted as zoom rather than scroll. Keep the argument only for
        # device-interface compatibility; modifiers remain supported by Drag.
        del holding
        self.gui.moveTo(x1, y1)
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) >= abs(dx):
            # Pulling content down (positive dy) reveals content above,
            # which is a positive PyAutoGUI scroll.
            clicks = self._scroll_clicks(dy)
            if clicks:
                self.gui.scroll(clicks)
        elif hasattr(self.gui, "hscroll"):
            # Horizontal scroll direction is opposite to finger motion.
            clicks = self._scroll_clicks(-dx)
            if clicks:
                self.gui.hscroll(clicks)
        else:
            # Some PyAutoGUI backends have no horizontal wheel support.
            # Preserve a useful gesture rather than silently doing nothing.
            self.gui.dragTo(x2, y2, duration=duration_ms / 1000, button="left")

    @staticmethod
    def _scroll_clicks(distance: int) -> int:
        """Convert a screen-distance gesture to wheel ticks without rounding to 0."""
        if not distance:
            return 0
        return (1 if distance > 0 else -1) * max(1, round(abs(distance) / 100))

    def input_text(self, text: str) -> None:
        """Paste Unicode through the clipboard; a trailing newline presses Enter."""
        try:
            import pyperclip
        except ImportError as exc:
            raise DesktopDeviceError("Desktop text input requires pyperclip. Run: pip install -e .") from exc

        text = text.replace("\\n", "\n")
        submit = text.endswith("\n")
        previous = pyperclip.paste()
        try:
            pyperclip.copy(text[:-1] if submit else text)
            self.hotkey(["ctrl", "v"])
            if submit:
                self.gui.press("enter")
        finally:
            pyperclip.copy(previous)

    def hotkey(self, keys: Sequence[str], repeat: int = 1) -> None:
        normalized = [self._key(key) for key in keys]
        for _ in range(repeat):
            self.gui.hotkey(*normalized)

    def enter(self) -> None:
        self.gui.press("enter")

    @contextmanager
    def _hold(self, keys: Sequence[str]) -> Iterator[None]:
        normalized = [self._key(key) for key in keys]
        for key in normalized:
            self.gui.keyDown(key)
        try:
            yield
        finally:
            for key in reversed(normalized):
                self.gui.keyUp(key)

    def _key(self, key: str) -> str:
        key = str(key).lower()
        return "command" if self.is_macos and key == "ctrl" else key
