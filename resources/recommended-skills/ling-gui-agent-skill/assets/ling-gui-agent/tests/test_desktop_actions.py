"""Desktop action parsing and dispatch tests, with no live GUI interaction."""

from __future__ import annotations

import unittest

from ling_gui_agent.executor import GUIExecutor
from ling_gui_agent.planner import ActionResult, LingTemplate


class RecordingDesktop:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


class DesktopActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = LingTemplate(platform="desktop")
        self.gui_state = {"metadata": {"screen_width": 1440, "screen_height": 900}}
        # Avoid GUIExecutor.__init__, which creates a real PyAutoGUI client and LLM.
        self.executor = object.__new__(GUIExecutor)
        self.executor.device = RecordingDesktop()

    def parse(self, action: str):
        return self.template.parse_model_output(f"<action>{action}</action>", self.gui_state)

    def assert_dispatch(self, action: str, method: str, args: tuple, kwargs: dict | None = None) -> None:
        result = self.parse(action)
        self.executor.device.calls.clear()
        self.executor._execute_action(result)
        self.assertEqual(self.executor.device.calls, [(method, args, kwargs or {})])

    def test_desktop_pointer_actions_are_parsed_and_dispatched(self) -> None:
        self.assert_dispatch("Click(box=(500, 500), holding=['ctrl'])", "tap", (720, 450, ['ctrl']))
        self.assert_dispatch("DoubleClick(box=(100, 200))", "double_click", (144, 180))
        self.assert_dispatch("TripleClick(box=(100, 200))", "triple_click", (144, 180))
        self.assert_dispatch("RightClick(box=(100, 200))", "click", (144, 180, "right"))
        self.assert_dispatch("MiddleClick(box=(100, 200))", "click", (144, 180, "middle"))
        self.assert_dispatch("Hover(box=(100, 200))", "hover", (144, 180))

    def test_desktop_drag_and_swipe_are_parsed_and_dispatched(self) -> None:
        self.assert_dispatch(
            "Drag(start=(100, 800), end=(900, 200), holding=['shift'])",
            "drag", (144, 720, 1297, 180), {"duration_ms": 700, "holding": ['shift']},
        )
        self.assert_dispatch(
            "Swipe(start=(500, 800), end=(500, 200), holding=[])",
            "swipe", (720, 720, 720, 180), {"duration_ms": 300},
        )

    def test_model_coordinate_999_maps_to_the_last_screen_pixel(self) -> None:
        result = self.parse("Click(box=(999, 999))")
        self.assertEqual(result.params, {"x": 1439, "y": 899})

    def test_swipe_coordinates_use_logical_screen_size_not_uploaded_image_size(self) -> None:
        result = self.template.parse_model_output(
            "<action>Swipe(start=(999, 999), end=(0, 0))</action>",
            {"metadata": {"screen_width": 1512, "screen_height": 982}},
        )
        self.assertEqual(result.params, {"x1": 1511, "y1": 981, "x2": 0, "y2": 0})

    def test_desktop_keyboard_actions_are_parsed_and_dispatched(self) -> None:
        self.assert_dispatch("Type(content='hello')", "input_text", ("hello",))
        self.assert_dispatch("Hotkey(keys=['ctrl', 'c'], repeat=2)", "hotkey", (['ctrl', 'c'], 2))
        self.assert_dispatch("PressEnter()", "enter", ())

    def test_desktop_enter_presses_the_enter_key(self) -> None:
        class FakeGui:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def press(self, *args): self.calls.append(("press", *args))

        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        device.enter()
        self.assertEqual(device.gui.calls, [("press", "enter")])

    def test_macos_permission_error_names_missing_permissions(self) -> None:
        from unittest.mock import patch
        from ling_gui_agent.device.desktop_client import (
            DesktopDeviceError,
            require_macos_permissions,
        )

        with patch(
            "ling_gui_agent.device.desktop_client.macos_permission_status",
            return_value=(False, False),
        ):
            with self.assertRaises(DesktopDeviceError) as context:
                require_macos_permissions()

        message = str(context.exception)
        self.assertIn("Accessibility", message)
        self.assertIn("Screen Recording", message)
        self.assertIn("System Settings > Privacy & Security", message)
        self.assertIn("Codex, Claude Code, or Terminal", message)

    def test_desktop_screenshot_preserves_native_retina_pixels(self) -> None:
        class FakeImage:
            size = (3024, 1964)

            def __init__(self) -> None:
                self.saved_to = None

            def save(self, path) -> None:
                self.saved_to = path

        class FakeGui:
            def __init__(self) -> None:
                self.image = FakeImage()

            def screenshot(self): return self.image
            def size(self): return type("Size", (), {"width": 1512, "height": 982})()

        from pathlib import Path
        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        with self.subTest("logical action coordinate size"):
            self.assertEqual(device.screenshot(Path("/tmp/native.png")), (1512, 982))
        self.assertEqual(device.gui.image.size, (3024, 1964))

    def test_annotation_scales_logical_coordinates_to_retina_image(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from PIL import Image

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "output.png"
            Image.new("RGB", (400, 200), "white").save(source)
            result = ActionResult(action="long_press", params={"x": 50, "y": 10})
            self.executor._annotate_screenshot(source, result, output, (100, 50))
            with Image.open(output) as annotated:
                self.assertEqual(annotated.getpixel((200, 40)), (255, 0, 0))
                self.assertEqual(annotated.getpixel((50, 10)), (255, 255, 255))

    def test_annotation_click_marks_center_and_appends_subtitle_panel(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from PIL import Image

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "output.png"
            Image.new("RGB", (400, 200), "white").save(source)
            result = ActionResult(action="click", params={"x": 50, "y": 10})
            self.executor._annotate_screenshot(source, result, output, (100, 50))
            with Image.open(output) as annotated:
                self.assertEqual(annotated.getpixel((200, 40)), (255, 255, 0))
                self.assertGreater(annotated.height, 200)

    def test_visualization_html_contains_step_action_and_reasoning(self) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "step_001_action.json").write_text(
                json.dumps({
                    "step_index": 1,
                    "action": "click",
                    "params": {"x": 12, "y": 34},
                    "extras": {
                        "raw_action_block": "Click(box=(12, 34))",
                        "reasoning_content": "Open the target folder",
                    },
                }, ensure_ascii=False), encoding="utf-8",
            )
            (run_dir / "step_001_out.png").write_bytes(b"placeholder")
            self.executor._write_visualization_html(run_dir)
            document = (run_dir / "visualization.html").read_text(encoding="utf-8")
            self.assertIn("Step 1", document)
            self.assertIn("Click(box=(12, 34))", document)
            self.assertIn("Open the target folder", document)
            self.assertIn("step_001_out.png", document)

    def test_desktop_swipe_uses_wheel_not_mouse_drag(self) -> None:
        class FakeGui:
            def __init__(self) -> None:
                self.PAUSE = 0
                self.calls: list[tuple] = []

            def moveTo(self, *args): self.calls.append(("moveTo", *args))
            def scroll(self, *args): self.calls.append(("scroll", *args))
            def keyDown(self, *args): self.calls.append(("keyDown", *args))
            def keyUp(self, *args): self.calls.append(("keyUp", *args))

        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        device.is_macos = False
        device.quartz = None
        device.swipe(500, 800, 500, 200, holding=["ctrl"])
        self.assertEqual(device.gui.calls, [("moveTo", 500, 800), ("scroll", -6)])

    def test_desktop_drag_uses_mouse_drag(self) -> None:
        class FakeGui:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def moveTo(self, *args): self.calls.append(("moveTo", *args))
            def dragTo(self, *args, **kwargs): self.calls.append(("dragTo", *args, kwargs))
            def keyDown(self, *args): self.calls.append(("keyDown", *args))
            def keyUp(self, *args): self.calls.append(("keyUp", *args))

        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        device.is_macos = False
        device.drag(100, 200, 300, 400, duration_ms=700)
        self.assertEqual(
            device.gui.calls,
            [("moveTo", 100, 200), ("dragTo", 300, 400, {"duration": 0.7, "button": "left"})],
        )

    def test_desktop_multi_clicks_use_a_50ms_interval(self) -> None:
        class FakeGui:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def doubleClick(self, *args, **kwargs): self.calls.append(("doubleClick", *args, kwargs))
            def click(self, *args, **kwargs): self.calls.append(("click", *args, kwargs))

        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        device.is_macos = False
        device.quartz = None
        device.double_click(100, 200)
        device.triple_click(300, 400)
        self.assertEqual(
            device.gui.calls,
            [
                ("doubleClick", 100, 200, {"interval": 0.05}),
                ("click", 300, 400, {"clicks": 3, "interval": 0.05}),
            ],
        )

    def test_macos_double_click_uses_two_clicks_with_a_50ms_gap(self) -> None:
        class FakeGui:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def click(self, *args, **kwargs): self.calls.append(("click", *args, kwargs))

        from unittest.mock import patch
        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = FakeGui()
        device.is_macos = True
        device.quartz = None
        with patch("ling_gui_agent.device.desktop_client.time.sleep") as sleep:
            device.double_click(100, 200)
        self.assertEqual(device.gui.calls, [("click", 100, 200, {}), ("click", 100, 200, {})])
        sleep.assert_called_once_with(0.05)

    def test_macos_quartz_double_click_sets_incrementing_click_state(self) -> None:
        class FakeQuartz:
            kCGEventLeftMouseDown = "down"
            kCGEventLeftMouseUp = "up"
            kCGMouseButtonLeft = "left"
            kCGMouseEventClickState = "click_state"
            kCGHIDEventTap = "hid"

            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def CGEventCreateMouseEvent(self, _source, event_type, point, button):
                event = {"type": event_type, "point": point, "button": button}
                self.calls.append(("create", event))
                return event

            def CGEventSetIntegerValueField(self, event, field, value):
                event[field] = value
                self.calls.append(("state", event["type"], value))

            def CGEventPost(self, tap, event):
                self.calls.append(("post", tap, event["type"], event["click_state"]))

        from unittest.mock import patch
        from ling_gui_agent.device.desktop_client import DesktopDeviceClient

        device = object.__new__(DesktopDeviceClient)
        device.gui = object()
        device.is_macos = True
        device.quartz = FakeQuartz()
        with patch("ling_gui_agent.device.desktop_client.time.sleep") as sleep:
            device.double_click(100, 200)
        self.assertEqual(
            device.quartz.calls,
            [
                ("create", {"type": "down", "point": (100, 200), "button": "left", "click_state": 1}),
                ("state", "down", 1), ("post", "hid", "down", 1),
                ("create", {"type": "up", "point": (100, 200), "button": "left", "click_state": 1}),
                ("state", "up", 1), ("post", "hid", "up", 1),
                ("create", {"type": "down", "point": (100, 200), "button": "left", "click_state": 2}),
                ("state", "down", 2), ("post", "hid", "down", 2),
                ("create", {"type": "up", "point": (100, 200), "button": "left", "click_state": 2}),
                ("state", "up", 2), ("post", "hid", "up", 2),
            ],
        )
        sleep.assert_called_once_with(0.05)


if __name__ == "__main__":
    unittest.main()
