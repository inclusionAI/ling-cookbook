import os
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from ling_gui_agent.config import LingConfig
from ling_gui_agent.log_visualizer import visualize_log_dir
from ling_gui_agent.planner.image_utils import _bounded_dimensions
from ling_gui_agent.planner.ling import LingTemplate


class ImageSizingTests(unittest.TestCase):
    def test_scales_within_budget_on_32_pixel_grid(self) -> None:
        width, height = _bounded_dimensions(3024, 1964, 2_621_440)
        self.assertEqual((width, height), (1984, 1280))
        self.assertLessEqual(width * height, 2_621_440)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_does_not_upscale_small_images(self) -> None:
        self.assertEqual(_bounded_dimensions(1280, 832, 2_621_440), (1280, 832))

    def test_custom_max_pixels_changes_uploaded_dimensions(self) -> None:
        width, height = _bounded_dimensions(3024, 1964, 65536)
        self.assertLessEqual(width * height, 65536)
        self.assertEqual((width % 32, height % 32), (0, 0))

    def test_messages_set_pixel_budget_on_each_image(self) -> None:
        template = LingTemplate(
            min_image_pixels=102_400, max_image_pixels=2_621_440, platform="desktop",
        )
        messages = template.build_messages(
            user_goal="test",
            plan_instruction="test",
            gui_state={"image_url": "https://example.com/current.png"},
            history_turns=[],
        )
        image = messages[-1]["content"][1]
        self.assertEqual(image["image_url"], {"url": "https://example.com/current.png"})
        self.assertEqual(image["min_pixels"], 102_400)
        self.assertEqual(image["max_pixels"], 2_621_440)

    def test_pixel_budgets_are_read_from_environment(self) -> None:
        with patch.dict(os.environ, {
            "LING_MIN_PIXELS": "2048",
            "LING_MAX_PIXELS": "65536",
        }):
            config = LingConfig.from_env()
        self.assertEqual((config.min_pixels, config.max_pixels), (2048, 65536))

    def test_pixel_budget_rejects_invalid_order(self) -> None:
        with patch.dict(os.environ, {
            "LING_MIN_PIXELS": "65536",
            "LING_MAX_PIXELS": "2048",
        }):
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                LingConfig.from_env()

    def test_log_visualizer_generates_derived_artifacts_without_api(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            Image.new("RGB", (128, 96), "white").save(run_dir / "step_001_in.png")
            (run_dir / "step_001_action.json").write_text(json.dumps({
                "step_index": 1,
                "action": "click",
                "params": {"x": 32, "y": 24},
                "extras": {"reasoning_content": "test"},
            }), encoding="utf-8")
            result = visualize_log_dir(run_dir)
            self.assertEqual(result["actions"], 1)
            self.assertTrue((run_dir / "step_001_out.png").is_file())
            self.assertTrue((run_dir / "visualization.html").is_file())

    def test_log_visualizer_handles_screenshot_without_action(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            Image.new("RGB", (128, 96), "white").save(run_dir / "step_001_in.png")
            result = visualize_log_dir(run_dir)
            document = (run_dir / "visualization.html").read_text(encoding="utf-8")
            self.assertEqual(result["actions"], 0)
            self.assertIn("no parsed action", document)
            self.assertTrue((run_dir / "sequence.mp4").is_file() or result["video"] is None)

    def test_first_user_turn_merges_task_and_first_screenshot(self) -> None:
        template = LingTemplate(platform="desktop")
        messages = template.build_messages(
            user_goal="open the browser",
            plan_instruction="open the browser",
            gui_state={"image_url": "https://example.com/current.png"},
            history_turns=[],
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(
            [part["type"] for part in messages[-1]["content"]],
            ["text", "image_url"],
        )
        self.assertNotIn("Previous turn, screen not shown", messages[-1]["content"][0]["text"])

    def test_history_keeps_assistant_between_user_turns(self) -> None:
        template = LingTemplate(platform="desktop")
        messages = template.build_messages(
            user_goal="open the browser",
            plan_instruction="open the browser",
            gui_state={"image_url": "https://example.com/current.png"},
            history_turns=[
                {
                    "screenshot_url": "https://example.com/previous.png",
                    "action_block": "<action>Click(box=(1, 2))</action>",
                }
            ],
        )
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )


if __name__ == "__main__":
    unittest.main()
