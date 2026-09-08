"""Configuration and skill-runner behavior tests without GUI interaction."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from PIL import Image

from ling_gui_agent.config import AgentConfig
from ling_gui_agent.executor import GUIExecutor
from ling_gui_agent.planner import ActionResult
from ling_gui_agent.planner.llm_client import LLMResponse


_RUNNER_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "ling-gui-agent-skill" / "scripts" / "run.py",
    Path(__file__).resolve().parents[3] / "scripts" / "run.py",
)
RUNNER_PATH = next((path for path in _RUNNER_CANDIDATES if path.is_file()), _RUNNER_CANDIDATES[0])


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_skill_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgentConfigTests(unittest.TestCase):
    def test_debug_defaults_to_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(AgentConfig.from_env().debug)

    def test_debug_accepts_common_true_and_false_values(self) -> None:
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(os.environ, {"LING_DEBUG": value}, clear=True):
                self.assertTrue(AgentConfig.from_env().debug)
        for value in ("0", "false", "NO", "off"):
            with self.subTest(value=value), patch.dict(os.environ, {"LING_DEBUG": value}, clear=True):
                self.assertFalse(AgentConfig.from_env().debug)

    def test_invalid_debug_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"LING_DEBUG": "sometimes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                AgentConfig.from_env()

    def test_visualization_is_only_invoked_in_debug_mode(self) -> None:
        class FakeDevice:
            def discover_device(self):
                return "desktop"

            def ensure_session(self, _device):
                return "desktop"

            def screenshot(self, path):
                Image.new("RGB", (64, 64), "white").save(path)
                return 64, 64

        for debug in (False, True):
            with self.subTest(debug=debug), TemporaryDirectory() as temp_dir:
                executor = object.__new__(GUIExecutor)
                executor.config = SimpleNamespace(
                    max_steps=1,
                    screenshot_dir=temp_dir,
                    step_delay_seconds=0,
                    debug=debug,
                )
                executor.device = FakeDevice()
                executor.planner = SimpleNamespace(
                    build_messages=lambda **_kwargs: [],
                    parse_model_output=lambda _content, _state: ActionResult(
                        action="terminate", params={"content": "done"}
                    ),
                )
                executor.llm = SimpleNamespace(
                    model="test",
                    complete=lambda **_kwargs: LLMResponse(content="<action>Finished(content='done')</action>"),
                )
                executor.llm_max_tokens = 16
                executor.llm_extra_payload = {}
                executor.history_turns = []
                executor._execute_action = Mock()
                executor._annotate_screenshot = Mock()
                executor._write_visualization_html = Mock()
                executor._finalize_visualization = Mock()

                result = executor.run("finish")

                self.assertTrue(result["success"])
                self.assertEqual(executor._annotate_screenshot.called, debug)
                self.assertEqual(executor._write_visualization_html.called, debug)
                self.assertEqual(executor._finalize_visualization.called, debug)


class SkillRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()

    def test_missing_env_stops_before_launch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / ".env"
            with patch.object(self.runner, "ENV_FILE", missing):
                with self.assertRaisesRegex(SystemExit, "Copy .env.example to .env"):
                    self.runner.validate_env()

    def test_incomplete_env_lists_missing_fields_without_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("LING_BASE_URL=https://api.vendor.test/v1\nLING_MODEL=\n", encoding="utf-8")
            with patch.object(self.runner, "ENV_FILE", env_file):
                with self.assertRaises(SystemExit) as context:
                    self.runner.validate_env()
            message = str(context.exception)
            self.assertIn("LING_MODEL", message)
            self.assertIn("LING_API_KEY", message)
            self.assertNotIn("https://api.vendor.test", message)

    def test_placeholder_or_malformed_base_url_is_rejected(self) -> None:
        for base_url in ("https://example.com/v1", "api.vendor.test/v1", "https://api.vendor.test"):
            with self.subTest(base_url=base_url), TemporaryDirectory() as temp_dir:
                env_file = Path(temp_dir) / ".env"
                env_file.write_text(
                    f"LING_BASE_URL={base_url}\n"
                    "LING_MODEL=ling\n"
                    "LING_API_KEY=secret\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SystemExit, "OpenAI-compatible API base URL"):
                    self.runner.validate_env(env_file)

    def test_complete_env_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "LING_BASE_URL=https://api.vendor.test/v1\n"
                "LING_MODEL=ling\n"
                "LING_API_KEY=secret\n",
                encoding="utf-8",
            )
            with patch.object(self.runner, "ENV_FILE", env_file):
                self.runner.validate_env()

    def test_explicit_config_has_precedence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "explicit.env"
            other = Path(temp_dir) / "other.env"
            with patch.dict(os.environ, {"LING_CONFIG_FILE": str(other)}):
                self.assertEqual(self.runner.resolve_config_file(str(explicit)), explicit.resolve())

    def test_environment_config_has_precedence_over_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            configured = Path(temp_dir) / "external.env"
            with patch.dict(os.environ, {"LING_CONFIG_FILE": str(configured)}):
                self.assertEqual(self.runner.resolve_config_file(), configured.resolve())

    def test_device_lock_rejects_live_owner_and_recovers_stale_owner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "device.lock"
            with patch.object(self.runner, "_lock_path", return_value=lock_path):
                lock_path.write_text(str(os.getpid()), encoding="ascii")
                with self.assertRaisesRegex(SystemExit, "Device is busy"):
                    with self.runner.device_lock("desktop", {}):
                        pass

                lock_path.write_text("999999999", encoding="ascii")
                with self.runner.device_lock("desktop", {}):
                    self.assertEqual(lock_path.read_text(encoding="ascii"), str(os.getpid()))
                self.assertFalse(lock_path.exists())

    def test_default_task_timeout_is_ten_minutes(self) -> None:
        self.assertEqual(self.runner.DEFAULT_TASK_TIMEOUT_SECONDS, 600.0)

    def test_launch_timeout_stops_child_and_returns_124(self) -> None:
        child = Mock()
        child.pid = 123
        child.wait.side_effect = self.runner.subprocess.TimeoutExpired("agent", 1)
        child.poll.return_value = None
        with patch.object(self.runner.subprocess, "Popen", return_value=child), patch.object(
            self.runner, "_stop_child"
        ) as stop_child:
            result = self.runner.launch(
                "goal", sys.executable, "desktop", None, Path("config.env"), 1
            )
        self.assertEqual(result, 124)
        stop_child.assert_called_with(child)

    def test_preflight_does_not_start_agent_task(self) -> None:
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")
        with patch.object(self.runner.subprocess, "run", return_value=completed) as run:
            self.runner.check_runtime(sys.executable, "desktop", Path("config.env"))
        command = run.call_args.args[0]
        self.assertEqual(command[1], "-c")
        self.assertNotIn("-m", command)
        self.assertIn("require_macos_permissions()", command[2])

    def test_windows_candidates_use_scripts_python(self) -> None:
        with patch.object(self.runner.os, "name", "nt"):
            candidates = self.runner._python_candidates(None)
        self.assertTrue(any(candidate.endswith("Scripts/python.exe") for candidate in candidates))

    def test_parent_probe_recognizes_current_process(self) -> None:
        from ling_gui_agent.main import _parent_exists

        self.assertTrue(_parent_exists(os.getpid()))


if __name__ == "__main__":
    unittest.main()
