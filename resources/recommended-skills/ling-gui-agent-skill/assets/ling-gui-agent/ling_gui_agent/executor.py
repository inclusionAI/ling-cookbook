"""Multi-step GUI executor: capture, plan, act, and repeat."""

from __future__ import annotations

import json
import html
import shutil
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .config import AgentConfig, LingConfig
from .device.appium_client import AppiumDeviceClient, AppiumDeviceError
from .device.desktop_client import DesktopDeviceClient, DesktopDeviceError
from .planner import (
    ActionParseError,
    ActionResult,
    ChatCompletionsLLM,
    LingTemplate,
)


class GUIExecutor:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.device = (
            DesktopDeviceClient()
            if config.platform in ("desktop", "web")
            else AppiumDeviceClient(config.appium_server_url, udid=config.appium_udid)
        )

        ling_cfg = LingConfig.from_env()
        self.llm = ChatCompletionsLLM(
            base_url=ling_cfg.base_url,
            api_key=ling_cfg.api_key,
            model=ling_cfg.model,
            timeout_seconds=ling_cfg.timeout_seconds,
        )
        self.llm_max_tokens = ling_cfg.max_tokens
        self.llm_extra_payload = {
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": ling_cfg.temperature,
            "top_p": 0.95,
            "top_k": 20,
        }
        self.planner = LingTemplate(
            max_visible_screenshots=ling_cfg.max_visible_screenshots,
            min_image_pixels=ling_cfg.min_pixels,
            max_image_pixels=ling_cfg.max_pixels,
            platform=config.platform,
        )
        self.history_turns: list[dict[str, Any]] = []

    def run(self, user_goal: str) -> dict[str, Any]:
        udid = self.device.discover_device()
        print(f"[device] {udid}  | Ling")
        self.device.ensure_session(udid)

        base_dir = Path(self.config.screenshot_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir = base_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] saving artifacts to {run_dir}")

        for step_index in range(1, self.config.max_steps + 1):
            prefix = f"step_{step_index:03d}"
            in_path = run_dir / f"{prefix}_in.png"
            out_path = run_dir / f"{prefix}_out.png"

            try:
                width, height = self.device.screenshot(in_path)
            except (AppiumDeviceError, DesktopDeviceError) as exc:
                return {"success": False, "step": step_index, "error": f"screenshot failed: {exc}"}

            gui_state = {
                "screenshot_path": str(in_path),
                "metadata": {"screen_width": width, "screen_height": height},
            }

            messages = self.planner.build_messages(
                user_goal=user_goal,
                plan_instruction=user_goal,
                gui_state=gui_state,
                history_turns=self.history_turns,
            )

            print(f"[step {step_index}] calling model ({len(messages)} messages) ...")
            llm_response = self.llm.complete(
                messages=messages,
                max_tokens=self.llm_max_tokens,
                extra_payload=self.llm_extra_payload,
            )
            try:
                result = self.planner.parse_model_output(llm_response.content, gui_state)
            except ActionParseError as exc:
                print(f"[step {step_index}] action parse failed ({type(exc).__name__})")
                return {"success": False, "step": step_index, "error": str(exc)}
            if llm_response.reasoning_content:
                result.extras["reasoning_content"] = llm_response.reasoning_content
            result.extras["step_index"] = step_index

            print(f"[step {step_index}] action={result.action} params={result.params}")
            self._execute_action(result)
            if self.config.debug:
                self._annotate_screenshot(in_path, result, out_path, (width, height))
            self._save_step_artifacts(
                run_dir=run_dir,
                prefix=prefix,
                step_index=step_index,
                messages=messages,
                llm_response=llm_response,
                result=result,
            )
            self._record_turn(result, in_path)
            if self.config.debug:
                self._write_visualization_html(run_dir)

            if result.action in ("terminate", "finished", "calluser", "answer"):
                if self.config.debug:
                    self._finalize_visualization(run_dir)
                summary = result.params.get("content") or result.extras.get("conclusion", "done")
                return {"success": True, "steps": step_index, "summary": summary}

            delay = self.config.step_delay_seconds
            print(f"[step {step_index}] waiting {delay}s for page transition ...")
            time.sleep(delay)

        if self.config.debug:
            self._finalize_visualization(run_dir)
        return {"success": False, "steps": self.config.max_steps, "summary": "Maximum steps reached before completion"}

    def _record_turn(self, result: ActionResult, screenshot_path: Path) -> None:
        reasoning = result.extras.get("reasoning_content", "")
        action_block = result.extras.get("raw_action_block", "")
        self.history_turns.append({
            "screenshot_path": str(screenshot_path),
            "action_block": action_block,
            "reasoning_content": reasoning,
            "assistant_text": "\n\n".join(filter(None, (str(reasoning), str(action_block)))),
        })

    def _save_step_artifacts(
        self,
        run_dir: Path,
        prefix: str,
        step_index: int,
        messages: list[dict[str, Any]],
        llm_response: Any,
        result: ActionResult,
    ) -> None:
        """Save each step's request, response, and parsed action."""
        request_payload = {
            "model": self.llm.model,
            "messages": messages,
            "max_tokens": self.llm_max_tokens,
            "extra_payload": self.llm_extra_payload,
        }
        (run_dir / f"{prefix}_request.json").write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        response_payload = dict(llm_response.raw) if llm_response.raw else {}
        (run_dir / f"{prefix}_response.json").write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        action_summary = {
            "step_index": step_index,
            "action": result.action,
            "params": result.params,
            "extras": {k: v for k, v in result.extras.items() if k != "raw_text"},
        }
        (run_dir / f"{prefix}_action.json").write_text(
            json.dumps(action_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _execute_action(self, result: ActionResult) -> None:
        action = result.action
        params = result.params or {}

        try:
            if action == "click":
                self.device.tap(int(params["x"]), int(params["y"]), params.get("holding", ()))
            elif action == "double_click":
                self.device.double_click(int(params["x"]), int(params["y"]))
            elif action == "triple_click":
                self.device.triple_click(int(params["x"]), int(params["y"]))
            elif action in ("right_click", "middle_click"):
                self.device.click(int(params["x"]), int(params["y"]), action.removesuffix("_click"))
            elif action == "hover":
                self.device.hover(int(params["x"]), int(params["y"]))
            elif action == "long_press":
                self.device.long_press(int(params["x"]), int(params["y"]))
            elif action in ("swipe", "drag"):
                if "x1" in params and "y1" in params and "x2" in params and "y2" in params:
                    duration = 700 if action == "drag" else 300
                    gesture = self.device.drag if action == "drag" else self.device.swipe
                    gesture_kwargs = {"duration_ms": duration}
                    if action == "drag":
                        gesture_kwargs["holding"] = params.get("holding", ())
                    gesture(
                        int(params["x1"]), int(params["y1"]),
                        int(params["x2"]), int(params["y2"]),
                        **gesture_kwargs,
                    )
                elif params.get("direction") == "down":
                    width, height = self._screen_size()
                    self.device.swipe(width // 2, height * 3 // 4, width // 2, height // 4)
                elif params.get("direction") == "up":
                    width, height = self._screen_size()
                    self.device.swipe(width // 2, height // 4, width // 2, height * 3 // 4)
                else:
                    print(f"[warn] unhandled {action} params: {params}")
            elif action == "type":
                self.device.input_text(str(params.get("content", "")))
            elif action == "hotkey":
                self.device.hotkey(params.get("keys", ()), int(params.get("repeat", 1)))
            elif action == "back":
                self.device.back()
            elif action == "home":
                self.device.home()
            elif action == "enter":
                self.device.enter()
            elif action == "menu":
                self.device.menu()
            elif action == "launch":
                app_id = params.get("app") or params.get("app_id")
                if app_id:
                    self.device.launch_app(app_id)
                else:
                    print(f"[warn] launch missing app id: {params}")
            elif action == "get_screenshot":
                # Ling GetScreenshot(): capture one additional image.
                path = Path(self.config.screenshot_dir) / f"extra_{int(time.time())}.png"
                self.device.screenshot(path)
                print(f"[info] saved extra screenshot to {path}")
            elif action in ("wait", "terminate", "finished", "calluser", "answer"):
                # wait / terminal actions need no device interaction
                pass
            else:
                print(f"[warn] unsupported action: {action} {params}")
        except (AppiumDeviceError, DesktopDeviceError) as exc:
            # Preserve the diagnostic in the step artifact without exposing a
            # potentially very long Appium server traceback on stdout.
            result.extras["execution_error"] = f"{type(exc).__name__}: {exc}"

    def _screen_size(self) -> tuple[int, int]:
        path = Path(self.config.screenshot_dir) / "size_probe.png"
        width, height = self.device.screenshot(path)
        return width, height

    def _annotate_screenshot(
        self,
        screenshot_path: Path,
        result: ActionResult,
        output_path: Path,
        logical_screen_size: tuple[int, int] | None = None,
    ) -> None:
        """Save a screenshot annotated with the executed action."""
        try:
            with Image.open(screenshot_path) as img:
                draw = ImageDraw.Draw(img)
                action = result.action
                params = result.params or {}
                logical_width, logical_height = logical_screen_size or (img.width, img.height)
                logical_width = max(1, logical_width)
                logical_height = max(1, logical_height)
                scale_x = img.width / logical_width
                scale_y = img.height / logical_height
                scale = (scale_x + scale_y) / 2

                # Prefer a Unicode font and include common Windows locations.
                font_paths = (
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/Hiragino Sans GB.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",
                    "/Library/Fonts/Arial Unicode.ttf",
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/segoeui.ttf",
                    "C:/Windows/Fonts/msyh.ttc",
                )

                def _load_font(size: int):
                    for font_path in font_paths:
                        try:
                            return ImageFont.truetype(font_path, size)
                        except OSError:
                            continue
                    return ImageFont.load_default()

                font = _load_font(max(1, round(36 * scale)))

                def _point(x: int, y: int) -> tuple[int, int]:
                    return round(x * scale_x), round(y * scale_y)

                def _text(x: int, y: int, text: str) -> None:
                    draw.text(_point(x, y), text, fill="red", font=font)

                def _circle(x: int, y: int, radius: int = 30, fill: bool = False) -> None:
                    color = "red"
                    x, y = _point(x, y)
                    radius = max(1, round(radius * scale))
                    bbox = [x - radius, y - radius, x + radius, y + radius]
                    if fill:
                        draw.ellipse(bbox, fill=color)
                    else:
                        draw.ellipse(bbox, outline=color, width=max(1, round(4 * scale)))

                def _center_dot(x: int, y: int) -> None:
                    cx, cy = _point(x, y)
                    dot = max(2, round(4 * scale))
                    draw.ellipse(
                        [cx - dot, cy - dot, cx + dot, cy + dot],
                        fill="yellow", outline="red", width=max(1, round(scale)),
                    )

                def _arrow(x1: int, y1: int, x2: int, y2: int) -> None:
                    x1, y1 = _point(x1, y1)
                    x2, y2 = _point(x2, y2)
                    draw.line([(x1, y1), (x2, y2)], fill="red", width=max(1, round(6 * scale)))
                    # Draw the two short sides of the arrowhead.
                    import math

                    angle = math.atan2(y2 - y1, x2 - x1)
                    arrow_len = 20 * scale
                    a1 = angle - math.pi / 6
                    a2 = angle + math.pi / 6
                    draw.line(
                        [(x2, y2), (x2 - arrow_len * math.cos(a1), y2 - arrow_len * math.sin(a1))],
                        fill="red",
                        width=max(1, round(4 * scale)),
                    )
                    draw.line(
                        [(x2, y2), (x2 - arrow_len * math.cos(a2), y2 - arrow_len * math.sin(a2))],
                        fill="red",
                        width=max(1, round(4 * scale)),
                    )

                if action in ("click", "double_click", "triple_click", "right_click", "middle_click", "hover"):
                    x, y = int(params.get("x", 0)), int(params.get("y", 0))
                    _circle(x, y, radius=16)
                    _center_dot(x, y)
                    if action == "double_click":
                        _circle(x, y, radius=26)
                    elif action == "triple_click":
                        _circle(x, y, radius=34)
                    _text(x + 22, y - 26, action.upper())
                elif action == "long_press":
                    x, y = int(params.get("x", 0)), int(params.get("y", 0))
                    _circle(x, y, radius=30, fill=True)
                    _text(x + 30, y - 30, "LONG")
                elif action in ("swipe", "drag"):
                    if "x1" in params and "x2" in params:
                        x1, y1 = int(params["x1"]), int(params["y1"])
                        x2, y2 = int(params["x2"]), int(params["y2"])
                        _arrow(x1, y1, x2, y2)
                        _text(x1, y1 - 30, action.upper())
                elif action == "type":
                    text = str(params.get("content", ""))[:20]
                    _text(20, 20, f"TYPE: {text}")
                elif action == "launch":
                    app = params.get("app") or params.get("app_id") or ""
                    _text(20, 20, f"LAUNCH: {app}")
                elif action in ("back", "home", "enter", "menu"):
                    _text(20, 20, f"KEY: {action.upper()}")
                else:
                    # Text is sufficient for actions without coordinates.
                    _text(20, 20, f"ACTION: {action}")

                # Append a subtitle panel below the screenshot instead of
                # obscuring the UI that the model acted on.
                reasoning = str(params.get("reasoning_content") or result.extras.get("reasoning_content", ""))
                action_text = action.upper()
                if params:
                    action_text += " " + json.dumps(params, ensure_ascii=False, separators=(",", ":"))[:140]
                panel_height = max(1, round(150 * scale))
                canvas = Image.new("RGB", (img.width, img.height + panel_height), (20, 22, 28))
                canvas.paste(img.convert("RGB"), (0, 0))
                subtitle_draw = ImageDraw.Draw(canvas)
                subtitle_font = _load_font(max(1, round(24 * scale)))
                panel_top = img.height
                subtitle_draw.text(
                    (round(24 * scale_x), panel_top + round(12 * scale)),
                    f"STEP {result.extras.get('step_index', '')}  {action_text[:160]}",
                    fill=(255, 220, 80), font=subtitle_font,
                )
                if reasoning:
                    lines = textwrap.wrap(reasoning, width=max(20, round(90 / scale)))[:3]
                    for line_index, line in enumerate(lines):
                        subtitle_draw.text(
                            (round(24 * scale_x), panel_top + round((48 + 32 * line_index) * scale)),
                            line, fill=(255, 255, 255), font=subtitle_font,
                        )

                canvas.save(output_path)
        except Exception as exc:
            print(f"[warn] annotate screenshot failed: {exc}")

    def _write_visualization_html(self, run_dir: Path) -> None:
        """Write a browsable step timeline next to the per-step artifacts."""
        cards: list[str] = []
        for action_path in sorted(run_dir.glob("step_*_action.json")):
            try:
                action_data = json.loads(action_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            step = html.escape(str(action_data.get("step_index", "")))
            action = html.escape(str(action_data.get("action", "")))
            params = html.escape(json.dumps(action_data.get("params", {}), ensure_ascii=False))
            extras = action_data.get("extras", {})
            reasoning = html.escape(str(extras.get("reasoning_content", "")))
            raw_action = html.escape(str(extras.get("raw_action_block", "")))
            image_name = action_path.name.replace("_action.json", "_out.png")
            cards.append(
                f'<article class="step"><h2>Step {step} · {action}</h2>'
                f'<img loading="lazy" src="{html.escape(image_name)}" alt="Step {step} screenshot">'
                f'<p class="action"><b>Action:</b> {raw_action or action} <code>{params}</code></p>'
                f'<details><summary>Reasoning</summary><p>{reasoning or "(none)"}</p></details></article>'
            )
        if not cards:
            for input_path in sorted(run_dir.glob("step_*_in.png")):
                step = html.escape(input_path.stem.removesuffix("_in"))
                cards.append(
                    f'<article class="step"><h2>{step} · no parsed action</h2>'
                    f'<img loading="lazy" src="{html.escape(input_path.name)}" alt="{step} source screenshot">'
                    '<p class="action">No action JSON was recorded for this screenshot.</p></article>'
                )
        video = ""
        if (run_dir / "sequence.mp4").is_file():
            video = '<section class="video"><h2>Sequence video</h2><video controls src="sequence.mp4"></video></section>'
        document = """<!doctype html>
<html><head><meta charset="utf-8"><title>GUI Agent Run</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#16181d;color:#eee;margin:0;padding:24px}.step{background:#252932;border-radius:12px;padding:16px;margin:0 0 24px;max-width:1100px}.step img{display:block;max-width:100%;height:auto;border-radius:8px}.action{color:#ffd85a;word-break:break-word}details{color:#cbd1dc}code{color:#b9e1ff}.video video{max-width:100%;border-radius:8px}</style></head>
<body><h1>Ling GUI Agent Run</h1>""" + video + "<main>" + "\n".join(cards) + "</main></body></html>"
        try:
            (run_dir / "visualization.html").write_text(document, encoding="utf-8")
        except OSError as exc:
            print(f"[warn] visualization html failed: {exc}")

    def _build_sequence_video(self, run_dir: Path) -> None:
        frames = sorted(run_dir.glob("step_*_out.png"))
        pattern = "step_*_out.png"
        if not frames:
            frames = sorted(run_dir.glob("step_*_in.png"))
            pattern = "step_*_in.png"
        if not frames:
            return
        ffmpeg = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None)
        if not ffmpeg:
            print("[warn] ffmpeg not found; skipped sequence.mp4")
            return
        output = run_dir / "sequence.mp4"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-framerate", "1",
                 "-pattern_type", "glob", "-i", str(run_dir / pattern),
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(output)],
                check=True, timeout=180,
            )
            print(f"[info] sequence video: {output}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[warn] sequence video failed: {exc}")

    def _finalize_visualization(self, run_dir: Path) -> None:
        self._build_sequence_video(run_dir)
        self._write_visualization_html(run_dir)
