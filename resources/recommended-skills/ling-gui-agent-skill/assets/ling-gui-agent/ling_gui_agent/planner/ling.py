"""Multi-turn conversation template for Ling models."""

from __future__ import annotations

import re
from typing import Any, Tuple

from .action_result import ActionResult
from .base import PromptTemplate


LING_MOBILE_SYSTEM_PROMPT = """
**You are a GUI Agent.**
Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions.

### Available Actions
You may execute one of the following functions:
- Click(box=(x1, y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Drag(start=(x1, y1), end=(x2, y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting app layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Swipe(start=(x1, y1), end=(x2, y2))
> Perform a swipe action by dragging from the start coordinate to the end coordinate. This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- DoubleClick(box=(x1, y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- LongPress(box=(x1, y1))
> Perform a long-press action at the specified screen coordinate for a certain duration. This can be used to trigger additional options, such as copy, forward, delete, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Type(content='')
> Enter the specified text into the currently active input field.
- LaunchApp(app='')
> Launch the target app. Use this action when the target app is not currently visible on the screen.
- Wait()
> Wait for the current page, animation, or content to finish loading.
- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.
- GetScreenshot()
> Take a screenshot and save it to the device's photo album.
- PressBack()
> Return to the previous screen.
- PressHome()
> Return to the system home screen.
- PressEnter()
> Perform an Enter key action.
- PressRecent()
> Open the system recent apps screen.
- Answer(content='')
> Answer the user's questions as requested.
- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Swipe` action with different directions to reveal additional content.
- To copy text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
"""

# Keep the old constant as a compatibility alias for callers that imported it.
LING_SYSTEM_PROMPT = LING_MOBILE_SYSTEM_PROMPT


LING_DESKTOP_SYSTEM_PROMPT = """
**You are a GUI Agent.**
Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions on a desktop operating system.

### Available Actions
You may execute one of the following functions. Coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Click(box=(x1, y1), holding=[])
> Perform a left-click. `holding` may contain modifier keys such as `shift` or `ctrl`.
- DoubleClick(box=(x1, y1))
> Perform a double-click.
- TripleClick(box=(x1, y1))
> Perform a triple-click to select a line or all text in a single-line input.
- RightClick(box=(x1, y1))
> Perform a right-click to open a context menu.
- MiddleClick(box=(x1, y1))
> Perform a middle-click.
- Hover(box=(x1, y1))
> Move the cursor without clicking to reveal a submenu, flyout, dropdown, or tooltip.
- Drag(start=(x1, y1), end=(x2, y2), holding=[])
> Hold and drag between coordinates. `holding` may contain modifier keys.
- Swipe(start=(x1, y1), end=(x2, y2))
> Scroll by swiping between coordinates.
- Type(content='')
> Enter text into the active input field. End content with `\\n` to press Enter.
- Hotkey(keys=['ctrl', 'c'], repeat=1)
> Press a keyboard shortcut. Use `repeat` for repeated key presses.
- Wait()
> Wait for a page, animation, or slow operation to finish.
- CallUser(content='')
> Request user takeover or additional information.
- Finished(content='')
> Mark the task as completed and report its status.

### Instructions
- Carefully examine the current screenshot and execute only one action per step.
- To replace existing input, use `TripleClick` followed by `Type`.
- To open a submenu, `Hover` over its parent, then `Click` the revealed item.
- To use a context menu, `RightClick` the target, then `Click` an entry.
- For range selection, click the first point then click the end point with `holding=['shift']`; for multi-selection use `holding=['ctrl']`.
- Use `Hotkey` for shortcuts such as copy (`ctrl+c`), paste (`ctrl+v`), save (`ctrl+s`), undo (`ctrl+z`), and find (`ctrl+f`).
- After launching an app, downloading, or another slow operation, use `Wait()` before continuing.
"""


class LingTemplate(PromptTemplate):
    """Build Ling messages and translate its fixed 0-999 coordinate space.

    The sequence contains the system prompt, one initial user task, alternating
    historical observations and assistant actions, and the current screenshot.
    The first screenshot is merged into the initial user turn. Only the newest
    configured screenshots remain visible; older ones use a text placeholder.
    """

    def __init__(
        self,
        coord_transform: str = "none",
        max_visible_screenshots: int = 3,
        min_image_pixels: int = 102_400,
        max_image_pixels: int = 2_621_440,
        platform: str = "mobile",
    ) -> None:
        super().__init__(coord_transform=coord_transform)
        self.max_visible_screenshots = max(max_visible_screenshots, 1)
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.platform = (platform or "mobile").lower()

    def _resolve_system_prompt(self, gui_state: dict[str, Any]) -> str:
        """Select mobile, desktop, or automatic prompt by screen orientation."""
        if self.platform == "mobile":
            return LING_MOBILE_SYSTEM_PROMPT
        if self.platform in ("web", "desktop"):
            return LING_DESKTOP_SYSTEM_PROMPT
        width, height = self._get_screen_size(gui_state)
        return LING_MOBILE_SYSTEM_PROMPT if height > width else LING_DESKTOP_SYSTEM_PROMPT

    def build_messages(
        self,
        user_goal: str,
        plan_instruction: str,
        gui_state: dict[str, Any],
        history_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        current_screenshot_url = self._resolve_screenshot_url(gui_state)

        # Reserve one visible-image slot for the current screenshot.
        visible_history_count = min(
            self.max_visible_screenshots - 1,
            len(history_turns),
        )
        first_visible_history_index = len(history_turns) - visible_history_count

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._resolve_system_prompt(gui_state)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._first_user_text(user_goal),
                    }
                ],
            },
        ]

        def _append_user_content(content: list[dict[str, Any]]) -> None:
            # The task text and the first observation are one user turn. This
            # also keeps consecutive user messages from appearing when a
            # history item has no assistant text to follow it.
            if messages[-1]["role"] == "user" and isinstance(messages[-1].get("content"), list):
                messages[-1]["content"].extend(content)
            else:
                messages.append({"role": "user", "content": content})

        for idx, turn in enumerate(history_turns):
            screenshot_url = self._resolve_url(
                turn.get("screenshot_path") or turn.get("screenshot_url")
            )
            if idx >= first_visible_history_index and screenshot_url:
                user_content: list[dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_url},
                        "min_pixels": self.min_image_pixels,
                        "max_pixels": self.max_image_pixels,
                    }
                ]
            else:
                user_content = [
                    {"type": "text", "text": "(Previous turn, screen not shown)"}
                ]
            _append_user_content(user_content)

            action_block = str(turn.get("action_block", "") or "").strip()
            reasoning_content = str(turn.get("reasoning_content", "") or "").strip()
            if action_block or reasoning_content:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                }
                if action_block:
                    assistant_msg["content"] = [{"type": "text", "text": action_block}]
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)

        # Append the current screenshot.
        if current_screenshot_url:
            _append_user_content(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": current_screenshot_url},
                        "min_pixels": self.min_image_pixels,
                        "max_pixels": self.max_image_pixels,
                    }
                ]
            )

        return messages

    def parse_model_output(self, output_text: str, gui_state: dict[str, Any]) -> ActionResult:
        from .action_result import ActionParseError

        action_block_inner = self._extract_tag(output_text, "action")
        action_block_full = f"<action>{action_block_inner}</action>" if action_block_inner else ""
        print("[Ling Output]", output_text)

        if not action_block_inner:
            raise ActionParseError(
                f"No <action> block found in model output: {output_text[:200]}"
            )

        try:
            action_name, action_params = self._parse_answer(action_block_inner)
        except Exception as e:
            raise ActionParseError(
                f"Failed to parse action block '{action_block_inner}': {e}"
            ) from e

        action_json = {"action": action_name, "params": action_params}
        action_json = self._convert_coordinate(action_json, gui_state)
        internal_action, internal_params = self.convert_to_internal_format(
            action_json["action"], action_json["params"]
        )
        return ActionResult(
            action=internal_action,
            params=internal_params,
            extras={
                "raw_action_block": action_block_full,
                "raw_text": output_text,
                "parse_mode": "action_block_parsed",
            },
        )

    def _first_user_text(self, user_goal: str) -> str:
        return f"### User Task\n{user_goal}"

    def _resolve_screenshot_url(self, gui_state: dict[str, Any]) -> str:
        return self._resolve_url(gui_state)

    def _resolve_url(self, value: Any) -> str:
        from .image_utils import resolve_image_url

        if isinstance(value, dict):
            return resolve_image_url(value, max_pixels=self.max_image_pixels)
        if isinstance(value, str):
            # Local paths become data URLs; remote and data URLs pass through.
            return resolve_image_url({"screenshot_path": value}, max_pixels=self.max_image_pixels)
        return ""

    def _extract_tag(self, text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>", text)
        return m.group(1).strip() if m else ""

    def _parse_answer(self, answer_text: str) -> Tuple[str, dict[str, Any]]:
        answer_text = answer_text.strip()
        pattern = r"^(\w+)\((.*)\)$"
        m = re.match(pattern, answer_text, re.DOTALL)
        if not m:
            raise ValueError(f"Cannot parse action block: {answer_text}")

        action_type = m.group(1)
        params_str = m.group(2).strip()
        params: dict[str, Any] = {}

        if params_str:
            for pair in self._split_parameters(params_str):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    params[key.strip()] = value.strip().strip("'")
                else:
                    params[pair.strip()] = None

        action_lower = action_type.lower()

        if action_lower in (
            "click", "doubleclick", "tripleclick", "rightclick", "middleclick", "hover", "longpress",
        ):
            p_x, p_y = self._parse_coordinates(params.get("box", ""))
            if p_x is not None and p_y is not None:
                out: dict[str, Any] = {"box": (p_x, p_y)}
                if "holding" in params:
                    out["holding"] = self._parse_list(params["holding"])
                return action_lower, out
            raise ValueError(f"Unknown {action_type} params: {params!r}")

        if action_lower in ("drag", "swipe"):
            s_x, s_y = self._parse_coordinates(params.get("start", ""))
            e_x, e_y = self._parse_coordinates(params.get("end", ""))
            if all(v is not None for v in (s_x, s_y, e_x, e_y)):
                out = {"start": (s_x, s_y), "end": (e_x, e_y)}
                if "holding" in params:
                    out["holding"] = self._parse_list(params["holding"])
                return action_lower, out
            raise ValueError(f"Unknown {action_type} params: {params!r}")

        if action_lower == "type":
            type_text = params.get("content")
            if type_text is not None:
                return action_lower, {"content": type_text}
            raise ValueError(f"Unknown type params: {params!r}")

        if action_lower in ("calluser", "answer", "finished"):
            text = params.get("content", "")
            return action_lower, {"content": text}

        if action_lower == "launchapp":
            return "launchapp", {"app": params.get("app", "")}

        if action_lower == "getscreenshot":
            return "getscreenshot", {}

        if action_lower == "hotkey":
            keys = self._parse_list(params.get("keys", ""))
            if not keys:
                raise ValueError(f"Unknown Hotkey params: {params!r}")
            try:
                repeat = int(params.get("repeat", "1"))
            except ValueError as exc:
                raise ValueError(f"Invalid Hotkey repeat: {params!r}") from exc
            if repeat < 1:
                raise ValueError(f"Hotkey repeat must be positive: {params!r}")
            return action_lower, {"keys": keys, "repeat": repeat}

        if action_lower in ("wait", "pressback", "presshome", "pressenter", "pressrecent"):
            return action_lower, {}

        raise ValueError(f"Unknown action: {action_type}")

    def _split_parameters(self, params_str: str) -> list[str]:
        parts: list[str] = []
        current = ""
        in_quotes = False
        quote_char = None
        bracket_level = 0

        for char in params_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif not in_quotes:
                if char in "([":
                    bracket_level += 1
                elif char in ")]":
                    bracket_level -= 1
                elif char == "," and bracket_level == 0:
                    parts.append(current.strip())
                    current = ""
                    continue
            current += char

        if current.strip():
            parts.append(current.strip())
        return parts

    def _parse_coordinates(self, coord_str: str) -> Tuple[Any, Any]:
        if not coord_str:
            return None, None
        coord_str = coord_str.replace(" ", "")
        match = re.match(r"\(([\d.]+),([\d.]+)\)", coord_str)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    def _parse_list(self, value: Any) -> list[str]:
        """Parse the simple single-quoted lists used by Ling action syntax."""
        if isinstance(value, list):
            return [str(item) for item in value]
        raw = str(value).strip()
        if not raw.startswith("[") or not raw.endswith("]"):
            return []
        return [item.strip().strip("'\"") for item in self._split_parameters(raw[1:-1]) if item.strip()]

    def _rescale_coordinate(
        self,
        x: float,
        y: float,
        orig_size: Tuple[int, int],
        resized_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        o_w, o_h = orig_size
        r_w, r_h = resized_size
        x_scaled = int(x * o_w / r_w)
        y_scaled = int(y * o_h / r_h)
        # PyAutoGUI's valid coordinates are 0..width-1 and 0..height-1.
        # The model's inclusive 999 edge otherwise maps one pixel past them.
        return max(0, min(x_scaled, o_w - 1)), max(0, min(y_scaled, o_h - 1))

    def _convert_coordinate(self, action_json: dict, gui_state: dict[str, Any]) -> dict:
        original_width, original_height = self._get_screen_size(gui_state)
        # Ling uses a fixed virtual coordinate space from 0 through 999.
        resized_width, resized_height = 999, 999
        if not all([original_width, original_height, resized_width, resized_height]):
            return action_json

        orig_size = (original_width, original_height)
        resized_size = (resized_width, resized_height)
        action_type = action_json["action"].lower()

        if action_type in (
            "click", "doubleclick", "tripleclick", "rightclick", "middleclick", "hover", "longpress",
        ):
            x, y = action_json["params"]["box"]
            action_json["params"]["box"] = self._rescale_coordinate(x, y, orig_size, resized_size)
        elif action_type in ("drag", "swipe"):
            x1, y1 = action_json["params"]["start"]
            x2, y2 = action_json["params"]["end"]
            action_json["params"]["start"] = self._rescale_coordinate(x1, y1, orig_size, resized_size)
            action_json["params"]["end"] = self._rescale_coordinate(x2, y2, orig_size, resized_size)
        return action_json

    def convert_to_internal_format(self, action: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action = action.lower()
        out = dict(params or {})
        action_alias = {
            "doubleclick": "double_click",
            "tripleclick": "triple_click",
            "rightclick": "right_click",
            "middleclick": "middle_click",
            "longpress": "long_press",
            "launchapp": "launch",
            "pressback": "back",
            "presshome": "home",
            "pressenter": "enter",
            "pressrecent": "menu",
            "answer": "answer",
            "finished": "terminate",
        }
        action = action_alias.get(action, action)

        if action in (
            "click", "double_click", "triple_click", "right_click", "middle_click", "hover", "long_press",
        ):
            box = out.pop("box", None)
            if isinstance(box, (list, tuple)) and len(box) >= 2:
                out["x"] = box[0]
                out["y"] = box[1]

        elif action in ("drag", "swipe"):
            start = out.pop("start", None)
            end = out.pop("end", None)
            if isinstance(start, (list, tuple)) and len(start) >= 2:
                out["x1"] = start[0]
                out["y1"] = start[1]
            if isinstance(end, (list, tuple)) and len(end) >= 2:
                out["x2"] = end[0]
                out["y2"] = end[1]

        return action, out
