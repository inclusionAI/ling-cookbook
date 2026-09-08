"""Build Ling request messages and optionally call a real API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ling_gui_agent.config import LingConfig
from ling_gui_agent.planner import LingTemplate
from ling_gui_agent.planner.llm_client import ChatCompletionsLLM


def main() -> int:
    cfg = LingConfig.from_env()
    platform = os.environ.get("PLATFORM", "mobile")
    template = LingTemplate(
        max_visible_screenshots=cfg.max_visible_screenshots,
        platform=platform,
    )

    # Simulate history with one user screenshot and assistant action per turn.
    history = [
        {
            "screenshot_url": "https://example.com/screenshot_1.png",
            "action_block": "<action>LaunchApp(app='Markor')</action>",
        },
        {
            "screenshot_url": "https://example.com/screenshot_2.png",
            "action_block": "<action>Click(box=(500, 400))</action>",
        },
        {
            "screenshot_url": "https://example.com/screenshot_3.png",
            "action_block": "<action>LongPress(box=(360, 560))</action>",
        },
        {
            "screenshot_url": "https://example.com/screenshot_4.png",
            "action_block": "<action>Click(box=(850, 80))</action>",
        },
    ]

    current_screenshot_url = "https://example.com/current.png"
    user_goal = (
        "Open Markor and look at the Documents folder. If there are at least two notes, "
        "delete the two most recently modified ones. If there is only one note or none, do nothing. "
        "After deletion, confirm the remaining notes are still intact."
    )

    gui_state = {
        "image_url": current_screenshot_url,
        "metadata": {"screen_width": 1080, "screen_height": 2400},
    }

    messages = template.build_messages(
        user_goal=user_goal,
        plan_instruction=user_goal,
        gui_state=gui_state,
        history_turns=history,
    )

    print(f"Total messages: {len(messages)}")
    print(f"Platform: {platform}")
    print(f"Max visible screenshots: {cfg.max_visible_screenshots}")
    print("\n--- Messages (last 6) ---")
    for i, msg in enumerate(messages[-6:], start=len(messages) - 6):
        content_summary = summarize_content(msg.get("content"))
        print(f"[{i}] {msg['role']}: {content_summary}")

    payload = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    print("\n--- Full payload ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if os.environ.get("LING_RUN_API_TEST") == "1":
        print("\n--- Calling Ling API ---")
        client = ChatCompletionsLLM(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            timeout_seconds=cfg.timeout_seconds,
        )
        try:
            llm_response = client.complete(messages=messages, max_tokens=cfg.max_tokens)
            print("Response content:", llm_response.content)
            if llm_response.reasoning_content:
                print("Reasoning content:", llm_response.reasoning_content)
            result = template.parse_model_output(llm_response.content, gui_state)
            print("Parsed action:", result.action, result.params)
        except Exception as exc:
            print(f"API call failed: {exc}", file=sys.stderr)
            return 1

    return 0


def summarize_content(content: object) -> str:
    if isinstance(content, str):
        return content[:80]
    if isinstance(content, list):
        parts = []
        for item in content:
            t = item.get("type", "?")
            if t == "text":
                parts.append(f"text={item.get('text', '')[:40]!r}")
            elif t == "image_url":
                url = item.get("image_url", {}).get("url", "")
                parts.append(f"image_url=[{len(url)} chars]")
            else:
                parts.append(t)
        return " | ".join(parts)
    return str(content)[:80]


if __name__ == "__main__":
    raise SystemExit(main())
