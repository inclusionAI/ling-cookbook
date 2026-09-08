"""Minimal chat-completions LLM client."""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx


DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ChatCompletionsLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **dict(extra_payload or {}),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM request failed: {response.status_code} {response.text}")

        data = response.json()
        self._log_response(data)
        self._print_reasoning(data)
        return self._extract_response(data)

    def _extract_response(self, data: dict[str, Any]) -> LLMResponse:
        if isinstance(data.get("error"), dict):
            msg = str(data["error"].get("message") or "").strip()
            if msg:
                raise LLMError(msg)

        content = ""
        try:
            content = self._content_to_text(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            pass

        if not content:
            try:
                content = self._content_to_text(data["choices"][0]["text"])
            except (KeyError, IndexError, TypeError):
                pass

        if not content and "output_text" in data:
            content = self._content_to_text(data["output_text"])

        if not content:
            raise LLMError(f"Unexpected LLM response shape: {json.dumps(data)[:400]}")

        reasoning_content = ""
        try:
            reasoning_content = self._content_to_text(
                data["choices"][0]["message"].get("reasoning_content", "")
            )
        except Exception:
            pass

        return LLMResponse(content=content, reasoning_content=reasoning_content, raw=data)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def _log_response(self, data: dict[str, Any]) -> None:
        """Persist the raw LLM response for reasoning-field diagnostics."""
        log_dir = os.environ.get("LING_GUI_LOG_DIR", str(DEFAULT_LOG_DIR / "llm_logs"))
        if not log_dir:
            return
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_model = str(self.model or "unknown").replace("/", "_").replace(":", "_")
        path = Path(log_dir) / f"{safe_model}_{ts}.json"
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _print_reasoning(self, data: dict[str, Any]) -> None:
        """Print reasoning content or embedded thinking tags to stdout."""
        try:
            msg = data["choices"][0]["message"]
            reasoning = msg.get("reasoning_content")
            if reasoning:
                text = self._content_to_text(reasoning)
                print("[reasoning_content]", text[:4000])

            content = self._content_to_text(msg.get("content"))
            if content:
                # Ling may embed its reasoning directly in <think> tags.
                import re
                think_match = re.search(r"<think>\s*([\s\S]*?)\s*</think>", content)
                if think_match:
                    print("[think]", think_match.group(1).strip()[:4000])
        except Exception:
            pass
