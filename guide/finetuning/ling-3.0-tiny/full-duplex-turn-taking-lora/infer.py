"""Query the dialogue brain through any OpenAI-compatible endpoint.

Single decision:
    python infer.py --agent-state silent --user "so I was thinking we could" --silence 0.8

Scripted call (replays a few moments of a conversation, one request per moment):
    python infer.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdb_brain.prompt import build_messages, parse_action  # noqa: E402

DEMO = [
    # (agent_state, what the assistant is saying, user speech so far, silence in seconds)
    ("silent", "", "Hey, can you help me plan a", 0.8),
    ("silent", "", "Hey, can you help me plan a weekend trip to Kyoto?", 0.8),
    ("speaking", "Sure! Kyoto is lovely in autumn. On day one you could visit", "yeah", 0.3),
    ("speaking", "Sure! Kyoto is lovely in autumn. On day one you could visit", "honey, can you grab my keys?", 0.3),
    ("speaking", "Sure! Kyoto is lovely in autumn. On day one you could visit",
     "wait, actually make it Osaka instead", 0.3),
    ("silent", "", "So last time I went to Japan I got completely lost in the subway, and then", 0.8),
    ("silent", "", "So last time I went to Japan I got completely lost in the subway, and then "
     "a stranger walked me all the way to my hotel.", 0.8),
]


def decide(client: httpx.Client, model: str, *, agent_state: str, history: str, user: str,
           silence: float) -> tuple[str | None, str]:
    point = {"agent_state": agent_state, "history": history, "user": user, "silence_s": silence}
    r = client.post("/v1/chat/completions", json={
        "model": model, "messages": build_messages(point), "max_tokens": 128, "temperature": 0.0,
    })
    r.raise_for_status()
    return parse_action(r.json()["choices"][0]["message"]["content"] or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--agent-state", choices=["silent", "speaking"], default="silent")
    ap.add_argument("--history", default="", help="what the assistant is currently saying")
    ap.add_argument("--user", help="user speech so far (ASR text)")
    ap.add_argument("--silence", type=float, default=0.8)
    args = ap.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        model = client.get("/v1/models").json()["data"][0]["id"]
        moments = DEMO if args.demo else [(args.agent_state, args.history, args.user, args.silence)]
        for state, history, user, silence in moments:
            action, reply = decide(client, model, agent_state=state, history=history, user=user,
                                   silence=silence)
            print(f"[{state:8s}] user: {user!r}")
            print(f"           -> {action}" + (f"  {reply!r}" if reply else ""))


if __name__ == "__main__":
    main()
