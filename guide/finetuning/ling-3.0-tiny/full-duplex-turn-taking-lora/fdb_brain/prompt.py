"""Shared prompt format for the full-duplex dialogue brain (training data, inference, eval)."""

from __future__ import annotations

import re

ACTIONS = ("LISTEN", "BACKCHANNEL", "SPEAK", "STOP", "IGNORE")

SYSTEM_PROMPT = """You are the turn-taking brain of a full-duplex voice assistant. You receive a live speech-to-text stream and decide, right now, what the assistant does next.

Answer with exactly one action tag first:
[LISTEN] The user has not finished their turn (mid-sentence pause, hesitation, trailing "and", "um"). Stay silent.
[BACKCHANNEL] The user is in the middle of a long turn. Give a very short acknowledgement such as "mm-hmm" without taking the floor.
[SPEAK] The user has finished a turn addressed to you. Take the floor and reply.
[STOP] You are speaking and the user interrupts you with a new or changed request. Stop and respond to the new request.
[IGNORE] The new speech does not need a response from you: a brief acknowledgement ("yeah", "I see"), the user talking to someone else, or background speech. Keep doing what you were doing.

After [SPEAK] or [STOP], write the spoken reply in one or two short sentences. After [BACKCHANNEL], write only the acknowledgement. After [LISTEN] or [IGNORE], write nothing else."""

_TAG = re.compile(r"\[(LISTEN|BACKCHANNEL|SPEAK|STOP|IGNORE)\]")


def user_message(*, agent_state: str, user: str, silence_s: float, history: str = "") -> str:
    if agent_state == "speaking":
        state = f'speaking (currently answering: "{history}")'
    else:
        state = "silent"
    return (
        f"Assistant state: {state}\n"
        f'New user speech: "{user}"\n'
        f"Silence since the last word: {silence_s:.1f}s\n"
        "Action:"
    )


def build_messages(point: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_message(
                agent_state=point["agent_state"],
                user=point["user"],
                silence_s=float(point["silence_s"]),
                history=point.get("history", ""),
            ),
        },
    ]


def parse_action(text: str) -> tuple[str | None, str]:
    """Return (action, reply). Reasoning before a closing </think> is ignored."""
    answer = text.rsplit("</think>", 1)[-1]
    m = _TAG.search(answer)
    if m is None:
        return None, answer.strip()
    return m.group(1), answer[m.end():].strip()
