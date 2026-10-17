"""Generate spoken voice-assistant scenario scripts with an OpenAI-compatible LLM.

Each scenario is one JSON object holding the raw material for several decision
points (see build_train.py): a spoken request and reply, an interruption, side
talk, background speech, and a rambling personal story. FDB is never shown to
the generator; build_train.py additionally drops scenarios that overlap FDB.

Example:
    python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 1200 --out data/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx

TOPICS = [
    "cooking and recipes", "weather and clothing", "smart home control", "music playback", "podcasts",
    "travel planning", "flights and hotels", "local restaurants", "fitness and workouts", "sleep",
    "personal finance", "budgeting", "shopping lists", "online orders", "calendar and meetings",
    "reminders and alarms", "timers", "kids homework help", "pet care", "gardening", "car maintenance",
    "commuting and traffic", "movies and TV", "books", "video games", "sports scores", "news",
    "health symptoms", "medication reminders", "language learning", "math questions", "history facts",
    "science questions", "space and astronomy", "job search", "resume writing", "emails and messages",
    "phone calls", "birthday planning", "gift ideas", "home cleaning", "laundry", "furniture and DIY",
    "tech support for a laptop", "wifi problems", "photography", "hiking", "camping", "coffee",
    "baking", "wine and drinks", "meditation", "mental wellbeing", "relationships advice", "moving house",
    "renting an apartment", "public transport", "learning guitar", "painting and drawing", "fashion",
]

PERSONAS = [
    "a busy parent", "a college student", "a retired teacher", "a software engineer", "a nurse on night shifts",
    "a small business owner", "a teenager", "a first-time traveler", "a fitness enthusiast", "a home cook",
    "a remote worker", "a new dog owner", "a grandparent", "a musician", "a shift worker at a warehouse",
]

GEN_PROMPT = """Write raw material for training a voice assistant, as natural SPOKEN English (contractions, casual phrasing, no markdown).

Speaker: {persona}. Topic: {topic}.

Return ONLY a JSON object with these string fields:
"request": a complete spoken request or question to the assistant, 8 to 25 words.
"reply": the assistant's spoken answer to "request", one or two short sentences.
"interrupt": while the assistant is answering, the user cuts in to change or replace the request (for example "Actually, wait, ..." or "Hold on, can you ... instead?"), 6 to 20 words.
"interrupt_reply": the assistant's short spoken answer to "interrupt", one or two sentences.
"side_talk": the user says something to another person in the room, not to the assistant, 4 to 14 words (for example talking to a kid, partner or roommate).
"background": one line of unrelated speech from a TV, radio or someone else nearby, 4 to 14 words.
"story": the user tells the assistant a rambling personal story or opinion, 4 to 6 sentences, with fillers like "um", "like", "you know", "so". It must sound unfinished until its last sentence.
"story_reply": the assistant's short, warm spoken response once the story is complete, one or two sentences.
"topic_switch": while the assistant is answering, the user cuts in with something UNRELATED to the request that still needs the assistant, a new question or a request to change the subject, 6 to 20 words. Vary how it opens; do not always start with "By the way".
"topic_switch_reply": the assistant's short spoken answer to "topic_switch", one or two sentences.
"statement": a complete statement (NOT a question) telling the assistant about a plan, wish or situation related to the topic, 5 to 14 words.
"statement_reply": the assistant's short spoken response to "statement", one or two sentences.
"short_story": a short, complete personal anecdote or opinion told to the assistant, 2 or 3 sentences with a filler or two, ending with a clear concluding sentence.
"short_story_reply": the assistant's short spoken response to "short_story", one or two sentences."""

FIELDS = ("request", "reply", "interrupt", "interrupt_reply", "side_talk", "background", "story", "story_reply",
          "topic_switch", "topic_switch_reply", "statement", "statement_reply", "short_story", "short_story_reply")


def _extract_json(text: str) -> dict | None:
    text = text.rsplit("</think>", 1)[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in FIELDS):
        return None
    return {k: " ".join(obj[k].split()) for k in FIELDS}


async def main_async(args) -> None:
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    have = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    jobs = [(rng.choice(TOPICS), rng.choice(PERSONAS)) for _ in range(args.n * 2)]
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        state = {"ok": have, "bad": 0}
        out = open(out_path, "a")

        async def one(topic: str, persona: str) -> None:
            if state["ok"] >= args.n:
                return
            async with sem:
                if state["ok"] >= args.n:
                    return
                r = await client.post("/v1/chat/completions", json={
                    "model": model,
                    "messages": [{"role": "user", "content": GEN_PROMPT.format(topic=topic, persona=persona)}],
                    "max_tokens": 1100, "temperature": 1.0, "top_p": 0.95,
                })
            obj = _extract_json(r.json()["choices"][0]["message"]["content"] or "") if r.status_code == 200 else None
            async with lock:
                if obj is None:
                    state["bad"] += 1
                    return
                if state["ok"] >= args.n:
                    return
                state["ok"] += 1
                out.write(json.dumps({"topic": topic, "persona": persona, **obj}, ensure_ascii=False) + "\n")
                out.flush()

        await asyncio.gather(*(one(t, p) for t, p in jobs))
        out.close()
    print(f"scenarios ok={state['ok']} rejected={state['bad']} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data/scenarios.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
