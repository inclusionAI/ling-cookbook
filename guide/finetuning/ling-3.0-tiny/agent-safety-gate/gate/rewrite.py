"""Grow the human folds with a served Ling-3.0-tiny.

Repeating a human record teaches its wording. So each training record is also rewritten:
`transfer` moves the same behaviour to another domain and keeps the label, and `flip` removes
the fault a reviewer named, which turns an unsafe record into a safe one. Only records that may
be trained on are read, so dev records from test families and every test record stay out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time

import httpx

from allin import trains_on
from benches import clip, load_all, norm

DOMAINS = (
    "retail banking", "hospital records", "smart home control", "cloud infrastructure", "online shopping",
    "legal case files", "human resources", "travel booking", "social media management", "school administration",
    "source code repositories", "insurance claims", "freight logistics", "city government services",
    "power grid operations", "laboratory research", "email and calendar", "cryptocurrency wallets",
    "video game servers", "customer support", "pharmacy orders", "factory robots", "newsroom publishing",
    "tax preparation", "real estate listings", "vehicle fleet management",
)

FORMAT = """Answer in exactly this format and nothing else:
User request: <the user's request>
Step 1 [action]: <text>
Step 2 [observation]: <text>
Final response: <the agent's closing message, or none>

Each step tag is one of: action, observation, thought, user."""

TRANSFER = """Below is a record of an AI agent working on a task, followed by a reviewer's note.

{record}

Reviewer's note: {note}

Write a new record in the domain "{domain}" where the agent behaves the same way. {keep}
Use a different user, different tools, and different names and values. Keep about the same number of steps, and keep tool calls as tool calls. Never mention the review, and do not comment on whether the agent is safe.

{format}"""

KEEP_UNSAFE = "It must make the same kind of mistake the reviewer found, without any hint that it is a mistake."
KEEP_SAFE = "It must stay just as well behaved as the agent in this record."

FLIP = """Below is a record of an AI agent working on a task, followed by a reviewer's objection.

{record}

Reviewer's objection: {note}

Rewrite the record with the smallest change that leaves the reviewer nothing to object to. For example the agent asks the user first, checks the source, refuses the harmful part, or keeps the secret out of its reply. Keep the same user request and the same style of steps. Never mention the review.

{format}"""


def as_text(record: dict) -> str:
    lines = ["User request: " + clip(record["request"], 700)]
    for index, (kind, text) in enumerate(record["steps"][:14], 1):
        lines.append(f"Step {index} [{kind}]: " + clip(text, 320))
    if record.get("response"):
        lines.append("Final response: " + clip(record["response"], 320))
    return "\n".join(lines)


def parse(content: str):
    request, response, steps, section = "", "", [], None
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        step = re.match(r"step\s*\d+\s*\[(\w+)\]\s*:\s*(.*)", line, re.I)
        if step:
            kind = step.group(1).lower()
            steps.append([kind if kind in ("action", "observation", "thought", "user") else "action", step.group(2)])
            section = "step"
        elif line.lower().startswith("user request:"):
            request, section = line.split(":", 1)[1].strip(), "request"
        elif line.lower().startswith("final response:"):
            response, section = line.split(":", 1)[1].strip(), "response"
        elif section == "step":
            steps[-1][1] += " " + line
        elif section == "request":
            request += " " + line
        elif section == "response":
            response += " " + line
    if response.lower() in ("none", "none.", "n/a"):
        response = ""
    if len(request) < 12 or not steps or re.search(r"review(er|'s)", content, re.I):
        return None
    return request, steps, response


async def one(client, args, job, semaphore, sink, stats):
    body = {
        "model": "policy",
        "messages": [{"role": "user", "content": job["prompt"]}],
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": args.max_tokens,
    }
    async with semaphore:
        try:
            reply = await client.post(f"{args.base_url}/v1/chat/completions", json=body)
            reply.raise_for_status()
        except Exception:  # noqa: BLE001
            stats["error"] += 1
            return
    data = reply.json()
    stats["tokens"] += (data.get("usage") or {}).get("completion_tokens") or 0
    parsed = parse(data["choices"][0]["message"].get("content") or "")
    source = job["source"]
    if parsed is None or (job["mode"] == "transfer" and norm(parsed[0]) == norm(source["request"])):
        stats["bad"] += 1
        return
    request, steps, response = parsed
    row = {
        "source": source["id"], "bench": source["bench"], "mode": job["mode"], "domain": job["domain"], "group": source["group"],
        "label": job["label"], "target": job["target"], "profile": source.get("profile") or "", "env": "",
        "request": request, "steps": steps, "response": response,
    }
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.flush()
    stats["kept"] += 1
    if stats["kept"] % 200 == 0:
        elapsed = time.time() - stats["start"]
        print(f"kept={stats['kept']} bad={stats['bad']} err={stats['error']} {stats['kept'] / elapsed:.2f}/s {stats['tokens'] / elapsed:.0f} tok/s", flush=True)


def jobs_for(args) -> list[dict]:
    rng = random.Random(args.seed)
    data = load_all(args.bench_root)
    rjudge_groups = {record["group"] for record in data["rjudge"]}
    jobs = []
    for bench, copies in (item.split(":") for item in args.plan.split(",")):
        for index, record in enumerate(data[bench]):
            if not trains_on(record, bench, rjudge_groups, True):
                continue
            ambiguous = bench == "asse-safety" and data["asse-strict"][index]["label"] != data["asse-loose"][index]["label"]
            label, note = record["label"], clip(record.get("rationale") or "", 420)
            target = 0.1 if not label else 0.9
            if ambiguous:
                target = 0.6 if label else 0.4
            if not note:
                note = "The agent's actions carry a risk the user did not accept." if label else "The agent stays within the request."
            for domain in rng.sample(DOMAINS, int(copies)):
                keep = KEEP_UNSAFE if label else KEEP_SAFE
                prompt = TRANSFER.format(record=as_text(record), note=note, domain=domain, keep=keep, format=FORMAT)
                jobs.append({"source": record, "mode": "transfer", "domain": domain, "label": label, "target": target, "prompt": prompt, "order": rng.random()})
            if label and not ambiguous and record.get("rationale"):
                prompt = FLIP.format(record=as_text(record), note=note, format=FORMAT)
                jobs.append({"source": record, "mode": "flip", "domain": "", "label": 0, "target": 0.15, "prompt": prompt, "order": rng.random()})
    jobs.sort(key=lambda job: job["order"])
    return jobs


async def run(args) -> None:
    jobs = jobs_for(args)
    print("jobs", len(jobs), flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"kept": 0, "bad": 0, "error": 0, "tokens": 0, "start": time.time()}
    limits = httpx.Limits(max_connections=args.concurrency + 8)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        with open(args.out, "a", encoding="utf-8") as sink:
            pending = set()
            for job in jobs:
                if args.stop_at and time.time() >= args.stop_at:
                    print("stop_at reached", flush=True)
                    break
                pending.add(asyncio.create_task(one(client, args, job, semaphore, sink, stats)))
                if len(pending) >= args.concurrency * 2:
                    _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if pending:
                await asyncio.wait(pending)
    print(f"final {stats}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8107")
    parser.add_argument("--plan", default="asse-safety:3,rjudge:3,preex:1", help="bench:transfers per record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--stop-at", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
