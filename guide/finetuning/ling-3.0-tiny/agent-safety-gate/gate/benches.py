"""Load public agent-safety benches into one record shape.

Record fields: bench, id, group, split, label (1 = unsafe), ambiguous, category,
request, profile, env, steps [(kind, text)], response, raw_input.

`group` hashes the first user request so that records sharing a request land in
the same dev/test slice. `split` is stable: 1 group in 5 is dev, the rest test.
Our own model may only look at dev while it is being developed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re

RISK_CATEGORIES = {
    "sensitive_data_privacy_violations": "Unauthorized collection, use, disclosure, or leakage of sensitive, confidential, or personal information.",
    "property_financial_loss": "Actions that directly or indirectly cause financial loss or damage to assets.",
    "misinformation_unsafe_content": "Generation or dissemination of false, misleading, harmful, or unsafe information.",
    "compromised_availability": "Disruption or degradation of service availability, including resource exhaustion or denial of service.",
    "unintended_unauthorized_actions": "Execution of actions beyond user intent or permissions, including escalation of privileges and unauthorized automation.",
    "external_adversarial_attack": "Susceptibility to prompt injection, jailbreak attempts, or other malicious inputs that cause unintended agent behavior.",
    "bias_discrimination": "Outputs or decisions reflecting unfair bias, discrimination, or lack of fairness.",
    "lack_accountability_traceability": "Insufficient logging or explainability, making it difficult to audit or assign responsibility for agent actions.",
}


def norm(text) -> str:
    return re.sub(r"\W+", "", str(text).lower())


def group_of(text) -> str:
    return hashlib.sha1(norm(text).encode()).hexdigest()[:16]


def family_of(request: str) -> str:
    """Template family: ASSE-Security has 817 records but only 95 opening sentences."""
    first = re.split(r"[.!?\n]", str(request).strip(), maxsplit=1)[0][:80]
    return group_of(first)


def split_of(group: str, family: str | None = None) -> str:
    """dev is 1 request group in 5, unchanged since it was first inspected.

    For ASSE the rest is divided by template family, so no family spans train and test.
    Benches without a human training fold pass no family and keep everything in test.
    """
    if int(group, 16) % 5 == 0:
        return "dev"
    if family is None:
        return "test"
    return "train" if int(family, 16) % 2 == 0 else "test"


def clip(text, limit: int) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    keep = max(1, limit // 2)
    return text[:keep] + " ...[clipped]... " + text[-keep:]


def _category(output: str) -> str | None:
    flat = norm(output)
    best, where = None, -1
    for name in RISK_CATEGORIES:
        at = flat.rfind(norm(name))
        if at > where:
            best, where = name, at
    return best


def load_preex(root: str) -> list[dict]:
    with open(os.path.join(root, "preex", "dataset.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    records = []
    for index, row in enumerate(rows):
        data = json.loads(row["input"])
        actions = data.get("actions") or {}
        agent_action = actions.get("agent_action") or []
        if isinstance(agent_action, str):
            agent_action = [agent_action]
        env = data.get("env_info") or ""
        if isinstance(env, dict):
            env = str(env.get("description") or env.get("name") or "")
        label = 0 if "no risks" in row["output"].lower()[:80] else 1
        group = group_of(data.get("user_request", ""))
        records.append(
            {
                "bench": "preex",
                "id": f"preex-{index}",
                "group": group,
                "split": split_of(group, family_of(data.get("user_request", ""))),
                "label": label,
                "rationale": row["output"] if label else "",
                "ambiguous": False,
                "category": _category(row["output"]) if label else None,
                "request": str(data.get("user_request", "")),
                "profile": "",
                "env": str(env),
                "steps": [("action", str(step)) for step in agent_action],
                "response": str(actions.get("agent_response") or ""),
                "raw_input": row["input"],
            }
        )
    return records


def _parse_contents(contents) -> tuple[str, list[tuple[str, str]]]:
    request, steps = None, []
    for round_ in contents:
        for turn in round_ if isinstance(round_, list) else [round_]:
            role = turn.get("role")
            if role == "user":
                text = turn.get("content") or ""
                if request is None:
                    request = str(text)
                else:
                    steps.append(("user", str(text)))
            elif role == "agent":
                for key, kind in (("thought", "thought"), ("action", "action"), ("content", "response")):
                    value = turn.get(key)
                    if value:
                        steps.append((kind, str(value)))
            elif role == "environment" and turn.get("content"):
                steps.append(("observation", str(turn["content"])))
    return request or "", steps


def _transcript(row: dict, bench: str, index: int, label: int) -> dict:
    request, steps = _parse_contents(row["contents"])
    response = ""
    if steps and steps[-1][0] == "response":
        response = steps.pop()[1]
    group = group_of(request)
    return {
        "bench": bench,
        "id": f"{bench}-{row.get('id', index)}-{index}",
        "group": group,
        "split": split_of(group, family_of(request)),
        "label": int(label),
        "rationale": str(row.get("risk_description") or ""),
        "ambiguous": bool(row.get("ambiguous")),
        "category": None,
        "request": request,
        "profile": str(row.get("profile") or ""),
        "env": "",
        "steps": steps,
        "response": response,
        "raw_input": "",
    }


def load_asse(root: str, name: str) -> list[dict]:
    path = os.path.join(root, f"AgentJudge-{name}.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [_transcript(row, f"asse-{name}", index, row["label"]) for index, row in enumerate(rows)]


def load_rjudge(root: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(root, "rjudge", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        stem = os.path.basename(path)[:-5]
        for index, row in enumerate(rows):
            record = _transcript(row, "rjudge", index, row["label"])
            record["id"] = f"rjudge-{stem}-{row.get('id', index)}"
            record["attack_type"] = row.get("attack_type")
            records.append(record)
    return records


def load_all(root: str) -> dict[str, list[dict]]:
    data = {
        "preex": load_preex(root),
        "asse-security": load_asse(root, "security"),
        "asse-safety": load_asse(root, "safety"),
        "asse-strict": load_asse(root, "strict"),
        "asse-loose": load_asse(root, "loose"),
        "rjudge": load_rjudge(root),
    }
    # ASSE-Safety reuses R-Judge records. Those stay out of training, which keeps R-Judge a
    # clean test and keeps its unlicensed text out of any adapter we publish.
    rjudge_groups = {record["group"] for record in data["rjudge"]}
    for name in ("asse-security", "asse-safety", "asse-strict", "asse-loose"):
        for record in data[name]:
            if record["split"] == "train" and record["group"] in rjudge_groups:
                record["split"] = "unused"
    return data


def render_state(record: dict, *, total_limit: int = 6000) -> str:
    """One text view of a trajectory, shared by every Ling arm and by training."""
    head = []
    if record.get("profile"):
        head.append("Agent role: " + clip(record["profile"], 300))
    if record.get("env"):
        head.append("Environment: " + clip(record["env"], 400))
    head.append("User request: " + clip(record["request"], 1500))
    for step_limit in (900, 500, 280, 160, 90):
        lines = [f"{i}. {kind}: {clip(text, step_limit)}" for i, (kind, text) in enumerate(record["steps"], 1)]
        tail = ["Agent steps:", *lines] if lines else ["Agent steps: none"]
        if record.get("response"):
            tail.append("Agent final response: " + clip(record["response"], min(step_limit, 600)))
        text = "\n".join(head + tail)
        if len(text) <= total_limit:
            return text
    return clip(text, total_limit)


if __name__ == "__main__":
    import sys

    data = load_all(sys.argv[1])
    for name, rows in data.items():
        positive = sum(r["label"] for r in rows)
        dev = sum(r["split"] == "dev" for r in rows)
        groups = len({r["group"] for r in rows})
        lengths = sorted(len(render_state(r)) for r in rows)
        empty = sum(1 for r in rows if not r["steps"] and not r["response"])
        print(
            f"{name:14s} n={len(rows):5d} unsafe={positive:5d} groups={groups:5d} dev={dev:4d} "
            f"state_chars p50={lengths[len(lengths) // 2]} p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]} no_steps={empty}"
        )
    print(render_state(data["asse-safety"][383])[:900])
