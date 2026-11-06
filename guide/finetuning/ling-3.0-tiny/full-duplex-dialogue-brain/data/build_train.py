"""Expand synthetic scenarios into labelled decision points for SFT.

One scenario -> up to 18 decision points covering all five actions:

    request cut mid-sentence          (silent)    -> [LISTEN]
    request complete                  (silent)    -> [SPEAK] reply
    story cut at an early clause      (silent)    -> [LISTEN]
    long story cut mid-sentence       (silent)    -> [LISTEN]
    story cut after a full sentence   (silent)    -> [BACKCHANNEL] mm-hmm   (sampled)
    2x story window ending on a complete-sounding clause before the
       final sentence (hard negatives) (silent)   -> [LISTEN]
    story complete                    (silent)    -> [SPEAK] story_reply
    short acknowledgement             (silent)    -> [LISTEN]      (sampled)
    interrupt                         (speaking)  -> [STOP] interrupt_reply
    short acknowledgement             (speaking)  -> [IGNORE]
    side talk                         (speaking)  -> [IGNORE]
    background speech                 (speaking)  -> [IGNORE]
  scenarios with the r2 fields additionally give:
    topic switch                      (speaking)  -> [STOP] topic_switch_reply
    statement cut mid-sentence        (silent)    -> [LISTEN]
    statement complete                (silent)    -> [SPEAK] statement_reply
    short story complete              (silent)    -> [SPEAK] short_story_reply

User texts are randomly rendered in ASR style (bracketed [UM]/[UH] fillers, no
punctuation) so the model does not rely on clean written punctuation.

Rows are written pre-tokenized ({"tokens", "prompt_mask"}) with the same chat template
`areno serve --disable-thinking` renders, so training and inference see identical
prompts. (AReno's raw-text SFT path does not recognise Ling's `<role>` markers and
would wrap an already-formatted prompt a second time.)

Example:
    python data/build_train.py --scenarios data/scenarios.jsonl data/scenarios_r2.jsonl --fdb-eval data/fdb_eval.jsonl \
        --tokenizer /path/to/Ling-3.0-tiny --out-dir data/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdb_brain.prompt import build_messages  # noqa: E402

ACKS = ["yeah", "mm-hmm", "uh-huh", "okay", "right", "I see", "yes", "sure", "got it", "oh nice",
        "yeah yeah", "mhm", "okay okay", "true", "makes sense"]
BACKCHANNELS = ["mm-hmm", "uh-huh", "yeah", "right", "I see", "oh wow", "mhm", "okay"]
# Short replies the user gives while the assistant is silent: the user keeps the floor.
SILENT_ACKS = ["yeah", "uh huh", "mhm", "no", "yeah um", "right", "okay", "yes", "oh yeah", "um",
               "yeah yeah", "sure", "uh", "hmm", "no no"]
P_SILENT_ACK = 0.5
P_BRACKET_FILLERS = 0.35
P_NO_PUNCT = 0.35
P_BACKCHANNEL = 0.6
N_MIDTURN_NEG = 2
SILENCE_S = 0.8
OVERLAP_S = 0.3
_SENT = re.compile(r"(?<=[.!?])\s+")
_FILLERS = {"um", "uh", "like", "so", "and", "but", "or", "you", "know", "the", "a", "to", "of", "my", "i"}


def _cut_mid(text: str, rng: random.Random) -> str | None:
    """Prefix ending mid-thought: drop at least 3 trailing words and strip end punctuation."""
    words = text.split()
    if len(words) < 6:
        return None
    k = rng.randint(max(2, len(words) // 3), len(words) - 3)
    prefix = " ".join(words[:k]).rstrip(",;:")
    # Prefer ending on a function word or filler so the cut sounds like a real pause.
    for j in range(k, max(1, k - 4), -1):
        if words[j - 1].lower().strip(",.") in _FILLERS:
            prefix = " ".join(words[:j]).rstrip(",;:")
            break
    return prefix.rstrip(".!?")


_FILLER_WORD = re.compile(r"\b(um+|uh+|er+)\b,?", re.I)
_PUNCT = re.compile(r"[.,!?;:\"]")


def _asr_style(text: str, rng: random.Random) -> str:
    """Randomly render text like a streaming ASR transcript."""
    if rng.random() < P_BRACKET_FILLERS:
        text = _FILLER_WORD.sub(lambda m: "[UH]" if m.group(1).lower().startswith("u") else "[UM]", text)
    if rng.random() < P_NO_PUNCT:
        text = _PUNCT.sub("", text)
    return " ".join(text.split())


_CLAUSE_END = re.compile(r"(?<=,)\s+(?=\S)")


def _midturn_clause(sents: list[str], rng: random.Random) -> str:
    """A window of a rambling turn that ends on a clause/sentence boundary before the last sentence.

    Half of the windows start mid-sentence (as when a streaming segment begins inside an
    ongoing turn), half start at the beginning of the turn.
    """
    end = rng.randint(1, len(sents) - 1)  # exclusive; never includes the final sentence
    pieces = sents[:end]
    last_clauses = _CLAUSE_END.split(pieces[-1])
    if len(last_clauses) > 1 and rng.random() < 0.5:
        pieces[-1] = " ".join(last_clauses[: rng.randint(1, len(last_clauses) - 1)])
    text = " ".join(pieces).rstrip(",;: ")
    if rng.random() < 0.5:
        words = text.split()
        if len(words) > 10:
            text = " ".join(words[rng.randint(2, len(words) // 2):])
            text = text[0].lower() + text[1:]
    return text


def _point(agent: str, user: str, silence: float, history: str = "") -> dict:
    return {"agent_state": agent, "user": user, "silence_s": silence, "history": history}


def expand(sc: dict, rng: random.Random) -> list[tuple[dict, str, str]]:
    out: list[tuple[dict, str, str]] = []
    req, story = sc["request"], sc["story"]

    def silent(user: str, gold: str, target: str) -> None:
        out.append((_point("silent", _asr_style(user, rng), SILENCE_S), gold, target))

    def speaking(user: str, gold: str, target: str) -> None:
        out.append((_point("speaking", _asr_style(user, rng), OVERLAP_S, req), gold, target))

    if (cut := _cut_mid(req, rng)) is not None:
        silent(cut, "LISTEN", "[LISTEN]")
    silent(req, "SPEAK", f"[SPEAK] {sc['reply']}")

    sents = [s for s in _SENT.split(story) if s.strip()]
    if len(sents) >= 3:
        if (cut := _cut_mid(sents[0] + " " + sents[1], rng)) is not None:
            silent(cut, "LISTEN", "[LISTEN]")
        # Long turn cut inside a later sentence: the user is clearly still going.
        k = rng.randint(2, len(sents) - 1)
        if (cut := _cut_mid(sents[k], rng)) is not None:
            silent(" ".join(sents[:k]) + " " + cut, "LISTEN", "[LISTEN]")
        k = rng.randint(1, len(sents) - 2)
        if rng.random() < P_BACKCHANNEL:
            silent(" ".join(sents[: k + 1]), "BACKCHANNEL", f"[BACKCHANNEL] {rng.choice(BACKCHANNELS)}")
        # Hard negatives: the turn so far ends on a complete-sounding clause, but the
        # speaker is mid-way through a longer turn, so the assistant keeps listening.
        for _ in range(N_MIDTURN_NEG):
            silent(_midturn_clause(sents, rng), "LISTEN", "[LISTEN]")
        silent(story, "SPEAK", f"[SPEAK] {sc['story_reply']}")
    if rng.random() < P_SILENT_ACK:
        silent(rng.choice(SILENT_ACKS), "LISTEN", "[LISTEN]")

    speaking(sc["interrupt"], "STOP", f"[STOP] {sc['interrupt_reply']}")
    speaking(rng.choice(ACKS), "IGNORE", "[IGNORE]")
    speaking(sc["side_talk"], "IGNORE", "[IGNORE]")
    speaking(sc["background"], "IGNORE", "[IGNORE]")

    if "topic_switch" in sc:
        speaking(sc["topic_switch"], "STOP", f"[STOP] {sc['topic_switch_reply']}")
        if (cut := _cut_mid(sc["statement"], rng)) is not None:
            silent(cut, "LISTEN", "[LISTEN]")
        silent(sc["statement"], "SPEAK", f"[SPEAK] {sc['statement_reply']}")
        silent(sc["short_story"], "SPEAK", f"[SPEAK] {sc['short_story_reply']}")
    return out


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, nargs="+")
    ap.add_argument("--fdb-eval", required=True, help="used only to drop overlapping texts")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from areno.api.tokenizer import normalize_token_ids

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    fdb_texts = {_norm(json.loads(line)["user"]) for line in open(args.fdb_eval)}
    fdb_texts |= {_norm(json.loads(line).get("history", "")) for line in open(args.fdb_eval)}
    fdb_texts.discard("")

    scenarios = [json.loads(line) for path in args.scenarios for line in open(path)]
    seen, kept, leak = set(), [], 0
    for sc in scenarios:
        key = _norm(sc["request"])
        if key in seen:
            continue
        seen.add(key)
        if any(_norm(sc[k]) in fdb_texts for k in ("request", "interrupt", "side_talk", "background",
                                                    "topic_switch", "statement", "short_story") if k in sc):
            leak += 1
            continue
        kept.append(sc)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac))
    splits = {"val": kept[:n_val], "train": kept[n_val:]}

    out_dir = Path(args.out_dir)
    for name, rows in splits.items():
        counts: Counter[str] = Counter()
        with open(out_dir / f"{name}.jsonl", "w") as f_sft, open(out_dir / f"{name}_points.jsonl", "w") as f_pts:
            for i, sc in enumerate(rows):
                for j, (point, gold, target) in enumerate(expand(sc, rng)):
                    prompt_ids = normalize_token_ids(tok.apply_chat_template(
                        build_messages(point), tokenize=True, add_generation_prompt=True, enable_thinking=False))
                    target_ids = tok.encode(target, add_special_tokens=False) + [tok.eos_token_id]
                    f_sft.write(json.dumps({"tokens": prompt_ids + target_ids,
                                            "prompt_mask": [True] * len(prompt_ids) + [False] * len(target_ids),
                                            "target": target}) + "\n")
                    f_pts.write(json.dumps({"id": f"synth/{name}/{i}/{j}", "source": f"synth_{name}",
                                            "gold": gold, "accept": [gold], **point}, ensure_ascii=False) + "\n")
                    counts[gold] += 1
        print(f"{name}: scenarios={len(rows)} points={sum(counts.values())} {dict(sorted(counts.items()))}")
    print(f"dropped duplicates={len(scenarios) - len(seen)} fdb_overlap={leak}")


if __name__ == "__main__":
    main()
