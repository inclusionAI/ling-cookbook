"""Turn Full-Duplex-Bench (v1.0 / v1.5) annotations into text-level decision points.

Each decision point is one moment in a call where the dialogue brain must act:

    agent state  (silent / speaking) + what the user said so far + silence length
        -> LISTEN | BACKCHANNEL | SPEAK | STOP | IGNORE

Only FDB text annotations (word-level transcripts and turn texts) are used; no audio.
FDB is evaluation-only: nothing here is used for training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Agent-silent decisions all use the same silence so the label cannot be read off
# the pause length: telling a mid-sentence pause from a finished turn must come
# from what the user said.
SILENT_GAP_S = 0.8
OVERLAP_GAP_S = 0.3
ICC_MIN_GAP_S = 0.3


def _samples(d: Path) -> list[Path]:
    """Numbered sample directories, skipping stray files such as .DS_Store."""
    return sorted((p for p in d.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))


def _words(transcript: list[dict]) -> list[tuple[str, float, float]]:
    return [(w["text"], float(w["timestamp"][0]), float(w["timestamp"][1])) for w in transcript]


def _text_until(words: list[tuple[str, float, float]], t: float) -> str:
    return " ".join(w for w, _, end in words if end <= t + 1e-6)


def _point(source: str, sample: str, gold: str, *, agent: str, user: str, silence: float,
           history: str = "", accept: list[str] | None = None) -> dict:
    return {
        "id": f"{source}/{sample}/{gold.lower()}",
        "source": source,
        "gold": gold,
        "accept": accept or [gold],
        "agent_state": agent,
        "history": history,
        "user": user,
        "silence_s": silence,
    }


def pause_points(root: Path, subset: str, *, end_of_turn: bool) -> list[dict]:
    """[PAUSE] spans -> LISTEN; synthetic clips also end on a finished turn -> SPEAK."""
    out = []
    for d in _samples(root / "v1.0" / subset):
        tr = d / "transcription.json"
        if not tr.exists():
            continue
        words = _words(json.loads(tr.read_text()))
        for i, pause in enumerate(json.loads((d / "pause.json").read_text())):
            user = _text_until(words, float(pause["timestamp"][0]))
            if user:
                p = _point(subset, f"{d.name}.{i}", "LISTEN", agent="silent", user=user, silence=SILENT_GAP_S)
                p["id"] += f".{i}"
                out.append(p)
        if end_of_turn and words:
            out.append(_point(subset, d.name, "SPEAK", agent="silent", user=" ".join(w for w, _, _ in words),
                              silence=SILENT_GAP_S))
    return out


def icc_points(root: Path) -> list[dict]:
    """Mid-turn gaps in long ICC monologues: listening or a backchannel is fine, taking the floor is not."""
    out = []
    for d in _samples(root / "v1.0" / "icc_backchannel"):
        tr = d / "transcription.json"
        if not tr.exists():
            continue
        words = _words(json.loads(tr.read_text()))
        for i in range(len(words) - 1):
            if words[i + 1][1] - words[i][2] >= ICC_MIN_GAP_S:
                p = _point("icc_backchannel", f"{d.name}.{i}", "BACKCHANNEL", agent="silent",
                           user=" ".join(w for w, _, _ in words[: i + 1]), silence=SILENT_GAP_S,
                           accept=["BACKCHANNEL", "LISTEN"])
                p["id"] += f".{i}"
                out.append(p)
    return out


V15 = {
    # subset -> (metadata key with the overlapping speech, gold while the agent is talking)
    "user_interruption": ("current_turn_text", "STOP"),
    "user_backchannel": ("backchannel_text", "IGNORE"),
    "talking_to_other": ("current_turn_text", "IGNORE"),
    "background_speech": ("background_text", "IGNORE"),
}


def v15_points(root: Path) -> list[dict]:
    """Each v1.5 clip: the user's request (agent silent -> SPEAK), then overlapping speech while the agent answers."""
    out = []
    for subset, (key, gold) in V15.items():
        for d in _samples(root / "v1.5" / subset):
            meta = json.loads((d / "metadata.json").read_text())
            request = meta["context_text"].strip()
            overlap = meta[key].strip()
            out.append(_point(f"v1.5_{subset}", d.name, "SPEAK", agent="silent", user=request,
                              silence=SILENT_GAP_S))
            out.append(_point(f"v1.5_{subset}", d.name, gold, agent="speaking", user=overlap,
                              silence=OVERLAP_GAP_S, history=request))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdb-root", required=True, help="directory containing v1.0/ and v1.5/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.fdb_root)

    points = (
        pause_points(root, "synthetic_pause_handling", end_of_turn=True)
        + pause_points(root, "candor_pause_handling", end_of_turn=False)
        + icc_points(root)
        + v15_points(root)
    )
    ids = Counter(p["id"] for p in points)
    dup = [k for k, n in ids.items() if n > 1]
    assert not dup, f"duplicate ids: {dup[:5]}"
    with open(args.out, "w") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(points)} decision points -> {args.out}")
    for (src, gold), n in sorted(Counter((p["source"], p["gold"]) for p in points).items()):
        print(f"  {src:32s} {gold:12s} {n}")


if __name__ == "__main__":
    main()
