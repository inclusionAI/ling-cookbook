"""Download the three public benchmarks into a bench directory and check them against our copies.

    python fetch_benches.py --out bench

Sources (none of them is redistributed here):
  ASSEBench     github.com/Astarojth/AgentAuditor-ASSEBench    Apache-2.0
  Pre-Ex-Bench  github.com/HowieHwong/Agentic-Guardian         CC BY-NC 4.0
  R-Judge       github.com/Lordog/R-Judge                      no license file: evaluation only

`bench_sha256.json` holds the hashes of the files every reported number was computed on.
A mismatch means upstream changed the file since then; the run continues but says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

ASSE = "https://raw.githubusercontent.com/Astarojth/AgentAuditor-ASSEBench/main/ASSEBench/dataset/AgentJudge-{}.json"
PREEX = "https://raw.githubusercontent.com/HowieHwong/Agentic-Guardian/main/Pre-Ex-Bench/dataset.json"
RJUDGE = "https://raw.githubusercontent.com/Lordog/R-Judge/main/data/{}/{}.json"
RJUDGE_FILES = {
    "Application": ["chatbot", "dh_app", "ds_app", "mail", "medical", "phone", "productivity", "socialapp"],
    "Finance": ["bitcoin", "dh_finance", "ds_finance", "moneymanagement", "webshop"],
    "IoT": ["household", "phone_iot", "trafficdispatch"],
    "Program": ["code_agentmonitor", "dh_program", "ds_program", "phone_program", "security", "software", "terminal"],
    "Web": ["dh_web", "ds_web", "webbrowser", "websearch"],
}


def targets() -> dict[str, str]:
    files = {f"AgentJudge-{name}.json": ASSE.format(name) for name in ("security", "safety", "strict", "loose")}
    files["preex/dataset.json"] = PREEX
    for category, names in RJUDGE_FILES.items():
        for name in names:
            files[f"rjudge/{category}__{name}.json"] = RJUDGE.format(category, name)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench")
    parser.add_argument("--proxy", default="", help="for example http://127.0.0.1:7890")
    args = parser.parse_args()
    handlers = [urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy} if args.proxy else {})]
    opener = urllib.request.build_opener(*handlers)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "bench_sha256.json"), encoding="utf-8") as handle:
        expected = json.load(handle)
    changed = 0
    for relative, url in targets().items():
        path = os.path.join(args.out, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with opener.open(url, timeout=120) as reply, open(path, "wb") as sink:
                sink.write(reply.read())
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected.get(relative):
            changed += 1
            print(f"changed upstream: {relative}")
    print(f"{len(targets())} files in {args.out}, {changed} differ from the files the reported numbers used")


if __name__ == "__main__":
    main()
