"""自对弈胜率评测：6 席全模型 + 脚本裁判（「谁是卧底」标准规则）。

补上 issue #11 评测口径的第 4 项指标：**自对弈胜率**。整局游戏由模型自己
发言/投票，脚本只做裁判（发牌、泄词/超字数判定、计票、出局、胜负），
与 Avernet 侧 `undercover.py` 的规则语义对齐但独立实现（不依赖 Avernet）。

指标（--policy model 时）：
- civilian_win_rate / undercover_win_rate：平民/卧底胜率；
- vote_parse_rate / vote_self_rate：票面可解析率 / 自票率（自票按废票计）；
- speech_len_ok / speech_leak_rate / speech_natural：发言合规（同 evaluate.py 口径）。

随机基线（--policy random）：投票均匀随机、发言用固定中性句，给出
"纯运气" 的平民胜率参照——胜率要跟它比才有意义。

用法：
    python selfplay.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path ../../adapters-v2-run3 \
        --games 20 --seed 2026 --out selfplay-v2.json
    python selfplay.py --model ~/models/Ling-3.0-tiny-4bit --policy random \
        --games 200 --seed 2026 --out selfplay-random.json

依赖：与训练同一个 venv（mlx-lm）。
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from prompting import build_speak_prompt, build_vote_prompt, clean_output  # noqa: E402
from word_pairs import TEST_PAIR_INDICES, WORD_PAIRS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import first_line, looks_natural  # noqa: E402

VOTE_RE = re.compile(r"我投(\d+)号")
MAX_ROUNDS = 6          # 对齐实战牌桌配置（6 人局最多 6 轮）
N_PLAYERS = 6
CANNED_SPEECH = "我觉得大家说的都有道理，再观察一下"  # 随机基线的固定发言


def parse_vote(text, own_seat, alive=None):
    """返回 (target, ok)：target=座位号或 None（弃权）；ok=票面是否规范。"""
    m = VOTE_RE.fullmatch(text.strip())
    if m:
        t = int(m.group(1))
        if t == own_seat or not (1 <= t <= N_PLAYERS) or (alive is not None and t not in alive):
            return None, False      # 自票/越界票 = 废票
        return t, True
    if text.strip() == "我弃权":
        return None, True
    return None, False


def run_game(model_ctx, policy, game_seed, pair, prompt_protocol="legacy"):
    """打一整局，返回对局记录。policy: 'model' | 'random'。"""
    if prompt_protocol not in {"legacy", "state-v2"}:
        raise ValueError("Unknown prompt protocol")
    rng = random.Random(game_seed)
    seats = list(range(1, N_PLAYERS + 1))
    undercover_seat = rng.choice(seats)
    roles = {s: ("卧底" if s == undercover_seat else "平民") for s in seats}
    words = {s: (pair["undercover"] if s == undercover_seat else pair["civilian"]) for s in seats}

    completed_rounds = []
    alive = set(seats)
    actions = []          # 全部动作的合规统计
    winner = None
    rounds_played = 0

    for r in range(1, MAX_ROUNDS + 1):
        rounds_played = r
        cur = {"round": r, "utterances": []}
        # 发言轮：按座位顺序，后发言者能看到本轮之前的发言
        for s in sorted(alive):
            if policy == "random":
                out = CANNED_SPEECH
            else:
                state = {"own_seat": s, "alive_seats": sorted(alive)} if prompt_protocol == "state-v2" else {}
                prompt = build_speak_prompt(roles[s], words[s], completed_rounds + [cur], **state)
                out = model_ctx.generate(prompt)
            out = clean_output(out)
            leak = any(ch in out for ch in words[s])
            len_ok = 1 <= len(out) <= 25 and len(out.splitlines()) == 1
            natural = looks_natural(out)
            actions.append({"phase": "speak", "seat": s, "output": out,
                            "len_ok": len_ok, "leak": leak, "natural": natural,
                            "format_ok": len_ok and not leak and natural})
            cur["utterances"].append((s, out))

        # 投票轮：全员对本轮完整历史投票
        votes = {}
        for s in sorted(alive):
            if policy == "random":
                choices = [x for x in sorted(alive) if x != s]
                target, ok = (rng.choice(choices), True) if choices else (None, True)
                out = f"我投{target}号" if target else "我弃权"
            else:
                state = {"own_seat": s, "alive_seats": sorted(alive)} if prompt_protocol == "state-v2" else {}
                prompt = build_vote_prompt(roles[s], words[s], completed_rounds + [cur], **state)
                out = clean_output(model_ctx.generate(prompt))
                target, ok = parse_vote(out, s, alive)
            actions.append({"phase": "vote", "seat": s, "output": out,
                            "format_ok": ok, "self_or_bad": not ok})
            votes[s] = target

        completed_rounds.append(cur)
        counts = {}
        for t in votes.values():
            if t is not None:
                counts[t] = counts.get(t, 0) + 1
        eliminated = None
        if counts:
            top = max(counts.values())
            leaders = [s for s, c in counts.items() if c == top]
            if len(leaders) == 1:
                eliminated = leaders[0]
                alive.discard(eliminated)

        if eliminated == undercover_seat:
            winner = "civilian"
            break
        if len(alive) <= 2:
            winner = "undercover"
            break
    else:
        winner = "undercover"   # 打满 6 轮没抓到卧底

    return {
        "game_seed": game_seed, "pair": pair, "undercover_seat": undercover_seat,
        "winner": winner, "rounds_played": rounds_played,
        "n_actions": len(actions), "actions": actions,
    }


class ModelCtx:
    def __init__(self, model_path, adapter_path, seed, max_tokens=256):
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler

        self.mx = mx
        self.generate_fn = generate
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.sampler = make_sampler(temp=0.7, top_p=0.9)
        self.max_tokens = max_tokens
        self.seed = seed
        self._step = 0

    def generate(self, prompt):
        # 每个动作用派生种子，保证逐局可复现
        self.mx.random.seed(self.seed * 100003 + self._step)
        self._step += 1
        text = self.generate_fn(
            self.model, self.tokenizer,
            prompt=self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False, thinking_option="on",
            ),
            max_tokens=self.max_tokens, sampler=self.sampler, verbose=False,
        )
        self.mx.clear_cache()
        return clean_output(text, require_think_end=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--policy", choices=["model", "random"], default="model")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-protocol", choices=["legacy", "state-v2"], default="legacy")
    ap.add_argument("--pairs", choices=["heldout", "all"], default="heldout",
                    help="heldout=只用 6 组留出词对（测泛化）；all=30 组全用")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.games < 1:
        ap.error("--games must be positive")

    if args.pairs == "heldout":
        pool = [WORD_PAIRS[i] for i in TEST_PAIR_INDICES]
    else:
        pool = list(WORD_PAIRS)

    model_ctx = None
    if args.policy == "model":
        model_ctx = ModelCtx(args.model, args.adapter_path, args.seed, args.max_tokens)
        print(f"loaded: {args.model} (adapter={args.adapter_path})", file=sys.stderr)

    games = []
    for g in range(args.games):
        pair = pool[g % len(pool)]
        rec = run_game(model_ctx, args.policy, args.seed + g, pair, args.prompt_protocol)
        games.append(rec)
        print(f"  game {g + 1}/{args.games}: {rec['winner']} win, "
              f"{rec['rounds_played']} rounds", file=sys.stderr)

    speaks = [a for rec in games for a in rec["actions"] if a["phase"] == "speak"]
    votes = [a for rec in games for a in rec["actions"] if a["phase"] == "vote"]
    n_civ = sum(1 for rec in games if rec["winner"] == "civilian")
    summary = {
        "protocol": "ling-thinking-on-strict-v1",
        "prompt_protocol": args.prompt_protocol,
        "thinking_option": "on" if args.policy == "model" else None,
        "max_tokens": args.max_tokens if args.policy == "model" else None,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "policy": args.policy,
        "decoding": "temp=0.7,top_p=0.9" if args.policy == "model" else "random",
        "seed": args.seed,
        "n_games": len(games),
        "civilian_win_rate": round(n_civ / len(games), 4),
        "undercover_win_rate": round(1 - n_civ / len(games), 4),
        "avg_rounds": round(sum(r["rounds_played"] for r in games) / len(games), 2),
        "n_speak": len(speaks),
        "speech_len_ok": round(sum(a["len_ok"] for a in speaks) / max(1, len(speaks)), 4),
        "speech_leak_rate": round(sum(a["leak"] for a in speaks) / max(1, len(speaks)), 4),
        "speech_natural": round(sum(a["natural"] for a in speaks) / max(1, len(speaks)), 4),
        "n_vote": len(votes),
        "vote_parse_rate": round(sum(a["format_ok"] for a in votes) / max(1, len(votes)), 4),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "games": games}, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
