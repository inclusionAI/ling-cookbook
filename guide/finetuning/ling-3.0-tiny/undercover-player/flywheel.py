"""实战对局 → 训练数据 飞轮：从牌桌 log 生成带真值标签的新样本。

动机：合成训练集的投票标签是规则启发式打的（文本属性有歧义，噪声不可约，
评测证明该任务没吸收）；而**实战对局里卧底身份是上帝视角真值**——从对局
状态生成的投票样本无标签噪声。同时实战历史带着真实对手文本（含跑偏输出），
天然是历史编码的鲁棒性增强。

三条采样规则（v1）：
1. vote 样本 = 每个「存活的平民座位」× 每个已完成投票轮，标签 = 「我投<真实卧底>号」
   （DAgger 式专家修正：状态取自实战分布，动作取自上帝视角专家）。
   跳过：本人是卧底的座位、卧底已出局的轮、PK 轮、无效票轮之后的轮照常（真值不依赖票面）。
2. speak 样本 = 只模仿我们自己（--me 座位）的实际发言，且必须过四关：
   无违规标记、≤25 字、looks_natural、clean_output 后非空。
   己方（平民阵营）获胜的局全收；否则以 0.3 概率下采样（mlx_lm 不认样本权重，
   权重用伯努利下采样兑现）。
3. 留出纪律：词对命中 test.jsonl 的留出词对 → 整局跳过（防评测污染）。

产出物带 meta 溯源（session/round/weight/iter）；合成集与实战集分开存放，
用 merge 子命令按比例合并——每轮迭代的合并上限默认 30%，防自我模仿淹没原分布。

用法（一轮迭代）：
    python flywheel.py build --state-dir ~/.openclaw/undercover-game \
        --out data/live-v1.jsonl
    python flywheel.py merge --train data/train.jsonl --live data/live-v1.jsonl \
        --out data/train-v2.jsonl          # 然后重训 + 固定 test.jsonl 评测门槛
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "data"))
from prompting import build_messages, clean_output  # noqa: E402

MAX_SPEAK_CHARS = 25
LOSE_SPEAK_KEEP_P = 0.3


def looks_natural(text):
    """与评测同款的「非思考前缀垃圾」检查（序号/markdown/尾冒号均不自然）。"""
    if not text:
        return False
    if re.match(r"^\s*(\d+[.、]|[#*`\-])", text):
        return False
    if text.rstrip().endswith(("：", ":")):
        return False
    return True


def load_holdout(test_path):
    keys = set()
    for line in open(test_path):
        keys.add(json.loads(line)["pair_key"])
    return keys


def iter_games(state_dir):
    for p in sorted(Path(state_dir).glob("*.json")):
        try:
            g = json.load(open(p))
        except Exception:
            continue
        if g.get("words") and g.get("seats") and g.get("rounds"):
            yield g


def game_to_samples(game, me, holdout, rng, iter_tag):
    civilian = game["words"]["civilian"]
    undercover = game["words"]["undercover"]
    pair_key = f"{civilian}/{undercover}"
    if pair_key in holdout:
        return []

    seats = {s["display"]: s for s in game["seats"]}
    uc_seat = next((s["seat"] for s in game["seats"] if s["word"] == undercover), None)
    my = seats.get(me)
    if uc_seat is None or my is None:
        return []

    win_side = (game.get("result") or {}).get("winner") or "?"
    sid_short = game["session_id"].split(":")[-1][:8]
    samples = []

    for r in game["rounds"]:
        speeches = r.get("speeches") or {}
        votes = r.get("votes") or {}
        if r.get("pk"):
            continue
        order = r.get("order") or sorted(int(k) for k in speeches)
        # 历史按「说话时刻」裁剪：
        #  vote 时刻 = 本轮发言全部结束之后 -> 含本轮全部发言；
        #  speak 时刻 = 轮到自己之前       -> 只含本轮排在自己前面的发言（自己的话是标签，绝不能进历史）。
        def history_through(up_to_round, precede_seat=None):
            hist = []
            for rr in game["rounds"]:
                if not (rr.get("speeches") or {}) or rr["round"] > up_to_round:
                    continue
                utts = []
                for k, v in sorted((rr["speeches"]).items(), key=lambda kv: int(kv[0])):
                    seat_i = int(k)
                    if rr["round"] == up_to_round and precede_seat is not None:
                        if seat_i == precede_seat:
                            continue
                        if precede_seat in order and seat_i in order and \
                                order.index(seat_i) > order.index(precede_seat):
                            continue
                    utts.append((seat_i, v["raw"] if isinstance(v, dict) else v))
                if utts:
                    hist.append({"round": rr["round"], "utterances": utts})
            return hist

        # --- vote：全存活平民座位 × 本轮（真值标签） ---
        if votes:
            for s in game["seats"]:
                if s["word"] == undercover:
                    continue  # 卧底视角的投票目标无真值
                if s.get("eliminated_round") is not None and s["eliminated_round"] < r["round"]:
                    continue
                msgs = build_messages("vote", "平民", s["word"],
                                      history_through(r["round"]),
                                      f"我投{uc_seat}号")
                samples.append({
                    "id": f"live-vote-{pair_key}-{sid_short}-{r['round']}-s{s['seat']}",
                    "task": "vote", "pair_key": pair_key, "role": "平民",
                    "word": s["word"], "messages": msgs,
                    "meta": {"source": "live", "session": game["session_id"],
                             "round": r["round"], "seat": s["seat"], "weight": 1.0,
                             "label": "ground-truth-undercover", "iter": iter_tag},
                })

        # --- speak：只模仿我们自己座位的干净发言 ---
        mine = speeches.get(str(my["seat"]))
        if isinstance(mine, dict) and mine.get("raw") and not mine.get("violation"):
            text = clean_output(mine["raw"])
            # 贪心坍缩守卫：同轮有别的座位说过一模一样的话 -> 是退化输出不是好样本
            #（实测一桌 5 人同说「蒸着吃吧」），丢弃（对齐合成数据的近重复守卫）。
            others = {clean_output(v["raw"]) for k, v in speeches.items()
                      if int(k) != my["seat"] and isinstance(v, dict) and v.get("raw")}
            if text and len(text) <= MAX_SPEAK_CHARS and looks_natural(text) \
                    and not any(ch in text for ch in my["word"]) and text not in others:
                my_win = (win_side == "civilian" and my["word"] == civilian) or \
                         (win_side == "undercover" and my["word"] == undercover)
                if my_win or rng.random() < LOSE_SPEAK_KEEP_P:
                    role = "平民" if my["word"] == civilian else "卧底"
                    msgs = build_messages("speak", role, my["word"],
                                          history_through(r["round"], precede_seat=my["seat"]),
                                          text)
                    samples.append({
                        "id": f"live-speak-{pair_key}-{sid_short}-{r['round']}",
                        "task": "speak", "pair_key": pair_key, "role": role,
                        "word": my["word"], "messages": msgs,
                        "meta": {"source": "live", "session": game["session_id"],
                                 "round": r["round"], "seat": my["seat"],
                                 "weight": 1.0 if my_win else LOSE_SPEAK_KEEP_P,
                                 "label": "self-imitation", "iter": iter_tag},
                    })
    return samples


def cmd_build(args):
    rng = random.Random(args.seed)
    holdout = load_holdout(args.test)
    out = []
    for g in iter_games(args.state_dir):
        out.extend(game_to_samples(g, args.me, holdout, rng, args.iter))
    # 精确去重（整条样本：task+word+user 历史+assistant）+ 每局上限。
    # 注意键必须含 user 历史：2026-11-13 踩坑——只按 (task, word, assistant)
    # 去重会把「不同历史、同标签」的 DAgger 样本折叠掉（正是价值所在），
    # 9 局实战只剩 15 条。历史是多样状态，不是重复。
    seen, kept, per_game = set(), [], {}
    for s in out:
        key = (s["task"], s["word"], s["messages"][0]["content"],
               s["messages"][1]["content"])
        sid = s["meta"]["session"]
        if key in seen or per_game.get(sid, 0) >= args.per_game_cap:
            continue
        seen.add(key)
        per_game[sid] = per_game.get(sid, 0) + 1
        kept.append(s)
    with open(args.out, "w") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    nv = sum(1 for s in kept if s["task"] == "vote")
    print(f"games={len(list(iter_games(args.state_dir)))} kept={len(kept)} "
          f"(vote={nv}, speak={len(kept) - nv}) -> {args.out}")
    print("注意：先跑评测门槛再合并——见 README「数据飞轮」一节。")


def cmd_merge(args):
    rng = random.Random(args.seed)
    train = [json.loads(l) for l in open(args.train)]
    live = []
    for path in args.live:
        live.extend(json.loads(l) for l in open(path))
    cap = int(len(train) * args.ratio)
    if len(live) > cap:
        live = rng.sample(live, cap)
    merged = train + live
    with open(args.out, "w") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"train={len(train)} + live={len(live)} (cap={cap}, ratio={args.ratio}) "
          f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="从对局状态生成实战样本")
    b.add_argument("--state-dir", default=str(Path.home() / ".openclaw" / "undercover-game"))
    b.add_argument("--test", default=str(HERE / "data" / "test.jsonl"))
    b.add_argument("--out", default=str(HERE / "data" / "live-v1.jsonl"))
    b.add_argument("--me", default="MLX小林", help="我们玩家的 display 名")
    b.add_argument("--iter", default="v1")
    b.add_argument("--per-game-cap", type=int, default=24)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=cmd_build)
    m = sub.add_parser("merge", help="按比例合并实战集进训练集")
    m.add_argument("--train", default=str(HERE / "data" / "train.jsonl"))
    m.add_argument("--live", required=True, nargs="+",
                   help="增量样本 JSONL，可多个（实战飞轮 + 教师蒸馏）")
    m.add_argument("--out", required=True)
    m.add_argument("--ratio", type=float, default=0.3,
                   help="实战样本占原训练集的上限比例（防自我模仿淹没）")
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_merge)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
