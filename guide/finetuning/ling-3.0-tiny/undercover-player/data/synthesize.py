"""「谁是卧底」SFT 数据合成脚本。

程序化合成两类任务的训练数据：
- speak：轮到我发言时，生成 ≤25 字、不泄词的描述；
- vote：投票环节，根据发言一致性输出「我投N号」或「我弃权」。

用法：
    python3 synthesize.py --out-dir . --seed 42

输出：
    train.jsonl / test.jsonl（test 使用未见过的词对，考察泛化）
每行包含 messages（供 MLX/TRL 直接训练）与 meta（供评测与分析）。
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_pairs import WORD_PAIRS, TEST_PAIR_INDICES  # noqa: E402
from prompting import build_messages  # noqa: E402

SPEAK_TEMPLATES_1 = [
    "{a1}",
    "{a1}吧",
    "我觉得{a1}",
    "怎么说呢，{a1}",
    "很明显{a1}",
    "{a1}，没错",
    "最大的特点就是{a1}",
]

SPEAK_TEMPLATES_2 = [
    "{a1}，{a2}",
    "{a1}，还有{a2}",
    "{a1}，另外{a2}",
    "{a1}，而且{a2}",
    "一个是{a1}，还有{a2}",
]

VAGUE_TEMPLATES = [
    "和大家说的差不多",
    "我这个也是这样",
    "前面说的挺接近的",
    "总之就是常见的那种",
    "我的特点刚才有人提过",
    "都差不多，不重复了",
    "同意前面几位的说法",
]


def valid_clue(text, banned_chars):
    if not (1 <= len(text) <= 25):
        return False
    return not any(ch in text for ch in banned_chars)


def gen_clue(rng, attrs, banned_chars, used_phrases, vague_ok, used_attrs_round=None):
    """从属性池生成一条不泄词、不与他人重复的发言。

    used_attrs_round：本轮已用过的属性短语，避免同轮多人说同一属性。
    """
    used_attrs_round = used_attrs_round or set()
    candidates = [a for a in attrs
                  if not any(ch in a for ch in banned_chars) and a not in used_attrs_round]
    for _ in range(60):
        if vague_ok and rng.random() < 0.35:
            text = rng.choice(VAGUE_TEMPLATES)
        elif len(candidates) >= 2 and rng.random() < 0.45:
            a1, a2 = rng.sample(candidates, 2)
            if a1 == a2:
                continue
            text = rng.choice(SPEAK_TEMPLATES_2).format(a1=a1, a2=a2)
        elif candidates:
            a1 = rng.choice(candidates)
            text = rng.choice(SPEAK_TEMPLATES_1).format(a1=a1)
        else:
            text = rng.choice(VAGUE_TEMPLATES)
        if valid_clue(text, banned_chars) and text not in used_phrases:
            used_phrases.add(text)
            return text
    # 兜底：纯模糊发言
    for text in VAGUE_TEMPLATES:
        if text not in used_phrases and valid_clue(text, banned_chars):
            used_phrases.add(text)
            return text
    return "和大家差不多"


def make_game(rng, pair, n_players, my_seat, my_role, n_rounds):
    """生成一局游戏：返回各玩家每轮发言、属性使用记录、词语映射。"""
    roles = ["平民"] * n_players
    if my_role == "卧底":
        uc_seat = my_seat
    else:
        uc_seat = rng.choice([s for s in range(1, n_players + 1) if s != my_seat])
        roles[uc_seat - 1] = "卧底"

    words = {}
    attr_pools = {}
    for s in range(1, n_players + 1):
        if s == uc_seat:
            words[s] = pair["undercover"]
            attr_pools[s] = pair["shared"] * 2 + pair["undercover_attrs"] * 2
        else:
            words[s] = pair["civilian"]
            attr_pools[s] = pair["shared"] * 2 + pair["civilian_attrs"] * 2

    banned = {s: set(words[s]) for s in words}
    used_phrases = set()
    attrs_used = {s: [] for s in words}
    rounds = []
    all_attrs = set(pair["shared"] + pair["civilian_attrs"] + pair["undercover_attrs"])
    for r in range(1, n_rounds + 1):
        utts = []
        round_used = set()
        for s in range(1, n_players + 1):
            pool = attr_pools[s]
            text = gen_clue(rng, pool, banned[s], used_phrases,
                            vague_ok=(r >= 2), used_attrs_round=round_used)
            utts.append((s, text))
            for a in all_attrs:
                if a in text:
                    attrs_used[s].append(a)
                    round_used.add(a)
        rounds.append({"round": r, "utterances": utts})

    return {
        "roles": roles,           # 下标 0 对应 1 号
        "uc_seat": uc_seat,
        "words": words,
        "rounds": rounds,
        "attrs_used": attrs_used,
    }


def vote_label(rng, game, my_seat, my_role, pair):
    """规则式投票标签。

    平民：投票给发言中使用卧底词显著属性最多的玩家；无证据则弃权。
    卧底：把票甩给发言信息量最弱的平民（自保），小概率弃权。
    """
    uc_attrs = set(pair["undercover_attrs"])
    civ_attrs = set(pair["civilian_attrs"])
    suspicion = {
        s: sum(1 for a in game["attrs_used"][s] if a in uc_attrs)
        for s in game["attrs_used"]
    }
    if my_role == "平民":
        top = max(suspicion.values())
        if top == 0:
            return "我弃权", None
        seat = min(s for s, v in suspicion.items() if v == top)
        return f"我投{seat}号", seat

    # 卧底：甩锅给 civilian 属性用得最少的其他玩家
    if rng.random() < 0.2:
        return "我弃权", None
    strength = {
        s: sum(1 for a in game["attrs_used"][s] if a in civ_attrs)
        for s in game["attrs_used"]
        if s != my_seat
    }
    floor = min(strength.values())
    seat = min(s for s, v in strength.items() if v == floor)
    return f"我投{seat}号", seat


def build_split(rng, pair_indices, pairs, n_speak, n_vote):
    samples = []
    for idx in pair_indices:
        pair = pairs[idx]
        pair_key = f"{pair['civilian']}/{pair['undercover']}"
        for _ in range(n_speak):
            n_players = rng.randint(4, 6)
            my_seat = rng.randint(1, n_players)
            my_role = rng.choice(["平民", "卧底"])
            n_rounds = rng.randint(1, 6)  # 对齐实战「最多6轮」深度：≥4 轮会触发历史窗口折叠
            game = make_game(rng, pair, n_players, my_seat, my_role, n_rounds)

            r = n_rounds
            # 我的历史：前 r-1 轮 + 第 r 轮我之前的人（渲染由 prompting 窗口化折叠）
            history = [{"round": x["round"], "utterances": x["utterances"]}
                       for x in game["rounds"][: r - 1]]
            current = [(s, t) for s, t in game["rounds"][r - 1]["utterances"] if s < my_seat]
            history.append({"round": r, "utterances": current})

            target = dict(game["rounds"][r - 1]["utterances"])[my_seat]
            messages = build_messages("speak", my_role, game["words"][my_seat], history, target)
            samples.append({
                "id": f"speak-{pair_key}-{len(samples)}",
                "task": "speak",
                "pair_key": pair_key,
                "role": my_role,
                "word": game["words"][my_seat],
                "messages": messages,
                "meta": {"my_seat": my_seat, "n_players": n_players,
                         "uc_seat": game["uc_seat"], "target": target},
            })

        for _ in range(n_vote):
            n_players = rng.randint(4, 6)
            my_seat = rng.randint(1, n_players)
            my_role = rng.choice(["平民", "卧底"])
            n_rounds = rng.randint(1, 6)  # 对齐实战「最多6轮」深度：≥4 轮会触发历史窗口折叠
            game = make_game(rng, pair, n_players, my_seat, my_role, n_rounds)

            target, voted_seat = vote_label(rng, game, my_seat, my_role, pair)
            messages = build_messages("vote", my_role, game["words"][my_seat],
                                      game["rounds"], target)
            samples.append({
                "id": f"vote-{pair_key}-{len(samples)}",
                "task": "vote",
                "pair_key": pair_key,
                "role": my_role,
                "word": game["words"][my_seat],
                "messages": messages,
                "meta": {"my_seat": my_seat, "n_players": n_players,
                         "uc_seat": game["uc_seat"], "target": target,
                         "label_seat": voted_seat},
            })
    rng.shuffle(samples)
    return samples


def dump(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speak-per-pair", type=int, default=20)
    ap.add_argument("--vote-per-pair", type=int, default=12)
    ap.add_argument("--vote-repeat", type=int, default=3,
                    help="训练集中投票样本重复次数（任务平衡；valid/test 不受影响）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    test_idx = sorted(TEST_PAIR_INDICES)
    train_idx = [i for i in range(len(WORD_PAIRS)) if i not in TEST_PAIR_INDICES]

    train = build_split(rng, train_idx, WORD_PAIRS,
                        args.speak_per_pair, args.vote_per_pair)
    test = build_split(rng, test_idx, WORD_PAIRS,
                       max(2, args.speak_per_pair // 2),
                       max(2, args.vote_per_pair // 2))

    # 从训练集切 10% 做验证集（mlx_lm.lora 需要 train/valid/test 三个文件）
    rng.shuffle(train)
    n_valid = max(1, len(train) // 10)
    valid, train = train[:n_valid], train[n_valid:]

    # 投票任务决策难度高于发言，训练集过采样投票样本做任务平衡
    # （只动 train；valid/test 保持自然分布，保证评测口径可比）
    if args.vote_repeat > 1:
        train = [s for s in train
                 for _ in range(args.vote_repeat if s["task"] == "vote" else 1)]
        rng.shuffle(train)

    os.makedirs(args.out_dir, exist_ok=True)
    dump(os.path.join(args.out_dir, "train.jsonl"), train)
    dump(os.path.join(args.out_dir, "valid.jsonl"), valid)
    dump(os.path.join(args.out_dir, "test.jsonl"), test)

    for name, data in (("train", train), ("valid", valid), ("test", test)):
        n_speak = sum(1 for s in data if s["task"] == "speak")
        print(f"{name}: {len(data)} 条（speak {n_speak} / vote {len(data) - n_speak}）")
    print("输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
