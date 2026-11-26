"""「谁是卧底」玩家提示词模板。

数据合成、评测脚本与推理样例共用此格式；
Avernet 玩家 Bot（inclusionAI/Avernet#2611）的适配层也应复用同一格式，
以保证微调模型收到的输入与训练分布一致。

历史渲染是窗口化的（见 render_history）：第 1 轮 + 最近 2 轮，
中间折叠——prompt 有界（≤~1k 字符），避免随轮次线性膨胀。
"""

RULES_SPEAK = (
    "规则：\n"
    "1. 发言不超过25个字\n"
    "2. 不要说出你词语里的任何一个字\n"
    "3. 描述要有信息量，但别让人一眼看穿你的词"
)

RULES_VOTE = (
    "规则：\n"
    "1. 根据大家的发言找出可疑的玩家\n"
    "2. 只输出「我投N号」或「我弃权」，不要输出其他内容"
)


HISTORY_WINDOW = 2  # 历史窗口：保留第 1 轮 + 最近 2 轮，中间折叠（推理 prompt 有界）


def render_player_state(own_seat=None, alive_seats=None):
    """旧调用不增加字段；新协议要求同时给出本人座位和完整存活名单。"""
    if own_seat is None and alive_seats is None:
        return ""
    if type(own_seat) is not int or own_seat < 1 or alive_seats is None:
        raise ValueError("own_seat 和 alive_seats 必须同时提供")
    seats = list(alive_seats)
    if not seats or any(type(s) is not int or s < 1 for s in seats):
        raise ValueError("存活座位必须为非空的正整数列表")
    if len(set(seats)) != len(seats) or own_seat not in seats:
        raise ValueError("本人必须存活，座位不能重复")
    return (f"你的座位：{own_seat}号\n"
            f"存活玩家：{'、'.join(f'{s}号' for s in sorted(seats))}\n"
            "投票只能选择其他存活玩家，也可以弃权。\n")


def render_history(rounds, window=HISTORY_WINDOW):
    """rounds: [{"round": 1, "utterances": [(seat, text), ...]}, ...]

    窗口化序列化：保留第 1 轮（初始描述信息量最大）与最近 window 轮，
    中间轮折叠成一行「（第x-y轮略）」；window=None 时输出全量历史。
    上限 ~3 轮历史，6 人局打满 prompt 也能压在 1k 字符内——排队的是
    OpenClaw 框架的 2 万字提示词，我们这侧只求有界与训练/推理分布一致。
    """
    if not rounds:
        return "（暂无）"
    if window is not None and len(rounds) > window + 1:
        kept = [rounds[0]] + rounds[-window:]
    else:
        kept = list(rounds)
    lines, prev = [], None
    for r in kept:
        if prev is not None and r["round"] > prev + 1:
            lines.append(f"（第{prev + 1}-{r['round'] - 1}轮略）")
        lines.append(f"第{r['round']}轮：")
        for seat, text in r["utterances"]:
            lines.append(f"{seat}号：{text}")
        prev = r["round"]
    return "\n".join(lines)


def build_speak_prompt(role, word, rounds, window=HISTORY_WINDOW, *, own_seat=None, alive_seats=None):
    history = render_history(rounds, window)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{render_player_state(own_seat, alive_seats)}"
        f"{RULES_SPEAK}\n"
        f"历史发言：\n{history}\n"
        "现在轮到你发言。\n"
        "请只输出你的发言内容（不超过25个字）："
    )


def build_vote_prompt(role, word, rounds, window=HISTORY_WINDOW, *, own_seat=None, alive_seats=None):
    history = render_history(rounds, window)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{render_player_state(own_seat, alive_seats)}"
        f"{RULES_VOTE}\n"
        f"历史发言：\n{history}\n"
        "现在开始投票。\n"
        "请只输出「我投N号」或「我弃权」："
    )


def build_messages(task, role, word, rounds, target, window=HISTORY_WINDOW, *, own_seat=None, alive_seats=None):
    """构造一条 SFT 样本的 messages 字段。"""
    if task == "speak":
        prompt = build_speak_prompt(role, word, rounds, window, own_seat=own_seat, alive_seats=alive_seats)
    elif task == "vote":
        prompt = build_vote_prompt(role, word, rounds, window, own_seat=own_seat, alive_seats=alive_seats)
    else:
        raise ValueError(f"未知任务：{task}")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]


def clean_output(text, require_think_end=False):
    """提取思考结束标记后的正文，并移除回合结束标记。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif require_think_end or "<think>" in text:
        return ""  # 思考尚未结束，不把推理文本当作答案
    text = text.split("<|role_end|>")[0]
    return text.strip()
