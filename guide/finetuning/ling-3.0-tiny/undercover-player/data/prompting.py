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


def build_speak_prompt(role, word, rounds, window=HISTORY_WINDOW):
    history = render_history(rounds, window)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{RULES_SPEAK}\n"
        f"历史发言：\n{history}\n"
        "现在轮到你发言。\n"
        "请只输出你的发言内容（不超过25个字）："
    )


def build_vote_prompt(role, word, rounds, window=HISTORY_WINDOW):
    history = render_history(rounds, window)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{RULES_VOTE}\n"
        f"历史发言：\n{history}\n"
        "现在开始投票。\n"
        "请只输出「我投N号」或「我弃权」："
    )


def build_messages(task, role, word, rounds, target, window=HISTORY_WINDOW):
    """构造一条 SFT 样本的 messages 字段。"""
    if task == "speak":
        prompt = build_speak_prompt(role, word, rounds, window)
    else:
        prompt = build_vote_prompt(role, word, rounds, window)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]


def clean_output(text):
    """剥掉 chat 模板结构残渣，只留模型内容。

    背景（见 README Troubleshooting）：Ling 的 chat 模板带 thinking 块，
    mlx_lm 的 mask_prompt 偏移会把闭合标签留在监督区，模型可能先输出
    闭合标签或尾随 <|role_end|>。这些是模板结构，不是模型说的话。
    """
    text = text.split("<|role_end|>")[0]
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()
