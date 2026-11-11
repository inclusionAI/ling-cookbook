"""「谁是卧底」玩家提示词模板。

数据合成、评测脚本与推理样例共用此格式；
Avernet 玩家 Bot（inclusionAI/Avernet#2611）的适配层也应复用同一格式，
以保证微调模型收到的输入与训练分布一致。
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


def render_history(rounds):
    """rounds: [{"round": 1, "utterances": [(seat, text), ...]}, ...]"""
    lines = []
    for r in rounds:
        lines.append(f"第{r['round']}轮：")
        for seat, text in r["utterances"]:
            lines.append(f"{seat}号：{text}")
    return "\n".join(lines) if lines else "（暂无）"


def build_speak_prompt(role, word, rounds):
    history = render_history(rounds)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{RULES_SPEAK}\n"
        f"历史发言：\n{history}\n"
        "现在轮到你发言。\n"
        "请只输出你的发言内容（不超过25个字）："
    )


def build_vote_prompt(role, word, rounds):
    history = render_history(rounds)
    return (
        "你在玩「谁是卧底」游戏。\n"
        f"你的身份：{role}\n"
        f"你的词语：{word}\n"
        f"{RULES_VOTE}\n"
        f"历史发言：\n{history}\n"
        "现在开始投票。\n"
        "请只输出「我投N号」或「我弃权」："
    )


def build_messages(task, role, word, rounds, target):
    """构造一条 SFT 样本的 messages 字段。"""
    if task == "speak":
        prompt = build_speak_prompt(role, word, rounds)
    else:
        prompt = build_vote_prompt(role, word, rounds)
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
