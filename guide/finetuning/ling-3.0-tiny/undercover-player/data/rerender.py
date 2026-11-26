"""格式迁移：把旧（全量历史）渲染的增量样本重渲染成当前窗口格式。

历史渲染从「全量」改为「第 1 轮 + 最近 2 轮」（见 prompting.render_history）
后，已采集的增量样本（flywheel 实战 / distill 蒸馏）里存的是旧格式 prompt。
重调 API 重新标注既贵又引入噪声；而**动作标签（assistant）不依赖历史渲染方式**，
所以把旧 prompt 解析回结构化 rounds，再用当前格式重渲染 user 部分即可。

安全性靠无损往返校验：对每条样本，先用 window=None（全量模式）重渲染解析出的
rounds，必须与原 prompt **逐字节相等**才允许写出——解析有任何歧义直接报错。

用法（就地重写，git 里留有旧版本）：
    python data/rerender.py data/live-v1.jsonl data/api-distill-v1.jsonl
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompting import build_messages  # noqa: E402

RE_ROUND = re.compile(r"^第(\d+)轮：$")
RE_UTT = re.compile(r"^(\d+)号：(.*)$", re.S)


def parse_prompt(text):
    """旧格式 prompt → (task, role, word, rounds)。只认自己写出的格式。"""
    m = re.search(r"你的身份：(\S+)", text)
    role = m.group(1) if m else ""
    m = re.search(r"你的词语：(\S+)", text)
    word = m.group(1) if m else ""
    task = "speak" if "现在轮到你发言" in text else "vote"
    hist = text.split("历史发言：\n", 1)[1].split("\n现在", 1)[0]
    rounds = []
    for line in hist.splitlines():
        if line == "（暂无）":
            continue
        m = RE_ROUND.match(line)
        if m:
            rounds.append({"round": int(m.group(1)), "utterances": []})
            continue
        m = RE_UTT.match(line)
        if m and rounds:
            rounds[-1]["utterances"].append((int(m.group(1)), m.group(2)))
            continue
        raise ValueError(f"无法解析的历史行：{line!r}")
    return task, role, word, rounds


def rerender_record(rec):
    user = rec["messages"][0]["content"]
    label = rec["messages"][1]["content"]
    task, role, word, rounds = parse_prompt(user)
    # 无损往返校验：全量模式必须逐字节还原旧 prompt
    check = build_messages(task, role, word, rounds, label, window=None)[0]["content"]
    if check != user:
        raise ValueError(f"往返校验失败（id={rec['id']}），解析有歧义，拒绝重渲染")
    msgs = build_messages(task, role, word, rounds, label)
    out = dict(rec)
    out["messages"] = msgs
    return out, len(msgs[0]["content"])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        p = Path(path)
        rows = [json.loads(l) for l in p.open()]
        # 先全量校验+渲染进内存，任何一条解析歧义都不落盘（避免写一半）
        results = [rerender_record(rec) for rec in rows]
        with p.open("w") as f:
            for out, _ in results:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        sizes = [n for _, n in results]
        folded = sum(1 for out, _ in results if "轮略）" in out["messages"][0]["content"])
        print(f"{p.name}: {len(rows)} 条重渲染（{folded} 条触发窗口折叠），"
              f"新 prompt {min(sizes)}~{max(sizes)} 字符")


if __name__ == "__main__":
    main()
