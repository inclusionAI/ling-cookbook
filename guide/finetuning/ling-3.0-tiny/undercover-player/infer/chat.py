"""「谁是卧底」玩家推理样例。

加载 4-bit 量化模型（可选挂 LoRA adapter），与微调模型进行单轮对话。

用法：
    # 基线（不挂 adapter）
    python chat.py --model ~/models/Ling-3.0-tiny-4bit

    # 微调后
    python chat.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path ../adapters

    # 新协议：--prompt-protocol state-v2，JSON 同时提供 own_seat 和 alive_seats
    # 交互模式
    python chat.py --model ~/models/Ling-3.0-tiny-4bit --interactive
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from prompting import build_speak_prompt, build_vote_prompt, clean_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--prompt-protocol", choices=["legacy", "state-v2"], default="legacy")
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    print(f"loaded: {args.model} (adapter={args.adapter_path})")

    def chat_once(task, role, word, rounds, own_seat=None, alive_seats=None):
        state = {}
        if args.prompt_protocol == "state-v2":
            if own_seat is None or alive_seats is None:
                raise ValueError("state-v2 requires own_seat and alive_seats")
            state = {"own_seat": own_seat, "alive_seats": alive_seats}
        prompt = (build_speak_prompt if task == "speak" else build_vote_prompt)(
            role, word, rounds, **state
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=False, thinking_option="on",  # 与默认训练模板一致
        )
        out = generate(model, tokenizer, prompt=text,
                       max_tokens=args.max_tokens,
                       sampler=make_sampler(temp=0.7, top_p=0.9), verbose=False)
        return clean_output(out, require_think_end=True)

    # 内置演示样例：第 2 轮发言 + 投票
    demo_rounds = [
        {"round": 1, "utterances": [
            (1, "甜的，我很喜欢"),
            (2, "超市里一年四季都有"),
            (3, "可以榨汁喝"),
        ]},
    ]
    print("\n=== speak 任务演示（身份=平民，词语=苹果）===")
    print("输出:", chat_once("speak", "平民", "苹果", demo_rounds, 4, [1, 2, 3, 4]))

    demo_rounds_vote = [
        {"round": 1, "utterances": [
            (1, "甜的，我很喜欢"),
            (2, "超市里一年四季都有"),
            (3, "黄色居多，皮很薄"),
            (4, "可以榨汁喝"),
        ]},
    ]
    print("\n=== vote 任务演示（身份=平民，词语=苹果）===")
    print("输出:", chat_once("vote", "平民", "苹果", demo_rounds_vote, 4, [1, 2, 3, 4]))

    if args.interactive:
        print("\n=== 交互模式（输入 JSON: {\"task\": \"speak|vote\", \"role\": ..., \"word\": ..., \"rounds\": [...]}）===")
        import json
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            try:
                q = json.loads(line)
                print("输出:", chat_once(q["task"], q["role"], q["word"], q["rounds"], q.get("own_seat"), q.get("alive_seats")))
            except Exception as e:
                print("解析失败:", e)


if __name__ == "__main__":
    main()
