"""「谁是卧底」玩家决策组件 — 微调前后自动评测脚本。

对 test.jsonl 逐条推理，计算：
- speak 任务：字数合规率（≤25 字）、泄词率（输出含自己词语任一字）、完全匹配率；
- vote 任务：格式合法率（「我投N号」或「我弃权」）、投票正确率（与规则式标签一致）。

用法（基线 vs 微调对比）：
    python evaluate.py --model ~/models/Ling-3.0-tiny-4bit --data ../data/test.jsonl --out base.json
    python evaluate.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path ../adapters \
        --data ../data/test.jsonl --out lora.json

依赖：与训练同一个 venv（mlx-lm）。
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from prompting import clean_output  # noqa: E402

VOTE_RE = re.compile(r"我投(\d+)号")


def first_line(text):
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def looks_natural(out):
    """发言应像一句正常的话，而非思考前缀（如「1. **分析请求：**」）。"""
    if re.match(r"^\s*\d+[\.、)]", out):        # 序号开头
        return False
    if "**" in out or "##" in out or "```" in out:  # markdown 标记
        return False
    if out.endswith(("：", ":")):                 # 未完成的标题/前缀
        return False
    return True


def eval_sample(sample, output):
    task = sample["task"]
    target = sample["messages"][1]["content"]
    out = first_line(clean_output(output))
    if task == "speak":
        word = sample["word"]
        len_ok = 1 <= len(out) <= 25
        leak = any(ch in out for ch in word)
        natural = looks_natural(out)
        return {
            "task": "speak", "output": out, "raw": output, "target": target,
            "len_ok": len_ok, "leak": leak, "natural": natural,
            "format_ok": len_ok and not leak and natural,
            "exact_match": out == target,
        }
    # vote
    m = VOTE_RE.search(out)
    if m:
        pred, fmt_ok = f"我投{m.group(1)}号", True
    elif "我弃权" in out:
        pred, fmt_ok = "我弃权", True
    else:
        pred, fmt_ok = out, False
    return {
        "task": "vote", "output": out, "raw": output, "target": target,
        "format_ok": fmt_ok, "correct": pred == target,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（4-bit 量化目录或 BF16 目录）")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter 目录；不传即评基线")
    ap.add_argument("--data", required=True, help="test.jsonl 路径")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    print(f"loaded: {args.model} (adapter={args.adapter_path})", file=sys.stderr)

    samples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    # 解码用固定种子的采样（temp=0.7, top_p=0.9）：贪心会让模型坍缩到单一
    # 安全输出（发言全用万能句、投票全弃权），无法反映决策质量。
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=0.7, top_p=0.9)

    records = []
    for i, s in enumerate(samples):
        prompt = tokenizer.apply_chat_template(
            s["messages"][:1], add_generation_prompt=True, tokenize=False,
            thinking_option="off",  # 与训练监督区对齐，见 data/prompting.py clean_output
        )
        out = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        records.append(eval_sample(s, out))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)}", file=sys.stderr)
        mx.clear_cache()

    speak = [r for r in records if r["task"] == "speak"]
    vote = [r for r in records if r["task"] == "vote"]
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "decoding": f"temp=0.7,top_p=0.9,seed={args.seed}",
        "n_speak": len(speak),
        "speak_len_ok": round(sum(r["len_ok"] for r in speak) / max(1, len(speak)), 4),
        "speak_leak_rate": round(sum(r["leak"] for r in speak) / max(1, len(speak)), 4),
        "speak_natural": round(sum(r["natural"] for r in speak) / max(1, len(speak)), 4),
        "speak_format_ok": round(sum(r["format_ok"] for r in speak) / max(1, len(speak)), 4),
        "speak_exact_match": round(sum(r["exact_match"] for r in speak) / max(1, len(speak)), 4),
        "n_vote": len(vote),
        "vote_format_ok": round(sum(r["format_ok"] for r in vote) / max(1, len(vote)), 4),
        "vote_accuracy": round(sum(r["correct"] for r in vote) / max(1, len(vote)), 4),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
