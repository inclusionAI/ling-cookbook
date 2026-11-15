#!/usr/bin/env python3
"""
eval/evaluate.py
================
中文 NL2SQL 评测脚本：在留出测试集上计算三种指标，对比微调前后。

指标:
  1. EX (Execution Accuracy, 执行准确率)  —— 在内存 SQLite 中执行 gold 与预测 SQL, 比对结果集;
  2. EM (Exact Match, 精确匹配)           —— 标准化后的 SQL 字符串是否一致;
  3. Legal Rate (SQL 合法率)              —— 预测 SQL 可被解析/执行的占比。

用法:
    # 评测未微调基座 (baseline)
    python3 eval/evaluate.py --messages-file ../data/test.jsonl --database ../data/database.sql

    # 评测微调后 (加载 LoRA 适配器)
    python3 eval/evaluate.py --messages-file ../data/test.jsonl --database ../data/database.sql \
        --adapter ../train/output

    # 仅校验数据/评测逻辑 (不加载模型, 用 gold 自测, 应得到 100%)
    python3 eval/evaluate.py --messages-file ../data/test.jsonl --database ../data/database.sql --gold-only

    # 快速冒烟 (只测前 N 条)
    python3 eval/evaluate.py ... --max-samples 20
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "data").resolve()


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = sql.rstrip(";")
    sql = re.sub(r"\s+", " ", sql).lower()
    return sql


def norm_value(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def norm_rows(rows):
    return sorted(tuple(norm_value(x) for x in row) for row in rows)


def execute(cur: sqlite3.Cursor, sql: str):
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        return True, norm_rows(rows)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def manual_format(messages) -> str:
    parts = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    return "".join(parts)


def load_generator(args):
    if args.gold_only:
        return lambda messages: messages[-1]["content"]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    use_chat_template = bool(getattr(tokenizer, "chat_template", None))

    def generate(messages):
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = manual_format(messages) + "<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
        sql = sql.split(";")[0].strip() + ";"
        return sql

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages-file", type=str, default=str(DATA_DIR / "test.jsonl"))
    ap.add_argument("--database", type=str, default=str(DATA_DIR / "database.sql"))
    ap.add_argument("--base-model", type=str, default="inclusionAI/Ling-3.0-tiny")
    ap.add_argument("--adapter", type=str, default=None, help="LoRA 适配器目录 (可选)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--gold-only", action="store_true", help="不加载模型, 用 gold 自测校验评测逻辑")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    messages_file = Path(args.messages_file)
    database_file = Path(args.database)
    if not messages_file.is_absolute():
        messages_file = (BASE_DIR.parent / messages_file).resolve()
    if not database_file.is_absolute():
        database_file = (BASE_DIR.parent / database_file).resolve()

    samples = [json.loads(line) for line in messages_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples:
        samples = samples[: args.max_samples]

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(database_file.read_text(encoding="utf-8"))

    generate = load_generator(args)

    n = len(samples)
    em_hit = 0
    ex_hit = 0
    legal = 0

    for i, s in enumerate(samples):
        messages = s["messages"]
        gold = messages[-1]["content"]
        pred = generate(messages)

        gold_ok, gold_rows = execute(cur, gold)
        pred_ok, pred_rows = execute(cur, pred)

        if pred_ok:
            legal += 1
        if normalize_sql(gold) == normalize_sql(pred):
            em_hit += 1
        if pred_ok and gold_ok and pred_rows == gold_rows:
            ex_hit += 1

        if i < 3:
            user_content = messages[-2]["content"]
            question = user_content.split("问题：")[-1].strip() if "问题：" in user_content else "(见 user 字段)"
            print(f"[sample {i}] question: {question}")
            print(f"  gold: {gold}")
            print(f"  pred: {pred}")
            print(f"  EX={pred_ok and gold_ok and pred_rows == gold_rows} legal={pred_ok}")

    print("\n=== 评测结果 ===")
    print(f"样本数          : {n}")
    print(f"SQL 合法率      : {legal / n:.2%} ({legal}/{n})")
    print(f"精确匹配 EM     : {em_hit / n:.2%} ({em_hit}/{n})")
    print(f"执行准确率 EX   : {ex_hit / n:.2%} ({ex_hit}/{n})")
    print("\n提示: 分别对基座(--adapter 省略)与微调后(--adapter 指定)各跑一次, 即可对比微调前后的 EX/EM。")


if __name__ == "__main__":
    main()