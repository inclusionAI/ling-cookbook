#!/usr/bin/env bash
# MLX QLoRA 训练启动脚本（MacBook 16G 内存配置）
# 前置：已按 README 下载 BF16 权重并完成 4-bit 量化（见 README「模型准备」一节）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Ling-3.0-tiny-4bit}"
ITERS="${ITERS:-1000}"

exec "$ROOT_DIR/.venv/bin/python" -m mlx_lm lora \
  --model "$MODEL_DIR" \
  --train \
  --data "$ROOT_DIR/data" \
  --fine-tune-type lora \
  --optimizer adamw \
  --num-layers 16 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --max-seq-length 1024 \
  --mask-prompt \
  --grad-checkpoint \
  --adapter-path "$ROOT_DIR/adapters" \
  --save-every 200 \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --val-batches 16 \
  --seed 42 \
  --trust-remote-code
