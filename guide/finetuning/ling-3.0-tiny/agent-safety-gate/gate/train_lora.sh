#!/usr/bin/env bash
# Usage: train_lora.sh <gpu> <name> <train.jsonl> <max_steps> [extra areno train args...]
# Single-GPU LoRA JEV run on full-precision weights. Training and scoring then see the same
# weights (the CUDA scorer loads full precision), and with activation recompute it took about
# 13 GB at --batch-size 32 --mini-bs 64. Turning recompute off reached 55-71 GB and ran out of memory.
set -u
. "$(dirname "$(readlink -f "$0")")/../env.sh"
gpu=$1 name=$2 data=$3 steps=$4
shift 4
cd "$L"
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 setsid nohup areno train \
  --algo jev --backend cuda \
  --ckpt "$L/models/Ling-3.0-tiny" \
  --dataset-path "$data" \
  --save-path "$L/runs/$name" \
  --world-size 1 --tp-size 1 --attn-backend flash \
  --lora-rank 8 --lora-alpha 16 --activation-checkpointing \
  --batch-size 8 --mini-bs 4 --max-context-len 1536 \
  --epochs 20 --max-steps "$steps" --save-interval 200 --seed "${SEED:-17}" \
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-decay-steps "$steps" \
  --metrics-log-dir "$L/runs/$name/tb" \
  "$@" > "$L/logs/train_$name.log" 2>&1 < /dev/null &
echo "$!" > "$L/logs/train_$name.pid"
echo "started $name pid $!"
