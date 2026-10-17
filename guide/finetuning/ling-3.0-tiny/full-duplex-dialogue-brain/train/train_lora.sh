#!/usr/bin/env bash
# LoRA SFT of Ling-3.0-tiny into the full-duplex dialogue brain with AReno.
# Usage: train/train_lora.sh <base_model_path> <train.jsonl> <output_dir> [gpus]
set -eu
model=$1 data=$2 out=$3 gpus=${4:-0,1,2,3}
here=$(cd "$(dirname "$0")" && pwd)
AR=${ARENO_VENV:+$ARENO_VENV/bin/}areno  # set ARENO_VENV, or have `areno` on PATH
ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
epochs=${EPOCHS:-3} bs=64
rows=$(wc -l < "$data")
# AReno SFT saves only every --save-interval steps, so point it at the last step.
steps=$(( (rows / bs) * epochs ))

# Run outside any AReno source checkout so the installed (compiled) package is imported.
cd /tmp
CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 "$AR" train \
  --algo sft \
  --ckpt "$model" \
  --dataset-path "$data" \
  --dataset-loader-fn "$here/sft_loader.py" \
  --tp-size "${TP:-$ngpu}" --world-size "$ngpu" \
  --lora-rank 32 --lora-alpha 64 \
  --batch-size "$bs" --mini-bs 16 \
  --epochs "$epochs" \
  --max-prompt-tokens 512 --max-new-tokens 96 \
  --lr "${LR:-2e-4}" --min-lr 2e-5 --lr-decay-style cosine --lr-decay-steps "$steps" \
  --attn-backend flash \
  --save-path "$out" --save-interval "${SAVE_INTERVAL:-$steps}"
