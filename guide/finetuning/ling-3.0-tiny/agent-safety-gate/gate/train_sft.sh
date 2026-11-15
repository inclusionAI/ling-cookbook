#!/usr/bin/env bash
# Usage: train_sft.sh <gpu> <name> <text.jsonl> <max_steps> [extra areno train args...]
# Text-LoRA arm: same data and QLoRA settings as JEV, but the model learns to write "safe" or "unsafe".
# QLORA= (set and empty) trains on full-precision weights instead.
set -u
. "$(dirname "$(readlink -f "$0")")/../env.sh"
gpu=$1 name=$2 data=$3 steps=$4
shift 4
cd "$L"
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 LING_MODEL_DIR="$L/models/Ling-3.0-tiny" setsid nohup areno train \
  --algo sft --backend cuda \
  --ckpt "$L/models/Ling-3.0-tiny" \
  --dataset-path "$data" --dataset-loader-fn "$L/gate/sft_loader.py" \
  --save-path "$L/runs/$name" \
  --world-size 1 --tp-size 1 --attn-backend flash \
  ${QLORA---qlora} --lora-rank 8 --lora-alpha 16 \
  --batch-size 8 --mini-bs 8 --shuffle --seed "${SEED:-17}" \
  --epochs 20 --max-steps "$steps" --save-interval 200 \
  --lr 2e-4 --min-lr 2e-5 --lr-decay-style cosine --lr-decay-steps "$steps" \
  --metrics-log-dir "$L/runs/$name/tb" \
  "$@" > "$L/logs/train_$name.log" 2>&1 < /dev/null &
echo "$!" > "$L/logs/train_$name.pid"
echo "started $name pid $!"
