#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to inclusionAI/Ling-3.0-tiny or a local model path}"
: "${ADAPTER:?Set ADAPTER to the LoRA directory or Hub model ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python eval/run_eval.py --mode base --model "$MODEL" --cases eval/cases.json \
  --attn-implementation eager --output results/base-eager.json
python eval/run_eval.py --mode lora --model "$MODEL" --adapter "$ADAPTER" \
  --cases eval/cases.json --attn-implementation eager \
  --output results/lora-eager-with-train.json
python eval/score_outputs.py results/base-eager.json results/lora-eager-with-train.json \
  --output results/scores.json
