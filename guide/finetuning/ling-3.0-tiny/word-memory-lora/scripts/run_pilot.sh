#!/usr/bin/env bash
# Run from the word-memory-lora directory. MODEL may be a local path or Hub ID.
set -euo pipefail
: "${MODEL:?Set MODEL to inclusionAI/Ling-3.0-tiny or a local model path}"
: "${TRAIN_FILE:?Set TRAIN_FILE to your authorized train.jsonl}"
: "${OUTPUT_DIR:=work/lora-r8-pilot}"
: "${ATTN_IMPLEMENTATION:=eager}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_lora.py \
  --model "$MODEL" --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" \
  --limit 4 --max-steps 48 --rank 8 --learning-rate 1e-4 \
  --micro-batch-size 2 --gradient-accumulation-steps 1 --max-length 896 \
  --padding dynamic --gradient-checkpointing --attn-implementation "$ATTN_IMPLEMENTATION"
python eval/run_eval.py --mode lora --model "$MODEL" --adapter "$OUTPUT_DIR/final" \
  --cases eval/cases.json --attn-implementation eager \
  --output results/lora-r8-eager-with-train.json
python eval/score_outputs.py results/base-eager.json results/lora-r8-eager-with-train.json \
  --output results/pilot-scores.json
