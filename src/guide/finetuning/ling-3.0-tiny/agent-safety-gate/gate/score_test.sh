#!/bin/bash
# score_test.sh <gpu> arm=checkpoint ... : score test slices once per arm. Run only after freeze files exist.
. "$(dirname "$(readlink -f "$0")")/../env.sh"
cd $L/gate
gpu=$1; shift
mkdir -p $L/runs/preds_test
for item in "$@"; do
  arm=${item%%=*}; ckpt=${item#*=}
  if [ -f "$L/runs/preds_test/$arm.rjudge.jsonl" ]; then echo "refusing to rerun $arm"; continue; fi
  # BATCH lowers the records per forward pass when the GPU is shared; it does not change the scores.
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python score_jev_batch.py --arm $arm --model $L/models/Ling-3.0-tiny --checkpoint $ckpt --bench-root $L/bench \
    --out-dir $L/runs/preds_test --slice test --batch "${BATCH:-32}" --benches preex,asse-security,asse-safety,rjudge \
    > "$L/logs/test_$arm.log" 2>&1
  grep -E "Error|Traceback" "$L/logs/test_$arm.log" | tail -3
  echo "TESTDONE $arm $(date +%T)"
done
echo TESTQUEUEDONE
