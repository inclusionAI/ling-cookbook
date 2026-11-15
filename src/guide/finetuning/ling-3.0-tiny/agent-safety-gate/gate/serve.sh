#!/usr/bin/env bash
# Usage: serve.sh <gpu> <port> <name> [extra areno serve args...]
# Starts a detached AReno server on one GPU and returns when it is ready.
set -u
. "$(dirname "$(readlink -f "$0")")/../env.sh"
gpu=$1 port=$2 name=$3
shift 3
log=$L/logs/serve_$name.log
cd "$L"
# Ling-8k is a symlink copy of the weights whose config caps max_position_embeddings at 8192:
# `areno serve` sizes each slot's KV cache from that value, and 131072 costs about 7.5 GiB per slot.
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 setsid nohup areno serve \
  --model-path "${MODEL:-$L/models/Ling-8k}" --tp-size 1 --world-size 1 \
  --host 127.0.0.1 --port "$port" --attn-backend flash "$@" > "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$L/logs/serve_$name.pid"
for _ in $(seq 1 60); do
  grep -q "Uvicorn running\|Application startup complete" "$log" 2>/dev/null && { echo "ready name=$name port=$port pid=$pid"; exit 0; }
  kill -0 "$pid" 2>/dev/null || { echo "server exited"; tail -25 "$log"; exit 1; }
  sleep 5
done
echo "timeout waiting for server"
tail -5 "$log"
exit 1
