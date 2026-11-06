#!/usr/bin/env bash
# Usage: serve.sh <model_path> <gpus> <port> <log> [extra areno serve args...]
# Starts areno serve detached; returns once the server is ready.
set -u
model=$1 gpus=$2 port=$3 log=$4
shift 4
AR=${ARENO_VENV:+$ARENO_VENV/bin/}areno  # set ARENO_VENV, or have `areno` on PATH
ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)

# Run outside any AReno source checkout so the installed (compiled) package is imported.
cd /tmp
CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 setsid nohup "$AR" serve \
  --model-path "$model" --tp-size "$ngpu" --world-size "$ngpu" \
  --host 127.0.0.1 --port "$port" --attn-backend flash \
  --max-running-prompts 16 "$@" > "$log" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 90); do
  grep -q "Uvicorn running" "$log" && { echo "ready port=$port pid=$pid"; exit 0; }
  kill -0 "$pid" 2>/dev/null || { echo "server exited"; tail -20 "$log"; exit 1; }
  sleep 10
done
echo "timeout waiting for server"; exit 1
