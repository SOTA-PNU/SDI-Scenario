#!/usr/bin/env bash
# Serve one model and stay up, leaving room on the card for Isaac.
#
# The turn loop alternates generation and simulation on the same GPU, so vLLM
# cannot take its usual 90%: gpt-oss-120b's mxfp4 weights are ~61 GiB of the
# card's 95.6 GiB, and Isaac needs single-digit GiB on top. 0.72 covers the
# weights plus a working KV cache and still leaves Isaac ~17 GiB. 0.72 was
# too tight: vLLM started, then refused for want of cache blocks.
#
#   bash gpu_serve.sh <hf-id> <tag> [reasoning-parser] [gpu-util]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/jun/vllm-bench/.venv
MODEL="$1"; TAG="$2"; RP="${3:-}"; UTIL="${4:-0.82}"
export PATH="$VENV/bin:$VENV/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="$HERE/results/generated/${TAG}.turnserve.log"

cmd=("$VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port 8000
     --served-model-name "$MODEL" --max-model-len 32768
     --gpu-memory-utilization "$UTIL")
[ -n "$RP" ] && cmd+=(--reasoning-parser "$RP")
echo "\$ ${cmd[*]}"
CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" >"$LOG" 2>&1 &
SRV=$!
t0=$SECONDS
until curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; do
  kill -0 $SRV 2>/dev/null || { echo "[!] 서버 종료"; tail -25 "$LOG"; exit 1; }
  [ $((SECONDS - t0)) -gt 1800 ] && { echo "[!] 기동 실패"; exit 1; }
  sleep 4
done
echo "[srv] ready in $((SECONDS - t0))s  pid=$SRV  util=$UTIL"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
wait $SRV
