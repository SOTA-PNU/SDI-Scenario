#!/usr/bin/env bash
# Serve one model on the local GPU with vLLM and collect the same k samples
# the NPU arm collected, then shut the server down.
#
# The NPU arm ran furiosa-llm's router; this is the same weights (furiosa-ai
# republishes the upstream safetensors byte-identically) and the same two
# files as input, so the only thing that differs is the backend.
#
#   bash gpu_generate.sh <hf-id> <tag> [reasoning-parser] [extra vllm args...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/jun/vllm-bench/.venv
MODEL="$1"; TAG="$2"; RP="${3:-}"; shift 3 2>/dev/null || shift $#
PORT=8000
# The system nvcc is 12.0; Blackwell (sm_120) needs >= 12.9 or vLLM silently
# falls back and then OOMs. The venv ships a matching CUDA 13.0 toolchain.
export PATH="$VENV/bin:$VENV/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"   # venv bin carries ninja, which flashinfer JIT shells out to
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="$HERE/results/generated/${TAG}.serve.log"
mkdir -p "$HERE/results/generated"

cmd=("$VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT"
     --served-model-name "$MODEL"
     --max-model-len 32768 --gpu-memory-utilization 0.90)
[ -n "$RP" ] && cmd+=(--reasoning-parser "$RP")
cmd+=("$@")

echo "\$ ${cmd[*]}"
CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" >"$LOG" 2>&1 &
SRV=$!
trap 'kill -TERM -'"$SRV"' 2>/dev/null; kill $SRV 2>/dev/null' EXIT

t0=$SECONDS
until curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[!] 서버가 떴다가 죽었습니다. 로그 끝:"; tail -30 "$LOG"; exit 1
  fi
  [ $((SECONDS - t0)) -gt 1800 ] && { echo "[!] 30분 안에 안 뜸"; tail -20 "$LOG"; exit 1; }
  sleep 4
done
echo "[srv] ready in $((SECONDS - t0))s"

for track in a b; do
  echo "=== $TAG track $track ==="
  "$VENV/bin/python" "$HERE/generate.py" "$MODEL" --track "$track" --k 5 \
      --tag "$TAG" --base "http://127.0.0.1:$PORT/v1" --warmup
done
echo "[done] $TAG"
