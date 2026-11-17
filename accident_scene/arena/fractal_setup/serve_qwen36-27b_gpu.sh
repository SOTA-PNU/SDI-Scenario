#!/usr/bin/env bash
# Serve Qwen3.6-27B (BF16, text only, thinking mode) with vLLM on 127.0.0.1:${PORT:-8420} - the C-6 analyst.
# Same port as gpt-oss-120b: only one of the two fits on the GPU next to Isaac, stop the other first.
#   tmux new -d -s vllmqwen 'bash accident_scene/arena/fractal_setup/serve_qwen36-27b_gpu.sh > ~/logs/vllm_qwen36.log 2>&1'
V="${VLLM_VENV:-$HOME/venvs/vllm}"
export CUDA_HOME="${V}-cuda-home"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
MODEL="${MODEL:-$HOME/models/Qwen3.6-27B}"
exec "$V/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "${PORT:-8420}" \
  --served-model-name qwen3.6-27b --max-model-len "${MAX_LEN:-131072}" --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --reasoning-parser qwen3 --language-model-only ${EXTRA_ARGS:-}
