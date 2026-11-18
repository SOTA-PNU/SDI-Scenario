#!/usr/bin/env bash
# Serve EXAONE 4.5 33B (BF16, 68.7 GB, text only) with vLLM on 127.0.0.1:${PORT:-8420} - the C-8 analyst.
# Thinking is on by default (reasoning parser qwen3, as the model card says). KV cache as fp8 like Gemma: the weights
# leave even less room. Same port as the other analyst models: stop the other first.
# License: EXAONE AI Model License (research / non-commercial).
#   tmux new -d -s vllmexaone 'bash accident_scene/arena/fractal_setup/serve_exaone45-33b_gpu.sh > ~/logs/vllm_exaone45.log 2>&1'
V="${VLLM_VENV:-$HOME/venvs/vllm}"
export CUDA_HOME="${V}-cuda-home"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
MODEL="${MODEL:-$HOME/models/EXAONE-4.5-33B}"
exec "$V/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "${PORT:-8420}" \
  --served-model-name exaone-4.5-33b --max-model-len "${MAX_LEN:-131072}" --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --reasoning-parser qwen3 --max-num-seqs "${MAX_SEQS:-4}" ${EXTRA_ARGS---language-model-only --kv-cache-dtype fp8}
