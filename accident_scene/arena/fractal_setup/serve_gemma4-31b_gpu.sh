#!/usr/bin/env bash
# Serve Gemma 4 31B (it, BF16) with vLLM on 127.0.0.1:${PORT:-8420} - the C-7 analyst. Thinking is switched on per
# request (chat_template_kwargs.enable_thinking, set in agent_loop.post). Same port as the other analyst models:
# only one fits on the GPU next to Isaac, stop the other first. KV cache is stored as fp8: at 0.80 a BF16 cache for
# 131072 tokens needs 23.3 GiB and only 15.0 GiB is left after the 62.6 GB of weights.
#   tmux new -d -s vllmgemma 'bash accident_scene/arena/fractal_setup/serve_gemma4-31b_gpu.sh > ~/logs/vllm_gemma4.log 2>&1'
V="${VLLM_VENV:-$HOME/venvs/vllm}"
export CUDA_HOME="${V}-cuda-home"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
MODEL="${MODEL:-$HOME/models/gemma-4-31B-it}"
exec "$V/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "${PORT:-8420}" \
  --served-model-name gemma-4-31b --max-model-len "${MAX_LEN:-131072}" --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --reasoning-parser gemma4 --max-num-seqs "${MAX_SEQS:-4}" ${EXTRA_ARGS---language-model-only --kv-cache-dtype fp8}
