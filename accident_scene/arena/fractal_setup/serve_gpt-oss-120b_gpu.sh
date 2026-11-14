#!/usr/bin/env bash
# Serve gpt-oss-120b with vLLM on 127.0.0.1:${PORT:-8420} - the endpoint run_b5*_sequential.sh calls.
# Flags are fractal's ~/models/serve_gpt-oss-120b.sh (TRITON_ATTN is required on SM120; 0.80 leaves room for Isaac).
# Needs setup_vllm.sh first. Run it detached, e.g.:
#   tmux new -d -s vllm120b 'bash accident_scene/arena/fractal_setup/serve_gpt-oss-120b_gpu.sh > ~/logs/vllm120b.log 2>&1'
# MODEL may be a local directory (default ~/models/gpt-oss-120b) or an HF id.
V="${VLLM_VENV:-$HOME/venvs/vllm}"
export CUDA_HOME="${V}-cuda-home"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"   # nvcc and ninja for flashinfer JIT
MODEL="${MODEL:-$HOME/models/gpt-oss-120b}"
exec "$V/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "${PORT:-8420}" \
  --served-model-name gpt-oss-120b --max-model-len 131072 --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser openai \
  --reasoning-parser openai_gptoss
