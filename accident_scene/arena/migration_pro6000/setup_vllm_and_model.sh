#!/usr/bin/env bash
# Run on the PRO 6000 server (giga-w773-s-3) as jun. Rebuilds fractal's vLLM venv and the gpt-oss-120b
# weights exactly (README.md). Safe to re-run: finished steps are skipped.
#   nohup bash setup_vllm_and_model.sh > ~/migration_logs/setup.log 2>&1 &
set -euo pipefail
M=$(cd "$(dirname "$0")" && pwd)
LOG=$HOME/migration_logs; mkdir -p "$LOG"
export PATH="$HOME/.local/bin:$PATH"
log() { echo "$(date '+%F %T') $*"; }

# 1. uv 0.12.3, the version that built fractal's venv; user-level. UV_NO_MODIFY_PATH=1 keeps the
#    installer out of ~/.bashrc and the other shell profiles (PATH is set above instead)
if ! command -v uv >/dev/null || [ "$(uv --version | awk '{print $2}')" != 0.12.3 ]; then
  log "installing uv 0.12.3"
  curl -LsSf https://astral.sh/uv/0.12.3/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
# 2. venv on a uv-managed CPython 3.12.3 (this server's python is 3.10), the 206 packages pinned from fractal,
#    installed the way fractal got them (~/furio-gpu/logs/pip.log and download-kernels.log there):
#    everything from PyPI, then flashinfer-cubin from flashinfer.ai/whl and the +cu130 flashinfer-jit-cache
#    (with its six -sm* parts) from flashinfer.ai/whl/cu130 - local versions PyPI does not serve
if [ ! -x "$HOME/furio-gpu/venv/bin/vllm" ]; then
  log "building ~/furio-gpu/venv"
  mkdir -p "$HOME/furio-gpu"
  uv venv --python 3.12.3 "$HOME/furio-gpu/venv"
  PY="$HOME/furio-gpu/venv/bin/python"
  grep -v -E '^flashinfer-(cubin|jit-cache)' "$M/vllm_venv_lock.txt" > "$LOG/lock_pypi.txt"
  uv pip install --python "$PY" -r "$LOG/lock_pypi.txt" 2>&1 | tee "$LOG/pip.log"
  uv pip install --python "$PY" --no-deps --index-url https://flashinfer.ai/whl flashinfer-cubin==0.7.0.post1 2>&1 | tee -a "$LOG/pip.log"
  uv pip install --python "$PY" --index-url https://flashinfer.ai/whl/cu130 flashinfer-jit-cache==0.7.0.post1+cu130 2>&1 | tee -a "$LOG/pip.log"
  # the result must be exactly fractal's package set
  "$PY" -c "import importlib.metadata as m; print('\n'.join(sorted({f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions()}, key=str.lower)))" > "$LOG/lock_built.txt"
  diff "$M/vllm_venv_lock.txt" "$LOG/lock_built.txt" && log "venv matches fractal's 206 packages" || { log "venv differs from fractal - see the diff above"; exit 1; }
fi
"$HOME/furio-gpu/venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(0))"

# 3. the model at fractal's revision, the same 22 files only (hf takes one --include per pattern)
log "downloading openai/gpt-oss-120b @ b5c939de"
mkdir -p "$HOME/models/gpt-oss-120b"
"$HOME/furio-gpu/venv/bin/hf" download openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --local-dir "$HOME/models/gpt-oss-120b" --include "model-*.safetensors" --include "model.safetensors.index.json" \
  --include "config.json" --include "generation_config.json" --include "chat_template.jinja" \
  --include "tokenizer.json" --include "tokenizer_config.json" --include "special_tokens_map.json"
log "verifying sha256 of the 22 files against fractal's"
(cd "$HOME/models/gpt-oss-120b" && sha256sum -c "$M/gpt-oss-120b_SHA256SUMS")
cp -n "$M/serve_gpt-oss-120b.sh" "$HOME/models/"
log "done: venv and model match fractal"
