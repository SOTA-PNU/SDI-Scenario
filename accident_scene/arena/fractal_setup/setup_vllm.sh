#!/usr/bin/env bash
# Build the vLLM env used to serve gpt-oss-120b on an RTX PRO 6000 (sm_120), without sudo.
#
#   bash accident_scene/arena/fractal_setup/setup_vllm.sh
#
# Result:
#   ~/venvs/vllm            uv venv, CPython 3.12 (/usr/bin/python3.12), vllm==0.32.0 (torch 2.13.0+cu130),
#                           the same pin set fractal's ~/furio-gpu/venv was built from
#   ~/venvs/vllm-cuda-home  symlink tree used as CUDA_HOME by serve_gpt-oss-120b_gpu.sh
#
# Why the extra steps (all found on 2026-11-14, each one only fails at engine init after ~90 s of weight loading):
#   * flashinfer JIT-compiles its sampling kernels at startup and needs nvcc. The apt nvidia-cuda-toolkit on
#     Ubuntu 24.04 is CUDA 12.0 (no sm_120), so we use the pip CUDA toolkit inside the venv.
#   * pip resolves nvidia-cuda-nvcc 13.5 next to the 13.0 runtime headers -> "CUDA compiler and CUDA toolkit
#     headers are incompatible". Pin nvcc/crt/nvvm to 13.0.88 (fractal's vllm-bench venv did the same on 2026-09-15).
#   * the pip layout has lib/ (not lib64/) and only libcudart.so.13 -> "cannot find -lcudart". The symlink tree adds
#     lib64/ and libcudart.so without touching the installed packages.
#   * ninja lives in the venv's bin/, which the serve script puts on PATH.
set -euo pipefail
V="${VLLM_VENV:-$HOME/venvs/vllm}"
H="${V}-cuda-home"
command -v uv >/dev/null || { echo "uv not found (fractal has /snap/bin/uv; else: curl -LsSf https://astral.sh/uv/0.12.3/install.sh | sh)"; exit 1; }
[ -x /usr/bin/python3.12 ] || { echo "/usr/bin/python3.12 missing"; exit 1; }

[ -x "$V/bin/python" ] || uv venv -p /usr/bin/python3.12 "$V"
uv pip install --python "$V" vllm==0.32.0
uv pip install --python "$V" nvidia-cuda-crt==13.0.88 nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88

C="$V/lib/python3.12/site-packages/nvidia/cu13"
[ -x "$C/bin/nvcc" ] || { echo "pip CUDA toolkit not found at $C"; exit 1; }
rm -rf "$H"; mkdir -p "$H/lib64"
ln -s "$C/bin" "$H/bin"; ln -s "$C/include" "$H/include"; ln -s "$C/nvvm" "$H/nvvm"
for f in "$C"/lib/*; do ln -s "$f" "$H/lib64/"; done
ln -s libcudart.so.13 "$H/lib64/libcudart.so"
[ -e /usr/lib/x86_64-linux-gnu/libcuda.so ] || ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$H/lib64/libcuda.so"

"$H/bin/nvcc" --version | tail -1
"$V/bin/python" -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
echo "OK: $V and $H"
