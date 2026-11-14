#!/usr/bin/env bash
# Shared helpers + the single source of truth for this package's paths.
# Every setup/*.sh sources this file. It never runs anything by itself.

# --- paths (override by exporting before you call any script) ----------------
# CARTER_WS   : where the 15 GB Isaac Sim micromamba workspace lives
# CARTER_REPO : where the benchmark repo (carter-scenario-levels) is deployed
export CARTER_WS="${CARTER_WS:-$HOME/carter_ws}"
export CARTER_REPO="${CARTER_REPO:-$HOME/carter-scenario-levels}"

# Directory this package was extracted into (parent of setup/).
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PKG_DIR

export MAMBA_ROOT_PREFIX="$CARTER_WS/mamba"
export MICROMAMBA="$CARTER_WS/tools/bin/micromamba"
export ISAAC_ENV_NAME="isaacsim"
export ISAAC_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$ISAAC_ENV_NAME"
export ISAAC_SITE="$ISAAC_ENV_PREFIX/lib/python3.10/site-packages"

# Isolate the conda env from ~/.local/lib/python3.10/site-packages.
# Ubuntu 22.04's system python is also 3.10, so the user site otherwise lands on
# sys.path AHEAD of the env and silently shadows numpy/torch. Measured on the
# source host: numpy 2.2.6 (user site) instead of the env's pinned 1.26.4.
export PYTHONNOUSERSITE=1

# --- pinned versions --------------------------------------------------------
export ISAACSIM_SPEC="isaacsim[all,extscache]==4.5.0"
export NVIDIA_INDEX="https://pypi.nvidia.com"
export MICROMAMBA_VERSION="2.8.1"
export PYTHON_VERSION="3.10"
# Deps the source host's install silently skipped because it found them in
# ~/.local. Without these the env's torch cannot import on a clean machine.
export REPAIR_PINS="typing_extensions==4.15.0 filelock==3.29.1 fsspec==2026.4.0"

export ASSET_ROOT_URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"

# --- output helpers ---------------------------------------------------------
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_BAD=$'\033[31m'; C_WARN=$'\033[33m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_OK=''; C_BAD=''; C_WARN=''; C_DIM=''; C_OFF=''
fi

say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  OK  %s %s\n' "$C_OK" "$C_OFF" "$*"; }
bad()  { printf '%s FAIL %s %s\n' "$C_BAD" "$C_OFF" "$*"; }
warn() { printf '%s WARN %s %s\n' "$C_WARN" "$C_OFF" "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }

# Run python inside the Isaac env. Always via the absolute micromamba path —
# micromamba is not on PATH on a fresh host.
isaac_py() { "$MICROMAMBA" run -n "$ISAAC_ENV_NAME" python "$@"; }

have() { command -v "$1" >/dev/null 2>&1; }
