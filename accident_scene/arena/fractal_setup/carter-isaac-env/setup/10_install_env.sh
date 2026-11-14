#!/usr/bin/env bash
# Build the Isaac Sim 4.5.0 micromamba environment at $CARTER_WS.
# Idempotent: re-running skips whatever is already correct.
#
# This reproduces the source host's env exactly as it was BUILT, plus the three
# dependency repairs that host silently skipped (it found them in ~/.local, so
# they never landed in the env and its torch cannot import on a clean machine).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

die() { bad "$1"; exit 1; }

step "Target"
say "    CARTER_WS = $CARTER_WS"
say "    env       = $ISAAC_ENV_PREFIX"
say "    spec      = $ISAACSIM_SPEC  (+ --extra-index-url $NVIDIA_INDEX)"

mkdir -p "$CARTER_WS/tools/bin" || die "cannot create $CARTER_WS/tools/bin"

step "1. micromamba $MICROMAMBA_VERSION"
if [ -x "$MICROMAMBA" ] && "$MICROMAMBA" --version 2>/dev/null | grep -q "^$MICROMAMBA_VERSION$"; then
  ok "already installed: $MICROMAMBA ($("$MICROMAMBA" --version))"
elif [ -x "$PKG_DIR/bin/micromamba" ]; then
  cp -a "$PKG_DIR/bin/micromamba" "$MICROMAMBA" || die "could not copy micromamba"
  chmod +x "$MICROMAMBA"
  ok "installed from this package: $("$MICROMAMBA" --version)"
else
  warn "bin/micromamba missing from the package — downloading instead"
  curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/$MICROMAMBA_VERSION" \
    | tar -xj -C "$CARTER_WS/tools" bin/micromamba || die "micromamba download failed"
  chmod +x "$MICROMAMBA"
  ok "downloaded: $("$MICROMAMBA" --version)"
fi

step "2. conda env '$ISAAC_ENV_NAME' (python $PYTHON_VERSION)"
# Literal command recorded in the source host's conda-meta/history.
if [ -x "$ISAAC_ENV_PREFIX/bin/python3.10" ]; then
  ok "env already exists: $("$ISAAC_ENV_PREFIX/bin/python3.10" --version 2>&1)"
else
  "$MICROMAMBA" create -y -n "$ISAAC_ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip \
    || die "micromamba create failed"
  ok "created: $("$ISAAC_ENV_PREFIX/bin/python3.10" --version 2>&1)"
fi

PYVER="$(isaac_py -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
[ "$PYVER" = "3.10" ] || die "env python is $PYVER, must be exactly 3.10 — Isaac Sim 4.5 publishes cp310 wheels only. Delete $ISAAC_ENV_PREFIX and re-run."

step "3. Isaac Sim $ISAACSIM_SPEC   (~7.2 GB download, ~15 GB installed)"
# Both halves are load-bearing:
#   --extra-index-url  : pypi.org has NO isaacsim 4.5.x and zero extscache wheels
#   [all,extscache]    : [all] alone omits the three extscache wheels (4.0 GB)
#   ==4.5.0            : without the pin pip silently takes isaacsim 6.x from
#                        pypi.org, a different API this harness is untested on
if isaac_py -c "import isaacsim, sys; sys.exit(0 if isaacsim.__file__ else 1)" >/dev/null 2>&1 \
   && isaac_py -m pip show isaacsim-extscache-kit >/dev/null 2>&1; then
  ok "isaacsim $(isaac_py -m pip show isaacsim 2>/dev/null | awk '/^Version:/{print $2}') already installed (incl. extscache)"
else
  say "    downloading — this takes 15-40 minutes depending on the link"
  isaac_py -m pip install "$ISAACSIM_SPEC" --extra-index-url "$NVIDIA_INDEX" \
    || die "isaacsim install failed. Check that $NVIDIA_INDEX is reachable and that python is 3.10 / glibc >= 2.34."
  ok "isaacsim installed"
fi

step "4. Repair the three deps the source host's install skipped"
# On the source host pip found these in ~/.local and printed "Requirement
# already satisfied", so they were never written into the env.
isaac_py -m pip install $REPAIR_PINS >/dev/null 2>&1 \
  || warn "could not install $REPAIR_PINS — check pip output manually"
ok "ensured: $REPAIR_PINS"

step "5. Verify the env is self-contained"
CHECK="$(isaac_py -m pip check 2>&1)"
if [ -z "$CHECK" ]; then
  ok "pip check clean (no missing dependencies)"
else
  warn "pip check reported:"
  printf '%s\n' "$CHECK" | sed 's/^/         /'
  say  "         (torch-only complaints are harmless — torch is never on the benchmark path —"
  say  "          but anything mentioning numpy or isaacsim must be fixed.)"
fi

step "6. Verify the pinned versions actually win"
isaac_py - <<'PY'
import sys, numpy
print(f"    python : {sys.version.split()[0]}")
print(f"    numpy  : {numpy.__version__}   <- must be 1.x (isaacsim-core pins numpy<2.0.0)")
print(f"    from   : {numpy.__file__}")
leaked = [p for p in sys.path if ".local/lib" in p]
print(f"    user-site leak : {leaked if leaked else 'none (PYTHONNOUSERSITE is working)'}")
sys.exit(0 if numpy.__version__.startswith("1.") and not leaked else 1)
PY
if [ $? -eq 0 ]; then ok "env isolation verified"; else die "the env is being shadowed by another site-packages — PYTHONNOUSERSITE=1 is not taking effect"; fi

step "7. Boot Isaac Sim headless (first boot compiles shaders: 2-3 minutes)"
# OMNI_KIT_ACCEPT_EULA is a RUNTIME gate. Without it Kit calls input() and any
# non-interactive run dies with "EOFError: EOF when reading a line".
#
# Two attempts. The first is the stock boot. If it dies, the second retries with
# the viewport suppressed, which is the documented escape on hosts whose driver
# is from the R590 (595.x) or R610 branch: Kit 106.5 segfaults inside
# librtx.scenedb.plugin.so a millisecond after the RTX SceneDbContext is built,
# and that context is only built once a viewport texture exists. Measured on
# fractal (RTX PRO 6000 Blackwell, driver 595.84). See PORTING-NOTES-fractal.md.
boot_isaac() {
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  EXTRA_BOOT_ARGS="$1" \
  isaac_py - <<'PY'
import os
from isaacsim import SimulationApp
# The driver check misreads any driver whose minor >= 256 (Vulkan packs the
# minor into 8 bits, so 535.309.01 reads as 535.53 and trips a banned range).
# carter_env.py passes this same flag on every run. Do not remove it.
args = ["--/rtx/verifyDriverVersion/enabled=false"]
args += [a for a in os.environ.get("EXTRA_BOOT_ARGS", "").split() if a]
app = SimulationApp({
    "headless": True,
    "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
    "extra_args": args,
})
print("BOOT OK")
app.close(wait_for_replicator=False)
PY
}

NO_VIEWPORT_ARGS="--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true \
--/exts/omni.kit.viewport.window/windowMenu/entryCount=0 \
--/exts/omni.kit.viewport.menubar.render/autoManage/enabledList= \
--/renderer/enabled=none \
--/rtx/hydra/mdlMaterialWarmup=false \
--/app/extensions/excluded/0=omni.physx.ui"

if boot_isaac ""; then
  ok "Isaac Sim boots headless"
else
  warn "stock headless boot failed — retrying with the viewport suppressed"
  if boot_isaac "$NO_VIEWPORT_ARGS"; then
    ok "Isaac Sim boots headless with the viewport suppressed"
    say "    This host needs the no-viewport workaround. After 20_deploy_repo.sh run:"
    say "        bash $PKG_DIR/tools/enable_no_viewport.sh"
    say "    Details: $PKG_DIR/PORTING-NOTES-fractal.md"
  else
    die "Isaac Sim failed to boot both ways. Re-run setup/00_preflight.sh — this is almost always a missing Vulkan ICD or GL userspace, not a python problem."
  fi
fi

step "8. Write $CARTER_WS/env.sh"
cat > "$CARTER_WS/env.sh" <<EOF
# Source this before running anything. Generated by 10_install_env.sh.
export CARTER_WS="$CARTER_WS"
export CARTER_REPO="$CARTER_REPO"
export MAMBA_ROOT_PREFIX="$CARTER_WS/mamba"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PATH="$CARTER_WS/tools/bin:\$PATH"
EOF
ok "wrote $CARTER_WS/env.sh"

step "Done"
ok "environment ready at $CARTER_WS  ($(du -sh "$ISAAC_ENV_PREFIX" 2>/dev/null | cut -f1))"
say "    next: setup/20_deploy_repo.sh"
