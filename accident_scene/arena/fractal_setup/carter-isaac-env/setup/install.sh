#!/usr/bin/env bash
# Full install: preflight -> environment -> repo -> smoke test.
# Stops at the first failing stage. Every stage is idempotent, so after fixing
# whatever it complained about you can just run this again.
#
#   bash setup/install.sh
#   CARTER_WS=/data/carter_ws CARTER_REPO=/data/carter-scenario-levels bash setup/install.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

say "Carter / Isaac Sim 4.5.0 benchmark environment"
say "  CARTER_WS   = $CARTER_WS      (~24 GB when built)"
say "  CARTER_REPO = $CARTER_REPO"
say "  package     = $PKG_DIR"

run_stage() {
  local script="$1" label="$2"
  step "$label"
  bash "$PKG_DIR/setup/$script"
  local rc=$?
  if [ $rc -ne 0 ]; then
    bad "$label failed (exit $rc). Fix the cause above and re-run: bash setup/install.sh"
    exit $rc
  fi
}

run_stage 00_preflight.sh   "Stage 1/4 — host preflight"
run_stage 10_install_env.sh "Stage 2/4 — Isaac Sim environment"
run_stage 20_deploy_repo.sh "Stage 3/4 — benchmark repo"
run_stage 30_smoke_test.sh  "Stage 4/4 — smoke test"

step "Install complete"
ok "environment ready"
say ""
say "Run the benchmark and compare against the source host:"
say "    bash $PKG_DIR/setup/40_run_benchmark.sh"
say ""
say "Run a single scenario:"
say "    $CARTER_WS/tools/bin/carter-run levels/level0/solution_carter_run.py"
