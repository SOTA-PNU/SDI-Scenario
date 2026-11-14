#!/usr/bin/env bash
# Run the benchmark matrix serially, then compare against the source-host
# reference results.
#
#   setup/40_run_benchmark.sh            # base + solution, levels 0-3 (8 runs, ~20 min)
#   setup/40_run_benchmark.sh --quick    # level0 solution only (1 run, ~2 min)
#   setup/40_run_benchmark.sh --full     # adds the prompted variants (12 runs, ~30 min)
#
# Deliberately NOT `set -e`: every base_* run legitimately exits 1, because its
# verdict is FAIL by design. That is the benchmark working, not an error.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODE="${1:---default}"
case "$MODE" in
  --quick)   RUNS=( "levels/level0/solution_carter_run.py" ) ;;
  --full)    RUNS=(
               levels/level0/base_carter_run.py levels/level0/solution_carter_run.py levels/level0/prompted_carter_run.py
               levels/level1/base_carter_run.py levels/level1/solution_carter_run.py levels/level1/prompted_carter_run.py
               levels/level2/base_carter_run.py levels/level2/solution_carter_run.py levels/level2/prompted_carter_run.py
               levels/level3/base_carter_run.py levels/level3/solution_carter_run.py levels/level3/prompted_carter_run.py ) ;;
  *)         RUNS=(
               levels/level0/base_carter_run.py levels/level0/solution_carter_run.py
               levels/level1/base_carter_run.py levels/level1/solution_carter_run.py
               levels/level2/base_carter_run.py levels/level2/solution_carter_run.py
               levels/level3/base_carter_run.py levels/level3/solution_carter_run.py ) ;;
esac

step "Benchmark matrix (${#RUNS[@]} runs, serial)"
say "    Expected: every base_* FAILS and every solution_*/prompted_* PASSES."
say "    A base run exiting 1 is correct behaviour."
say ""

# Prefer the installed wrapper; fall back to the one in this package so the
# script also works before 20_deploy_repo.sh has run.
RUNNER_BIN="$CARTER_WS/tools/bin/carter-run"
[ -x "$RUNNER_BIN" ] || RUNNER_BIN="$PKG_DIR/tools/carter-run"
[ -r "$RUNNER_BIN" ] || { bad "carter-run not found at $CARTER_WS/tools/bin or $PKG_DIR/tools"; exit 66; }

LOGDIR="$CARTER_WS/benchmark_logs"
mkdir -p "$LOGDIR"
START=$SECONDS
declare -a SUMMARY=()

for script in "${RUNS[@]}"; do
  NAME="$(echo "$script" | sed 's|levels/||; s|/|_|g; s|_carter_run.py||')"
  step "run: $script"
  bash "$RUNNER_BIN" "$script" 2>&1 | tee "$LOGDIR/$NAME.log"
  RC="${PIPESTATUS[0]}"

  if [ "$RC" -eq 2 ]; then
    bad "ROS 2 bridge failed (exit 2) — environment fault, stopping here."
    say "    Fix the environment and re-run; partial results are already on disk."
    exit 2
  fi
  if [ "$RC" -eq 75 ]; then
    bad "another Isaac process is running — stopping. Runs must be serial."
    exit 75
  fi

  VERD="$(grep -o ' -> [A-Z]*' "$LOGDIR/$NAME.log" 2>/dev/null | tail -1 | tr -d ' ->')"
  SUMMARY+=("$NAME ${VERD:-?}")
done

ELAPSED=$((SECONDS - START))
step "All runs finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"
for s in "${SUMMARY[@]}"; do say "    $s"; done

step "Parity check against the source host"
python3 "$PKG_DIR/tools/verify_parity.py" --repo "$CARTER_REPO" \
        --reference "$PKG_DIR/reference/expected_results.json"
PARITY=$?

step "Result"
if [ "$PARITY" -eq 0 ]; then
  ok "this host reproduces the source host's benchmark results"
  say "    logs: $LOGDIR"
  say "    To restore the pristine reference results:  cd $CARTER_REPO && git checkout -- levels/"
  say "    (a read-only copy also lives at $CARTER_WS/reference_baseline)"
else
  warn "parity check reported differences — read the detail above before drawing conclusions."
  say "    A differing *verdict* means something is genuinely wrong."
  say "    Sub-tolerance float drift on a different GPU model can be legitimate."
fi
exit "$PARITY"
