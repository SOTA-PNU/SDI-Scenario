#!/usr/bin/env bash
# One Isaac boot that exercises everything the benchmark needs, without writing
# any result file. Run this before spending 20+ minutes on the full matrix.
#
# probe_scene.py checks, in one boot:
#   [A] robot spawn pose          [D] physics/render dt
#   [B] floor prim paths          [E] in-process rclpy: /cmd_vel -> robot moves
#   [C] free-space probes         [F] PhysX contact reports (the collision oracle)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

step "Smoke test: $CARTER_REPO/levels/common/probe_scene.py"
say "    First boot on a new GPU recompiles shaders — allow 3-5 minutes."
say "    Later boots settle at roughly 55-80 seconds."
say ""

# Prefer the installed wrapper; fall back to this package's copy so the smoke
# test also works before 20_deploy_repo.sh has installed it.
RUNNER_BIN="$CARTER_WS/tools/bin/carter-run"
[ -x "$RUNNER_BIN" ] || RUNNER_BIN="$PKG_DIR/tools/carter-run"
[ -r "$RUNNER_BIN" ] || { bad "carter-run not found at $CARTER_WS/tools/bin or $PKG_DIR/tools"; exit 66; }

LOG="$CARTER_WS/smoke_test.log"
bash "$RUNNER_BIN" levels/common/probe_scene.py 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"

step "Result"
if [ "$RC" -eq 2 ]; then
  bad "the ROS 2 bridge failed to start (exit 2)."
  say "    The simulator would look healthy while publishing nothing. Check that no"
  say "    system ROS or RoboStack environment is sourced in this shell, then re-run."
  exit 2
fi

# probe_scene prints a section per check; a dead run has none of them.
FOUND=0
for tag in "spawn" "rclpy" "contact"; do
  grep -qi "$tag" "$LOG" && FOUND=$((FOUND + 1))
done

if [ "$RC" -eq 0 ] && [ "$FOUND" -ge 2 ]; then
  ok "Isaac Sim boots, the ROS 2 bridge is live, and the robot responds to /cmd_vel"
  say "    full log: $LOG"
  say "    next: setup/40_run_benchmark.sh"
  exit 0
fi

bad "smoke test did not complete cleanly (exit $RC)"
say "    log: $LOG"
say ""
say "    Most common causes, in order:"
say "      1. driver installed without GL/Vulkan userspace -> re-run setup/00_preflight.sh"
say "      2. no outbound network to the asset server      -> the scene USD streams on every run"
say "      3. a system ROS sourced in this shell           -> start a clean shell"
exit 1
