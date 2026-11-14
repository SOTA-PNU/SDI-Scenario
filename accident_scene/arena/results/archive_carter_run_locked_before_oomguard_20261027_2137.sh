#!/usr/bin/env bash
# carter-run behind an exclusive lock (REPORT.md 실험 11, 2026-10-13).
#
# carter-run's serial guard scans /proc for a running simulator and starts if it sees
# none. Two callers that check within the same few milliseconds both see none and both
# start: at 11:51:05 two gradings started 3 ms apart and ran side by side. flock makes
# the check-and-start one step for every caller that goes through this wrapper. When
# the lock is held it exits 75, the code carter-run's own guard uses, so callers that
# already wait-and-retry on 75 (turn_loop.grade, record_a1.sh) need no other change.
exec flock -n -E 75 "${CARTER_LOCK:-/tmp/carter_run_serial.lock}" \
  "$HOME/carter_ws/tools/bin/carter-run" "$@"
