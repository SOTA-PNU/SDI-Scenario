#!/usr/bin/env bash
# carter-run behind an exclusive lock (REPORT.md 실험 11, 2026-10-13).
#
# carter-run's serial guard scans /proc for a running simulator and starts if it sees
# none. Two callers that check within the same few milliseconds both see none and both
# start: at 11:51:05 two gradings started 3 ms apart and ran side by side. flock makes
# the check-and-start one step for every caller that goes through this wrapper. When
# the lock is held it exits 75, the code carter-run's own guard uses, so callers that
# already wait-and-retry on 75 (turn_loop.grade, record_a1.sh) need no other change.
#
# GPU memory guard (REPORT.md 실험 23, 2026-10-27). A grading that started while the GPU
# was short of memory (the 120b server plus another user's job) lost its CUDA contexts at
# startup: omni.replicator.core did not start, so no lidar or camera topics were published,
# yet the robot drove and the run was graded. Two checks now treat that like a busy lock:
#   - before starting, fewer than CARTER_MIN_FREE_MIB (default 10000) MiB free -> exit 75;
#   - while running, a CUDA out-of-memory line in the output -> stop the run, keep its output
#     under results/isaac_oom/, exit 75.
# The caller then waits and calls again, so a run is only ever graded with a healthy start.
#
# Wall-time limit (REPORT.md 실험 32, 2026-10-29). turn_loop.grade gives up after 1800 s, but that kills only this
# wrapper: the setsid'd simulator would keep running and holding the lock (fd 9), and the agent loop would die on
# the exception. The wrapper now stops the whole simulator group itself after CARTER_WALL_LIMIT_S (default 1700)
# and ends with exit 124 and one 'TimeoutError:' line, which the loop reports as the run's error.
set -u
LOCK="${CARTER_LOCK:-/tmp/carter_run_serial.lock}"
exec 9>"$LOCK"
flock -n 9 || exit 75

MIN_FREE="${CARTER_MIN_FREE_MIB:-10000}"
OOM_RE='out of memory|Failed to create primary CUDA context'
OOM_DIR="$(cd "$(dirname "$0")" && pwd)/results/isaac_oom"

free_mib=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', *' 'NR==1 {print $1 - $2}')
if [ -n "${free_mib:-}" ] && [ "$free_mib" -lt "$MIN_FREE" ]; then
  echo "carter_run_locked: only ${free_mib} MiB GPU memory free (< ${MIN_FREE}); not starting" >&2
  exit 75
fi

out=$(mktemp) err=$(mktemp)
trap 'rm -f "$out" "$err"' EXIT
WALL_LIMIT="${CARTER_WALL_LIMIT_S:-1700}"
setsid "$HOME/carter_ws/tools/bin/carter-run" "$@" >"$out" 2>"$err" &
pid=$!
t_start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
  if [ $(( $(date +%s) - t_start )) -ge "$WALL_LIMIT" ]; then
    kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    cat "$out"; cat "$err" >&2
    echo "TimeoutError: 시험 주행이 벽시계 ${WALL_LIMIT} s 안에 끝나지 않아 하네스가 멈췄습니다. controller() 한 번 호출이 오래 걸렸을 수 있습니다." >&2
    exit 124
  fi
  if grep -q -E "$OOM_RE" "$out" "$err"; then
    kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    mkdir -p "$OOM_DIR"
    stem="$(date +%Y%m%d_%H%M%S)_$(basename "${1:-run}" .py)"
    cp "$out" "$OOM_DIR/$stem.out"; cp "$err" "$OOM_DIR/$stem.err"
    echo "carter_run_locked: CUDA out of memory at startup; run stopped, output in $OOM_DIR/$stem.*" >&2
    exit 75
  fi
  sleep 2
done
wait "$pid"; rc=$?
cat "$out"; cat "$err" >&2
exit $rc
