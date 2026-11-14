#!/usr/bin/env bash
# Grade a set of submissions serially in Isaac.
#
# Serial is not a convenience: concurrent Isaac processes change the timing and
# therefore the verdict, which is why carter-run carries its own guard.
#
#   bash grade_batch.sh 'gpu_*_carter_run.py' [--force]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PAT="${1:-gpu_*_carter_run.py}"
FORCE="${2:-}"
export CARTER_REPO="$REPO"
unset CARTER_WHEEL_TOE_DEG

n=0; ok=0
for f in "$HERE"/$PAT; do
  [ -e "$f" ] || continue
  v="$(basename "$f")"; v="${v%_carter_run.py}"
  res="$HERE/results/${v}_result.json"
  if [ -f "$res" ] && [ "$FORCE" != "--force" ]; then
    echo "건너뜀(기존 결과): $v"; continue
  fi
  n=$((n+1))
  printf '[%d] %s ... ' "$n" "$v"
  "$HOME/carter_ws/tools/bin/carter-run" "accident_scene/arena/${v}_carter_run.py" \
      >"$HERE/results/${v}.isaac.log" 2>&1
  rc=$?
  if [ -f "$res" ]; then
    python3 - "$res" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{})
print(f"{d['verdict']:4} dist={m.get('final_dist_m')} col={m.get('collision_count')} "
      f"ttg={m.get('time_to_goal_s')} partners={(m.get('collision_partners') or [])[:1]}")
PY
    ok=$((ok+1))
  else
    echo "결과 없음 (rc=$rc) — $HERE/results/${v}.isaac.log"
  fi
done
echo "채점 $ok/$n"
