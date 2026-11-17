#!/usr/bin/env bash
# M-1 (DESIGN_MULTI_AGENT.md, 2026-11-15): the B-5-29 task (base v17, speeds x5) with harness + coder + analyst.
# M-1b (2026-11-15): every compile or test drive is judged by the analyst (static failures included), and the
# harness records each change (per-function diff). M-1 v1 run stopped: results/turns/prompt_m1_stopped_20261115_105052.
# M-1c (2026-11-15): base v18 (goal-align turn capped at 0.4 rad/s), digest states how a run ended and where the time
# went (ended_by, time_budget). M-1b stopped: results/turns/prompt_m1b_stopped_20261115_114231.
# M-1d (2026-11-15): the coder answers a reason + only the functions it changes (harness swaps them in, plan_path must
# stay implemented), diagnoses that propose forbidden methods are refused, the analyst traces the failing code line by
# line. M-1c a0 stopped after 19 iterations: results/turns/prompt_m1c.
set -u
set -o pipefail
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_m1d
TAG=gpu_gpt_oss_120b_pM1d
mkdir -p "$D/isaac_logs" || exit 1
exec 8>"$D/.sequential.lock"
flock -n 8 || { echo 'M-1 sequential runner is already active' >&2; exit 75; }

# Do not overwrite any previous model, grading, or snapshot artifacts.
shopt -s nullglob
existing=("$D"/"$TAG"* "$D/isaac_logs"/"$TAG"* results/"$TAG"* "$TAG"*.py
          "$D/sequential.log" "$D/isaac_logs/omniverse_logs_offsets_start.json")
for f in "${existing[@]}"; do
  if [ -e "$f" ]; then
    echo "Refusing to overwrite existing M-1 result: $f" >&2
    exit 73
  fi
done
echo "$(date '+%F %T') start" >> "$D/sequential.log"
python3 - <<'PY' || exit 1
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f)
                  for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
with open("results/turns/prompt_m1d/isaac_logs/omniverse_logs_offsets_start.json", "x") as f:
    json.dump(snap, f, indent=1)
PY

for s in 0 1 2 3; do
  python3 multi_agent_loop.py --model gpt-oss-120b --tag "$TAG" --sample "$s" --base http://127.0.0.1:8420/v1 \
    --out-dir "$D" --base-file base_carter_run_ros_map_lidartime_objects_check_v18.py --movers v1 \
    --test-sim-s 200 --max-iters 20 --max-wall-h 3 --max-grades 2 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/${TAG}_a$s.log" 2>&1 < /dev/null
  rc=$?
  # Read grading records, never a test-drive pass or exit status.
  grade_turn=$(python3 - "$D/${TAG}_a${s}_turns.json" "$s" <<'PY'
import json, sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text())
    if data.get("tag") != "gpu_gpt_oss_120b_pM1d" or data.get("sample") != int(sys.argv[2]):
        raise ValueError("run identity does not match")
    passing = [r["turn"] for r in data["turns"] if r.get("verdict") == "pass" and type(r.get("turn")) is int]
    print("PASS", min(passing), data.get("stop"), data.get("iterations")) if passing else \
        print("FAIL -", data.get("stop"), data.get("iterations"))
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"Cannot read grading record: {exc}", file=sys.stderr)
    print("FAIL - - -")
PY
  )
  read -r grade turn stop iters <<< "$grade_turn"
  echo "$(date '+%F %T') a$s exit=$rc grade=${grade:-FAIL} turn=${turn:--} stop=${stop:--} iters=${iters:--}" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
