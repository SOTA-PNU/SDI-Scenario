#!/usr/bin/env bash
# B-5-22, 실험 48: a0/a1, then a2/a3 only when both first samples pass grading.
# agent_loop calls carter_run_locked.sh for every Isaac execution.
set -u
set -o pipefail
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_22
TAG=gpu_gpt_oss_120b_pB522
mkdir -p "$D/isaac_logs" || exit 1
exec 8>"$D/.sequential.lock"
flock -n 8 || { echo 'B-5-22 sequential runner is already active' >&2; exit 75; }

# Do not overwrite any previous model, grading, or snapshot artifacts.
shopt -s nullglob
existing=("$D"/"$TAG"* "$D/isaac_logs"/"$TAG"* results/"$TAG"* "$TAG"*.py
          "$D/sequential.log" "$D/isaac_logs/omniverse_logs_offsets_start.json")
for f in "${existing[@]}"; do
  if [ -e "$f" ]; then
    echo "Refusing to overwrite existing B-5-22 result: $f" >&2
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
with open("results/turns/prompt_b5_22/isaac_logs/omniverse_logs_offsets_start.json", "x") as f:
    json.dump(snap, f, indent=1)
PY

first_pair_pass=1
for s in 0 1 2 3; do
  if [ "$s" -eq 2 ] && [ "$first_pair_pass" -ne 1 ]; then
    break
  fi
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_22.md --base-file base_carter_run_ros_map_lidartime_objects_check_v8.py --turn-note-file PROMPT_B5_10_turn_note.md --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB522 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/${TAG}_a$s.log" 2>&1 < /dev/null
  rc=$?
  # Read submitted-code grading records, never a test-drive pass or exit status.
  grade_turn=$(python3 - "$D/${TAG}_a${s}_turns.json" "$s" <<'PY'
import json, sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text())
    if data.get("tag") != "gpu_gpt_oss_120b_pB522" or data.get("sample") != int(sys.argv[2]):
        raise ValueError("run identity does not match")
    passing = [r["turn"] for r in data["turns"]
               if r.get("verdict") == "pass" and type(r.get("turn")) is int
               and 1 <= r["turn"] <= 2]
    print("PASS", min(passing)) if passing else print("FAIL -")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"Cannot read grading record: {exc}", file=sys.stderr)
    print("FAIL -")
PY
  )
  read -r grade turn <<< "$grade_turn"
  grade=${grade:-FAIL}
  turn=${turn:--}
  echo "$(date '+%F %T') a$s exit=$rc grade=$grade turn=$turn" >> "$D/sequential.log"
  if [ "$s" -lt 2 ] && [ "$grade" != PASS ]; then
    first_pair_pass=0
  fi
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
