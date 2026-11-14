#!/usr/bin/env bash
# B-5-28 (REPORT.md 실험 54) samples a4-a7, after a0-a3 (4/4). User 2026-11-13: "혹시 모르니 a4~a7 실험도 진행해줘".
# Same prompt, base, turn note, flags and seed rule as run_b528_sequential.sh, all-link contacts, same results folder.
# a0-a3 files are not touched. All four samples run regardless of earlier results.
# $1: PID to wait for first (the B-5-28 pass-video recorder), so Isaac never runs twice.
set -u
set -o pipefail
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_28
TAG=gpu_gpt_oss_120b_pB528
WAIT_PID=${1:-}
mkdir -p "$D/isaac_logs" || exit 1
exec 8>"$D/.sequential.lock"
flock -n 8 || { echo 'B-5-28 sequential runner is already active' >&2; exit 75; }

shopt -s nullglob
existing=("$D/${TAG}_a"[4-7]* "$D/isaac_logs/${TAG}_a"[4-7]* results/"${TAG}_a"[4-7]* "${TAG}_a"[4-7]*.py
          "$D/isaac_logs/omniverse_logs_offsets_start_a4a7.json")
for f in "${existing[@]}"; do
  if [ -e "$f" ]; then
    echo "Refusing to overwrite existing B-5-28 a4-a7 result: $f" >&2
    exit 73
  fi
done
if [ -n "$WAIT_PID" ]; then
  echo "$(date '+%F %T') a4-a7: waiting for pid $WAIT_PID" >> "$D/sequential.log"
  while [ -e "/proc/$WAIT_PID" ]; do sleep 20; done
fi
echo "$(date '+%F %T') a4-a7 start" >> "$D/sequential.log"
python3 - <<'PY' || exit 1
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f)
                  for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
with open("results/turns/prompt_b5_28/isaac_logs/omniverse_logs_offsets_start_a4a7.json", "x") as f:
    json.dump(snap, f, indent=1)
PY

for s in 4 5 6 7; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_28.md --base-file base_carter_run_ros_map_lidartime_objects_check_v16.py --turn-note-file PROMPT_B5_10_turn_note.md --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB528 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/${TAG}_a$s.log" 2>&1 < /dev/null
  rc=$?
  # Read submitted-code grading records, never a test-drive pass or exit status.
  grade_turn=$(python3 - "$D/${TAG}_a${s}_turns.json" "$s" <<'PY'
import json, sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text())
    if data.get("tag") != "gpu_gpt_oss_120b_pB528" or data.get("sample") != int(sys.argv[2]):
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
done
echo "$(date '+%F %T') a4-a7 done" >> "$D/sequential.log"
