#!/usr/bin/env bash
# Project C, B-5-12 (REPORT.md 실험 38) samples a2 and a3, after a0 and a1 (both passed in turn 1). User 2026-10-29:
# "5-12로 a2 a3도 해봐야겠다. 5-13은 프롬프트 만들어만 두고 앞에꺼 먼저 해줘". Same prompt, base, flags and seeds as
# run_b512_sequential.sh, all-link contacts; the same results folder (a0/a1 files are not touched).
# $1: PID of any process to wait for first (the B-5-12 pass-video recorder), so Isaac never runs twice.
set -u
# Every robot rigid body reports contacts (chassis, drive wheels, casters): levels/common/carter_env.py, 실험 38.
# User 2026-10-29 chose to stop the first B-5-12 launch and restart with this count (B-5-11 a2 'passed' while M1
# pushed its drive wheel 0.86 m; chassis-only contacts missed it). Test drives and gradings both use it.
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_12
mkdir -p "$D/isaac_logs"
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "$(date '+%F %T') a2-a3: waiting for pid $WAIT_PID" >> "$D/sequential.log"
  while [ -e "/proc/$WAIT_PID" ]; do sleep 20; done
fi
echo "$(date '+%F %T') a2-a3 start" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_12/isaac_logs/omniverse_logs_offsets_start_a2a3.json", "w"), indent=1)
PY
for s in 2 3; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_12.md --base-file base_carter_run_ros_map_dbg_heapq_lidartime.py --turn-note-file PROMPT_B5_10_turn_note.md --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB512 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB512_a$s.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date '+%F %T') a$s exit=$rc" >> "$D/sequential.log"
done
echo "$(date '+%F %T') a2-a3 done" >> "$D/sequential.log"
