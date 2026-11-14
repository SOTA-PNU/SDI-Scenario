#!/usr/bin/env bash
# Project C, B-5-10 (REPORT.md 실험 35): the B-5-9 prompt and base unchanged; --turn-note-file PROMPT_B5_10_turn_note.md
# replaces the closing sentence of every turn feedback ("고쳐 다시 제출") with "test first, submit the passing code".
# Same flags and base as B-5-9. Four samples a0-a3. User, 2026-10-28: "waypoint를 찍지 목하게 하면 되려나.
# 너가 알아서 발전시키면서 해줘봐. 정답은 주지말고". a0 to a3, one after the other.
# Optional $1: PID of a running run_b5*_sequential.sh to wait for first (no parallel Isaac runs, no shared vLLM batches).
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_10
mkdir -p "$D/isaac_logs"
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "$(date '+%F %T') waiting for runner pid $WAIT_PID" >> "$D/sequential.log"
  while [ -e "/proc/$WAIT_PID" ] && grep -qa "run_b5.*_sequential" "/proc/$WAIT_PID/cmdline" 2>/dev/null; do sleep 20; done
fi
echo "$(date '+%F %T') start" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_10/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1 2 3; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_9.md --base-file base_carter_run_ros_map_dbg_heapq.py --turn-note-file PROMPT_B5_10_turn_note.md --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB510 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB510_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
