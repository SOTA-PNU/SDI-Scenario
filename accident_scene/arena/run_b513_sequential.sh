#!/usr/bin/env bash
# Project C, B-5-13 (REPORT.md 실험 39): PROMPT_B5_13.md (B-5-12 + the decision method: predict each tracked object a few
# seconds ahead, roll candidate commands with the robot's body, choose the one that does not touch and goes furthest) on
# base_carter_run_ros_map_lidartime_objects.py (B-5-12 base + observed_objects()). User 2026-10-29: "권하는 조합 중 1,2만 적용".
# Samples a0 and a1 only, as B-5-12 ("5-12는 a1까지 테스트"). Exit code logged is agent_loop's own.
# Optional $1: PID of a running run_b5*_sequential.sh to wait for first (no parallel Isaac runs, no shared vLLM batches).
set -u
# Every robot rigid body reports contacts (chassis, drive wheels, casters): levels/common/carter_env.py, 실험 38.
# Same scoring as the B-5-12 restart (chassis-only contacts missed a wheel push in B-5-11 a2).
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_13
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
json.dump(snap, open("results/turns/prompt_b5_13/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_13.md --base-file base_carter_run_ros_map_lidartime_objects.py --turn-note-file PROMPT_B5_10_turn_note.md --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB513 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB513_a$s.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date '+%F %T') a$s exit=$rc" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
