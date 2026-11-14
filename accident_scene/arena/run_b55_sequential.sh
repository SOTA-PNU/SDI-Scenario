#!/usr/bin/env bash
# Project C, prompt B-5-5 (REPORT.md 실험 30): B-5-4 plus a prompt-only work method after each test (pick out what
# decides the cause from the result and the DBG lines, state the cause, record it at the top of the next EDIT REGION).
# Same base (base_carter_run_ros_map.py) and flags as B-5-4. User, 2026-10-28: "프롬프트만을 수정해서 성공을 해보고 싶어서
# ... 그걸 프롬프트로 넣어주면 가능할까? 버전 하나에 시도해봐줘". a0 then a1, one after the other.
# Optional $1: PID of a running run_b54_sequential.sh to wait for first (no parallel Isaac runs, no shared vLLM batches).
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_5
mkdir -p "$D/isaac_logs"
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "$(date '+%F %T') waiting for run_b54_sequential.sh pid $WAIT_PID" >> "$D/sequential.log"
  while [ -e "/proc/$WAIT_PID" ] && grep -qa run_b54_sequential "/proc/$WAIT_PID/cmdline" 2>/dev/null; do sleep 20; done
fi
echo "$(date '+%F %T') start" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_5/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_5.md --base-file base_carter_run_ros_map.py --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB55 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB55_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
