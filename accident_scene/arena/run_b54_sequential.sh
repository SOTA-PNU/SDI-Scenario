#!/usr/bin/env bash
# Project C, prompt B-5-4 (REPORT.md 실험 29): B-5-3 plus the exact map in the base (base_carter_run_ros_map.py),
# the map principle, lessons and block-only answers in the prompt; agent_loop --b5 --b-test-movers, unlimited tests.
# Started right away (user, 2026-10-28: "성공시켜보자. 버전 번호는 높여가면서"). a0 then a1, one after the other.
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_4
mkdir -p "$D/isaac_logs"
echo "$(date '+%F %T') start" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_4/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_4.md --base-file base_carter_run_ros_map.py --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB54 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB54_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
