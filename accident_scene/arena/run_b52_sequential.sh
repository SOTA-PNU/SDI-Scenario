#!/usr/bin/env bash
# Project C, prompt B-5-2 (REPORT.md 실험 27): B-5-1 plus a table of what each sensor sees (measured),
# agent_loop --b5 --b-test-movers; base, sampling and the unlimited test drives as in B-5. a0 then a1, one after the other.
# Starts when the B-5-1 runs are done (results/turns/prompt_b5_2/sequential.log), so no Isaac run overlaps another.
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_2
mkdir -p "$D/isaac_logs"
until grep -q " done$" results/turns/prompt_b5_1/sequential.log 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start (B-5-1 done)" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_2/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_2.md --base-file base_carter_run_ros.py --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB52 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB52_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
