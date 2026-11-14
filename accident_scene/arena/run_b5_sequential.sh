#!/usr/bin/env bash
# Project C, prompt B-5 (REPORT.md 실험 25): B-4 + "use the sensor base code" and "detour around unexpected obstacles",
# base base_carter_run_ros.py = the harness-env-free base (env=None, user 2026-10-28), no test-drive limit (--test-budget 0, user 2026-10-28). a0 then a1, one after the other, so no Isaac test of
# one run overlaps the other run's generation on the shared GPU. Same settings as A-4 otherwise.
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5
mkdir -p "$D/isaac_logs"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --prompt-file PROMPT_B5.md --base-file base_carter_run_ros.py --model gpt-oss-120b --tag gpu_gpt_oss_120b_pB5 \
    --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB5_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
