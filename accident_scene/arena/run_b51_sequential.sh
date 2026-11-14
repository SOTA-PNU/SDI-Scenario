#!/usr/bin/env bash
# Project C, prompt B-5-1 (REPORT.md 실험 26): B-5 with the test drives on the grading scene too (moving obstacles v1),
# agent_loop --b5 --b-test-movers; base, sampling and the unlimited test drives as in B-5. a0 then a1, one after the other.
# Started at once (user, 2026-10-28 16:5x: "여섯번째 에서 멈추고, B-5-1 테스트 바로 ㄱㄱ") after B-5 a1 was stopped during
# its 7th turn-2 test drive; the B-5 videos come later, so no Isaac run overlaps another.
set -u
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_1
mkdir -p "$D/isaac_logs"
echo "$(date '+%F %T') start" >> "$D/sequential.log"
python3 - <<'PY'
import glob, json, os, time
d = os.path.expanduser("~/.nvidia-omniverse/logs")
snap = {"taken_epoch": time.time(), "taken": time.strftime("%F %T"), "dir": d,
        "files": {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(d + "/*")) if os.path.isfile(f)}}
json.dump(snap, open("results/turns/prompt_b5_1/isaac_logs/omniverse_logs_offsets_start.json", "w"), indent=1)
PY
for s in 0 1; do
  python3 agent_loop.py --b5 --b-test-movers --prompt-file PROMPT_B5_1.md --base-file base_carter_run_ros.py --model gpt-oss-120b \
    --tag gpu_gpt_oss_120b_pB51 --sample "$s" --base http://127.0.0.1:8420/v1 --out-dir "$D" --max-turns 2 --test-budget 0 \
    --max-tokens 16384 --temperature 0.7 --top-p 0.95 --seed-base 1000 \
    > "$D/gpu_gpt_oss_120b_pB51_a$s.log" 2>&1 < /dev/null
  echo "$(date '+%F %T') a$s exit=$?" >> "$D/sequential.log"
done
echo "$(date '+%F %T') done" >> "$D/sequential.log"
