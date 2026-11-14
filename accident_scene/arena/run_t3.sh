#!/usr/bin/env bash
# REPORT.md 실험 8 — launch T3 (reasoning_effort high, no tools), five samples, each detached
cd "$(dirname "$0")" && mkdir -p results/turns/transfer_t3
for i in 0 1 2 3 4; do
  setsid nohup python3 turn_loop.py --model gpt-oss-120b --tag npu_gpt_oss_120b_t3 --track a \
    --sample "$i" --max-turns 8 --base http://127.0.0.1:8400/v1 --max-tokens 16384 \
    --reasoning-effort high --turns-dir results/turns/transfer_t3 \
    > "results/turns/transfer_t3/npu_gpt_oss_120b_t3_a$i.log" 2>&1 < /dev/null &
done
echo "T3 started: samples 0-4"
