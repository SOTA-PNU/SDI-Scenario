#!/usr/bin/env bash
# REPORT.md 실험 10 — prompt A-1 (waypoints + active Isaac Sim use), NPU gpt-oss-120b, track A,
# five samples, cap 8, same sampling as the T0 baseline; every Isaac run's logs kept.
cd "$(dirname "$0")" && mkdir -p results/turns/prompt_a1/isaac_logs
for i in 0 1 2 3 4; do
  setsid nohup python3 turn_loop.py --model gpt-oss-120b --tag npu_gpt_oss_120b_pA1 --track a \
    --sample "$i" --max-turns 8 --base http://127.0.0.1:8400/v1 --max-tokens 16384 \
    --prompt-file PROMPT_A1.md --turns-dir results/turns/prompt_a1 \
    --isaac-log-dir results/turns/prompt_a1/isaac_logs \
    > "results/turns/prompt_a1/npu_gpt_oss_120b_pA1_a$i.log" 2>&1 < /dev/null &
done
echo "A-1 started: samples 0-4"
