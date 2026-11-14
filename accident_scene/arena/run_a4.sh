#!/usr/bin/env bash
# REPORT.md 실험 13 — prompt A-4 (route designed by the model, tracking recipe and fixed-structure map given, Isaac Sim
# test tool), NPU gpt-oss-120b, two runs, cap 2 turns, 10 test drives per turn; every model
# call, tool call and Isaac run is kept under results/turns/prompt_a4/.
cd "$(dirname "$0")" && mkdir -p results/turns/prompt_a4/isaac_logs
for i in 0 1; do
  setsid nohup python3 agent_loop.py --model gpt-oss-120b --tag npu_gpt_oss_120b_pA4 --sample "$i" \
    --prompt-file PROMPT_A4.md --base http://127.0.0.1:8400/v1 --out-dir results/turns/prompt_a4 \
    --max-turns 2 --test-budget 10 --max-tokens 16384 \
    > "results/turns/prompt_a4/npu_gpt_oss_120b_pA4_a$i.log" 2>&1 < /dev/null &
done
echo "A-4 started: samples 0-1"
