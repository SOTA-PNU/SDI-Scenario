#!/usr/bin/env bash
# REPORT.md 실험 8 — launch T2 (reasoning high + Python tool), three samples, each detached
cd "$(dirname "$0")" && mkdir -p results/turns/transfer_t2
for i in 0 1 2; do
  setsid nohup python3 npu_tool_loop.py --sample "$i" \
    > "results/turns/transfer_t2/npu_gpt_oss_120b_t2_a$i.log" 2>&1 < /dev/null &
done
echo "T2 started: samples 0 1 2"
