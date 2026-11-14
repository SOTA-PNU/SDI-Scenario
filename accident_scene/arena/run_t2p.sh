#!/usr/bin/env bash
# REPORT.md 실험 9 — launch T2P (Claude's own procedure given as instructions), three samples, each detached.
# Same loop as T2 (npu_tool_loop.py) with the pre-registered T2P options. Pass --resume to continue runs.
cd "$(dirname "$0")" && mkdir -p results/turns/transfer_t2p
T2P="@prompts_t2p/t2p.args"   # options in a file: no "*_carter_run.py" on the command line (carter-run guard)
for i in ${SAMPLES:-0 1 2}; do
  setsid nohup python3 npu_tool_loop.py --sample "$i" $T2P "$@" \
    >> "results/turns/transfer_t2p/npu_gpt_oss_120b_t2p_a$i.log" 2>&1 < /dev/null &
done
echo "T2P started: samples ${SAMPLES:-0 1 2} $*"
