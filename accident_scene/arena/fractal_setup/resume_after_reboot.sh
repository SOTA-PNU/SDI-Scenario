#!/usr/bin/env bash
# 2026-11-16 (2nd reboot): env and model are in place; serve with the handoff settings, then finish C-2 turn 15 and
# restart the autopilot at 16. A GPU watch (1 s, synced) runs in the background in case the host goes down again.
set -euo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [resume] $*"; }
nohup bash -c 'while true; do nvidia-smi --query-gpu=timestamp,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader >> ~/logs/gpu_watch.csv; sync ~/logs/gpu_watch.csv; sleep 1; done' >/dev/null 2>&1 &
say "gpu watch pid $!"
say "serve on 8420 (handoff settings)"
mv ~/logs/vllm120b.log ~/logs/vllm120b_$(date +%H%M)_prev.log 2>/dev/null || true
tmux new -d -s vllm120b "bash $A/fractal_setup/serve_gpt-oss-120b_gpu.sh > ~/logs/vllm120b.log 2>&1"
for i in $(seq 1 120); do
  curl -sf 127.0.0.1:8420/v1/models >/dev/null && break
  grep -q -E 'Error|Traceback' ~/logs/vllm120b.log && { say "vllm failed"; grep -E 'Error|OOM' ~/logs/vllm120b.log | tail -5; exit 1; }
  sleep 10
done
curl -sf 127.0.0.1:8420/v1/models >/dev/null || { say "vllm not up after 20 min"; exit 1; }
say "vllm up, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
say "finish turn 15"
cd $A
D=results/turns/prompt_c2_turn15; P=results/turns/prompt_c2_turn14
DIAG=$(ls $P/gpu_gpt_oss_120b_pC2t14_multi*_diagnosis.json | grep -v -E 'BAD|ARCH' | head -1)
ROS_DOMAIN_ID=37 python3 c1_cycle.py --series c2 --turn 15 --code $P/gpu_gpt_oss_120b_pC2t14_block.py --diagnosis $DIAG \
  --history-file $P/gpu_gpt_oss_120b_pC2t14_history.txt --reuse-block $D/gpu_gpt_oss_120b_pC2t15_block.py \
  --reuse-reason "$(cat $D/reason.txt)" > ~/logs/c2_turn15.log 2>&1 || say "turn 15 rc=$?"
say "autopilot from turn 16"
nohup python3 c2_auto.py --start 16 >> ~/logs/c2_auto.log 2>&1 &
say "autopilot pid $!"
