#!/usr/bin/env bash
# 2026-11-18: the EXAONE engine segfaulted (flashinfer prefill, fp8 KV cache) during the turn-1 analysis. Runs are kept;
# the analyst is asked again on the same input (server now on TRITON_ATTN), then the autopilot continues at turn 2.
cd ~/SDI-Scenario/accident_scene/arena
ROS_DOMAIN_ID=37 PYTHONFAULTHANDLER=1 python3 c1_cycle.py --series c8 --turn 1 --code results/turns/prompt_c8_turn0/start_block.py --diagnosis results/turns/prompt_c8_turn0/claude_exaone45_analyst_pC8t0_multi3_diagnosis.json \
  --history-file results/turns/prompt_c8_turn0/claude_exaone45_analyst_pC8t0_history.txt --reuse-block results/turns/prompt_c8_turn1/claude_exaone45_analyst_pC8t1_block.py --reuse-reason "$(cat results/turns/prompt_c8_turn1/reason.txt)" \
  --reuse-seeds 1673687014,321034601,346148729 --reuse-coder-log results/turns/prompt_c8_turn1/claude_exaone45_analyst_pC8t1_turn_before_reanalysis.json > ~/logs/c8_turn1_reanalysis.log 2>&1
echo "$(date '+%F %T') [c8] turn 1 re-analysis rc=$?"
PYTHONFAULTHANDLER=1 exec python3 c2_auto.py --series c8 --start 2 --max-turn 30
