#!/usr/bin/env bash
# 2026-11-18: C-7b turn 3 crashed in the harness's evidence check (regex took "." as a number). Runs are kept;
# the analyst is asked again on the same input, then the autopilot continues at turn 4.
cd ~/SDI-Scenario/accident_scene/arena
ROS_DOMAIN_ID=37 PYTHONFAULTHANDLER=1 python3 c1_cycle.py --series c71 --turn 3 --code results/turns/prompt_c71_turn2/claude_gemma4_analyst_pC71t2_block.py --diagnosis results/turns/prompt_c71_turn2/claude_gemma4_analyst_pC71t2_multi10_diagnosis.json \
  --history-file results/turns/prompt_c71_turn2/claude_gemma4_analyst_pC71t2_history.txt --reuse-block results/turns/prompt_c71_turn3/claude_gemma4_analyst_pC71t3_block.py --reuse-reason "$(cat results/turns/prompt_c71_turn3/reason.txt)" \
  --reuse-seeds 106926726,1959134779,712816087,581634200,1270477532 --reuse-coder-log results/turns/prompt_c71_turn3/claude_gemma4_analyst_pC71t3_turn_before_reanalysis.json > ~/logs/c71_turn3_reanalysis.log 2>&1
echo "$(date '+%F %T') [c71] turn 3 re-analysis rc=$?"
PYTHONFAULTHANDLER=1 exec python3 c2_auto.py --series c71 --start 4 --max-turn 30
