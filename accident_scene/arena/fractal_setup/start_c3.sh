#!/usr/bin/env bash
# C-3 (2026-11-16): the C-2 task, base, prompts and start code, with every role (coder, reviewer, analyst) played by
# Claude Opus 5.5 through `claude -p --tools ""` (agent_loop.claude_post). Turn 0 tests the start code as it is,
# then the autopilot runs from turn 1.   bash start_c3.sh [pid to wait for]
set -uo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [c3] $*"; }
[ -n "${1:-}" ] && { say "wait for pid $1"; while [ -e /proc/$1 ]; do sleep 10; done; }
cd $A
D=results/turns/prompt_c3_turn0
mkdir -p $D
cp results/turns/prompt_c2_turn0/start_block.py results/turns/prompt_c2_turn0/history0.txt $D/
say "turn 0 (start code as it is)"
ROS_DOMAIN_ID=37 PYTHONFAULTHANDLER=1 python3 c1_cycle.py --series c3 --turn 0 --code $D/start_block.py \
  --history-file $D/history0.txt --reuse-block $D/start_block.py \
  --reuse-reason "출발점 코드: A-4 경유점 추종, 속도 5배, 센서 미사용" > ~/logs/c3_turn0.log 2>&1
say "turn 0 rc=$?"
PYTHONFAULTHANDLER=1 exec python3 c2_auto.py --series c3 --start 1 --max-turn 30
