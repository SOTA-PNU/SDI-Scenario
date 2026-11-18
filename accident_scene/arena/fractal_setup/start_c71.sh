#!/usr/bin/env bash
# C-7b (2026-11-18): C-7 again (analyst Gemma 4 31B, coder and reviewer Claude Opus 5.5) from turn 0, with two facts
# added to the analyst input: how many batches ran / mean time against the 60 s needed, and the reviewer's concerns.
#   bash start_c71.sh [pid to wait for] [log that must say the goal was reached before C-7b starts]
set -uo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [c71] $*"; }
[ -n "${1:-}" ] && { say "wait for pid $1"; while [ -e /proc/$1 ]; do sleep 10; done; }
if [ -n "${2:-}" ] && ! grep -q "목표 달성" "$2"; then say "$2 has no 목표 달성 - C-7b not started"; exit 1; fi
curl -sf 127.0.0.1:8420/v1/models >/dev/null || { say "vLLM is not up on 8420 - C-7b not started"; exit 1; }
cd $A
D=results/turns/prompt_c71_turn0
mkdir -p $D
cp results/turns/prompt_c2_turn0/start_block.py results/turns/prompt_c2_turn0/history0.txt $D/
say "turn 0 (start code as it is)"
ROS_DOMAIN_ID=37 PYTHONFAULTHANDLER=1 python3 c1_cycle.py --series c71 --turn 0 --code $D/start_block.py \
  --history-file $D/history0.txt --reuse-block $D/start_block.py \
  --reuse-reason "출발점 코드: A-4 경유점 추종, 속도 5배, 센서 미사용" > ~/logs/c71_turn0.log 2>&1
say "turn 0 rc=$?"
PYTHONFAULTHANDLER=1 exec python3 c2_auto.py --series c71 --start 1 --max-turn 30
