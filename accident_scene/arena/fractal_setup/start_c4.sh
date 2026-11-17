#!/usr/bin/env bash
# C-4 (2026-11-17): as C-3 (same task, base, prompts, start code, harness rules), but the coder is gpt-oss-120b on
# vLLM (127.0.0.1:8420); reviewer and analyst stay Claude Opus 5.5 (`claude -p --tools ""`).
#   bash start_c4.sh [pid to wait for] [log that must say the goal was reached before C-4 starts]
set -uo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [c4] $*"; }
[ -n "${1:-}" ] && { say "wait for pid $1"; while [ -e /proc/$1 ]; do sleep 10; done; }
if [ -n "${2:-}" ] && ! grep -q "목표 달성" "$2"; then say "$2 has no 목표 달성 - C-4 not started"; exit 1; fi
curl -sf 127.0.0.1:8420/v1/models >/dev/null || { say "vLLM is not up on 8420 - C-4 not started"; exit 1; }
cd $A
D=results/turns/prompt_c4_turn0
mkdir -p $D
cp results/turns/prompt_c2_turn0/start_block.py results/turns/prompt_c2_turn0/history0.txt $D/
say "turn 0 (start code as it is)"
ROS_DOMAIN_ID=37 PYTHONFAULTHANDLER=1 python3 c1_cycle.py --series c4 --turn 0 --code $D/start_block.py \
  --history-file $D/history0.txt --reuse-block $D/start_block.py \
  --reuse-reason "출발점 코드: A-4 경유점 추종, 속도 5배, 센서 미사용" > ~/logs/c4_turn0.log 2>&1
say "turn 0 rc=$?"
PYTHONFAULTHANDLER=1 exec python3 c2_auto.py --series c4 --start 1 --max-turn 30
