#!/usr/bin/env bash
# REPORT.md 실험 8 — (re)start every unfinished T3/T2 run, each detached.
# Saved records -> continue from the interrupted turn (--resume); no records -> start
# fresh; finished or still running -> left alone. Logs are appended, never truncated.
cd "$(dirname "$0")" || exit 1
if ! curl -fsS --max-time 5 http://127.0.0.1:8400/v1/models >/dev/null 2>&1; then
  echo "NPU 서버(127.0.0.1:8400)에 닿지 않습니다 - 터널을 먼저 여십시오 (bash ~/tunnel.sh)"
  exit 1
fi
finished() {  # passed, or out of turns (8)
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get("passed_at_turn") or d.get("status") or len(d["turns"])>=8) else 1)' "$1"
}
T3="--model gpt-oss-120b --tag npu_gpt_oss_120b_t3 --track a --max-turns 8 --base http://127.0.0.1:8400/v1 --max-tokens 16384 --reasoning-effort high --turns-dir results/turns/transfer_t3"
for i in 0 1 2 3 4; do
  d=results/turns/transfer_t3; stem=npu_gpt_oss_120b_t3_a$i
  pgrep -f "turn_loop.py .*--tag npu_gpt_oss_120b_t3 .*--sample $i( |$)" >/dev/null && { echo "T3 a$i: 실행 중 - 그대로 둠"; continue; }
  if [ -f $d/${stem}_turns.json ]; then
    finished $d/${stem}_turns.json && { echo "T3 a$i: 종료됨"; continue; }
    mode=--resume; echo "=== $(date '+%F %T') resume" >> $d/$stem.log
  else mode=; fi
  setsid nohup python3 turn_loop.py $T3 --sample $i $mode >> $d/$stem.log 2>&1 < /dev/null &
  echo "T3 a$i: ${mode:-새로 시작}"
done
for i in 0 1 2; do
  d=results/turns/transfer_t2; stem=npu_gpt_oss_120b_t2_a$i
  pgrep -f "npu_tool_loop.py --sample $i( |$)" >/dev/null && { echo "T2 a$i: 실행 중 - 그대로 둠"; continue; }
  if [ -f $d/${stem}_state.json ]; then
    [ -f $d/${stem}_turns.json ] && finished $d/${stem}_turns.json && { echo "T2 a$i: 종료됨"; continue; }
    mode=--resume; echo "=== $(date '+%F %T') resume" >> $d/$stem.log
  else mode=; fi
  setsid nohup python3 npu_tool_loop.py --sample $i $mode >> $d/$stem.log 2>&1 < /dev/null &
  echo "T2 a$i: ${mode:-새로 시작}"
done
for i in 0 1 2; do
  d=results/turns/transfer_t2p; stem=npu_gpt_oss_120b_t2p_a$i
  [ -f $d/${stem}_state.json ] || continue
  pgrep -f "npu_tool_loop.py --sample $i @prompts_t2p" >/dev/null && { echo "T2P a$i: 실행 중 - 그대로 둠"; continue; }
  [ -f $d/${stem}_turns.json ] && finished $d/${stem}_turns.json && { echo "T2P a$i: 종료됨"; continue; }
  echo "=== $(date '+%F %T') resume" >> $d/$stem.log
  SAMPLES=$i ./run_t2p.sh --resume >/dev/null; echo "T2P a$i: --resume"
done
