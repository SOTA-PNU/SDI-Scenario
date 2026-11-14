#!/usr/bin/env bash
# REPORT.md 실험 10 — wait until the NPU endpoint answers, then start the A-1 turn loops once.
# Runs detached (setsid), so it survives the Claude session; gives up after 7 days.
cd "$(dirname "$0")"
LOG=results/turns/prompt_a1/supervisor.log
echo "$(date '+%F %T') waiting for http://127.0.0.1:8400/v1/models" >> "$LOG"
t0=$(date +%s)
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8400/v1/models)" = "200" ]; do
  if [ $(( $(date +%s) - t0 )) -gt 604800 ]; then echo "$(date '+%F %T') gave up after 7 days" >> "$LOG"; exit 1; fi
  sleep 30
done
echo "$(date '+%F %T') NPU answering after $(( $(date +%s) - t0 ))s - starting run_a1.sh" >> "$LOG"
curl -s --max-time 10 http://127.0.0.1:8400/v1/models >> "$LOG"; echo >> "$LOG"
./run_a1.sh >> "$LOG" 2>&1
