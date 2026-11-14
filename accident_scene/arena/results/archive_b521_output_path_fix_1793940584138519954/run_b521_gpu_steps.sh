#!/usr/bin/env bash
# Run outside the preparation sandbox after vLLM is started by the operator.
# The v8 200-second choose_command gate runs once through carter_run_locked.sh.
set -u
set -o pipefail
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_20
mkdir -p "$D" || exit 1
exec 7>"$D/.gpu_steps.lock"
flock -n 7 || { echo 'B-5-21 GPU steps are already active' >&2; exit 75; }
if [ -f "$D/gpu_steps.log" ]; then
  stamp=$(date '+%Y%m%d_%H%M%S')
  cp -p "$D/gpu_steps.log" "results/archive_gpu_steps_before_b521_${stamp}_$$.log" || exit 1
fi
log_stage() { echo "$(date '+%F %T') $*" >> "$D/gpu_steps.log"; }
# Only executed on the GPU host. The preparation sandbox cannot see its processes.
log_stage 'b520_wait start interval_s=30'
while true; do
  previous_count=$(
    ps -eo args | grep -c '[r]un_b520_gpu_steps.sh'
    previous_status=("${PIPESTATUS[@]}")
    [ "${previous_status[0]}" -eq 0 ] && [ "${previous_status[1]}" -le 1 ]
  )
  previous_rc=$?
  if [ "$previous_rc" -ne 0 ] || ! [[ "$previous_count" =~ ^[0-9]+$ ]]; then
    log_stage 'b520_wait FAIL process scan error; done'
    exit 1
  fi
  if [ "$previous_count" -eq 0 ]; then
    log_stage 'b520_wait PASS count=0'
    break
  fi
  log_stage "b520_wait WAIT count=$previous_count interval_s=30"
  if ! sleep 30; then
    log_stage 'b520_wait FAIL sleep interrupted; done'
    exit 1
  fi
done
agent_count=$(
  ps -eo args | grep -c '[a]gent_loop.py'
  scan_status=("${PIPESTATUS[@]}")
  [ "${scan_status[0]}" -eq 0 ] && [ "${scan_status[1]}" -le 1 ]
)
scan_rc=$?
if [ "$scan_rc" -ne 0 ] || ! [[ "$agent_count" =~ ^[0-9]+$ ]]; then
  log_stage 'agent_check FAIL process scan error; done'
  exit 1
fi
if [ "$agent_count" -ne 0 ]; then
  log_stage "agent_check BUSY count=$agent_count; done"
  exit 75
fi
log_stage 'agent_check PASS count=0'
MAX_WAIT_S=1800
deadline=$((SECONDS + MAX_WAIT_S))
log_stage 'models_wait start endpoint=http://127.0.0.1:8420/v1/models max_s=1800'
ready=0
while [ "$SECONDS" -lt "$deadline" ]; do
  remaining=$((deadline - SECONDS))
  request_timeout=5
  if [ "$remaining" -lt "$request_timeout" ]; then request_timeout=$remaining; fi
  if curl --fail --silent --max-time "$request_timeout" http://127.0.0.1:8420/v1/models |
      python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(m.get("id")=="gpt-oss-120b" for m in d.get("data", [])) else 1)' 2>/dev/null; then
    ready=1
    break
  fi
  remaining=$((deadline - SECONDS))
  if [ "$remaining" -le 0 ]; then break; fi
  delay=10
  if [ "$remaining" -lt "$delay" ]; then delay=$remaining; fi
  sleep "$delay"
done
if [ "$ready" -ne 1 ]; then
  log_stage 'models_wait FAIL timeout=1800s; done'
  exit 124
fi
log_stage 'models_wait PASS model=gpt-oss-120b'

verdict=results/analysis/b521_gate_isaac/verdict.txt
# The gate archives its complete previous output only when a new locked run
# actually starts. A unique token binds both result files to this invocation;
# even a zero exit with an untouched old PASS must not launch the model loop.
gate_run_id=$(python3 -c 'import uuid; print(uuid.uuid4().hex)') || exit 1
log_stage "gate start run_id=$gate_run_id base=v8 sim_seconds=200 launcher=carter_run_locked.sh collisions=0 median_ms<=3 errors=0"
B521_GATE_RUN_ID="$gate_run_id" python3 analysis/b521_gate_isaac.py
smoke_rc=$?
first_line=''
if [ -f "$verdict" ]; then IFS= read -r first_line < "$verdict" || :; fi
if [ "$smoke_rc" -ne 0 ] || [ "$first_line" != PASS ]; then
  log_stage "gate FAIL exit=$smoke_rc verdict=${first_line:-missing}; done"
  exit 1
fi
if ! python3 - "$gate_run_id" <<'PY'
import json, sys
from pathlib import Path
try:
    root = Path("results/analysis/b521_gate_isaac")
    verdict_lines = (root / "verdict.txt").read_text().splitlines()
    launch = json.loads((root / "launch.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert len(verdict_lines) >= 2 and verdict_lines[1] == "RUN_ID=" + sys.argv[1]
    assert launch.get("run_id") == sys.argv[1]
    assert summary.get("launch", {}).get("run_id") == sys.argv[1]
    assert summary.get("verdict") == "PASS"
    assert launch.get("base_file") == "base_carter_run_ros_map_lidartime_objects_check_v8.py"
except (OSError, ValueError, AssertionError, AttributeError, TypeError):
    sys.exit(1)
PY
then
  log_stage "gate FAIL stale_or_inconsistent_result run_id=$gate_run_id; done"
  exit 1
fi
log_stage "gate PASS exit=0 run_id=$gate_run_id"
log_stage 'sequential start samples=a0,a1; a2,a3_if_both_pass'
bash run_b521_sequential.sh
run_rc=$?
log_stage "sequential exit=$run_rc; done"
exit "$run_rc"
