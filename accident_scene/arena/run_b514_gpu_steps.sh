#!/usr/bin/env bash
# Run outside the preparation sandbox after vLLM is started by the operator.
# The accuracy helper runs A (follow) and B (stationary) through carter_run_locked.sh.
set -u
set -o pipefail
export CARTER_CONTACT_ALL_LINKS=1
cd "$(dirname "$0")" || exit 1
D=results/turns/prompt_b5_14
mkdir -p "$D" || exit 1
exec 7>"$D/.gpu_steps.lock"
flock -n 7 || { echo 'B-5-14 GPU steps are already active' >&2; exit 75; }
if [ -f "$D/gpu_steps.log" ]; then
  stamp=$(date '+%Y%m%d_%H%M%S')
  cp -p "$D/gpu_steps.log" "results/archive_gpu_steps_before_b514_v2_${stamp}_$$.log" || exit 1
fi
log_stage() { echo "$(date '+%F %T') $*" >> "$D/gpu_steps.log"; }
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

verdict=results/analysis/b514_accuracy_isaac/verdict.txt
# A previous verdict must never authorize a new run. Preserve it for review.
if [ -e "$verdict" ]; then
  log_stage "smoke FAIL existing_verdict=$verdict; done"
  exit 73
fi
log_stage 'smoke start runs=A_follow,B_stationary sim_seconds=60 launcher=carter_run_locked.sh median_abs_error_m<=0.15 false_overlap<=0.05 median_ms<=2 errors=0'
python3 analysis/b514_accuracy_isaac_check.py
smoke_rc=$?
first_line=''
if [ -f "$verdict" ]; then IFS= read -r first_line < "$verdict" || :; fi
if [ "$smoke_rc" -ne 0 ] || [ "$first_line" != PASS ]; then
  log_stage "smoke FAIL exit=$smoke_rc verdict=${first_line:-missing}; done"
  exit 1
fi
log_stage 'smoke PASS exit=0'
log_stage 'sequential start samples=a0,a1; a2,a3_if_both_pass'
bash run_b514_sequential.sh
run_rc=$?
log_stage "sequential exit=$run_rc; done"
exit "$run_rc"
