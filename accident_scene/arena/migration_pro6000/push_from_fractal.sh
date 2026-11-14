#!/usr/bin/env bash
# Run on fractal: push the B-experiment files to the PRO 6000 server (ssh alias pro6000, key
# ~/.ssh/id_ed25519_pro6000). Rate-limited for the lab uplink; re-run to resume after an interruption.
#   setsid nohup bash push_from_fractal.sh > /dev/null 2>&1 &
set -u
LOG=${LOG:-$HOME/.claude/jobs/fdd8b179/tmp/pro6000_rsync.log}
BW=${BW:-5000}            # KiB/s, about 40 Mbit/s
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "repo: ~/SDI-Scenario without the videos and frame images under results/"
rsync -aH --partial --mkpath --bwlimit="$BW" --info=stats1 \
  --exclude='/accident_scene/arena/results/**.mp4' --exclude='/accident_scene/arena/results/**.png' \
  --exclude='/accident_scene/arena/results/**.jpg' --exclude='__pycache__/' \
  "$HOME/SDI-Scenario/" pro6000:SDI-Scenario/ >> "$LOG" 2>&1
rc_repo=$?; log "repo rc=$rc_repo"
log "carter_ws: Isaac Sim 4.5.0.0 micromamba env + carter-run"
rsync -aH --partial --mkpath --bwlimit="$BW" --info=stats1 "$HOME/carter_ws/" pro6000:carter_ws/ >> "$LOG" 2>&1
rc_ws=$?; log "carter_ws rc=$rc_ws"
# rc 20 = stopped by a signal (e.g. the PRO 6000 side ended it); --partial keeps what arrived, re-run to resume
if [ "$rc_repo" = 0 ] && [ "$rc_ws" = 0 ]; then log "all done"; else log "NOT complete (repo rc=$rc_repo, carter_ws rc=$rc_ws) - re-run to resume"; fi
