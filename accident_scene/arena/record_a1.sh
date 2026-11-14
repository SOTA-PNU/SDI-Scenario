#!/usr/bin/env bash
# REPORT.md 실험 10-11 (prompts A-1, A-2): Isaac Sim videos of graded turns - visualization only.
#
#   FRAMES_ROOT=<scratch dir> bash record_a1.sh npu_gpt_oss_120b_pA1_a0 npu_gpt_oss_120b_pA1_a0_t2 ...
#   TURNS=results/turns/prompt_a2 LABEL="prompt A-2" FRAMES_ROOT=... bash record_a1.sh npu_gpt_oss_120b_pA2_a0 ...
#
# Project C (B prompts, 실험 19-): set ARENA_MOVERS=v1 so the recording runs on the grading map with the
# moving obstacles (labelled M1-M5 in the video); REC_EVERY (physics steps per frame, default 4) and
# REC_MODEL (caption, default "NPU gpt-oss-120b") override the A-series defaults.
#
# TURNS (default results/turns/prompt_a1) is the run directory, LABEL the caption. A turn-1
# graded result that was invalid (A-2: two gradings ran side by side) is compared against
# its serial re-grade, $TURNS/regrade/regrade_<VARIANT>_t1_result.json, when that exists.
#
# For each VARIANT (turn 1 = <stem>, turn n = <stem>_t<n>):
#  * wait until that turn is in <stem>_turns.json; a turn that never reached Isaac
#    (static_dq / empty_response) has no run to record and is skipped
#  * copy the graded submission (isaac_logs/<VARIANT>_carter_run.py) to
#    rec_<VARIANT>_carter_run.py, so the recording run writes rec_* result files and
#    never touches the graded ones
#  * record it (path tracer, top-down, 1920x1080, every 4th step = 15 fps), waiting on
#    the carter-run serial guard (exit 75) like turn_loop.py does
#  * draw labels from layout_v2.json and the recording run's own result/trajectory,
#    and compare its verdict with the graded one (a recording run can differ by a few
#    cm: the per-frame wall time changes how many frames the ROS 2 command delay spans)
#  * delete the frames and the rec_ copy
# Output: results/turns/prompt_a1/videos/<VARIANT>.mp4 (+ .rec_result.json,
# .rec_trajectory.csv, .rec_compare.json, .rec.carter-run.log), log in record.log.
set -u
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
TURNS=${TURNS:-results/turns/prompt_a1}
LABEL=${LABEL:-prompt A-1}
SRC=$TURNS/isaac_logs
OUT=$TURNS/videos
FR=${FRAMES_ROOT:-${TMPDIR:-/tmp}/a1_frames}
mkdir -p "$OUT" "$FR"
log() { echo "$(date '+%F %T') $*"; }

turn_state() {  # prints the verdict of turn N of STEM, or nothing if not there yet
  python3 - "$1" "$2" "$TURNS" <<'PY'
import json, sys
from pathlib import Path
f = Path(f"{sys.argv[3]}/{sys.argv[1]}_turns.json")
if f.exists():
    for t in json.loads(f.read_text()).get("turns", []):
        if t.get("turn") == int(sys.argv[2]):
            print(t.get("verdict")); break
PY
}

for v in "$@"; do
  if [[ $v =~ ^(.*)_t([0-9]+)$ ]]; then stem=${BASH_REMATCH[1]}; n=${BASH_REMATCH[2]}; else stem=$v; n=1; fi
  [ -f "$OUT/$v.mp4" ] && { log "$v: video exists - skipped"; continue; }
  waited=0
  until st=$(turn_state "$stem" "$n"); [ -n "$st" ]; do
    [ $waited -ge 43200 ] && break
    sleep 60; waited=$((waited + 60))
  done
  if [ -z "$st" ]; then log "$v: turn $n not graded within 12 h - skipped"; continue; fi
  if [ "$st" = static_dq ] || [ "$st" = empty_response ]; then
    log "$v: turn $n is $st - it never ran in Isaac, nothing to record"; continue
  fi
  if [ ! -f "$SRC/${v}_carter_run.py" ]; then log "$v: graded submission missing in $SRC - skipped"; continue; fi
  rec="rec_${v}"
  cp "$SRC/${v}_carter_run.py" "${rec}_carter_run.py"
  # REC_TIMEOUT_S: record only the first N sim seconds of a run that stands still afterwards
  # (A-2 a1 turn 2 stopped at 17.4 s and stood until 600 s); the caption says so
  [ -n "${REC_TIMEOUT_S:-}" ] && sed -i "s/timeout_s=600.0,/timeout_s=${REC_TIMEOUT_S},/" "${rec}_carter_run.py"
  rm -rf "${FR:?}/$v"; mkdir -p "$FR/$v"
  log "$v: recording (graded verdict $st)"
  guard=0; t0=$(date +%s)
  while true; do
    t0=$(date +%s)
    env -u CARTER_WHEEL_TOE_DEG CARTER_RECORD=1 CARTER_RECORD_EVERY="${REC_EVERY:-4}" CARTER_RECORD_SIZE=1920x1080 \
      CARTER_RECORD_DIR="$FR/$v" CARTER_REPO="$REPO" \
      ./carter_run_locked.sh "accident_scene/arena/${rec}_carter_run.py" \
      > "$OUT/$v.rec.carter-run.log" 2>&1
    rc=$?
    [ $rc -ne 75 ] && break
    guard=$((guard + 1)); sleep 30
  done
  t1=$(date +%s)
  for k in result.json trajectory.csv; do
    [ -f "results/${rec}_$k" ] && mv "results/${rec}_$k" "$OUT/$v.rec_$k"
  done
  if [ ! -f "$OUT/$v.rec_result.json" ]; then
    log "$v: recording run wrote no result (rc=$rc) - see $OUT/$v.rec.carter-run.log"
    rm -f "${rec}_carter_run.py"; continue
  fi
  ref="$SRC/${v}_result.json"
  [ "$n" = 1 ] && [ -f "$TURNS/regrade/regrade_${v}_t1_result.json" ] && ref="$TURNS/regrade/regrade_${v}_t1_result.json"
  python3 - "$v" "$ref" "$OUT/$v.rec_result.json" "$OUT/$v.rec_compare.json" \
      "$((t1 - t0))" "$guard" <<'PY'
import json, sys
v, graded, rec, out, wall, guard = sys.argv[1:7]
def pick(p):
    r = json.load(open(p)); m = r["metrics"]
    return {"verdict": r["verdict"], "final_dist_m": m["final_dist_m"], "collision_count": m["collision_count"],
            "collision_partners": m["collision_partners"], "first_collision_t": m["first_collision_t"],
            "time_to_goal_s": m["time_to_goal_s"], "sim_time_s": m["sim_time_s"], "final_pose": m["final_pose"]}
g, r = pick(graded), pick(rec)
same = {k: g[k] == r[k] for k in ("verdict", "collision_partners")}
cmp = {"variant": v, "graded": g, "recorded": r, "same_verdict_and_partners": all(same.values()),
       "final_dist_diff_m": round(abs(g["final_dist_m"] - r["final_dist_m"]), 4),
       "record_wall_s": int(wall), "serial_guard_waits": int(guard)}
open(out, "w").write(json.dumps(cmp, indent=1, ensure_ascii=False))
print(f"graded {g['verdict']} dist={g['final_dist_m']} col={g['collision_count']} {g['collision_partners']} | "
      f"recorded {r['verdict']} dist={r['final_dist_m']} col={r['collision_count']} {r['collision_partners']} | "
      f"same={cmp['same_verdict_and_partners']} wall={wall}s")
PY
  run=${stem##*_}
  ANNOT_EVERY="${REC_EVERY:-4}" ANNOT_MODE=arena ANNOT_LAYOUT=layout_v2.json ANNOT_MOVERS="${ARENA_MOVERS:-}" \
    ANNOT_CTRL="${REC_MODEL:-NPU gpt-oss-120b}  |  $LABEL  |  run $run, turn $n" \
    python3 annotate_video.py "$FR/$v" "$OUT/$v.rec_result.json" "$OUT/$v.rec_trajectory.csv" "$OUT/$v.mp4" \
    > "$OUT/$v.annotate.log" 2>&1 && log "$v: video $OUT/$v.mp4 ($(du -h "$OUT/$v.mp4" | cut -f1))" \
    || log "$v: annotate failed - frames kept in $FR/$v, see $OUT/$v.annotate.log"
  [ -f "$OUT/$v.mp4" ] && rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"
  rm -f "${rec}_carter_run.py"
done
log "record_a1.sh done: $*"
