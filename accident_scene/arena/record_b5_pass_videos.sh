#!/usr/bin/env bash
# Videos of every graded pass in the B-5 series that has none yet (user 2026-10-29: "성공한것들 영상 남겨줘"),
# recorded as replays of the graded trajectories with the path tracer (analysis/c_b4_replay_runner.py, as
# record_b5_replay.sh): the robot is placed on the graded trajectory, M1-M5 follow arena_movers v1, no controller and
# no physics, so the clip is the graded run itself. $1 = PID of a running run_b5*_sequential.sh to wait for first
# (no parallel Isaac runs). The pass list is read after that wait, so a pass of the waited-for run is included.
# ONLY_DIRS (space-separated results/turns folder names) limits the passes; user 2026-10-29: "b-5,b-5-8은 녹화 빼줘".
#   ONLY_DIRS=prompt_b5_11 FRAMES_ROOT=<scratch> setsid nohup bash record_b5_pass_videos.sh <runner pid> > <log> 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
REPO=$(cd ../.. && pwd)
FR=${FRAMES_ROOT:-/tmp/b5_pass_frames}
EVERY=4
WAIT_PID=${1:-}
log() { echo "$(date '+%F %T') $*"; }
if [ -n "$WAIT_PID" ]; then
  log "waiting for runner pid $WAIT_PID"
  while [ -e "/proc/$WAIT_PID" ] && grep -qa "run_b5.*_sequential" "/proc/$WAIT_PID/cmdline" 2>/dev/null; do sleep 20; done
fi
mkdir -p "$FR"
mapfile -t SPECS < <(python3 - <<'PY'
import glob, json, os
only = os.environ.get("ONLY_DIRS", "").split()
for f in sorted(glob.glob("results/turns/prompt_b5*/*_turns.json")):
    d = json.load(open(f)); tdir = os.path.dirname(f)
    if only and os.path.basename(tdir) not in only:
        continue
    name = os.path.basename(tdir).replace("prompt_b5", "B-5").replace("_", "-")
    for t in d.get("turns", []):
        v = t.get("variant")
        if t.get("verdict") == "pass" and not os.path.exists(f"{tdir}/videos/{v}.mp4"):
            run = v.split("_a")[-1].split("_")[0]
            print(f"{tdir}|{v}|{name}|run a{run}, turn {t.get('turn')}")
PY
)
log "passes to record: ${#SPECS[@]}"; printf '  %s\n' "${SPECS[@]}"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r tdir v name what <<< "$spec"
  LOGS=$tdir/isaac_logs; OUT=$tdir/videos; mkdir -p "$OUT"
  traj=$LOGS/${v}_trajectory.csv; res=$LOGS/${v}_result.json
  end=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['metrics']['sim_time_s'])" "$res")
  rec=rec_replay_${v}
  python3 analysis/c_b4_replay_runner.py "$LOGS/${v}_carter_run.py" "${rec}_carter_run.py" >/dev/null || { log "$v: runner not made"; continue; }
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; mkdir -p "$FR/$v"
  log "$v: replay recording to $end s"
  guard=0; t0=$(date +%s)
  while true; do
    env -u CARTER_WHEEL_TOE_DEG CARTER_RECORD=1 CARTER_RECORD_EVERY=$EVERY CARTER_RECORD_SIZE=1920x1080 \
      CARTER_RECORD_DIR="$FR/$v" CARTER_REPO="$REPO" ARENA_MOVERS=v1 REPLAY_PAUSE_S=0.3 \
      REPLAY_TRAJ="$PWD/$traj" REPLAY_END_S="$end" \
      ./carter_run_locked.sh "accident_scene/arena/${rec}_carter_run.py" > "$OUT/$v.replay.carter-run.log" 2>&1 < /dev/null
    rc=$?
    [ $rc -ne 75 ] && break
    guard=$((guard + 1)); sleep 30
  done
  want=$(python3 -c "import sys; print(int(round(float(sys.argv[1]) * 60)) // $EVERY)" "$end")
  got=$(ls "$FR/$v" | wc -l)
  log "$v: $got/$want frames, $(( $(date +%s) - t0 )) s wall, rc=$rc, guard waits $guard"
  if [ "$got" -ne "$want" ]; then log "$v: frame count wrong - frames kept in $FR/$v"; rm -f "${rec}_carter_run.py"; continue; fi
  ANNOT_EVERY=$EVERY ANNOT_MODE=arena ANNOT_LAYOUT=layout_v2.json ANNOT_MOVERS=v1 \
    ANNOT_CTRL="GPU gpt-oss-120b  |  prompt $name  |  $what  |  채점 궤적 재생" \
    python3 annotate_video.py "$FR/$v" "$res" "$traj" "$OUT/$v.mp4" > "$OUT/$v.annotate.log" 2>&1 < /dev/null \
    || { log "$v: annotate failed - frames kept, see $OUT/$v.annotate.log"; rm -f "${rec}_carter_run.py"; continue; }
  python3 - "$v" "$traj" "$res" "$end" "$got" "$EVERY" "$OUT/$v.replay.json" <<'PY'
import json, sys, time
v, traj, res, end, n, every, out = sys.argv[1:8]
json.dump({"variant": v, "what": "replay of the graded run: robot on the graded trajectory, M1-M5 from arena_movers v1",
           "trajectory": traj, "result": res, "end_s": float(end), "frames": int(n), "record_every": int(every),
           "render": "path tracer 8 spp + OptiX denoiser, 1920x1080, 0.3 s GPU pause per frame",
           "script": "record_b5_pass_videos.sh", "recorded": time.strftime("%Y-%m-%d %H:%M")}, open(out, "w"), indent=1, ensure_ascii=False)
PY
  log "$v: video $OUT/$v.mp4 ($(du -h "$OUT/$v.mp4" | cut -f1))"
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; rm -f "${rec}_carter_run.py"
done
log "done"
