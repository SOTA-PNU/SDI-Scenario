#!/usr/bin/env bash
# B-5 videos (실험 25, user 2026-10-28): one pass (a0 turn 1) and one failure (a1 turn 1), recorded as replays of
# the graded runs with the path tracer (analysis/c_b4_replay_runner.py, as record_b4_replay.sh). Waits for the
# B-5 runs to finish first, so no Isaac run of the experiment is held up.
# a1 turn 1 sat within 0.21 m from 17.7 s to the 600 s timeout, spinning back and forth; the clip stops at 90 s
# and says so. REPLAY_PAUSE_S idles the GPU between frames (the server died once during a path-traced replay).
#   FRAMES_ROOT=<scratch dir> setsid nohup bash record_b5_replay.sh > results/turns/prompt_b5/videos/record.log 2>&1 &
set -u
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
TURNS=results/turns/prompt_b5
LOGS=$TURNS/isaac_logs
OUT=$TURNS/videos
FR=${FRAMES_ROOT:-/tmp/b5_replay_frames}
EVERY=4
mkdir -p "$OUT" "$FR"
log() { echo "$(date '+%F %T') $*"; }
# (the wait for the B-5 runs was removed: the user stopped B-5 a1 at 16:5x and the videos were postponed to after B-5-3)
log "recording"
for spec in "gpu_gpt_oss_120b_pB5_a0|0|run a0, turn 1|" \
            "gpu_gpt_oss_120b_pB5_a1|90|run a1, turn 1|90초에서 자름: 이후 600초까지 같은 자리(0.2 m 안)에서 멈춤과 회전을 되풀이하다 시간 초과, 충돌 0"; do
  IFS='|' read -r v cut what note <<< "$spec"
  traj=$LOGS/${v}_trajectory.csv; res=$LOGS/${v}_result.json
  [ -f "$OUT/$v.mp4" ] && { log "$v: video exists - skipped"; continue; }
  end=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['metrics']['sim_time_s'])" "$res")
  [ "$cut" != 0 ] && end=$cut
  rec=rec_replay_${v}
  python3 analysis/c_b4_replay_runner.py "$LOGS/${v}_carter_run.py" "${rec}_carter_run.py" >/dev/null || { log "$v: runner not made"; continue; }
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; mkdir -p "$FR/$v"
  log "$v: replay recording to $end s"
  guard=0; t0=$(date +%s)
  while true; do
    env -u CARTER_WHEEL_TOE_DEG CARTER_RECORD=1 CARTER_RECORD_EVERY=$EVERY CARTER_RECORD_SIZE=1920x1080 \
      CARTER_RECORD_DIR="$FR/$v" CARTER_REPO="$REPO" ARENA_MOVERS=v1 REPLAY_PAUSE_S=0.3 \
      REPLAY_TRAJ="$PWD/$traj" REPLAY_END_S="$end" \
      ./carter_run_locked.sh "accident_scene/arena/${rec}_carter_run.py" > "$OUT/$v.replay.carter-run.log" 2>&1
    rc=$?
    [ $rc -ne 75 ] && break
    guard=$((guard + 1)); sleep 30
  done
  want=$(python3 -c "import sys; print(int(round(float(sys.argv[1]) * 60)) // $EVERY)" "$end")
  got=$(ls "$FR/$v" | wc -l)
  log "$v: $got/$want frames, $(( $(date +%s) - t0 )) s wall, rc=$rc, guard waits $guard"
  if [ "$got" -ne "$want" ]; then log "$v: frame count wrong - frames kept in $FR/$v"; rm -f "${rec}_carter_run.py"; continue; fi
  ANNOT_EVERY=$EVERY ANNOT_MODE=arena ANNOT_LAYOUT=layout_v2.json ANNOT_MOVERS=v1 \
    ANNOT_MAX_S=$([ "$cut" != 0 ] && echo "$cut") ANNOT_CUT_NOTE="$note" \
    ANNOT_CTRL="GPU gpt-oss-120b  |  prompt B-5  |  $what  |  채점 궤적 재생" \
    python3 annotate_video.py "$FR/$v" "$res" "$traj" "$OUT/$v.mp4" > "$OUT/$v.annotate.log" 2>&1 \
    || { log "$v: annotate failed - frames kept, see $OUT/$v.annotate.log"; rm -f "${rec}_carter_run.py"; continue; }
  python3 - "$v" "$traj" "$res" "$end" "$got" "$EVERY" "$OUT/$v.replay.json" <<'PY'
import json, sys, time
v, traj, res, end, n, every, out = sys.argv[1:8]
json.dump({"variant": v, "what": "replay of the graded run: robot on the graded trajectory, M1-M5 from arena_movers v1",
           "trajectory": traj, "result": res, "end_s": float(end), "frames": int(n), "record_every": int(every),
           "render": "path tracer 8 spp + OptiX denoiser, 1920x1080, 0.3 s GPU pause per frame",
           "script": "record_b5_replay.sh", "recorded": time.strftime("%Y-%m-%d %H:%M")}, open(out, "w"), indent=1, ensure_ascii=False)
PY
  log "$v: video $OUT/$v.mp4 ($(du -h "$OUT/$v.mp4" | cut -f1))"
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; rm -f "${rec}_carter_run.py"
done
python3 analysis/c_b_videos_collect.py
log "record_b5_replay.sh done"
