#!/usr/bin/env bash
# Re-record the two B-4 videos (실험 24, run a1 turns 1 and 2) as replays of the graded runs.
# User 2026-10-28: the realtime-rendered recordings were noisy, stalled and did not play on a Mac.
#
# The B-4 controllers steer by the RTX lidar, so a live re-run had to use the real-time renderer
# (heavy speckle in this headless capture path), and the a1 turn-1 re-run did not reproduce the
# graded box-06 contact. A replay needs neither: analysis/c_b4_replay_runner.py turns the graded
# file into a runner that stops physics, places the robot on the graded trajectory and M1-M5 on
# their v1 lanes at each frame's mission time, and renders every frame once with the path tracer.
# annotate_video.py then labels the frames from the graded result and trajectory.
#
#   FRAMES_ROOT=<scratch dir> bash record_b4_replay.sh
# Output: results/turns/prompt_b4/videos/<variant>.mp4 (+ _share.mp4 4x for turn 1,
# .replay.json, .replay.carter-run.log, .annotate.log); the realtime-era files of each variant
# are moved, not overwritten, to videos/archive_realtime_20261028/. Then results/b_videos is rebuilt.
set -u
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
TURNS=results/turns/prompt_b4
LOGS=$TURNS/isaac_logs
OUT=$TURNS/videos
ARCH=$OUT/archive_realtime_20261028
FR=${FRAMES_ROOT:-/tmp/b4_replay_frames}
EVERY=4
mkdir -p "$OUT" "$ARCH" "$FR"
log() { echo "$(date '+%F %T') $*"; }

for v in gpu_gpt_oss_120b_pB4_a1_t2 gpu_gpt_oss_120b_pB4_a1; do
  case $v in *_t2) turn=2 ;; *) turn=1 ;; esac
  traj=$LOGS/${v}_trajectory.csv; res=$LOGS/${v}_result.json
  end=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['metrics']['sim_time_s'])" "$res")
  for f in "$OUT/$v.mp4" "$OUT/${v}_share.mp4" "$OUT/${v}_mac.mp4" "$OUT/$v".rec_* "$OUT/$v.annotate.log"; do
    [ -e "$f" ] && mv "$f" "$ARCH/"
  done
  rec=rec_replay_${v}
  python3 analysis/c_b4_replay_runner.py "$LOGS/${v}_carter_run.py" "${rec}_carter_run.py" >/dev/null || { log "$v: runner not made"; continue; }
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; mkdir -p "$FR/$v"
  log "$v: replay recording to $end s"
  guard=0; t0=$(date +%s)
  while true; do
    env -u CARTER_WHEEL_TOE_DEG CARTER_RECORD=1 CARTER_RECORD_EVERY=$EVERY CARTER_RECORD_SIZE=1920x1080 \
      CARTER_RECORD_DIR="$FR/$v" CARTER_REPO="$REPO" ARENA_MOVERS=v1 \
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
    ANNOT_CTRL="GPU gpt-oss-120b  |  prompt B-4  |  run a1, turn $turn  |  채점 궤적 재생" \
    python3 annotate_video.py "$FR/$v" "$res" "$traj" "$OUT/$v.mp4" > "$OUT/$v.annotate.log" 2>&1 \
    || { log "$v: annotate failed - frames kept, see $OUT/$v.annotate.log"; rm -f "${rec}_carter_run.py"; continue; }
  python3 - "$v" "$traj" "$res" "$end" "$got" "$EVERY" "$OUT/$v.replay.json" <<'PY'
import json, sys, time
v, traj, res, end, n, every, out = sys.argv[1:8]
json.dump({"variant": v, "what": "replay of the graded run: robot on the graded trajectory, M1-M5 from arena_movers v1",
           "trajectory": traj, "result": res, "end_s": float(end), "frames": int(n), "record_every": int(every),
           "render": "path tracer 8 spp + OptiX denoiser, 1920x1080", "script": "record_b4_replay.sh",
           "recorded": time.strftime("%Y-%m-%d %H:%M")}, open(out, "w"), indent=1, ensure_ascii=False)
PY
  if [ "$turn" = 1 ]; then
    ffmpeg -y -v error -i "$OUT/$v.mp4" -vf "setpts=0.25*PTS" -r 30 -c:v libx264 -preset slow -crf 23 \
      -pix_fmt yuv420p -movflags +faststart "$OUT/${v}_share.mp4" && log "$v: 4x copy $(du -h "$OUT/${v}_share.mp4" | cut -f1)"
  fi
  log "$v: video $OUT/$v.mp4 ($(du -h "$OUT/$v.mp4" | cut -f1))"
  rm -rf "${FR:?}/$v" "${FR:?}/${v}_annot"; rm -f "${rec}_carter_run.py"
done
mkdir -p results/b_videos/archive_realtime_20261028
for f in results/b_videos/5_B-4_실패_gpu_gpt_oss_120b_pB4_a1.mp4 results/b_videos/6_B-4_성공_gpu_gpt_oss_120b_pB4_a1_t2.mp4; do
  [ -e "$f" ] && mv "$f" results/b_videos/archive_realtime_20261028/
done
python3 analysis/c_b_videos_collect.py
log "record_b4_replay.sh done"
