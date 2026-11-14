#!/usr/bin/env bash
# Project C videos (user, 2026-10-28): one representative graded run per prompt B-1, B-2, B-2-1, B-3,
# and for B-4 one failure and one success. All are final gradings on the moving-obstacle map, so the
# recording runs with ARENA_MOVERS=v1 too. record_a1.sh copies each graded file to rec_*, records it
# top-down, compares the recording's verdict with the graded one, and writes <TURNS>/videos/<VARIANT>.mp4.
# B-4 runs steer by the RTX lidar: record them with the real-time renderer the gradings used (the path
# tracer changes the global render mode); the others do not use lidar values and get the path tracer.
set -u
cd "$(dirname "$0")"
FR=${FRAMES_ROOT:?set FRAMES_ROOT to a scratch dir}
export ARENA_MOVERS=v1 REC_MODEL="GPU gpt-oss-120b"
TURNS=results/turns/prompt_b4 LABEL="prompt B-4" CARTER_RECORD_RENDER=realtime REC_EVERY=4 \
  FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB4_a1_t2
TURNS=results/turns/prompt_b4 LABEL="prompt B-4" CARTER_RECORD_RENDER=realtime REC_EVERY=12 \
  FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB4_a1
TURNS=results/turns/prompt_b1 LABEL="prompt B-1" FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB1_a1_t2
TURNS=results/turns/prompt_b2_all LABEL="prompt B-2" FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB2all_a0_t2
TURNS=results/turns/prompt_b2_1 LABEL="prompt B-2-1" FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB21_a1_t2
TURNS=results/turns/prompt_b3 LABEL="prompt B-3" FRAMES_ROOT="$FR" bash record_a1.sh gpu_gpt_oss_120b_pB3_a0_t2
echo "$(date '+%F %T') record_b_videos.sh done"
