#!/bin/bash
# NPU generation: one model resident at a time, both tracks back to back.
# Card swaps cost ~80 s (tp32 takes all four cards, LRU evicts), so the loop
# never revisits a model.  Qwen3-32B is first because it is already resident.
set -u
ARENA=/home/jun/SDI-Scenario/accident_scene/arena
T=/home/jun/.claude/jobs/fdd8b179/tmp
LOG=$T/npu_generation.log
BASE=http://127.0.0.1:8400/v1
K=5

for MODEL in \
  Qwen3-32B-FP8 \
  EXAONE-4.0-32B-FP8 \
  gpt-oss-120b \
  Solar-Open-100B-NVFP4A16 ; do

  TAG="npu_$(echo "$MODEL" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\+/_/g;s/_$//')"
  echo "[$(date +%H:%M:%S)] ===== $MODEL (tag=$TAG)" >> $LOG

  # warm the model so the card swap is not charged to sample 0
  S=$(date +%s)
  curl -s -m 600 $BASE/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2000,\"temperature\":0}" \
    -o /dev/null -w "  warmup http=%{http_code} %{time_total}s\n" >> $LOG 2>&1
  echo "[$(date +%H:%M:%S)] warm in $(( $(date +%s) - S ))s" >> $LOG

  for TRACK in a b; do
    python3 $ARENA/generate.py "$MODEL" --track $TRACK --k $K --tag "$TAG" \
        --base $BASE >> $LOG 2>&1
  done
  echo "[$(date +%H:%M:%S)] DONE $MODEL" >> $LOG
done
echo "[$(date +%H:%M:%S)] ALL NPU GENERATION DONE" >> $LOG
