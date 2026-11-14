#!/usr/bin/env bash
# gpt-oss-120b on this GPU for project C (REPORT.md 실험 19-). Same flags as the furio 20b service
# (~/furio-gpu/logs/restore_20b_service.sh) except model, port 8420 and GPU share 0.80 (Isaac needs ~6.3 GB).
# 0.85 until 2026-10-27 21:40: an Isaac grading then started short of GPU memory while another user's
# job held ~4 GB, came up without lidar/camera topics and was still graded (REPORT 실험 23). 0.80 leaves
# ~15 GB free beside that job; the KV cache (~170k tokens) still covers our one ~40k-token request.
cd ~/furio-gpu || exit 1
exec venv/bin/vllm serve /home/jun/models/gpt-oss-120b --host 127.0.0.1 --port 8420 \
  --served-model-name gpt-oss-120b --max-model-len 131072 --gpu-memory-utilization ${GPU_UTIL:-0.80} \
  --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser openai \
  --reasoning-parser openai_gptoss
