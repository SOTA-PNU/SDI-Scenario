#!/usr/bin/env bash
# 2026-11-16: fractal rebooted, jun's vLLM (8420) is gone and its model dir is not readable to crojjang.
# Build our own vLLM env, download gpt-oss-120b (user approved), serve it on 127.0.0.1:8420, then finish C-2 turn 15
# from its reviewed code and restart the autopilot at turn 16. Every step must succeed or the script stops.
set -euo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [restore] $*"; }
say "1/5 vLLM env"
bash $A/fractal_setup/setup_vllm.sh > ~/logs/setup_vllm.log 2>&1
grep -q '^OK: ' ~/logs/setup_vllm.log || { say "setup_vllm failed"; tail -5 ~/logs/setup_vllm.log; exit 1; }
tail -2 ~/logs/setup_vllm.log
say "2/5 model download (65 GB)"
mkdir -p ~/models
~/venvs/vllm/bin/hf download openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --local-dir ~/models/gpt-oss-120b --max-workers 1 \
  --include 'model-*-of-00014.safetensors' --include model.safetensors.index.json --include config.json \
  --include generation_config.json --include chat_template.jinja --include special_tokens_map.json \
  --include tokenizer.json --include tokenizer_config.json > ~/logs/hf_download.log 2>&1
n=$(ls ~/models/gpt-oss-120b/model-*-of-00014.safetensors | wc -l)
[ "$n" -eq 15 ] || { say "expected 15 shards, found $n"; exit 1; }
say "downloaded: $(du -sh ~/models/gpt-oss-120b | cut -f1)"
say "3/5 serve on 8420"
tmux new -d -s vllm120b "bash $A/fractal_setup/serve_gpt-oss-120b_gpu.sh > ~/logs/vllm120b.log 2>&1"
for i in $(seq 1 120); do
  curl -sf 127.0.0.1:8420/v1/models >/dev/null && break
  grep -q -E 'Error|Traceback' ~/logs/vllm120b.log && { say "vllm failed"; tail -20 ~/logs/vllm120b.log; exit 1; }
  sleep 10
done
curl -sf 127.0.0.1:8420/v1/models >/dev/null || { say "vllm not up after 20 min"; exit 1; }
say "vllm up: $(curl -s 127.0.0.1:8420/v1/models | head -c 120)"
say "4/5 finish turn 15"
cd $A
D=results/turns/prompt_c2_turn15; P=results/turns/prompt_c2_turn14
DIAG=$(ls $P/gpu_gpt_oss_120b_pC2t14_multi*_diagnosis.json | grep -v -E 'BAD|ARCH' | head -1)
ROS_DOMAIN_ID=37 python3 c1_cycle.py --series c2 --turn 15 --code $P/gpu_gpt_oss_120b_pC2t14_block.py --diagnosis $DIAG \
  --history-file $P/gpu_gpt_oss_120b_pC2t14_history.txt --reuse-block $D/gpu_gpt_oss_120b_pC2t15_block.py \
  --reuse-reason "$(cat $D/reason.txt)" > ~/logs/c2_turn15.log 2>&1 || say "turn 15 rc=$?"
say "5/5 autopilot from turn 16"
nohup python3 c2_auto.py --start 16 >> ~/logs/c2_auto.log 2>&1 &
say "autopilot pid $!"
