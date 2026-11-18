#!/usr/bin/env bash
# After C-7b ends (goal reached or turn 30): stop the Gemma server, serve EXAONE 4.5 33B, test one analyst request
# (a real C-7 input) and start C-8 only if the answer is a parsable diagnosis with reasoning.
# Context: 131072 first, then 114688, then 98304 (the analyst needs the input + 32768 for the answer).
#   tmux new -d -s c8wait "bash fractal_setup/switch_to_c8.sh > ~/logs/c8_switch.log 2>&1"
set -uo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [c8-switch] $*"; }
say "wait for C-7b to end"
until grep -q -E '종료|목표 달성' ~/logs/c71_auto.log 2>/dev/null && ! tmux has-session -t c71 2>/dev/null; do sleep 30; done
say "C-7b: $(tail -1 ~/logs/c71_auto.log)"
grep -q '^DONE rc=0' ~/logs/exaone45_download.log || { say "EXAONE download is not complete - C-8 not started"; exit 1; }
tmux kill-session -t vllmgemma 2>/dev/null; sleep 15
up=0
for len in 131072 114688 98304; do
  for extra in "--language-model-only --kv-cache-dtype fp8" "--kv-cache-dtype fp8"; do
    say "serve exaone (max len $len, extra args: '${extra}')"
    tmux kill-session -t vllmexaone 2>/dev/null; sleep 5
    tmux new -d -s vllmexaone "MAX_LEN=$len EXTRA_ARGS='${extra}' bash $A/fractal_setup/serve_exaone45-33b_gpu.sh > ~/logs/vllm_exaone45.log 2>&1"
    for i in $(seq 1 180); do
      curl -sf 127.0.0.1:8420/v1/models >/dev/null && { up=1; break; }
      tmux has-session -t vllmexaone 2>/dev/null || break
      sleep 5
    done
    [ $up = 1 ] && break 2
    say "did not come up: $(grep -E 'ValueError|Error:' ~/logs/vllm_exaone45.log | tail -1 | cut -c1-300)"
    cp ~/logs/vllm_exaone45.log ~/logs/vllm_exaone45_failed_$(date +%H%M%S).log
    grep -q 'KV cache' ~/logs/vllm_exaone45.log && grep -q 'ValueError' ~/logs/vllm_exaone45.log && break   # memory: next length
  done
done
[ $up = 1 ] || { say "exaone server failed - C-8 not started"; exit 1; }
say "server up (max len $len), GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
cd $A
python3 - <<'PY' || { echo "$(date '+%F %T') [c8-switch] test request failed - C-8 not started"; exit 1; }
import json, time, glob
import agent_loop as AL, c1_multi_analyze as M
U = "http://127.0.0.1:8420/v1"
p = json.load(open(sorted(glob.glob('results/turns/prompt_c7_turn8/*multi*_request.json'))[0]))
p["messages"] = p["messages"][:1]; p["model"] = "exaone-4.5-33b"
print("server context", M.server_context(U), "input tokens", M.count_tokens(U, "exaone-4.5-33b", p["messages"][0]["content"]), flush=True)
t = time.time(); r = AL.post(U, p, 7200)
m = r["choices"][0]["message"]; c = m.get("content") or ""
rs = m.get("reasoning_content") or m.get("reasoning") or ""
print(round(time.time() - t), "s", r.get("usage"), "finish", r["choices"][0].get("finish_reason"), "reasoning chars", len(rs), "content chars", len(c), flush=True)
d = json.loads(c[c.find("{"):c.rfind("}") + 1])
print("keys", list(d)); print("chosen:", str(d.get("chosen"))[:300]); print("direction:", str(d.get("direction"))[:500])
assert d.get("direction") and len(rs) > 200, "no direction or no reasoning"
PY
say "test ok - start C-8"
tmux new -d -s c8 "bash $A/fractal_setup/start_c8.sh > ~/logs/c8_auto.log 2>&1"
say "C-8 started (tmux c8, ~/logs/c8_auto.log)"
