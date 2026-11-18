#!/usr/bin/env bash
# After C-6 ends and the Gemma download is complete: stop the Qwen server, serve Gemma 4 31B, test one analyst
# request (a real C-6 input) and start C-7 only if the answer is a parsable diagnosis with reasoning.
#   tmux new -d -s c7wait "bash fractal_setup/switch_to_c7.sh > ~/logs/c7_switch.log 2>&1"
set -uo pipefail
A=~/SDI-Scenario/accident_scene/arena
say() { echo "$(date '+%F %T') [c7-switch] $*"; }
say "wait for C-6 to end and for the download"
until grep -q -E '종료|목표 달성' ~/logs/c6_auto.log && ! tmux has-session -t c6 2>/dev/null; do sleep 30; done
until grep -q '^DONE rc=0' ~/logs/gemma4_download.log; do
  grep -q '^DONE' ~/logs/gemma4_download.log && { say "download failed"; exit 1; }
  sleep 30
done
say "C-6: $(tail -1 ~/logs/c6_auto.log)"
tmux kill-session -t vllmqwen 2>/dev/null; sleep 15
up=0
for extra in "--language-model-only" ""; do
  say "serve gemma (extra args: '${extra}')"
  tmux kill-session -t vllmgemma 2>/dev/null; sleep 5
  tmux new -d -s vllmgemma "EXTRA_ARGS='${extra}' bash $A/fractal_setup/serve_gemma4-31b_gpu.sh > ~/logs/vllm_gemma4.log 2>&1"
  for i in $(seq 1 180); do
    curl -sf 127.0.0.1:8420/v1/models >/dev/null && { up=1; break; }
    tmux has-session -t vllmgemma 2>/dev/null || break
    sleep 5
  done
  [ $up = 1 ] && break
  say "did not come up: $(grep -E 'Error|error:' ~/logs/vllm_gemma4.log | tail -2)"
  cp ~/logs/vllm_gemma4.log ~/logs/vllm_gemma4_failed_$(date +%H%M%S).log
done
[ $up = 1 ] || { say "gemma server failed - C-7 not started"; exit 1; }
say "server up, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
cd $A
python3 - <<'PY' || { echo "$(date '+%F %T') [c7-switch] test request failed - C-7 not started"; exit 1; }
import json, time, glob, sys
import agent_loop as AL, c1_multi_analyze as M
U = "http://127.0.0.1:8420/v1"
p = json.load(open(sorted(glob.glob('results/turns/prompt_c6_turn8/*multi*_request.json'))[0]))
p["messages"] = p["messages"][:1]; p["model"] = "gemma-4-31b"
print("input tokens", M.count_tokens(U, "gemma-4-31b", p["messages"][0]["content"]), flush=True)
t = time.time(); r = AL.post(U, p, 7200)
m = r["choices"][0]["message"]; c = m.get("content") or ""
rs = m.get("reasoning_content") or m.get("reasoning") or ""
print(round(time.time() - t), "s", r.get("usage"), "finish", r["choices"][0].get("finish_reason"), "reasoning chars", len(rs), "content chars", len(c), flush=True)
d = json.loads(c[c.find("{"):c.rfind("}") + 1])
print("keys", list(d)); print("chosen:", str(d.get("chosen"))[:300]); print("direction:", str(d.get("direction"))[:500])
assert d.get("direction") and len(rs) > 200, "no direction or no reasoning"
PY
say "test ok - start C-7"
tmux new -d -s c7 "bash $A/fractal_setup/start_c7.sh > ~/logs/c7_auto.log 2>&1"
say "C-7 started (tmux c7, ~/logs/c7_auto.log)"
