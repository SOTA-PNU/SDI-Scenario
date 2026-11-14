"""Smoke test for the GPU gpt-oss-120b server before B runs: one chat request with the B tool
(run_isaac_sim, agent_loop.TOOL_B) that should come back as a tool call, then the tool result fed
back for a final answer. Prints which fields carry the reasoning and the tool call. No Isaac.

    python3 analysis/c_vllm_smoke.py [base_url]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8420/v1"


def post(payload):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), round(time.time() - t0, 1)


msgs = [{"role": "user", "content": "Call the run_isaac_sim tool once with code "
         "'def controller(t, pose, env):\\n    return 0.0, 0.0, True\\n' and sim_seconds 10. Do not answer before calling it."}]
p = {"model": "gpt-oss-120b", "messages": msgs, "tools": [A.TOOL_B], "tool_choice": "auto",
     "max_tokens": 2048, "temperature": 0.7, "top_p": 0.95, "logprobs": True, "top_logprobs": 1}
r, dt = post(p)
m = r["choices"][0]["message"]
print("1st call", dt, "s | finish", r["choices"][0]["finish_reason"], "| message keys", sorted(m.keys()))
print("  reasoning fields:", {k: len(m.get(k) or "") for k in ("reasoning", "reasoning_content") if k in m})
print("  tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False)[:300])
print("  usage:", r.get("usage"))
if m.get("tool_calls"):
    tc = m["tool_calls"][0]
    msgs.append({"role": "assistant", "content": m.get("content"), "reasoning": m.get("reasoning") or m.get("reasoning_content"),
                 "tool_calls": m["tool_calls"]})
    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"result": {"verdict": "pass"}})})
    p2 = dict(p, messages=msgs)
    r2, dt2 = post(p2)
    m2 = r2["choices"][0]["message"]
    print("2nd call", dt2, "s | finish", r2["choices"][0]["finish_reason"], "| content:", (m2.get("content") or "")[:200])
