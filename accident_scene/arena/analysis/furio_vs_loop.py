"""Same prompt (A-4), two harnesses: agent_loop.py (실험 13) vs furio/openclaude (실험 14).

Reads only the kept logs and prints one comparison table (and writes it as JSON):
  agent_loop  results/turns/prompt_a4/<stem>_c<NN>_request.json, _turns.json, _events.jsonl
  furio       results/turns/prompt_a4_furio/<stem>_llm/ (proxy: every request/response),
              <stem>_t<N>_stream.jsonl (openclaude events), <stem>_tool_calls.jsonl, _turns.json

    python3 analysis/furio_vs_loop.py   (writes results/analysis/furio_vs_loop.json)
"""
import glob
import json
import os
from collections import Counter
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
os.chdir(ARENA)


def sse_usage(raw_path):
    """usage of a streamed completion: the last chunk that carries it; else count logprob tokens."""
    usage, ntok = None, 0
    for line in open(raw_path, errors="replace"):
        if not line.startswith("data: ") or line.startswith("data: [DONE]"):
            continue
        try:
            j = json.loads(line[6:])
        except ValueError:
            continue
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            lp = (ch.get("logprobs") or {}).get("content") or []
            ntok += len(lp)
    return usage, ntok


def furio_run(stem, d="results/turns/prompt_a4_furio"):
    idx = [json.loads(l) for l in open(f"{d}/{stem}_llm/index.jsonl")]
    chats = [e for e in idx if e["method"] == "POST" and e["path"].endswith("/chat/completions")]
    first_req = None
    prompt_tok = comp_tok = 0
    usage_seen = 0
    for e in chats:
        req = json.load(open(f"{d}/{stem}_llm/{e['seq']:04d}_request.json"))["body"]
        if first_req is None:
            first_req = req
        u, ntok = sse_usage(f"{d}/{stem}_llm/{e['seq']:04d}_response.raw")
        if u:
            usage_seen += 1
            prompt_tok += u.get("prompt_tokens") or 0
            comp_tok += u.get("completion_tokens") or 0
        else:
            comp_tok += ntok
    tools, errors, denied, texts = Counter(), 0, 0, 0
    for f in sorted(glob.glob(f"{d}/{stem}_t*_stream.jsonl")):
        for line in open(f, errors="replace"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "assistant":
                for c in ev["message"].get("content", []):
                    if c.get("type") == "tool_use":
                        tools[c.get("name")] += 1
                    elif c.get("type") == "text":
                        texts += 1
            elif ev.get("type") == "user":
                for c in ev["message"].get("content", []):
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        s = c.get("content") if isinstance(c.get("content"), str) else json.dumps(c.get("content"))
                        errors += "<tool_use_error>" in (s or "")
                        denied += "has been denied" in (s or "")
    tc = [json.loads(l) for l in open(f"{d}/{stem}_tool_calls.jsonl")] if os.path.exists(f"{d}/{stem}_tool_calls.jsonl") else []
    turns = json.load(open(f"{d}/{stem}_turns.json")) if os.path.exists(f"{d}/{stem}_turns.json") else {}
    sysm = [m for m in (first_req or {}).get("messages", []) if m["role"] == "system"]
    return {
        "harness": "furio (openclaude)",
        "temperature": (first_req or {}).get("temperature"), "top_p": (first_req or {}).get("top_p"),
        "max_tokens": (first_req or {}).get("max_tokens"), "stream": (first_req or {}).get("stream"),
        "system_prompt_chars": sum(len(m["content"]) if isinstance(m["content"], str) else len(json.dumps(m["content"])) for m in sysm),
        "tools_offered": [t["function"]["name"] for t in (first_req or {}).get("tools", [])],
        "model_calls": len(chats), "other_requests": len(idx) - len(chats),
        "request_bytes_first_last": [chats[0]["req_bytes"], chats[-1]["req_bytes"]] if chats else None,
        "prompt_tokens_sum": prompt_tok if usage_seen else None, "completion_tokens_sum": comp_tok,
        "usage_reported": f"{usage_seen}/{len(chats)}",
        "gen_wall_s_sum": round(sum(e["wall_s"] for e in chats), 1),
        "tool_uses": dict(tools), "tool_errors": errors, "denied": denied, "text_only_replies": texts,
        "isaac_tests": len([t for t in tc if t.get("record")]),
        "test_verdicts": [((t.get("record") or {}).get("verdict"), (t.get("record") or {}).get("final_dist_m")) for t in tc],
        "turns": [(t.get("turn"), t.get("verdict"), t.get("final_dist_m")) for t in turns.get("turns", [])],
        "passed_at_turn": turns.get("passed_at_turn"), "finished": "ended" in turns,
        "started": turns.get("started"), "ended": turns.get("ended"),
    }


def loop_run(stem, d="results/turns/prompt_a4"):
    turns = json.load(open(f"{d}/{stem}_turns.json"))
    req = json.load(open(f"{d}/{stem}_c01_request.json"))
    calls = [c for t in turns["turns"] for c in t["calls"]]
    events = [json.loads(l) for l in open(f"{d}/{stem}_events.jsonl")]
    sysm = [m for m in req["messages"] if m["role"] == "system"]
    return {
        "harness": "agent_loop (직접 루프)",
        "temperature": req.get("temperature"), "top_p": req.get("top_p"), "max_tokens": req.get("max_tokens"),
        "stream": req.get("stream", False),
        "system_prompt_chars": sum(len(m["content"]) for m in sysm),
        "tools_offered": [t["function"]["name"] for t in req.get("tools", [])],
        "model_calls": len(calls), "other_requests": 0,
        "request_bytes_first_last": [os.path.getsize(f"{d}/{stem}_c01_request.json"),
                                     os.path.getsize(f"{d}/{stem}_c{len(calls):02d}_request.json")],
        "prompt_tokens_sum": sum(c.get("prompt_tokens") or 0 for c in calls),
        "completion_tokens_sum": sum(c.get("completion_tokens") or 0 for c in calls),
        "usage_reported": f"{len(calls)}/{len(calls)}",
        "gen_wall_s_sum": round(sum(c.get("gen_wall_s") or 0 for c in calls), 1),
        "tool_uses": {"run_isaac_sim": sum(c.get("tool_calls") or 0 for c in calls)},
        "tool_errors": 0, "denied": 0, "text_only_replies": sum(1 for c in calls if not c.get("tool_calls")),
        "recovered_tool_calls": sum(1 for e in events if e["type"] == "recovered_tool_call"),
        "isaac_tests": sum(len(t["tests"]) for t in turns["turns"]),
        "test_verdicts": [(x.get("verdict"), x.get("final_dist_m")) for t in turns["turns"] for x in t["tests"]],
        "turns": [(t.get("turn"), t.get("verdict"), t.get("final_dist_m")) for t in turns["turns"]],
        "passed_at_turn": turns.get("passed_at_turn"), "finished": "ended" in turns,
        "started": turns.get("started"), "ended": turns.get("ended"),
    }


def main():
    rows = {}
    for k in (0, 1):
        rows[f"A-4 loop a{k}"] = loop_run(f"npu_gpt_oss_120b_pA4_a{k}")
        if os.path.exists(f"results/turns/prompt_a4_furio/npu_furio_pA4_a{k}_llm/index.jsonl"):
            rows[f"A-4 furio a{k}"] = furio_run(f"npu_furio_pA4_a{k}")
    Path("results/analysis/furio_vs_loop.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    keys = ["harness", "temperature", "top_p", "max_tokens", "stream", "system_prompt_chars", "tools_offered",
            "model_calls", "other_requests", "request_bytes_first_last", "prompt_tokens_sum", "completion_tokens_sum",
            "usage_reported", "gen_wall_s_sum", "tool_uses", "tool_errors", "denied", "text_only_replies",
            "recovered_tool_calls", "isaac_tests", "test_verdicts", "turns", "passed_at_turn", "finished",
            "started", "ended"]
    names = list(rows)
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(rows[n].get(k, "-")) for n in names) + " |")


if __name__ == "__main__":
    main()
