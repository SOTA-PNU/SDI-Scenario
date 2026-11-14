"""Collect the reasoning (gpt-oss analysis channel) of every model call of prompt A-4
(REPORT.md 실험 13) from the generation text rebuilt from logprobs (*_raw_text.txt), with the
call's timing, token counts and action. Writes results/analysis/a4_reasoning_raw.md and
results/analysis/a4_reasoning_calls.json. No Isaac, no LLM.

    python3 analysis/a4_reasoning_extract.py [--turns-dir results/turns/prompt_a4] [--tag npu_gpt_oss_120b_pA4]
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--turns-dir", default="results/turns/prompt_a4")
ap.add_argument("--tag", default="npu_gpt_oss_120b_pA4")
ap.add_argument("--out-md", default="results/analysis/a4_reasoning_raw.md")
ap.add_argument("--out-json", default="results/analysis/a4_reasoning_calls.json")
a = ap.parse_args()

SEG = re.compile(r"(?:<\|start\|>assistant)?<\|channel\|>(\w+)(?:\s+to=([\w.]+))?\s*(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?=<\|end\|>|<\|call\|>|<\|return\|>|$)", re.S)
calls = []
for raw in sorted(glob.glob(os.path.join(a.turns_dir, f"{a.tag}_a*_c*_raw.json"))):
    stem = os.path.basename(raw)[: -len("_raw.json")]
    run, call = re.search(r"_(a\d+)_c(\d+)$", stem).groups()
    r = json.load(open(raw))
    msg, u = r["choices"][0]["message"], r.get("usage", {})
    text = open(raw.replace("_raw.json", "_raw_text.txt")).read()
    segs = [(ch, to, body) for ch, to, body in SEG.findall(text)]
    analysis = "\n".join(b for ch, _, b in segs if ch == "analysis")
    final = "\n".join(b for ch, _, b in segs if ch == "final")
    tool = [(ch, to) for ch, to, _ in segs if ch == "commentary"]
    ev = {}
    evf = os.path.join(a.turns_dir, f"{a.tag}_{run}_events.jsonl")
    for e in map(json.loads, open(evf)):
        if e.get("call") == int(call) and e["type"] in ("response", "recovered_tool_call"):
            ev.setdefault(e["type"], e)
    calls.append({
        "run": run, "call": int(call),
        "gen_s": (ev.get("response") or {}).get("gen_wall_s"),
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "server_reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "analysis_chars": len(analysis), "final_chars": len(final),
        "server_reasoning_field_equals_content": bool(msg.get("content")) and msg.get("reasoning") == msg.get("content"),
        "action": ("tool " + ",".join(f"{ch}->{to or '(수신자 없음)'}" for ch, to in tool)) if tool
                  else ("final answer" if final else "none"),
        "recovered_tool_call": "recovered_tool_call" in ev,
        "analysis": analysis,
    })
os.makedirs(os.path.dirname(a.out_md), exist_ok=True)
json.dump([{k: v for k, v in c.items() if k != "analysis"} for c in calls],
          open(a.out_json, "w"), ensure_ascii=False, indent=1)
with open(a.out_md, "w") as f:
    f.write("# A-4 추론 원문 (실험 13, NPU gpt-oss-120b)\n\n")
    f.write("모델이 분석 채널(`analysis`)에 쓴 추론을 생성 원문(`*_raw_text.txt`, logprobs 로 다시 만든 "
            f"글)에서 그대로 옮겼다. 영어 원문이며 고치지 않았다. 만든 스크립트: `analysis/a4_reasoning_extract.py`.\n\n")
    for c in calls:
        f.write(f"## {c['run']} c{c['call']:02d}\n\n")
        f.write(f"- 행동: {c['action']}" + (" (서버가 버린 호출을 루프가 원문에서 복원)" if c["recovered_tool_call"] else "") + "\n")
        f.write(f"- 입력 {c['prompt_tokens']} 토큰, 생성 {c['completion_tokens']} 토큰, 분석 채널 {c['analysis_chars']}자\n\n")
        if c["analysis"]:
            f.write("```text\n" + c["analysis"].strip() + "\n```\n\n")
        else:
            f.write("분석 채널 없음. 곧바로 최종 답 채널로 시작했다.\n\n")
print(json.dumps([{k: v for k, v in c.items() if k != "analysis"} for c in calls], ensure_ascii=False, indent=1))


# ---- tool calls and harness work, in time order (added 2026-10-16) ----
import datetime as _dt


def _hhmmss(epoch):
    return _dt.datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def _carter_header(variant):
    f = os.path.join(a.turns_dir, "isaac_logs", f"{variant}.carter-run.stdout.log")
    if not os.path.exists(f):
        return None
    for line in open(f):
        if line.startswith("# start_epoch="):
            kv = dict(p.split("=", 1) for p in line[2:].split())
            return {"start": _hhmmss(float(kv["start_epoch"])), "end": _hhmmss(float(kv["end_epoch"])),
                    "isaac_wall_s": float(kv["wall_s"]), "lock_waits": int(kv["serial_guard_waits"]),
                    "returncode": int(kv["returncode"])}
    return None


timeline = {}
for evf in sorted(glob.glob(os.path.join(a.turns_dir, f"{a.tag}_a*_events.jsonl"))):
    run = re.search(r"_(a\d+)_events", evf).group(1)
    rows = []
    for e in map(json.loads, open(evf)):
        t, ty = e["time"][11:], e["type"]
        if ty == "request":
            rows.append({"time": t, "step": f"c{e['call']:02d} 요청", "messages": e["messages"], "chars": e["chars"],
                         "tool_choice": e.get("tool_choice")})
        elif ty == "response":
            rows.append({"time": t, "step": f"c{e['call']:02d} 응답", "gen_wall_s": e["gen_wall_s"],
                         "finish": e["finish"], "tool_calls": e["tool_calls"], "content_chars": e.get("content_chars")})
        elif ty == "recovered_tool_call":
            rows.append({"time": t, "step": f"c{e['call']:02d} 도구 호출 복원", "name": e["name"],
                         "args_chars": e["args_chars"], "header": e["header"]})
        elif ty == "tool_call":
            tj = json.load(open(os.path.join(a.turns_dir, f"{a.tag}_{run}_c{e['call']:02d}_tool1.json")))
            code = tj["arguments"].get("code", "")
            rec = tj.get("record") or {}
            out = tj["result"]
            rr = out.get("result") or {}
            rows.append({"time": t, "step": f"c{e['call']:02d} 도구 호출 run_isaac_sim (시험 {e['test']})",
                         "call_id": (tj.get("tool_call") or {}).get("id"), "sim_seconds": e["sim_seconds"],
                         "code_chars": len(code), "code_is_region_only": "base_carter_run" not in code and "MISSION" not in code,
                         "runner": rec.get("variant", "") + "_carter_run.py", "isaac": _carter_header(rec.get("variant", "")),
                         "returned_keys": list(out.keys()), "verdict": rr.get("verdict"),
                         "partners": (rr.get("metrics") or {}).get("collision_partners"),
                         "time_to_goal_s": (rr.get("metrics") or {}).get("time_to_goal_s"),
                         "harness_events": out.get("harness_events"), "trajectory_rows": len(out.get("trajectory") or []),
                         "tool_wall_s": out.get("wall_s")})
        elif ty == "tool_result":
            rows.append({"time": t, "step": f"c{e['call']:02d} 도구 결과 반환", "chars": e["chars"], "error": e["error"]})
        elif ty == "submission":
            rows.append({"time": t, "step": "제출", "file": e["variant"] + "_carter_run.py", "static_ok": e["static_ok"],
                         "spliced_region_into_base": e["spliced"]})
        elif ty == "grading":
            rows.append({"time": t, "step": "채점", "verdict": e["verdict"], "final_dist_m": e["final_dist_m"],
                         "collisions": e["collisions"], "isaac": _carter_header(e["variant"])})
    timeline[run] = rows
json.dump(timeline, open(a.out_json.replace("_calls.json", "_tool_timeline.json"), "w"), ensure_ascii=False, indent=1)
print("timeline ->", a.out_json.replace("_calls.json", "_tool_timeline.json"))
