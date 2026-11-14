"""Reproduce completed B517 a3 termination evidence; no simulation or requests.

Reads only B517 artifacts and static harness/base sources. Writes one analysis
JSON, preserving any previous version in results/archive_*.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
DATA = ARENA / "results/turns/prompt_b5_17"
STEM = "gpu_gpt_oss_120b_pB517_a3"
OUT = ARENA / "results/analysis/b517_a3_t2_termination_evidence.json"


def reference(path, needle):
    path = Path(path)
    text = path.read_text()
    offset = text.index(needle)
    return {"file": str(path.relative_to(ARENA)), "line": text[:offset].count("\n") + 1}


def main():
    events_path = DATA / f"{STEM}_events.jsonl"
    events = [{**json.loads(line), "source_line": i}
              for i, line in enumerate(events_path.read_text().splitlines(), 1)]
    turns_path = DATA / f"{STEM}_turns.json"
    turns = json.loads(turns_path.read_text())
    rows = []
    for index in range(1, 13):
        request_path = DATA / f"{STEM}_c{index:02d}_request.json"
        raw_path = DATA / f"{STEM}_c{index:02d}_raw.json"
        request, raw = json.loads(request_path.read_text()), json.loads(raw_path.read_text())
        choice, usage = raw["choices"][0], raw["usage"]
        message = choice["message"]
        rows.append({
            "call": index, "turn": raw["_bench"]["turn"],
            "request_file": str(request_path.relative_to(ARENA)),
            "raw_file": str(raw_path.relative_to(ARENA)),
            "request_messages": len(request["messages"]),
            "tool_choice": request["tool_choice"], "max_tokens": request["max_tokens"],
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "reasoning_tokens": usage["completion_tokens_details"]["reasoning_tokens"],
            "finish_reason": choice["finish_reason"],
            "tool_calls": len(message.get("tool_calls") or []),
            "content_chars": len(message.get("content") or ""),
            "bench": raw["_bench"],
            "usage_reference": reference(raw_path, '"prompt_tokens":'),
            "finish_reference": reference(raw_path, '"finish_reason":'),
        })
    terminal = rows[-1]
    final_raw = json.loads((DATA / f"{STEM}_c12_raw.json").read_text())
    reply_path = DATA / f"{STEM}_t2_reply.md"
    reply = reply_path.read_text()
    # Exercise actual extraction/splicing/static checking in memory only.
    sys.path.insert(0, str(ARENA))
    import agent_loop as A
    base_path = ARENA / turns["base_file"]
    base = base_path.read_text()
    A.BASE = A.SP.BASE = base
    A.B_MODE = "b5"
    A.A21 = A.A22 = A.A23 = A.A4T = False
    assembled = A.build(A.TL.extract(reply))
    static_ok, detail = A.static_check(assembled)
    stored_submission = ARENA / turns["turns"][1]["submission"]
    assert stored_submission.read_text() == assembled
    assert not static_ok and detail == turns["turns"][1]["detail"]
    assert terminal["finish_reason"] == "stop" and terminal["tool_calls"] == 0
    assert turns["turns"][1]["verdict"] == "static_dq"
    assert len([e for e in events if e["type"] == "grading"]) == 1
    last_result = next(e for e in events if e["type"] == "tool_result" and e["record"]["test"] == 10)
    final_request = next(e for e in events if e["type"] == "request" and e["call"] == 12)
    final_response = next(e for e in events if e["type"] == "response" and e["call"] == 12)
    final_submission = next(e for e in events if e["type"] == "submission" and e["turn"] == 2)
    end = next(e for e in events if e["type"] == "end")
    first_t2_request = next(e for e in events if e["type"] == "request" and e["turn"] == 2)
    event_counts = dict(Counter(e["type"] for e in events))
    max_prompt = max(rows, key=lambda row: row["prompt_tokens"])
    max_output = max(rows, key=lambda row: row["completion_tokens"])
    max_total = max(rows, key=lambda row: row["total_tokens"])
    agent = ARENA / "agent_loop.py"
    output = {
        "scope": "Completed B517 a3 calls01–12 and tests01–10; no B518 runtime data, process inspection, HTTP or simulation.",
        "conclusion": {
            "second_isaac_grading_performed": False,
            "second_submission_static_check_performed": True,
            "terminal_verdict": "static_dq",
            "mechanism": "c12 returned a non-code final answer without a tool call. B5 accepted it as a submission, splice placed the prose in EDIT REGION, compile failed at line754, and the max_turns=2 loop ended without calling TL.grade for turn2.",
            "context_overflow_observed": False,
            "call_cap_reached": False,
            "output_token_limit_reached": False,
            "model_stated_reason_limit": "Final raw reasoning mentions limited time and difficulty. This is model text, not evidence of an actual wall-time, tool-call, or context limit being reached.",
        },
        "final_answer": {
            "text": reply, "reference": {"file": str(reply_path.relative_to(ARENA)), "line": 1},
            "raw_reasoning": final_raw["choices"][0]["message"].get("reasoning"),
            "reasoning_reference": reference(DATA / f"{STEM}_c12_raw.json", '"reasoning":'),
            "readable_raw_reference": {"file": str((DATA / f"{STEM}_c12_raw_text.txt").relative_to(ARENA)), "lines": [3, 5, 7, 9]},
        },
        "timeline": {
            "event_file": str(events_path.relative_to(ARENA)),
            "last_test_result": last_result, "final_request": final_request,
            "final_response": final_response, "static_submission": final_submission, "end": end,
            "seconds_from_last_test_result": {
                "request": round(final_request["ts"] - last_result["ts"], 3),
                "response": round(final_response["ts"] - last_result["ts"], 3),
                "static_submission": round(final_submission["ts"] - last_result["ts"], 3),
                "run_end": round(end["ts"] - last_result["ts"], 3),
            },
            "turn2_wall_seconds": round(end["ts"] - first_t2_request["ts"], 3),
            "sample_wall_seconds": round(end["ts"] - events[0]["ts"], 3),
            "precision_note": "Event epoch values are recorded to milliseconds; displayed text times round to whole seconds.",
        },
        "token_usage": {
            "maximum_input": {"call": max_prompt["call"], "prompt_tokens": max_prompt["prompt_tokens"]},
            "maximum_output": {"call": max_output["call"], "completion_tokens": max_output["completion_tokens"]},
            "maximum_total_one_call": {"call": max_total["call"], "prompt_tokens": max_total["prompt_tokens"], "completion_tokens": max_total["completion_tokens"], "total_tokens": max_total["total_tokens"]},
            "final_call_output_breakdown": {"completion_tokens": terminal["completion_tokens"], "reasoning_tokens": terminal["reasoning_tokens"], "requested_max_tokens": terminal["max_tokens"]},
            "largest_input_plus_requested_output_budget": max(row["prompt_tokens"] + row["max_tokens"] for row in rows),
            "request_messages_final": terminal["request_messages"], "request_chars_final": final_request["chars"],
            "actual_server_context_limit": "Not queried; the accepted c12 response and absence of context HTTP errors establish no observed overflow.",
            "per_call": rows,
        },
        "limits_and_events": {
            "max_turns": turns["max_turns"], "turns_used": turns["turns_used"],
            "test_budget": turns["test_budget"], "per_turn_call_cap": turns["test_budget"] + 6,
            "total_model_calls": turns["model_calls"], "turn2_model_calls": len(turns["turns"][1]["calls"]),
            "turn2_tests": turns["turns"][1]["tests_used"],
            "all_requests_tool_choice_auto": all(row["tool_choice"] == "auto" for row in rows),
            "event_type_counts": event_counts,
            "http_error_trim_lost_or_recovery_events": [e for e in events if e["type"] in {"http_error", "trim", "server_wait", "lost_response", "recovered_tool_call", "final_asked"}],
            "all_retry_waits_empty": all(not row["bench"]["retry_waits"] for row in rows),
            "default_character_budget": 300000,
            "default_character_budget_reference": reference(agent, 'ap.add_argument("--budget-chars", type=int, default=300000'),
            "all_tests_omitted_sim_seconds": all(e["sim_seconds"] is None for e in events if e["type"] == "tool_call"),
            "all_tests_used_600s": all(test["sim_seconds"] == 600.0 for test in turns["turns"][1]["tests"]),
        },
        "static_reproduction": {
            "ok": static_ok, "detail": detail, "assembled_line754": assembled.splitlines()[753],
            "identical_to_retained_submission": True,
            "outside_region_preserved": A.TL._outside(assembled) == A.TL._outside(base),
            "retained_submission": str(stored_submission.relative_to(ARENA)),
            "agent_loop_sha16": hashlib.sha256(agent.read_bytes()).hexdigest()[:16],
            "read_only_method": "A.build(A.TL.extract(reply)), then A.static_check in memory; no write_submission or grade invoked.",
        },
        "source_references": {
            "call_cap_formula": reference(agent, "def call_cap(a):"),
            "budget_zero_maps_to_one_million": reference(agent, "if a.test_budget == 0:"),
            "context_error_branch": reference(agent, 'if e.code == 400 and "context length" in body'),
            "non_tool_response_becomes_submission": {"file": "agent_loop.py", "lines": [1247, 1250, 1256]},
            "static_check_before_grade": {"file": "agent_loop.py", "lines": [1288, 1290, 1292, 1299, 1304]},
            "static_compile_error": reference(agent, 'compile(text, "<submission>", "exec")'),
            "outer_turn_loop": reference(agent, "for turn in range(start, a.max_turns + 1):"),
            "turn2_static_record": reference(turns_path, '"verdict": "static_dq"'),
            "sequential_terminal": reference(DATA / "sequential.log", "a3 exit=1 grade=FAIL turn=-"),
        },
        "proposal_for_parent": "[harness] Before consuming a B5 turn, return a bounded repair instruction for non-code final answers or invalid static submissions. Log explicit termination_reason=static_dq and distinguish this from an Isaac grade. Existing A22 pre-submission gating does not run in B5; B518 prose alone does not add this enforcement.",
    }
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_task13_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("wrote", OUT)
    print("last_test_to_end_s", output["timeline"]["seconds_from_last_test_result"]["run_end"])
    print("maximum_context", output["token_usage"]["maximum_total_one_call"])


if __name__ == "__main__":
    main()
