"""Read-only extraction of B524 a1 model revisions; no simulator or model calls."""
import ast
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ARENA = Path(__file__).resolve().parents[1]
D = ARENA / "results/turns/prompt_b5_24"
PREFIX = "gpu_gpt_oss_120b_pB524_a1"
OUT = ARENA / "results/analysis/b524_a1_model_revision_evidence.json"
HOOKS = ("plan_path", "lookahead_target", "follow_cmd", "controller")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def relative(path):
    return str(path.relative_to(ARENA))


def refs(path, needle):
    return [{"file": relative(path), "line": i, "text": line}
            for i, line in enumerate(path.read_text().splitlines(), 1) if needle in line]


def normal(fn):
    fn = ast.parse(ast.unparse(fn)).body[0]
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            item = node.body[0]
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                node.body = node.body[1:]
    return ast.dump(fn, include_attributes=False)


SUMMARIES = [
    ("grade_t1", "Raw map rectangle corners form the visibility graph; segment_clearance rejects edges ending at obstacle corners. No trial before submission.",
     "Read-only source: obstacle corners are not clearance-offset. Actual stdout reports empty_path. The next model request contains grading JSON but no DBG."),
    ("test01", "Replace visibility graph with 0.5m grid A*, testing cell centers only; insert exact start/goal. Lookahead uses current segment remainder capped at2m, minimum0.3m.",
     "No neighbor-edge/endpoint-connector check or validate_plan/check_path call. Actual first rejection is segment15, gap0.195982<margin0.25."),
    ("test02", "Planner execution AST unchanged; lookahead changed to exactly0.3m ahead.",
     "Model incorrectly infers acceptance at clearance0.4 from one clearance0.5 rejection line; changing lookahead cannot repair a rejected plan."),
    ("test03", "Replace A* with unchecked start-to-goal straight line; same0.3m lookahead.",
     "Model transfers the prior grid path's gap to an unrelated straight line. Actual straight-line gap=-0.25, rejecting all positive margins."),
    ("test04", "Return to center-only A* with1.0m cells; same0.3m lookahead.",
     "No edge validation. First gap0.070156<smallest margin0.10; later recorded retries return empty_path."),
    ("test05", "Return to0.5m cells and delete middle points when point-to-line deviation<=0.2m.",
     "Shortcut deletion checks no map clearance. Model continues attributing no motion to jagged tracking; actual output remains plan rejection."),
    ("test06", "Replace computed grid route with manual five-point corridor.",
     "No map or public validator call; actual gap=-0.25. The prompt explicitly forbids manual route coordinates."),
    ("test07", "Replace one manual vertical corridor with a new six-point manual corridor.",
     "No map or public validator call; actual first_bad_segment2 and gap=-0.25. Model guesses obstacle geometry rather than inspecting the returned path's failing segment."),
    ("grade_t2", "Submit the same four function execution ASTs as test07; comments/docstrings/spacing differ.",
     "Final reasoning acknowledges failure; no passing test exists. The final path still crosses a mapped obstacle according to actual check_path DBG."),
]

QUOTE_NEEDLES = {
    1: ["Nodes", "Now we need to produce code block only"],
    2: ["plan_path likely returned empty path", "coarse grid A*"],
    3: ["But main issue:", "Let's modify lookahead_target"],
    4: ["Thus modify plan_path", "The min_gap earlier"],
    5: ["So our assumption wrong", "Let's try step = 1.0"],
    6: ["Now plan_reject", "Tolerance maybe"],
    7: ["Robot still not moving", "manually define a simple corridor"],
    8: ["Plan rejected due to segment 1", "Thus use these waypoints"],
    9: ["Still failing due to segment 2", "However we haven't achieved pass"],
}


def run():
    sources = [D / f"{PREFIX}_turns.json", ARENA / "PROMPT_B5_24.md",
               ARENA / "base_carter_run_ros_map_lidartime_objects_check_v12.py"]
    turns = json.loads(sources[0].read_text())
    calls, revisions = [], []
    previous = None
    for call in range(1, 10):
        rawp = D / f"{PREFIX}_c{call:02d}_raw.json"
        textp = D / f"{PREFIX}_c{call:02d}_raw_text.txt"
        requestp = D / f"{PREFIX}_c{call:02d}_request.json"
        sources += [rawp, textp, requestp]
        raw, request = json.loads(rawp.read_text()), json.loads(requestp.read_text())
        msg = raw["choices"][0]["message"]
        suffix = "" if call == 1 else "_t2" if call == 9 else f"_t2_test{call-1:02d}"
        snapshot = D / "isaac_logs" / f"{PREFIX}{suffix}_carter_run.py"
        stdout = D / "isaac_logs" / f"{PREFIX}{suffix}.carter-run.stdout.log"
        resultp = D / "isaac_logs" / f"{PREFIX}{suffix}_result.json"
        sources += [snapshot, stdout, resultp]
        code = snapshot.read_text()
        fns = {n.name: n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in HOOKS}
        norms = {name: normal(fn) for name, fn in fns.items()}
        lines = code.splitlines()
        functions = {name: {"line": fn.lineno, "end_line": fn.end_lineno,
            "execution_ast_sha256": digest(norms[name].encode())} for name, fn in fns.items()}
        edit_calls = Counter(node.func.id for fn in fns.values() for node in ast.walk(fn)
                             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
        rej = refs(stdout, "DBG plan_reject ")
        result = json.loads(resultp.read_text())
        stage, change, explanation = SUMMARIES[call-1]
        item = {"call": call, "stage": stage, "snapshot": relative(snapshot),
            "source_sha256": digest(snapshot.read_bytes()), "functions": functions,
            "changed_execution_functions": list(norms) if previous is None else [k for k in norms if norms[k] != previous[k]],
            "relevant_edit_calls": {k: edit_calls[k] for k in ("map_rects", "map_clearance", "segment_clearance", "check_path", "validate_plan", "choose_command", "observed_objects")},
            "modification": change, "source_and_log_interpretation": explanation,
            "plan_source": [{"line": i, "text": lines[i-1]} for i in range(fns["plan_path"].lineno, fns["plan_path"].end_lineno+1)],
            "actual_stdout_reject_count": len(rej),
            "actual_stdout_reason_counts": dict(Counter(row["text"].split("reason=", 1)[1].split()[0] for row in rej)),
            "actual_stdout_first_last_rejects": rej[:2]+rej[-1:],
            "result_file": relative(resultp), "verdict": result["verdict"], "metrics": result["metrics"]}
        if previous is not None:
            item["execution_ast_equal_to_previous"] = norms == previous
        previous = norms
        quotes = [ref for needle in QUOTE_NEEDLES[call] for ref in refs(textp, needle)]
        callinfo = {"call": call, "turn": 1 if call == 1 else 2,
            "request": relative(requestp), "request_seed": request.get("seed"),
            "raw": relative(rawp), "raw_text": relative(textp), "finish_reason": raw["choices"][0]["finish_reason"],
            "tool_call_count": len(msg.get("tool_calls", [])), "reasoning_evidence": quotes,
            "model_reasoning_is_not_execution_evidence": True}
        if 2 <= call <= 8:
            toolp = D / f"{PREFIX}_c{call:02d}_tool1.json"
            sources.append(toolp)
            tool = json.loads(toolp.read_text())
            output = tool["result"]["controller_output"]
            callinfo["tool"] = {"file": relative(toolp), "json_keys": list(tool["arguments"]),
                "sim_seconds_argument_present": "sim_seconds" in tool["arguments"],
                "applied_sim_seconds": tool["result"]["sim_seconds"],
                "controller_output_lines": len(output), "omitted_lines": tool["result"]["controller_output_omitted_lines"],
                "delivered_plan_reject": [{"index": i, "text": line} for i, line in enumerate(output) if "DBG plan_reject " in line],
                "record": tool["record"]}
        calls.append(callinfo)
        revisions.append(item)
    feedback_request = json.loads((D / f"{PREFIX}_c02_request.json").read_text())
    feedback = feedback_request["messages"][-1]["content"]
    base = sources[2]
    prompt = sources[1]
    checks = [
        ("nine_calls_first_turn_zero_tests_second_turn_seven", len(calls)==9 and turns["turns"][0]["tests_used"]==0 and turns["turns"][1]["tests_used"]==7),
        ("all_seven_trial_json_omit_sim_seconds", all(not c["tool"]["sim_seconds_argument_present"] for c in calls if "tool" in c)),
        ("all_nine_runs_fail_no_collision_601s", all(r["verdict"]=="fail" and r["metrics"]["collision_count"]==0 and r["metrics"]["sim_time_s"]==601.0 for r in revisions)),
        ("all_nine_edits_ignore_public_validate_plan", all(r["relevant_edit_calls"]["validate_plan"]==0 for r in revisions)),
        ("test01_test02_same_planner_execution", revisions[1]["functions"]["plan_path"]["execution_ast_sha256"]==revisions[2]["functions"]["plan_path"]["execution_ast_sha256"]),
        ("follow_cmd_and_wrapper_execution_unchanged_all_nine", all(r["functions"][k]["execution_ast_sha256"]==revisions[0]["functions"][k]["execution_ast_sha256"] for r in revisions for k in ("follow_cmd", "controller"))),
        ("final_same_execution_as_test07", revisions[-1]["execution_ast_equal_to_previous"]),
        ("first_grading_feedback_has_no_reject_debug", "plan_reject" not in feedback and "controller_output" not in feedback),
        ("every_trial_delivers_three_reject_lines", all(len(c["tool"]["delivered_plan_reject"])==3 for c in calls if "tool" in c)),
    ]
    report = {"verdict": "pass" if all(ok for _,ok in checks) else "fail", "passed": sum(ok for _,ok in checks), "total": len(checks),
        "scope": "Saved completed B524 a1 model responses, requests, trial payloads and actual source/log snapshots only; no GPU, HTTP, processes or simulator replay.",
        "turns_file": relative(sources[0]), "first_turn_feedback": {"request": relative(D/f"{PREFIX}_c02_request.json"), "message_index": len(feedback_request["messages"])-1, "chars": len(feedback), "contains_DBG": "DBG" in feedback,
            "conclusion": "The model inferred empty-path failure from grading outcome; it was not sent first-turn plan-reject DBG."},
        "calls": calls, "revisions": revisions,
        "mechanism_refs": {"reason_only_throttle": refs(base, 'previous = _BASE_LOOP_ST["plan_note_t"].get(reason)'),
            "all_attempts_validate": refs(base, 'validation = validate_plan(proposed)'),
            "retry": refs(base, 'state["next_plan_t"] = t + cfg["plan_retry_s"]')},
        "existing_prompt_instructions": {needle: refs(prompt, needle) for needle in ("각 이웃 연결도", "모든 연결과 양 끝점을", "validate_plan(path)", "손으로 고른", "첫 응답은", '"sim_seconds": 200')},
        "conclusions": [
            "Rejection telemetry reached the model during all seven trials; plan acceptance was never proven by the single displayed clearance0.5 rejection.",
            "c03 explicitly treats the absence of other clearance lines as acceptance at0.4 and diagnoses lookahead/heading without plan-state evidence. Per-reason10s dedup hides same-reason later attempts.",
            "The model incorrectly transfers one grid path's reported gap to other clearance-dependent paths and then to an unrelated straight line.",
            "The original prompt already required valid neighbor edges, shortcut checks and validate_plan; all nine edits omit the public validator and later planners omit all segment checks.",
            "The final submission changes formatting/docstrings, not the failing test07 execution. Clearance relaxation cannot clear an actual -0.25 map gap from an obstacle-crossing centerline.",
        ],
        "suggestions_for_root_decision": [
            "Report a per-planning-batch summary for every attempted clearance and explicit accepted/new_path/old_path/no_path state, while keeping acceptance geometry unchanged.",
            "Make planner-first debugging explicit: validate every A* edge and shortcut at requested center clearance, call validate_plan before returning, and fix its failed segment before changing target/follow.",
            "State that absence of a per-reason DBG line is not acceptance; raw obstacle corner nodes have zero map clearance and must not be treated as clearance-valid nodes.",
        ],
        "limits": ["No actual sensor-state replay or physical cause inferred from tiny trajectory drift.",
            "The emitted rejection is only the first visible reason per10s and cannot establish all-clearance acceptance; separate actual-map CPU planner evidence supplies that detail.",
            "Reasoning quotes record the model's stated beliefs, not reliable geometry or executed control behavior."],
        "checks": [{"name": name, "pass": ok} for name,ok in checks],
        "source_sha256": {relative(p): digest(p.read_bytes()) for p in dict.fromkeys(sources)}}
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_task27_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"file": str(OUT), "verdict": report["verdict"], "passed": report["passed"], "total": report["total"]}))


if __name__ == "__main__":
    run()
