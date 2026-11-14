#!/usr/bin/env python3
"""Parse completed B524 a0/a1 artifacts; never execute model code or a simulator."""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

A = Path(__file__).resolve().parents[1]
D = Path("results/turns/prompt_b5_24")
L = D / "isaac_logs"
OUT = Path("results/analysis/b524_a1_run_evidence.json")
SHA = {}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    path = Path(path)
    raw = (A / path).read_bytes()
    SHA[path.as_posix()] = digest(raw)
    return raw.decode()


def load(path):
    return json.loads(read(path))


def numbered(lines, predicate):
    return [{"line": i, "text": text} for i, text in enumerate(lines, 1) if predicate(text)]


def reject_record(row):
    record = dict(row)
    # All tokens before error= are scalar diagnostics, not executable text.
    head = row["text"].split(" error=", 1)[0]
    for key, token in re.findall(r"(\w+)=([^ ]+)", head):
        if token == "None":
            value = None
        elif key in ("reason",):
            value = token
        else:
            value = float(token)
            if key == "segment":
                value = int(value)
        record[key] = value
    record["error_text"] = row["text"].split(" error=", 1)[1]
    return record


def source_description(path):
    source = read(path)
    first = source.index("# [EDIT REGION]")
    last = source.index("# ========================= [END EDIT REGION]")
    region = source[first:last]
    functions = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("plan_path", "lookahead_target", "follow_cmd", "controller"):
            functions[node.name] = {"line": node.lineno, "end_line": node.end_lineno}
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            functions[node.name]["executable_AST_sha256_excluding_docstring"] = digest(ast.dump(node, include_attributes=False).encode())
    return {"file": str(path), "sha256": digest(source.encode()), "edit_sha256": digest(region.encode()), "functions": functions}


def run_record(variant, label, reported_runtime_error):
    paths = {name: L / (variant + suffix) for name, suffix in {
        "result": "_result.json", "source": "_carter_run.py", "stdout": ".carter-run.stdout.log",
        "stderr": ".carter-run.stderr.log", "trajectory": "_trajectory.csv"}.items()}
    result = load(paths["result"])
    stdout, stderr = read(paths["stdout"]).splitlines(), read(paths["stderr"]).splitlines()
    # Isaac warning output can interleave with Python stdout on the same line.
    # Keep the raw line and its number while extracting the actual DBG suffix.
    dbg = [{"line": i, "text": line[line.index("DBG "):], "raw_text": line}
           for i, line in enumerate(stdout, 1) if "DBG " in line]
    rejects = [reject_record(row) for row in dbg if row["text"].startswith("DBG plan_reject")]
    choose = [row for row in dbg if row["text"].startswith("DBG choose")]
    counts = Counter(re.search(r"pick=(\S+) why=(\S+)", row["text"]).groups() for row in choose)
    trajectory = list(csv.DictReader(read(paths["trajectory"]).splitlines()))
    points = [(float(row["x"]), float(row["y"])) for row in trajectory]
    initial = points[0]
    timeouts = numbered(stdout, lambda line: "[mission] TIMEOUT" in line)
    first_by_reason = {}
    for row in rejects:
        first_by_reason.setdefault(row["reason"], row)
    transition = []
    for row in rejects:
        if not transition or transition[-1]["reason"] != row["reason"]:
            transition.append(row)
    return {"variant": variant, "label": label, "files": {key: str(value) for key, value in paths.items()},
            "verdict": result["verdict"], "criteria": result["criteria"], "metrics": result["metrics"],
            "oracles": result["oracles"], "reported_controller_runtime_error": reported_runtime_error,
            "simulator_seed": result.get("seed"), "source": source_description(paths["source"]),
            "rejections": {"printed_count": len(rejects), "reason_counts": dict(Counter(row["reason"] for row in rejects)),
                           "first_by_reason": first_by_reason, "reason_transitions": transition, "all_printed": rejects,
                           "limitation": "Reason-based 10s throttling hides subsequent clearances with the same reason. Printed rows are not a count of planner calls and do not prove later clearances passed."},
            "dbg": {"count": len(dbg), "choose_count": len(choose),
                    "extraction": "Locate DBG substring, including lines prefixed by interleaved Isaac warnings; preserve raw line and source line number.",
                    "interleaved_lines": [row for row in dbg if row["raw_text"] != row["text"]],
                    "choose_counts": [{"chosen": key[0], "reason": key[1], "count": value} for key, value in sorted(counts.items())],
                    "first_choose": choose[:1], "last_choose": choose[-1:],
                    "all_non_follow_choose": [row for row in choose if "pick=follow " not in row["text"]],
                    "other_lines": [row for row in dbg if not row["text"].startswith(("DBG choose", "DBG plan_reject"))]},
            "timeout_lines": timeouts,
            "trajectory": {"rows": len(trajectory), "first": trajectory[0], "last": trajectory[-1],
                           "max_distance_from_first_sample_m": max(math.dist(initial, p) for p in points),
                           "final_distance_from_first_sample_m": math.dist(initial, points[-1]),
                           "sampled_polyline_length_m": sum(math.dist(p, q) for p, q in zip(points, points[1:])),
                           "limitation": "Rounded CSV samples; sampled length is not full physical travel."},
            "stderr_diagnostics": numbered(stderr, lambda line: any(token in line for token in
                ("returncode=", "Traceback (most recent", "ModuleNotFoundError", "Failed to import python module", "/proc/"))),
            "stderr_note": "Startup extension errors, if present, are separate from the harness-reported controller runtime_error. Full mission duration is recorded independently."}


def delivered_tool_result(sample, call):
    prefix = f"gpu_gpt_oss_120b_pB524_a{sample}"
    tool_file = D / f"{prefix}_c{call:02d}_tool1.json"
    tool = load(tool_file)
    request_file = D / f"{prefix}_c{call + 1:02d}_request.json"
    request = load(request_file)
    tool_id = tool["tool_call"]["id"]
    matching = [(index, message) for index, message in enumerate(request["messages"])
                if message.get("role") == "tool" and message.get("tool_call_id") == tool_id]
    assert len(matching) == 1, (request_file, tool_id)
    index, message = matching[0]
    parsed = json.loads(message["content"])
    exact = parsed == tool["result"]
    assert exact, (request_file, "tool payload changed")
    output = parsed.get("controller_output", [])
    return {"tool_file": str(tool_file), "call": call, "variant": tool["record"]["variant"],
            "tool_call_id": tool_id, "argument_keys": list(tool["arguments"]),
            "sim_seconds_argument_present": "sim_seconds" in tool["arguments"],
            "sim_seconds_argument": tool["arguments"].get("sim_seconds"),
            "record": tool["record"], "static_check": parsed.get("static_check"),
            "model_delivery": {"next_request_file": str(request_file), "message_index_zero_based": index,
                               "same_decoded_result_as_saved_tool_file": exact,
                               "message_content_sha256": digest(message["content"].encode()),
                               "controller_output": output,
                               "plan_reject_lines": [line for line in output if line.startswith("DBG plan_reject")],
                               "controller_output_omitted_lines": parsed.get("controller_output_omitted_lines"),
                               "harness_events": parsed.get("harness_events"),
                               "runtime_error": parsed.get("runtime_error")}}


def main():
    turns = {sample: load(D / f"gpu_gpt_oss_120b_pB524_a{sample}_turns.json") for sample in (0, 1)}
    events = {}
    for sample in (0, 1):
        text = read(D / f"gpu_gpt_oss_120b_pB524_a{sample}_events.jsonl")
        events[sample] = [{"line": i, **json.loads(line)} for i, line in enumerate(text.splitlines(), 1)]
    a0 = run_record("gpu_gpt_oss_120b_pB524_a0", "a0_turn1_grading", None)
    first = run_record("gpu_gpt_oss_120b_pB524_a1", "a1_turn1_grading", None)
    tests = [run_record(f"gpu_gpt_oss_120b_pB524_a1_t2_test{test:02d}", f"a1_turn2_test{test:02d}",
                        turns[1]["turns"][1]["tests"][test - 1].get("runtime_error")) for test in range(1, 8)]
    final = run_record("gpu_gpt_oss_120b_pB524_a1_t2", "a1_turn2_grading", None)
    delivered = [delivered_tool_result(1, call) for call in range(2, 9)]
    a0_tool = delivered_tool_result(0, 1)
    a1_runs = [first, *tests, final]
    table = []
    for row in a1_runs:
        reject = row["rejections"]["all_printed"][0]
        table.append({"run": row["label"], "verdict": row["verdict"],
                      "sim_time_s": row["metrics"]["sim_time_s"], "final_dist_m": row["metrics"]["final_dist_m"],
                      "collision_count": row["metrics"]["collision_count"], "first_reject": reject,
                      "reason_counts": row["rejections"]["reason_counts"],
                      "max_start_displacement_m": row["trajectory"]["max_distance_from_first_sample_m"]})
    checks = {"a0_pass_at_turn1": turns[0]["passed_at_turn"] == 1 and a0["verdict"] == "pass",
              "a1_two_grades_and_seven_tests_failed": all(row["verdict"] == "fail" for row in a1_runs) and len(tests) == 7,
              "a1_all_nine_completed_601s_collision0": all(row["metrics"]["sim_time_s"] == 601.0 and row["metrics"]["collision_count"] == 0 for row in a1_runs),
              "all_seven_feedbacks_reached_next_model_request_unchanged": all(row["model_delivery"]["same_decoded_result_as_saved_tool_file"] for row in delivered),
              "all_seven_feedbacks_include_rejection_lines": all(row["model_delivery"]["plan_reject_lines"] for row in delivered),
              "all_seven_test_calls_omit_sim_seconds": all(not row["sim_seconds_argument_present"] for row in delivered)}
    assert all(checks.values()), checks
    SHA["analysis/b524_a1_run_evidence.py"] = digest(Path(__file__).read_bytes())
    report = {"task": 27, "scope": "Read-only parsing of completed B524 a0/a1 records. No model/controller execution, simulation, HTTP or process inspection.",
              "a0": a0, "a0_test_feedback": a0_tool, "a1_summary_table": table, "a1_runs": a1_runs,
              "a1_model_delivered_tool_feedback": delivered, "turns": turns, "events": events, "checks": checks,
              "limitations": ["Only printed rejection diagnostics are measured; later clearances suppressed for the same reason require separate original-function CPU analysis.",
                              "min_gap_map is map center clearance minus .25m; -.25 means a sampled center lies inside or on an obstacle and does not by itself locate the exact collision geometry.",
                              "Rejected planning requests zero nominal, but sampled choose records include occasional avoidance. Do not call these missions physically motionless.",
                              "A model receiving a rejection line is proven by its next request; its reasoning and source revisions are analyzed separately."],
              "source_sha256": SHA}
    out = A / OUT
    if out.exists():
        archive = A / "results" / ("archive_b524_a1_run_evidence_" + str(time.time_ns()))
        archive.mkdir()
        shutil.copy2(out, archive / out.name)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"file": str(OUT), "sha256": digest(out.read_bytes()), "checks": checks,
                      "a1_runs": len(a1_runs), "delivered_feedbacks": len(delivered)}))


if __name__ == "__main__":
    main()
