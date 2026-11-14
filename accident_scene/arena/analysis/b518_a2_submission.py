"""Compare completed B518 a2 submissions; no GPU, HTTP, processes or a3 reads."""
import ast
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
DATA = ARENA / "results/turns/prompt_b5_18"
LOGS = DATA / "isaac_logs"
STEM = "gpu_gpt_oss_120b_pB518_a2"
OUT = ARENA / "results/analysis/b518_a2_submission_evidence.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reference(path, needle):
    text = path.read_text()
    i = text.index(needle)
    return {"file": str(path.relative_to(ARENA)), "line": text[:i].count("\n") + 1}


def main():
    sys.path.insert(0, str(ARENA))
    import agent_loop as A
    A.BASE = A.SP.BASE = (ARENA / "base_carter_run_ros_map_lidartime_objects_check_v7.py").read_text()
    A.A21 = A.A22 = A.A23 = A.A4T = False
    A.B_MODE = "b5"
    turns_path = DATA / f"{STEM}_turns.json"
    turns = json.loads(turns_path.read_text())
    comparisons = []
    for turn, call, test_suffix in [(1, 1, "_t1_test01"), (2, 5, "_t2_test03")]:
        grade_suffix = "_t2" if turn == 2 else ""
        raw_path = DATA / f"{STEM}_c{call:02d}_raw.json"
        raw = json.loads(raw_path.read_text())
        args = json.loads(raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        code = args["code"]
        reply_path = DATA / f"{STEM}{grade_suffix}_reply.md"
        extracted = A.TL.extract(reply_path.read_text())
        test_path = LOGS / f"{STEM}{test_suffix}_carter_run.py"
        grade_path = LOGS / f"{STEM}{grade_suffix}_carter_run.py"
        test, grade = test_path.read_text(), grade_path.read_text()
        test_region, grade_region = A.region_of(test), A.region_of(grade)
        test_result_path = LOGS / f"{STEM}{test_suffix}_result.json"
        grade_result_path = LOGS / f"{STEM}{grade_suffix}_result.json"
        test_result, grade_result = json.loads(test_result_path.read_text()), json.loads(grade_result_path.read_text())
        path_logs = {}
        for label, suffix in [("test", test_suffix), ("submission", grade_suffix)]:
            log_path = LOGS / f"{STEM}{suffix}.carter-run.stdout.log"
            path_logs[label] = [{"file": str(log_path.relative_to(ARENA)), "line": i, "text": line}
                                for i, line in enumerate(log_path.read_text().splitlines(), 1) if "DBG path points=" in line]
        old_ast, new_ast = ast.parse(test_region), ast.parse(grade_region)
        raw_ast_equal = ast.dump(old_ast) == ast.dump(new_ast)
        renamed_unused_targets = 0
        for fn in old_ast.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "_plan_path":
                for n in ast.walk(fn):
                    if (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
                            and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                            and n.iter.func.id == "range" and len(n.iter.args) == 1
                            and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == "max_iter"):
                        assert not any(isinstance(k, ast.Name) and isinstance(k.ctx, ast.Load) and k.id == "i" for k in ast.walk(fn))
                        n.target.id = "_"
                        renamed_unused_targets += 1
        assert A.build(code) == test and A.build(extracted) == grade
        comparisons.append({
            "turn": turn, "test_call": call,
            "source_files": {"tool_raw": str(raw_path.relative_to(ARENA)), "reply": str(reply_path.relative_to(ARENA)),
                             "test_executable": str(test_path.relative_to(ARENA)), "grade_executable": str(grade_path.relative_to(ARENA))},
            "tool_argument_keys": sorted(args), "sim_seconds_explicit": "sim_seconds" in args,
            "tool_code_chars": len(code), "final_extracted_chars": len(extracted),
            "tool_code_sha256": sha(code), "final_extracted_sha256": sha(extracted),
            "raw_code_string_equal": code == extracted,
            "raw_difference_is_only_one_final_newline": extracted == code + "\n",
            "raw_normalized_equal": A.norm_region(code) == A.norm_region(extracted),
            "raw_ast_equal": ast.dump(ast.parse(code)) == ast.dump(ast.parse(extracted)),
            "stored_executable_byte_equal": test_path.read_bytes() == grade_path.read_bytes(),
            "stored_executable_sha256": {"test": sha(test), "submission": sha(grade)},
            "stored_edit_region_byte_equal": test_region.encode() == grade_region.encode(),
            "stored_edit_region_chars": {"test": len(test_region), "submission": len(grade_region)},
            "stored_edit_region_sha256": {"test": sha(test_region), "submission": sha(grade_region)},
            "stored_edit_region_normalized_equal": A.norm_region(test_region) == A.norm_region(grade_region),
            "stored_edit_region_ast_equal": raw_ast_equal,
            "ast_equal_after_unused_loop_target_rename": ast.dump(old_ast) == ast.dump(new_ast),
            "unused_loop_targets_renamed_in_comparison_only": renamed_unused_targets,
            "edit_unified_diff": "\n".join(difflib.unified_diff(test_region.splitlines(), grade_region.splitlines(), fromfile=test_path.name, tofile=grade_path.name)),
            "actual_harness_build_matches_both_retained_files": True,
            "static_checks": {"test": A.static_check(test), "submission": A.static_check(grade)},
            "test_result": {"file": str(test_result_path.relative_to(ARENA)), "verdict": test_result["verdict"], "oracles": test_result["oracles"], "metrics": test_result["metrics"]},
            "grade_result": {"file": str(grade_result_path.relative_to(ARENA)), "verdict": grade_result["verdict"], "oracles": grade_result["oracles"], "metrics": grade_result["metrics"]},
            "initial_path_debug": path_logs,
        })
    final = comparisons[1]
    assert final["stored_executable_byte_equal"] and final["stored_edit_region_byte_equal"]
    assert final["raw_difference_is_only_one_final_newline"]
    events_path = DATA / f"{STEM}_events.jsonl"
    events = [{**json.loads(line), "source_line": i} for i, line in enumerate(events_path.read_text().splitlines(), 1)]
    tool_events = [e for e in events if e["type"] == "tool_call"]
    last_test = next(e for e in events if e["type"] == "tool_result" and e["call"] == 5)
    final_submit = next(e for e in events if e["type"] == "submission" and e["turn"] == 2)
    final_grade = next(e for e in events if e["type"] == "grading" and e["turn"] == 2)
    output = {
        "scope": "Completed B518 a2 material only, plus static v7/harness. No a3 artifacts, live logs, processes, HTTP, or simulations inspected or run.",
        "conclusion": {
            "turn2_test03_and_submission_executable_identical": True,
            "turn2_edit_diff": "Empty: byte-identical stored EDIT REGION and full executable. Raw final fenced extraction adds only one terminal newline, removed by actual splice normalization.",
            "not_a_post_pass_controller_regression": True,
            "same_source_different_planned_paths": "Both files use random RRT planning; actual initial DBG reports test99points/gap0.351 and submission88points/gap0.348. The generated route differed before subsequent dynamics.",
            "path_gap_units": "DBG min_gap comes from check_path, after subtracting half-width0.25; center clearances are about0.601 and0.598, not0.351 and0.348.",
            "causal_limit": "Route difference is established. Without saved full planned points or sensor streams this comparison alone cannot assign all later timeout behavior to route randomness or exclude sensor timing effects.",
            "turn1_one_line": "시험01 PASS 뒤 제출은83.733s에 arena_mover_01과 처음 접촉하여 누적220건, 최종거리20.179m로 FAIL; t1 code differs in formatting/comments and an unused i→_ loop binding, with otherwise identical AST.",
            "test_duration": "All four a2 tool calls omit sim_seconds and run with600s; the t2 test03 PASS ttg285.717s is not a200s PASS.",
        },
        "comparisons": comparisons,
        "tool_timing": {
            "all_four_tool_argument_events": tool_events,
            "all_four_omit_sim_seconds": len(tool_events) == 4 and all(e["sim_seconds"] is None for e in tool_events),
            "executed_tests": [{"turn": t["turn"], "test": x["test"], "sim_seconds": x["sim_seconds"], "time_to_goal_s": x.get("time_to_goal_s"), "verdict": x["verdict"]} for t in turns["turns"] for x in t["tests"]],
            "last_test_result": last_test, "submission": final_submit, "grading": final_grade,
            "test_result_to_submission_seconds": round(final_submit["ts"] - last_test["ts"], 3),
            "event_file": str(events_path.relative_to(ARENA)),
        },
        "source_references": {
            "random_import": reference(LOGS / f"{STEM}_t2_carter_run.py", "import random"),
            "random_sampling": reference(LOGS / f"{STEM}_t2_carter_run.py", "if random.random() < 0.2:"),
            "plan_once": reference(LOGS / f"{STEM}_t2_carter_run.py", "if not _PATH_PLANNED:"),
            "check_path_debug": reference(LOGS / f"{STEM}_t2_carter_run.py", 'print(f"DBG path points='),
            "exact_submission_prose": reference(ARENA / "PROMPT_B5_18.md", "시험 주행에서 pass 를 확인했으면 그 시험의 `code`"),
            "200_second_argument_prose": reference(ARENA / "PROMPT_B5_18.md", '`sim_seconds` (매 호출 필수)'),
            "no_random_dependency_prose": reference(ARENA / "PROMPT_B5_18.md", "난수·벽시계·파일 입출력·스레드에 의존하지 말고"),
            "limited_static_checker": reference(ARENA / "agent_loop.py", "def static_check(text):"),
        },
        "proposal_candidates_for_parent": [
            "[harness] Validate the existing allowed-import/deterministic-planning contract before execution, including random use. Preserve the actually computed path in test and grading diagnostics so identical source is not mistaken for identical route.",
            "[harness] Require the top-level sim_seconds=200 argument or report an input error before execution; the285.717s PASS demonstrates that current prose alone does not enforce the intended test duration.",
        ],
    }
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_task14_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("wrote", OUT)
    print("t2_full_sha16", final["stored_executable_sha256"]["test"][:16])
    print("t2_edit_sha16", final["stored_edit_region_sha256"]["test"][:16])


if __name__ == "__main__":
    main()
