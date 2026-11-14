"""Read completed B521 a0 sources/logs only; no simulation/process/network use."""
import ast
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

A = Path(__file__).resolve().parents[1]
D = A / "results/turns/prompt_b5_21"
P = "gpu_gpt_oss_120b_pB521_a0"


def region(text):
    start = text.index("# [EDIT REGION]")
    end = text.index("# ========================= [END EDIT REGION]", start)
    return text[start:end]


def main():
    turns = json.loads((D / (P + "_turns.json")).read_text())["turns"]
    tests = {t["variant"]: t for t in turns[0]["tests"]}
    rows = []
    for suffix in [f"_t1_test{i:02d}" for i in range(1, 9)] + ["", "_t2"]:
        name = P + suffix
        code_path = D / "isaac_logs" / (name + "_carter_run.py")
        text = code_path.read_text()
        edit = region(text)
        tree = ast.parse(text)
        edit_start = text[:text.index("# [EDIT REGION]")].count("\n") + 1
        edit_end = text[:text.index("# ========================= [END EDIT REGION]", text.index("# [EDIT REGION]"))].count("\n") + 1
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and edit_start < n.lineno < edit_end]
        controller = next(n for n in funcs if n.name == "controller")
        log_path = D / "isaac_logs" / (name + ".carter-run.stdout.log")
        log_lines = log_path.read_text().splitlines()
        chooser = [{"line": i + 1, "text": line[line.index("DBG choose "):], **dict(re.findall(r"(\w+)=([^ ]+)", line[line.index("DBG choose "):] ))}
                   for i, line in enumerate(log_lines) if "DBG choose " in line]
        planning = [{"line": i + 1, "text": line[line.index("DBG plan_path "):]} for i, line in enumerate(log_lines) if "DBG plan_path " in line]
        result_path = D / "isaac_logs" / (name + "_result.json")
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        traj_path = D / "isaac_logs" / (name + "_trajectory.csv")
        trajectory = None
        if traj_path.exists():
            records = list(csv.DictReader(traj_path.open()))
            start = records[0]
            trajectory = {"samples": len(records), "first": start, "last": records[-1],
                          "max_displacement_from_first_m": max(math.hypot(float(r["x"]) - float(start["x"]), float(r["y"]) - float(start["y"])) for r in records)}
        patterns = ("while _PLAN_CLEARANCE", "max(0.5, _PLAN_CLEARANCE", "_NEXT_PLAN_T = t + 5.0", "replan_now =", "observed_objects()", "o['dist'] < 2.0", "target_xy = GOAL[:2]", "choose_command(v_follow", "def lookahead_target", "def plan_path", "check_path(path", "min_x = wx0 +", "if map_clearance(cx_w", "_REPLAN_PENDING = True", "v_follow, w_follow = 0.0, 0.0")
        citations = [{"line": i + 1, "text": line} for i, line in enumerate(text.splitlines()) if edit_start <= i + 1 < edit_end and any(p in line for p in patterns)]
        rows.append({"label": suffix or "grading1", "source": str(code_path.relative_to(A)), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "edit_sha256": hashlib.sha256(edit.encode()).hexdigest(), "edit_ast": ast.dump(ast.parse(edit)),
                     "functions": [{"name": n.name, "line": n.lineno} for n in funcs],
                     "controller_calls": [{"name": n.func.id, "line": n.lineno} for n in sorted(ast.walk(controller), key=lambda n: getattr(n, "lineno", 0)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("plan_path", "lookahead_target", "follow_cmd", "choose_command", "observed_objects")],
                     "citations": citations, "result": {"verdict": result["verdict"], "metrics": result["metrics"], "oracles": result["oracles"]} if result else None,
                     "test_record": tests.get(name), "trajectory": trajectory, "log": str(log_path.relative_to(A)),
                     "chooser_samples": len(chooser), "choice_counts": dict(Counter(r["pick"] for r in chooser)),
                     "reason_counts": dict(Counter(r["why"] for r in chooser)), "first_chooser": chooser[:5], "last_chooser": chooser[-2:],
                     "planning_counts": dict(Counter(r["text"].split(" clearance")[0] for r in planning)), "first_planning": planning[:3], "last_planning": planning[-2:]})
    tools = []
    for i in range(1, 9):
        p = D / f"{P}_c{i:02d}_tool1.json"
        record = json.loads(p.read_text())
        args = json.loads(record["tool_call"]["function"]["arguments"])
        tools.append({"call": i, "file": str(p.relative_to(A)), "argument_keys": list(args), "sim_seconds_present": "sim_seconds" in args, "sim_seconds": args.get("sim_seconds")})
    submission = rows[-2:]
    for r in submission:
        r["identical_test_edit_bytes"] = [other["label"] for other in rows[:-2] if other["edit_sha256"] == r["edit_sha256"]]
        r["identical_test_edit_ast"] = [other["label"] for other in rows[:-2] if other["edit_ast"] == r["edit_ast"]]
    for r in rows:
        del r["edit_ast"]
    data = {"scope": "Completed B521 a0 only; saved artifacts read without GPU, HTTP, process inspection or a1 access.",
            "rows": rows, "tool_arguments": tools,
            "findings": ["Both graded submissions retain one chooser call and return its command unchanged, but manually zero nominal inputs when any observed distance is below2m.",
                         "Grade1 logs repeated successful planning at1m; grade2 does not call its retained planner at all. A start-cell failure is not established as their stop cause.",
                         "Tests1-4 scan from the route beginning for a point farther than0.2m, without nearest-progress projection. Test5 adds a positive heading-dot check; test6 adds manual waiting and per-tick pending replanning; test7 removes that per-tick pending write; test8 removes lookahead in favor of an index.",
                         "All8 tool calls omit sim_seconds; the saved test records show600s. Six tests time out without contacts, one contacts mover02, and test6 hits the1700s wall-time guard.",
                         "A saturated clearance while-loop can fail to terminate at0.5m, but the saved graded logs do not show that branch occurring."],
            "limitations": ["Chooser DBG prints selected names and gaps, not the submitted nominal velocities, targets or full observed-object lists.",
                            "A follow choice may carry zero nominal velocity; counts are sampled DBG lines, not all control ticks.",
                            "Source mechanisms and saved trajectories establish structural deviations; sensor-state replay and map geometry are separate evidence.",
                            "Test6 has no completed result JSON or trajectory; its last chooser line is not an exact simulation termination time."]}
    out = A / "results/analysis/b521_a0_submission_evidence.json"
    if out.exists():
        shutil.copy2(out, A / "results" / f"archive_{out.name}_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(out)
    for r in submission:
        print(r["label"], "bytes", r["identical_test_edit_bytes"], "AST", r["identical_test_edit_ast"], "maxstart", r["trajectory"]["max_displacement_from_first_m"])


if __name__ == "__main__":
    main()
