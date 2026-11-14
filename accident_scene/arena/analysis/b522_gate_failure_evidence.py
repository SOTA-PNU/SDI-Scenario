"""Read saved B521/B522 gate sources and telemetry; evaluate static map plan only."""
import ast
from datetime import datetime
import difflib
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import sys

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A / "analysis"))
from b515_check_offline import load_ns


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    return [{"source_line": i + 1, **json.loads(line)} for i, line in enumerate(path.read_text().splitlines())]


def main():
    versions = {}
    texts = []
    for version in ("b521", "b522"):
        root = A / "results/analysis" / (version + "_gate_isaac")
        path = root / (version + "_gate_probe_carter_run.py")
        source = path.read_text()
        texts.append(source)
        tree = ast.parse(source)
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        begin = source.index("# [EDIT REGION]")
        end = source.index("# ========================= [END EDIT REGION]", begin)
        ns = load_ns(A / "base_carter_run_ros_map_lidartime_objects_check_v8.py")
        ns.update(heapq=heapq, START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"])
        exec(compile(ast.Module(body=[fns["_probe_plan"]], type_ignores=[]), "<saved map-only gate planner>", "exec"), ns)
        planned = ns["_probe_plan"]()
        rows = read_jsonl(root / "observations.jsonl")
        truth_rows = read_jsonl(root / "truth_observations.jsonl")
        chooser = []
        for i, line in enumerate((root / "stdout.log").read_text().splitlines(), 1):
            if "DBG choose t=" in line:
                text = line[line.index("DBG choose t="):]
                fields = dict(re.findall(r"(\w+)=([^ ]+)", text))
                if 11.917 <= float(fields["t"]) <= 16.917:
                    chooser.append({"line": i, "text": text, **fields})
        selected = []
        for t in (12., 12.5, 13., 13.5, 14., 14.5, 15., 15.1, 15.3, 15.5, 15.6, 15.7, 15.8, 15.9, 16., 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9):
            row = min(rows, key=lambda r: abs(r["t"] - t))
            truth = min(truth_rows, key=lambda r: abs(r["t"] - t))
            selected.append({**{k: row[k] for k in ("source_line", "t", "pose", "nominal", "target_xy", "command", "scan_points_t")},
                             "objects": [{**{k: o[k] for k in ("id", "x", "y", "vx", "vy", "vel_ok", "seen_t", "age_s", "dist")}, "n_points": len(o["points"])} for o in row["objects"]],
                             "info": row["info"], "postrun_2d_m2_body_gap_m": truth["true_body_gaps"]["west_corridor"]})
        result = json.loads((root / (version + "_gate_probe_result.json")).read_text())
        versions[version] = {"source": str(path.relative_to(A)), "source_sha256": sha(source),
                             "edit_sha256": sha(source[begin:end]),
                             "function_ast_sha256": {name: sha(ast.dump(fn)) for name, fn in fns.items()},
                             "gate_functions": {name: {"first_line": fns[name].lineno, "last_line": fns[name].end_lineno} for name in ("_probe_plan", "_probe_follow", "_gate_follow", "controller")},
                             "launch": json.loads((root / "launch.json").read_text()),
                             "result": {"verdict": result["verdict"], "metrics": result["metrics"]},
                             "static_saved_planner_path": planned, "static_path_check": ns["check_path"](planned),
                             "precontact_window_s": [11.917, 16.917], "precontact_chooser_lines": chooser,
                             "selected_observations": selected,
                             "empty_observation_times_14_to_contact": [r["t"] for r in rows if 14 <= r["t"] <= 16.917 and not r["objects"]]}
    one, two = versions["b521"], versions["b522"]
    data = {"scope": "Saved completed gates; no Isaac/GPU/HTTP/process access. Static map planner invocation only.",
            "all_source_functions_ast_identical": one["function_ast_sha256"] == two["function_ast_sha256"],
            "entire_source_identical_after_timing_env_key_rename": texts[0].replace("B521_TIMINGS_OUT", "B522_TIMINGS_OUT") == texts[1],
            "source_diff": list(difflib.unified_diff(texts[0].splitlines(), texts[1].splitlines(), fromfile="b521_snapshot", tofile="b522_snapshot", lineterm="")),
            "same_static_path": one["static_saved_planner_path"] == two["static_saved_planner_path"],
            "new_b522_fixed_scaffold_controller_was_used": False, "versions": versions,
            "limitations": ["Selected object and body gaps are horizon minima, not current physical-contact measurements.",
                            "Post-run truth fields are analytic 2D v1 geometry, never controller inputs or measured mesh-contact distances.",
                            "Recorded tracking outputs and per-sensor pts_t show loss/reacquisition, but raw lidar scans and pre-filter point sets were not saved; occlusion/filter/scan-phase causes cannot be uniquely identified.",
                            "Two different trajectories at the same time are not a controlled counterfactual comparison."]}
    out = A / "results/analysis/b522_gate_failure_evidence.json"
    if out.exists():
        shutil.copy2(out, A / "results" / f"archive_{out.name}_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(out)
    print("functions_same", data["all_source_functions_ast_identical"], "source_same_after_env_rename", data["entire_source_identical_after_timing_env_key_rename"], "path", one["static_saved_planner_path"])


if __name__ == "__main__":
    main()
