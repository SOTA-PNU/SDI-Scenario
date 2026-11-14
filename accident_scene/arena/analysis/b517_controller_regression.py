"""CPU diagnostics of a2 caller stop gates and its first two path simplifiers.

These are synthetic sensor/map fixtures, NOT replay of a2's unlogged sensors.
The actual saved EDIT controllers and actual v5/v6 chooser are used. The only
controller intervention removes the caller's object-based nominal-command
overwrite; the chooser still sees exactly the same synthetic observations.
No simulator, network, process inspection, or closed-loop safety claim.
"""
import ast
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from b515_check_offline import load_ns, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
SAVED = ARENA / "results/turns/prompt_b5_16/isaac_logs"
STEM = "gpu_gpt_oss_120b_pB516_a2"
OUT = ARENA / "results/analysis/b517_controller_regression.json"
BEGIN = "# [EDIT REGION] mission controller - modify ONLY this block"
END = "# ========================= [END EDIT REGION]"


def source_record(path):
    return {"path": str(path.relative_to(ARENA)),
            "sha16": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}


def edit_region(path):
    source = path.read_text()
    start = source.index(BEGIN)
    return source[start:source.index(END, start)], source[:start].count("\n")


def make_object(x, y):
    # A small observed surface with its tracker centroid at the points' mean.
    points = [[x, float(y + dy)] for dy in np.linspace(-0.1, 0.1, 21)]
    return {"id": 7001, "x": x, "y": y, "vx": 0.0, "vy": 0.0,
            "vel_ok": True, "radius": 0.1, "points": points}


def isolated_controller(base, path, intervention):
    ns = load_ns(base)
    ns["WALLS"] = (-100.0, 100.0, -100.0, 100.0)
    ns["_MAP_RECTS"][:] = [np.array([[50.0, 51.0, 50.0, 51.0]])]
    region, offset = edit_region(path)
    tree = ast.parse(region)
    actual = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "controller")
    start = next(i for i, n in enumerate(actual.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "objs" for t in n.targets))
    end = next(i for i in range(start + 1, len(actual.body))
               if isinstance(actual.body[i], ast.Assign)
               and isinstance(actual.body[i].value, ast.Call)
               and isinstance(actual.body[i].value.func, ast.Name)
               and actual.body[i].value.func.id == "choose_command")
    removed = actual.body[start:end]
    metadata = {"original_controller_ast_sha16": hashlib.sha256(ast.dump(actual).encode()).hexdigest()[:16],
                "manual_gate_source_lines": [offset + removed[0].lineno, offset + removed[-1].end_lineno],
                "intervention": "remove only objs/stop loop and nominal override" if intervention else "none"}
    exec(compile(tree, str(path), "exec"), ns)
    if intervention:
        changed = copy.deepcopy(actual)
        del changed.body[start:end]
        module = ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[]))
        exec(compile(module, str(path) + ":manual_gate_removed", "exec"), ns)
    # Give both variants the same synthetic map-validated straight path. Do not
    # execute the submitted hard-coded planner or use any actual mover geometry.
    ns["START"], ns["GOAL"] = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    ns["_path"], ns["_path_idx"] = [(0.0, 0.0), (5.0, 0.0)], 0
    return ns, metadata


def run_controller(base, source, obj, intervention):
    ns, metadata = isolated_controller(base, source, intervention)
    actual_choose, calls = ns["choose_command"], []
    debug = []
    ns["print"] = lambda *args, **kwargs: debug.append(" ".join(map(str, args)))

    def logged_choose(v, w, target):
        observed = ns["observed_objects"]()
        result = actual_choose(v, w, target)
        calls.append({"input": [v, w], "target": list(target), "objects": observed,
                      "command": list(result[:2]), "info": result[2]})
        return result

    ns["choose_command"] = logged_choose
    returned = []
    for k in range(20):
        now = 10.0 + k * 0.1
        seed(ns, [obj], pose=(0.0, 0.0, 0.0), now=now)
        returned.append(list(ns["controller"](now, (0.0, 0.0, 0.0), None)))
    return {**metadata, "held_pose": [0.0, 0.0, 0.0], "tick_count": len(calls),
            "note": "20 repeated held-pose decisions, not an integrated robot rollout",
            "first": calls[0], "last": calls[-1], "inputs": [r["input"] for r in calls],
            "commands": [r["command"] for r in calls], "returned": returned,
            "reasons": [r["info"]["reason"] for r in calls], "debug_first": debug[:1]}


def saved_simplifier(path, clearance):
    region, offset = edit_region(path)
    tree = ast.parse(region)
    planner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_plan_path")
    # Reuse the exact actual simplification tail, rather than rewriting its bug.
    idx = next(i for i, n in enumerate(planner.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "not raw_path")
    tail = copy.deepcopy(planner.body[idx:])
    fn = ast.FunctionDef(name="saved_simplify",
                         args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="raw_path")],
                                            kwonlyargs=[], kw_defaults=[], defaults=[]),
                         body=tail, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {"segment_clearance": clearance, "_CLEARANCE_THRESH": 0.6}
    exec(compile(module, str(path) + ":simplification_tail", "exec"), ns)
    return ns["saved_simplify"], [offset + tail[0].lineno, offset + tail[-1].end_lineno]


def anchored_los(points, clearance, threshold=0.6):
    kept, i = [points[0]], 0
    while i < len(points) - 1:
        visible = [j for j in range(i + 1, len(points)) if clearance(points[i], points[j]) >= threshold]
        if not visible:
            raise ValueError("input path has no safe next segment")
        i = max(visible)
        kept.append(points[i])
    return kept


def main():
    checks, controller_scenes, path_scenes = [], [], []

    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})

    bases = {v: ARENA / f"base_carter_run_ros_map_lidartime_objects_check_{v}.py" for v in ("v5", "v6")}
    controllers = {
        "cone": (SAVED / f"{STEM}_carter_run.py", make_object(2.4, 0.8)),
        "line_proximity": (SAVED / f"{STEM}_t2_carter_run.py", make_object(2.4, 0.45)),
    }
    for version, base in bases.items():
        for kind, (source, obj) in controllers.items():
            original = run_controller(base, source, obj, False)
            continuous_nominal = run_controller(base, source, obj, True)
            a, b = original["first"], continuous_nominal["first"]
            prefix = version + "_" + kind
            check(prefix + "_caller_overwrites_safe_follow_with_zero",
                  all(c == [0.0, 0.0] for c in original["inputs"])
                  and all(c == [0.0, 0.0] for c in original["commands"])
                  and all(r == "follow_safe" for r in original["reasons"])
                  and a["info"]["chosen"] == "follow")
            check(prefix + "_same_sensor_and_target_reach_real_chooser",
                  a["objects"] == b["objects"] and a["target"] == b["target"]
                  and len(a["objects"]) == 1 and len(a["objects"][0]["points"]) == 21)
            chosen = b["info"]["candidates"][b["info"]["chosen"]]
            check(prefix + "_path_nominal_allows_safe_forward_horizon",
                  all(c == [0.3, 0.0] for c in continuous_nominal["inputs"])
                  and b["command"][0] > 0.0 and chosen["t_hit"] is None
                  and b["info"]["candidates"]["follow"]["t_hit"] is None)
            controller_scenes.append({"name": prefix, "base": source_record(base),
                                      "controller": source_record(source), "synthetic_object": obj,
                                      "original": original, "manual_gate_removed": continuous_nominal})

    # A concave free-space route around an artificial square: every raw adjacent
    # edge is clear, while the direct start-goal segment penetrates the square.
    ns = load_ns(bases["v6"])
    ns["WALLS"] = (-5.0, 10.0, -5.0, 10.0)
    ns["_MAP_RECTS"][:] = [np.array([[1.0, 3.0, 1.0, 3.0]])]
    clearance = ns["segment_clearance"]
    raw = [(0.0, float(y)) for y in range(5)] + [(float(x), 4.0) for x in range(1, 5)]
    fixed = anchored_los(raw, clearance)
    for test_number in (1, 2):
        source = SAVED / f"{STEM}_t1_test{test_number:02d}_carter_run.py"
        simplify, lines = saved_simplifier(source, clearance)
        old = simplify(raw)
        prefix = f"test{test_number:02d}_simplifier"
        raw_min = min(clearance(a, b) for a, b in zip(raw, raw[1:]))
        fixed_min = min(clearance(a, b) for a, b in zip(fixed, fixed[1:]))
        implied_gap = clearance(raw[0], old[0])
        check(prefix + "_valid_detour_input", raw_min >= 0.6)
        check(prefix + "_actual_tail_loses_start_and_detour", old == [raw[-1]] and implied_gap < 0.6)
        check(prefix + "_anchored_los_retains_safe_start_goal_route",
              fixed[0] == raw[0] and fixed[-1] == raw[-1] and len(fixed) >= 3 and fixed_min >= 0.6)
        path_scenes.append({"source": source_record(source), "simplifier_source_lines": lines,
                            "synthetic_rectangle": [1.0, 3.0, 1.0, 3.0], "threshold": 0.6,
                            "raw_path": raw, "raw_min_segment_clearance": raw_min,
                            "actual_simplifier_output": old, "implied_start_to_target_clearance": implied_gap,
                            "anchored_los_output": fixed, "anchored_min_segment_clearance": fixed_min})
    report = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
              "execution": "CPU synthetic fixtures only; no GPU, HTTP, simulator, process checks, or actual a2 sensor replay",
              "interpretation": [
                  "The saved caller can erase a safe positive path-follow request using a centroid cone/line heuristic.",
                  "Both actual bases correctly retain that safe zero-speed request as follow_safe; the caller originates the stop.",
                  "Removing only the caller heuristic leaves identical observed points available to the real chooser and permits a positive command safe within its 3s horizon.",
                  "Repeated held-pose fixtures establish the decision mechanism, not actual a2 observation history or a guarantee of eventual goal attainment.",
                  "The saved first two simplifier tails can collapse a valid detour to the goal alone; anchoring the start and validating every retained segment prevents this synthetic failure."],
              "controller_scenes": controller_scenes, "path_scenes": path_scenes,
              "script_sha16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        archive = ARENA / "results" / ("archive_b517_controller_regression_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        shutil.copy2(OUT, archive)
    OUT.write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "file": str(OUT)}))
    for c in checks:
        if not c["pass"]:
            print(json.dumps(c))
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
