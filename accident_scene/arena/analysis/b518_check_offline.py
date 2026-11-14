"""B-5-18 CPU checks for path geometry, diagnostic timing and v6 choice parity."""
import ast
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import b518_build_base as BUILDER
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, object_fixture, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v7.py"
V6 = BUILDER.SRC
OUT = ARENA / "results/analysis/b518_check_offline.json"


def space(rect=(0.0, 2.0, 0.0, 2.0), base=BASE):
    ns = load_ns(base)
    ns["WALLS"] = (-100.0, 100.0, -100.0, 100.0)
    ns["_MAP_RECTS"][:] = [np.array([rect])]
    seed(ns, pose=(-5.0, -5.0, 0.0), now=0.0)
    return ns


def run():
    checks, examples = {}, {}

    def record(name, passed, **details):
        checks[name] = {"pass": bool(passed), **details}

    source, old_source = BASE.read_text(), V6.read_text()
    record("builder_reproduces_base", BUILDER.build(old_source) == source)
    new_ast, old_ast = ast.parse(source), ast.parse(old_source)
    new_nodes = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    old_nodes = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    added = {"check_path", "_base_note_map_stop"}
    unchanged = set(new_nodes) == set(old_nodes) | added
    for name in old_nodes:
        node = copy.deepcopy(new_nodes[name])
        if name == "choose_command":
            node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name) and n.value.func.id == "_base_note_map_stop")]
        unchanged &= ast.dump(node, include_attributes=False) == ast.dump(old_nodes[name], include_attributes=False)
    record("all_v6_functions_unchanged_except_diagnostic_call", unchanged)

    ns = space()
    path = ns["check_path"]
    result = path([(-1, 3), (3, 3)])
    record("path_straight_clearance_and_location", abs(result["min_gap_map"]-.75) < 1e-12
           and np.allclose(result["min_location"], (0, 3)) and result["first_bad_segment"] is None, result=result)
    result = path([(-2, 3), (1, 3), (1, -1)])
    record("path_first_bad_segment_and_global_minimum", result["first_bad_segment"] == 1
           and abs(result["min_gap_map"]+.25) < 1e-12 and np.allclose(result["min_location"], (1, 2)), result=result)
    examples["blocked_polyline"] = result
    result = path([(-.5, .5), (-.5, 1.5)])
    record("path_center_half_metre_boundary_passes", result["first_bad_segment"] is None
           and result["min_gap_map"] == .25 and result["margin_m"] == .25 and result["sampling_step_m"] == .1, result=result)
    inside = path([(-.499, .5), (-.499, 1.5)])
    outside = path([(-.501, .5), (-.501, 1.5)])
    record("path_both_sides_of_boundary", inside["first_bad_segment"] == 0 and outside["first_bad_segment"] is None)
    custom = path([(-.35, .5), (-.35, 1.5)], margin_m=.1)
    record("path_custom_margin", custom["first_bad_segment"] is None and abs(custom["min_gap_map"]-.1) < 1e-12)
    duplicate = path([(-.5, 1), (-.5, 1), (-.5, 1)])
    duplicate_inside = path([(1, 1), (1, 1)])
    record("duplicate_points_are_checked", duplicate["min_gap_map"] == .25 and duplicate["first_bad_segment"] is None
           and duplicate_inside["first_bad_segment"] == 0 and duplicate_inside["min_gap_map"] == -.25)
    endpoint = path([(-1, 1), (-.48, 1)])
    record("non_multiple_length_endpoint_included", np.allclose(endpoint["min_location"], (-.48, 1))
           and endpoint["first_bad_segment"] == 0 and abs(endpoint["min_gap_map"]-.23) < 1e-12)

    # Inspect actual sample coordinates, independent of a specific distance implementation.
    sampled = []
    saved_map = ns["map_clearance"]
    def capture_map(x, y):
        sampled.append(np.column_stack((x, y)))
        return np.full(np.asarray(x).shape, 1.0)
    ns["map_clearance"] = capture_map
    polyline = np.array([[0., 0.], [.23, .17], [.23, .17], [.23, 1.03]])
    path(polyline)
    q = np.concatenate(sampled)
    record("path_sample_step_and_all_vertices", np.max(np.linalg.norm(np.diff(q, axis=0), axis=1)) <= .1+1e-12
           and all(np.any(np.linalg.norm(q-p, axis=1) < 1e-12) for p in polyline), samples=len(q))
    ns["map_clearance"] = saved_map
    wall = space()
    wall["WALLS"] = (0., 10., 0., 10.)
    wall["_MAP_RECTS"][:] = [np.array([[50., 51., 50., 51.]])]
    w = wall["check_path"]([(.5, 2), (.5, 8)])
    record("path_wall_clearance", w["min_gap_map"] == .25 and w["first_bad_segment"] is None)

    invalid_paths = [[], [[0, 0]], [0, 1], [[0, 0, 0], [1, 1, 1]], [[0, 0], [1]],
                     [[0, 0], [math.nan, 1]], [[0, 0], [math.inf, 1]], [[0, 0], [-math.inf, 1]]]
    rejected = 0
    for points in invalid_paths:
        try: path(points)
        except ValueError: rejected += 1
    record("path_invalid_points_rejected", rejected == len(invalid_paths), rejected=rejected, cases=len(invalid_paths))
    invalid_margins = (-1., math.nan, math.inf, -math.inf)
    rejected = 0
    for margin in invalid_margins:
        try: path([(0, 0), (1, 1)], margin)
        except ValueError: rejected += 1
    record("path_invalid_margin_rejected", rejected == len(invalid_margins), rejected=rejected)

    # Actual chooser integration: follow/slow hit the mapped rectangle, stop is safe.
    runtime = space((.7, 2., -1., 1.))
    exec(source[source.index("# ---- DBG 기록 ----"):source.index("# [EDIT REGION] mission controller")], runtime)
    lines = []
    runtime["builtins"] = SimpleNamespace(print=lambda *args, **kwargs: lines.append(" ".join(str(a) for a in args)))
    boundaries, selected = {}, set()
    for frame in range(1202):
        t = frame/60.
        seed(runtime, pose=(0., 0., 0.), now=t)
        runtime["_BASE_DBG_STATE"]["t"] = t
        runtime["print"]("DBG ordinary line consumes global allowance")
        v, w, info = runtime["choose_command"](.3, 0., (5., 0.))
        selected.add((v, w, info["chosen"]))
        if frame in (600, 601, 1200, 1201):
            boundaries[str(frame)] = sum("follow 가 지도에 막힘" in line for line in lines)
    warnings = [line for line in lines if "follow 가 지도에 막힘" in line]
    record("warning_after_ten_seconds_then_every_ten", boundaries == {"600": 0, "601": 1, "1200": 1, "1201": 2}
           and selected == {(0.0, 0.0, "stop")}, boundary_counts=boundaries, warning_lines=warnings)
    record("warning_bypasses_global_dbg_throttle", len(warnings) == 2
           and runtime["_BASE_DBG_STATE"]["dropped"] > 0
           and any("DBG choose t=10.00" in line for line in lines), ordinary_dropped=runtime["_BASE_DBG_STATE"]["dropped"])

    # Timer reset semantics use the same helper called by choose_command.
    def warning_ns():
        test = space(); captured = []
        test["builtins"] = SimpleNamespace(print=lambda *a, **k: captured.append(" ".join(str(x) for x in a)))
        return test, captured
    candidates = {"follow": {"v": .3, "w": 0., "t_hit_map": .5}, "stop": {"v": 0., "w": 0.}}
    for name, alteration in (("different_selection", "follow"), ("map_clears", "clear"), ("nonzero_stop_command", "moving")):
        test, captured = warning_ns(); helper = test["_base_note_map_stop"]
        helper(0., "stop", candidates); helper(9., "stop", candidates)
        altered = copy.deepcopy(candidates)
        chosen = "follow" if alteration == "follow" else "stop"
        if alteration == "clear": altered["follow"]["t_hit_map"] = None
        if alteration == "moving": altered["stop"]["v"] = .1
        helper(9.5, chosen, altered); helper(10., "stop", candidates); helper(20., "stop", candidates)
        before = len(captured); helper(20.+1/60., "stop", candidates)
        record("warning_reset_"+name, before == 0 and len(captured) == 1)
    test, captured = warning_ns(); helper = test["_base_note_map_stop"]
    helper(100., "stop", candidates); helper(109., "stop", candidates); helper(0., "stop", candidates)
    helper(10., "stop", candidates); before = len(captured); helper(10.+1/60., "stop", candidates)
    record("warning_reset_time_rewinds", before == 0 and len(captured) == 1)
    test, captured = warning_ns(); helper = test["_base_note_map_stop"]
    for t in (0., 9., 10., 10.+1/60.):
        seed(test, pose=(t, -t, t), now=t)
        helper(t, "stop", candidates)
    record("warning_not_reset_by_pose_changes", len(captured) == 1)

    # Compare complete v6/v7 chooser returns, including full candidate diagnostics.
    old, new = load_ns(V6), load_ns(BASE)
    rng = np.random.default_rng(518)
    parity, count = True, 0
    for case in range(80):
        objects = [object_fixture(float(rng.uniform(-7, -3)), float(rng.uniform(-9, -5)),
                                  float(rng.uniform(-.8, .8)), float(rng.uniform(-.8, .8)), ident=i) for i in range(case % 5)]
        for obj in objects:
            obj["seen_t"] = 30.-float(rng.uniform(0, .5)); obj["vel_ok"] = bool(rng.integers(2))
        pose = (-8.+float(rng.uniform(0, 2)), -10.+float(rng.uniform(0, 2)), float(rng.uniform(-math.pi, math.pi)))
        args = (0.0 if case % 2 else .3, float(rng.uniform(-.4, .4)), (-3., -4.))
        seed(old, objects, pose=pose, now=30.); seed(new, objects, pose=pose, now=30.)
        parity &= old["choose_command"](*args) == new["choose_command"](*args)
        parity &= old["check_motion"](*args[:2]) == new["check_motion"](*args[:2]); count += 1
    record("v6_choice_and_motion_exact_parity", parity, cases=count)

    ns = load_ns(BASE)
    objects = [object_fixture(-6.+i*.25, -10.+i*.15, -.1+i*.02, .05, ident=i) for i in range(10)]
    seed(ns, objects, pose=(-8., -10., 0.), now=10.)
    performance = {}
    for label, vf, n in (("following", .3, 7), ("alignment", 0., 9)):
        for _ in range(50): ns["choose_command"](vf, .2, (-5., -7.))
        times = []
        for _ in range(400):
            start = time.perf_counter_ns(); ns["choose_command"](vf, .2, (-5., -7.))
            times.append((time.perf_counter_ns()-start)/1e6)
        row = dict(calls=len(times), objects=10, points_per_object=20, candidates=n, horizon_s=3.,
                   median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), target_ms=3.)
        performance[label] = row; record("performance_"+label, row["median_ms"] <= 3., **row)
    path_ns = load_ns(BASE)
    polyline = [(-9., -10.8), (-5., -7.), (-5., 3.), (2., 14.), (8., 16.5)]
    times = []
    for _ in range(100):
        start = time.perf_counter_ns(); path_ns["check_path"](polyline)
        times.append((time.perf_counter_ns()-start)/1e6)
    performance["path"] = dict(calls=100, points=len(polyline), median_ms=float(np.median(times)))

    passed = sum(check["pass"] for check in checks.values())
    out = dict(verdict="pass" if passed == len(checks) else "fail", passed=passed, total=len(checks),
               base_file=str(BASE), base_sha16=hashlib.sha256(BASE.read_bytes()).hexdigest()[:16],
               checks=checks, examples=examples, performance=performance,
               limitations=["check_path samples centerline clearance minus a fixed half-width; it does not model turning or chassis length.",
                            "No Isaac, HTTP or running B-5-17 artifacts used. CPU phase sweep is a separate result."])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(safe_json(out), ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    archive_before_write(OUT); OUT.write_text(encoded)
    print(f"{out['verdict']} {passed}/{len(checks)} sha={out['base_sha16']} "
          + " ".join(f"{label}={row['median_ms']:.4f}ms" for label,row in performance.items()))
    for name, check in checks.items():
        if not check["pass"]: print("FAIL", name, check)
    return out


if __name__ == "__main__":
    raise SystemExit(0 if run()["verdict"] == "pass" else 1)
