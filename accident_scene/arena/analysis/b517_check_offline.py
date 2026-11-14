"""B-5-17 CPU-only sensor fixtures, regression, invariance and performance checks."""
import ast
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import b517_build_base as BUILDER
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, object_fixture, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v6.py"
V4 = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v4.py"
OUT = ARENA / "results/analysis/b517_check_offline.json"


def space(objects=(), base=BASE):
    ns = load_ns(base)
    ns["WALLS"] = (-100.0, 100.0, -100.0, 100.0)
    ns["_MAP_RECTS"][:] = [np.array([[50.0, 51.0, 50.0, 51.0]])]
    seed(ns, objects, pose=(0.0, 0.0, 0.0), now=10.0)
    return ns


def run():
    checks, scenes = {}, {}

    def record(name, passed, **details):
        checks[name] = {"pass": bool(passed), **details}

    source, prior = BASE.read_text(), BUILDER.SRC.read_text()
    record("builder_reproduces_base", BUILDER.build(prior) == source)
    nodes = lambda s: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    now_nodes, old_nodes = nodes(source), nodes(prior)
    replaced = {"_base_choose_rank", "choose_command"}
    record("all_other_functions_preserved", set(now_nodes) == set(old_nodes)
           and all(now_nodes[n] == old_nodes[n] for n in old_nodes if n not in replaced))

    # Sensor fixtures use only visible points, observed velocities and timestamps.
    definitions = {
        "crossing_stop": ([object_fixture(1.3, 1.2, 0.0, -0.4)], (0.3, 0.0, (5.0, 0.0)), "stop"),
        "approaching_stopped_robot": ([object_fixture(-2.0, 0.0, 0.7, 0.0, "right")], (0.3, 0.0, (5.0, 0.0)), "fast"),
        "empty_road": ([], (0.3, 0.0, (5.0, 0.0)), "fast"),
        "all_unsafe": ([object_fixture(2.0, 0.0, -0.7, 0.0)], (0.3, 0.0, (5.0, 0.0)), "back"),
        "slow_priority": ([object_fixture(1.4, 0.0, 0.0, 0.0)], (0.3, 0.0, (5.0, 0.0)), "slow"),
        "alignment_approach_escape": ([object_fixture(-2.0, 0.0, 0.7, 0.0, "right")], (0.0, 0.4, (5.0, 5.0)), "toward"),
    }
    for name, (objects, args, expected) in definitions.items():
        ns = space(objects)
        v, w, info = ns["choose_command"](*args)
        cs = info["candidates"]
        count = 9 if args[0] == 0.0 else 7
        record(name, info["chosen"] == expected and len(cs) == count
               and (v, w) == (cs[expected]["v"], cs[expected]["w"]), expected=expected, actual=info["chosen"])
        scenes[name] = {"objects": objects, "command_args": args, "result": info}
        factory = ns["_base_command_candidates"]
        variants = (list(range(count - 1, -1, -1)), list(range(3, count)) + list(range(3)),
                    np.random.default_rng(516).permutation(count).tolist())
        invariant = True
        for order in variants:
            def reordered(*factory_args, order=order):
                candidates = factory(*factory_args)
                return [candidates[i] for i in order]
            ns["_base_command_candidates"] = reordered
            av, aw, ai = ns["choose_command"](*args)
            invariant &= (av, aw, ai) == (v, w, info)
        record(name + "_order_invariant", invariant, permutations=len(variants))

    # Reproduce the mechanism of a0 box06: the pose/map are saved evidence, but
    # this approaching surface is explicitly synthetic, not an inferred actual object.
    box_objects = [object_fixture(3.3, 12.8, 0.6, 0.0, "right")]
    comparisons = {}
    for label, path in (("v4", V4), ("v6", BASE)):
        ns = load_ns(path)
        seed(ns, box_objects, pose=(5.1901, 12.8011, 3.1318), now=96.5)
        comparisons[label] = ns["choose_command"](0.0, -0.4, (5.2, 14.35))[2]
    old, new = comparisons["v4"], comparisons["v6"]
    record("box_back_map_penetration_filtered",
           old["chosen"] == "back" and old["candidates"]["back"]["min_gap_map"] == 0.0
           and new["map_filtered"] and "back" in new["excluded_map_names"]
           and new["candidates"][new["chosen"]]["min_gap_map"] > 0.0,
           old_chosen=old["chosen"], new_chosen=new["chosen"], excluded=new["excluded_map_names"])
    scenes["box_back_regression"] = {"description": "Saved a0 pose/map; synthetic observed approaching surface, not actual replay.",
                                     "objects": box_objects, "results": comparisons}

    ns, oldns = space(), space(base=V4)
    factory = ns["_base_command_candidates"]
    record("positive_follow_candidates_preserved", all(factory(v, w, (5, 0)) == oldns["_base_command_candidates"](v, w)
           for v in (0.001, 0.3, 0.8) for w in (-0.4, 0.0, 0.4)))
    extras = {n: (v, w) for n, v, w in factory(0.0, 0.4, (0, -5))}
    record("zero_follow_retains_three_forward_candidates",
           extras["forward_slow"] == (0.15, 0.0) and extras["forward_fast"] == (1.0, 0.0)
           and extras["toward"] == (1.0, -0.4) and len(extras) == 9, candidates=extras)
    for name, args in (("goal_stop", (0.0, 0.0, (0.0, 0.0))), ("goal_alignment", (0.0, 0.4, (0.0, 0.0)))):
        v, w, info = ns["choose_command"](*args)
        record(name + "_preserves_safe_request", info["chosen"] == "follow" and (v, w) == args[:2]
               and len(info["candidates"]) == 9, result=info)

    rank = ns["_base_choose_rank"]
    def candidate(obj=-1.0, map_gap=0.5, obj_hit=0.3, map_hit=None, v=0.0):
        hits = [h for h in (obj_hit, map_hit) if h is not None]
        return dict(v=v, w=0.0, robust_min_gap_obj=obj, min_gap_map=map_gap,
                    t_hit_obj=obj_hit, t_hit_map=map_hit, t_hit=min(hits) if hits else None, end_pose=(0, 0, 0))
    blocked = {n: candidate(-10.0) for n in ("follow", "slow", "stop")}
    choices = {**blocked, "back": candidate(10.0, 0.0, None, 0.1), "turn_l": candidate(-0.5, 0.3)}
    got = rank(choices, (5, 0))
    record("zero_map_excluded_despite_better_object_gap", got == ("turn_l", "map_clearance_object_gap"), actual=got)
    choices["back"]["min_gap_map"] = -0.01
    record("negative_map_excluded", rank(choices, (5, 0))[0] == "turn_l")
    choices = {**blocked, "a": candidate(-0.2, 0.3), "b": candidate(-0.3, 20.0)}
    record("object_and_map_gaps_rank_separately", rank(choices, (5, 0))[0] == "a")
    for c in choices.values(): c.update(min_gap_map=0.1, t_hit_map=0.2, t_hit=0.2)
    choices["b"].update(robust_min_gap_obj=-0.2, t_hit_obj=None, t_hit_map=0.2, t_hit=0.2)
    record("none_object_ttc_ranks_as_infinity", rank(choices, (5, 0))[0] == "b")
    choices["a"] = copy.deepcopy(choices["b"])
    record("exact_ties_use_name_not_order", rank(choices, (5, 0)) == rank(dict(reversed(list(choices.items()))), (5, 0))
           == ("a", "emergency_object_gap"))
    for c in choices.values():
        c["min_gap_map"] = 0.0
    record("all_zero_map_still_returns_deterministic_candidate", rank(choices, (5, 0)) == ("a", "emergency_object_gap"))
    record("no_combined_min_gap_field", all("min_gap" not in c for c in new["candidates"].values()))

    # Physically consistent adversarial candidates: a small positive map gap is
    # unsafe under check_motion's 0.2m criterion even when its object gap is best.
    choices = {n: candidate(-10.0, .03, .3, .2) for n in ("follow", "slow", "stop")}
    choices["fast"] = candidate(10.0, .03, None, .2, v=1.0)
    choices["back"] = candidate(-1.0, .5, .3, None, v=-.2)
    got = rank(choices, (5, 0))
    record("positive_but_unsafe_map_gap_excluded", got == ("back", "map_clearance_object_gap"), actual=got)
    choices["back"].update(min_gap_map=.1, t_hit_map=.2)
    choices["fast"]["min_gap_map"] = 0.0
    got = rank(choices, (5, 0))
    record("no_clear_map_falls_back_to_positive_map", got == ("back", "map_filtered_object_gap"), actual=got)
    for c in choices.values(): c.update(min_gap_map=0.0, t_hit_map=0.0)
    got = rank(choices, (5, 0))
    record("no_positive_map_keeps_emergency_alternatives", got == ("fast", "emergency_object_gap"), actual=got)
    # Candidate geometry from the saved box00 approach at 4.5s, with an explicitly
    # synthetic approaching visible surface behind the robot. No actual lidar replay.
    pose = (-8.2967, -8.5422, 1.3321)
    x, y, yaw = pose; c, sn = math.cos(yaw), math.sin(yaw)
    ob = object_fixture(-1.5, 0.0, .4, 0.0, "right")
    ob["x"], ob["y"] = x-1.5*c, y-1.5*sn
    ob["vx"], ob["vy"] = .4*c, .4*sn
    ob["points"] = [[x+a*c-b*sn, y+a*sn+b*c] for a,b in ob["points"]]
    err = math.atan2(5-y, -5-x)-yaw
    wf = max(-.4, min(.4, 1.5*math.atan2(math.sin(err), math.cos(err))))
    comparisons = {}
    for label, path in (("v5", BUILDER.SRC), ("v6", BASE)):
        case = load_ns(path); seed(case, [ob], pose=pose, now=4.5)
        comparisons[label] = case["choose_command"](.3, wf, (-5,5))[2]
    old, new = comparisons["v5"], comparisons["v6"]
    record("box00_positive_gap_contact_buffer_regression", old["chosen"]=="fast"
           and 0 < old["min_gap_map"] < .03
           and new["chosen"]=="follow" and new["min_gap_map"]>.2
           and new["map_filter"]=="clearance" and new["excluded_map_names"]==["fast"]
           and all(c["t_hit"] is not None for c in new["candidates"].values()),
           old_choice=old["chosen"], old_map_gap=old["min_gap_map"], new_choice=new["chosen"], new_map_gap=new["min_gap_map"])
    record("filter_uses_unchanged_predictions", old["candidates"]==new["candidates"])
    scenes["box00_approach_regression"] = {"pose":pose,"objects":[ob],"command_args":[.3,wf,(-5,5)],
        "description":"Saved box00 approach pose/map with synthetic sensor points; fixes selection of map-buffer-violating escape, does not replay the full actual run.","results":comparisons}
    crowded = space([object_fixture(2,0,-.7,0)])
    info = crowded["choose_command"](.3,0,(5,0))[2]
    record("clearance_filter_without_exclusions_is_reported", info["map_filter"]=="clearance"
           and not info["map_filtered"] and info["excluded_map_names"]==[])

    # Raw and robust motion predictions must match preserved v4 for arbitrary commands.
    rng = np.random.default_rng(516)
    parity = True
    for _ in range(50):
        objects = [object_fixture(float(rng.uniform(-2, 3)), float(rng.uniform(-2, 2)),
                                  float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8)), ident=i)
                   for i in range(4)]
        for o in objects:
            o["seen_t"] = float(rng.uniform(9.5, 10.0))
            o["vel_ok"] = bool(rng.integers(2))
        seed(ns, objects, pose=(-8, -10, 0.2), now=10.0)
        seed(oldns, objects, pose=(-8, -10, 0.2), now=10.0)
        args = (float(rng.uniform(-0.2, 1.0)), float(rng.uniform(-0.4, 0.4)))
        parity &= ns["check_motion"](*args) == oldns["check_motion"](*args)
    record("v4_motion_and_uncertainty_parity", parity, cases=50)

    rejected = required = 0
    for bad in (math.nan, math.inf, -math.inf):
        for slot in range(5):
            args = [0.3, 0.2, 3.0, 0.1, 0.2]; args[slot] = bad; required += 1
            try: ns["check_motion"](*args)
            except ValueError: rejected += 1
        for args in ((bad, 0, (5, 0)), (0, bad, (5, 0)), (0, 0, (bad, 0)), (0, 0, (0, bad)), (0, 0, (5, 0), bad)):
            required += 1
            try: ns["choose_command"](*args)
            except ValueError: rejected += 1
        for field in ("vx", "vy", "seen_t"):
            ob = object_fixture(2, 0, 0, 0); ob[field] = bad; test = space([ob]); required += 1
            try: test["check_motion"](0, 0)
            except ValueError: rejected += 1
    record("nonfinite_values_rejected", rejected == required, rejected=rejected, required=required)

    runtime = space()
    exec(source[source.index("# ---- DBG 기록 ----"):source.index("# [EDIT REGION] mission controller")], runtime)
    captured = []
    runtime["builtins"] = SimpleNamespace(print=lambda *args, **kwargs: captured.append(" ".join(str(a) for a in args)))
    for t in (1.0, 1.1, 1.5, 1.6):
        seed(runtime, pose=(0, 0, 0), now=t); runtime["_BASE_DBG_STATE"]["t"] = t
        runtime["print"]("DBG other diagnostic")
        runtime["choose_command"](0, 0, (5, 0))
    chooser_lines = [s for s in captured if s.startswith("DBG choose ")]
    record("chooser_dbg_half_second_and_global_throttle", len(chooser_lines) == 2 and all("map=" in s for s in chooser_lines)
           and runtime["_BASE_DBG_STATE"]["dropped"] == 2, lines=chooser_lines)

    ns = load_ns(BASE)
    objects = [object_fixture(-6.0+i*0.25, -10.0+i*0.15, -0.1+i*0.02, 0.05, ident=i) for i in range(10)]
    seed(ns, objects, pose=(-8, -10, 0), now=10.0)
    performance = {}
    for label, vf, count in (("following", 0.3, 7), ("alignment", 0.0, 9)):
        for _ in range(50): ns["choose_command"](vf, 0.2, (-5, -7))
        times = []
        for _ in range(400):
            start = time.perf_counter_ns(); ns["choose_command"](vf, 0.2, (-5, -7))
            times.append((time.perf_counter_ns()-start)/1e6)
        p = dict(calls=len(times), objects=10, points_per_object=20, candidates=count, horizon_s=3.0,
                 median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), max_ms=max(times), target_ms=3.0)
        performance[label] = p
        record("performance_"+label, p["median_ms"] <= 3.0, **p)

    passed = sum(c["pass"] for c in checks.values())
    out = dict(verdict="pass" if passed == len(checks) else "fail", passed=passed, total=len(checks),
               base_file=str(BASE), base_sha16=hashlib.sha256(BASE.read_bytes()).hexdigest()[:16],
               checks=checks, scenes=scenes, performance=performance,
               expectation_changes=["All-unsafe ranking now honors the existing map clearance criterion before object-gap ranking; fallback fixtures retain consistent map TTC values."],
               limitations=["Synthetic regression matches failure mechanism, not missing actual lidar observations.",
                            "CPU-only helper verification; 2D phase sweep and Isaac gate are separate results."])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(safe_json(out), ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    archive_before_write(OUT); OUT.write_text(encoded)
    print(f"{out['verdict']} {passed}/{len(checks)} sha={out['base_sha16']} "
          + " ".join(f"{name}={p['median_ms']:.4f}ms" for name,p in performance.items()))
    for name, check in checks.items():
        if not check["pass"]: print("FAIL", name, check)
    return out


if __name__ == "__main__":
    raise SystemExit(0 if run()["verdict"] == "pass" else 1)
