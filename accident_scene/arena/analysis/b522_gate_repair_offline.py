"""B522 v9 retention repair checks plus preserved motion geometry and chooser tests."""
import ast
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import b519_build_base as HORIZON_BUILDER
import b522_build_base_v9 as BUILDER
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, object_fixture, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
BASE = BUILDER.OUT
V7 = HORIZON_BUILDER.SRC
OUT = ARENA / "results/analysis/b522_gate_repair_offline.json"


def space(objects=(), base=BASE, rect=(50., 51., 50., 51.)):
    ns = load_ns(base)
    ns["WALLS"] = (-100., 100., -100., 100.)
    ns["_MAP_RECTS"][:] = [np.array([rect])]
    seed(ns, objects, pose=(0., 0., 0.), now=10.)
    return ns


def run():
    checks, examples = {}, {}

    def record(name, passed, **details):
        checks[name] = {"pass": bool(passed), **details}

    source, old_source = BASE.read_text(), V7.read_text()
    record("builder_reproduces_base", BUILDER.build(HORIZON_BUILDER.build(old_source)) == source)
    nodes = lambda s: {n.name: n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    new_nodes, old_nodes = nodes(source), nodes(old_source)
    def executable(node):
        node=copy.deepcopy(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body=node.body[1:]
        return ast.dump(node,include_attributes=False)
    additions = {"_base_map_horizon", "_base_check_motion"}
    unchanged = set(new_nodes) == set(old_nodes) | additions
    unchanged &= all(executable(old_nodes[n]) == executable(new_nodes[n])
                     for n in old_nodes if n not in {"check_motion", "choose_command"})
    record("all_other_functions_preserved", unchanged)
    record("public_motion_signature_preserved", ast.dump(new_nodes["check_motion"].args) == ast.dump(old_nodes["check_motion"].args)
           and ast.dump(new_nodes["check_motion"].returns) == ast.dump(old_nodes["check_motion"].returns))
    object_block = lambda text: text[text.index("    objects = observed_objects()", text.index("def check_motion")):
                                   text.index("    perimeter = _BASE_BODY_PERIMETER", text.index("def check_motion"))]
    original_object_block = object_block(old_source)
    new_object_block = object_block(source).split("    # Object prediction, uncertainty")[0]
    record("object_geometry_uncertainty_block_unchanged", new_object_block == original_object_block,
           block_sha16=hashlib.sha256(original_object_block.encode()).hexdigest()[:16])

    old, new = load_ns(V7), load_ns(BASE)
    record("configuration_and_body_unchanged", all(old[k] == new[k] for k in
           ("_BASE_BODY_BOX", "_BASE_MOTION_UNCERTAINTY", "_BASE_ESCAPE_CONFIG")))
    cases = [(0., 0., 3., .8), (.3, .4, 3., .8), (1., .4, 3., 1.),
             (-.2, 0., 3., .8), (2., 0., 3., 1.5), (0., 5., 3., .5+5./2.68),
             (1., .4, .35, .35), (1., .4, 0., 0.)]
    formula_ok = all(abs(new["_base_map_horizon"](v, w, h)-expected) < 1e-12 for v, w, h, expected in cases)
    record("map_horizon_formula_and_short_public_horizon_cap", formula_ok, cases=cases,
           config=new["_BASE_MAP_HORIZON_CONFIG"])

    # Complete dict parity of the original public function, including all geometry,
    # unknown/known velocity uncertainty, timestamps, raw hit fields and end_pose.
    rng = np.random.default_rng(519)
    parity, core_equal_horizon, core_objects, metadata = True, True, True, True
    object_keys = ("min_gap_obj", "t_hit_obj", "hit_obj_id", "raw_t_hit_obj", "robust_min_gap_obj", "end_gap_obj",
                   "robust_end_gap_obj", "future_min_gap_obj", "end_pose")
    candidate_keys = tuple(k for k in object_keys if k not in ("hit_obj_id", "end_gap_obj"))
    for case in range(100):
        objects = [object_fixture(float(rng.uniform(-8., -3.)), float(rng.uniform(-10., -5.)),
                                  float(rng.uniform(-.8, .8)), float(rng.uniform(-.8, .8)), ident=i)
                   for i in range(case % 6)]
        for obj in objects:
            obj["seen_t"] = 30. - float(rng.uniform(0., .9))
            obj["vel_ok"] = bool(rng.integers(2))
        pose = (-8.+float(rng.uniform(0., 2.)), -10.+float(rng.uniform(0., 2.)), float(rng.uniform(-math.pi, math.pi)))
        seed(old, objects, pose=pose, now=30.); seed(new, objects, pose=pose, now=30.)
        v, w = float(rng.uniform(-.3, 1.)), float(rng.uniform(-.8, .8))
        horizon = (0., .03, .73, 1., 2.93, 3.)[case % 6]
        dt = (.07, .1, .3)[case % 3]
        margin = (.0, .2, .35)[case % 3]
        expected = old["check_motion"](v, w, horizon, dt, margin)
        actual = new["check_motion"](v, w, horizon, dt, margin)
        parity &= actual == expected
        core_equal_horizon &= new["_base_check_motion"](v, w, horizon, dt, margin, horizon) == expected
        mh = new["_base_map_horizon"](v, w, horizon)
        short = new["_base_check_motion"](v, w, horizon, dt, margin, mh)
        core_objects &= all(short[k] == expected[k] for k in object_keys)
        vf = 0. if case % 2 else .3
        _, _, info = new["choose_command"](vf, w, (-3., -4.), horizon)
        for c in info["candidates"].values():
            expected = old["check_motion"](c["v"], c["w"], horizon)
            core_objects &= all(c[k] == expected[k] for k in candidate_keys)
            metadata &= c["object_horizon_s"] == horizon and c["map_horizon_s"] == new["_base_map_horizon"](c["v"], c["w"], horizon)
        metadata &= info["object_horizon_s"] == horizon and info["map_horizon_s"] == info["candidates"][info["chosen"]]["map_horizon_s"]
    record("public_motion_exact_v7_parity", parity, cases=100)
    record("private_equal_horizon_exact_v7_parity", core_equal_horizon, cases=100)
    record("short_map_keeps_all_object_fields_and_full_end_pose", core_objects, cases=100)
    record("candidate_and_top_level_actual_horizon_metadata", metadata, cases=100)

    # The map-only horizon is shorter while a future object at2.3s remains unsafe.
    ns = space([object_fixture(1.3, 0., 0., 0.)])
    _, _, late = ns["choose_command"](.3, 0., (5., 0.))
    follow = late["candidates"]["follow"]
    record("object_collision_after_map_horizon_is_not_discarded", follow["map_horizon_s"] == .8
           and follow["object_horizon_s"] == 3. and follow["t_hit_obj"] > 2.
           and follow["t_hit_obj"] <= 3. and late["chosen"] == "slow"
           and late["candidates"]["slow"]["t_hit_obj"] is None
           and abs(follow["end_pose"][0]-.9) < 1e-12, follow=follow, chosen=late["chosen"])
    examples["late_object"] = late

    # Physical map obstruction remains a stop if it is inside even the slow horizon.
    ns = space(rect=(.42, 2., -1., 1.))
    v, w, blocked = ns["choose_command"](.3, 0., (5., 0.))
    record("immediate_map_obstacle_still_stops", (v, w, blocked["chosen"]) == (0., 0., "stop")
           and blocked["candidates"]["follow"]["t_hit_map"] is not None
           and blocked["candidates"]["slow"]["t_hit_map"] is not None
           and blocked["candidates"]["stop"]["t_hit_map"] is None, info=blocked)
    # A long-horizon-only map hit disappears from choose diagnostics, not from the public helper.
    ns = space(rect=(1., 2., -1., 1.))
    public = ns["check_motion"](.3, 0.)
    _, _, shorter = ns["choose_command"](.3, 0., (.5, 0.))
    short = shorter["candidates"]["follow"]
    record("map_hit_beyond_short_horizon_excluded_only_inside_chooser", public["t_hit_map"] is not None
           and public["t_hit_map"] > short["map_horizon_s"] and short["t_hit_map"] is None
           and short["min_gap_map"] > .2 and short["end_pose"] == public["end_pose"], public=public, follow=short)
    endpoint_ns = space(rect=(1., 2., -1., 1.))
    endpoint = endpoint_ns["_base_check_motion"](1., 0., 2., dt=.1, margin_m=.15, map_horizon_s=.73)
    record("short_map_grid_includes_fractional_endpoint", endpoint["t_hit_map"] == .73
           and abs(endpoint["min_gap_map"]-.14) < 1e-12, result=endpoint)
    wall = space(); wall["WALLS"] = (-10., 1., -10., 10.)
    r = wall["_base_check_motion"](1., 0., 3., map_horizon_s=.8)
    record("short_map_checks_walls_and_full_body", abs(r["t_hit_map"]-.7) < 1e-12 and abs(r["min_gap_map"]-.07) < 1e-12, result=r)

    # Existing sensor scenarios retain semantics in clear static space.
    definitions = {
        "crossing_stop": ([object_fixture(1.3, 1.2, 0., -.4)], (.3, 0., (5., 0.)), "stop"),
        "approaching_stopped_robot": ([object_fixture(-2., 0., .7, 0., "right")], (.3, 0., (5., 0.)), "fast"),
        "empty_road": ([], (.3, 0., (5., 0.)), "fast"),
        "all_unsafe": ([object_fixture(2., 0., -.7, 0.)], (.3, 0., (5., 0.)), "back"),
        "slow_priority": ([object_fixture(1.4, 0., 0., 0.)], (.3, 0., (5., 0.)), "slow"),
        "zero_follow_escape": ([object_fixture(-2., 0., .7, 0., "right")], (0., .4, (5., 5.)), "toward"),
    }
    scenes_ok, order_ok = True, True
    scene_out = {}
    for label, (objects, args, expected) in definitions.items():
        ns = space(objects)
        v, w, info = ns["choose_command"](*args)
        scenes_ok &= info["chosen"] == expected
        scene_out[label] = {"expected": expected, "actual": info["chosen"]}
        factory = ns["_base_command_candidates"]
        count = 9 if args[0] == 0. else 7
        for order in (list(range(count-1, -1, -1)), list(range(3, count))+list(range(3)), rng.permutation(count).tolist()):
            def reordered(*a, order=order):
                candidates = factory(*a)
                return [candidates[i] for i in order]
            ns["_base_command_candidates"] = reordered
            av, aw, ai = ns["choose_command"](*args)
            order_ok &= (av, aw, ai) == (v, w, info)
    record("six_sensor_scenes_preserved_in_clear_map", scenes_ok, scenes=scene_out)
    record("candidate_order_invariant", order_ok, scenes=6, permutations_per_scene=3)

    # Nonfinite commands, poses, observations and private bounds must fail early.
    ns = space(); failures, expected_failures = 0, 0
    calls = []
    for invalid in (math.nan, math.inf, -math.inf):
        for index in range(5):
            args = [.3, .2, 3., .1, .2]; args[index] = invalid
            calls.append(lambda args=args: ns["check_motion"](*args))
        calls.extend([lambda q=invalid: ns["choose_command"](q, .2, (5., 0.)),
                      lambda q=invalid: ns["choose_command"](.3, q, (5., 0.)),
                      lambda q=invalid: ns["choose_command"](.3, .2, (q, 0.)),
                      lambda q=invalid: ns["choose_command"](.3, .2, (5., 0.), q),
                      lambda q=invalid: ns["_base_check_motion"](.3, .2, map_horizon_s=q)])
    calls.extend([lambda: ns["_base_check_motion"](.3, .2, map_horizon_s=-.1),
                  lambda: ns["_base_check_motion"](.3, .2, map_horizon_s=3.1),
                  lambda: ns["check_motion"](.3, .2, -1.), lambda: ns["check_motion"](.3, .2, dt=0.),
                  lambda: ns["check_motion"](.3, .2, margin_m=-.1)])
    for call in calls:
        expected_failures += 1
        try: call()
        except ValueError: failures += 1
    for index in range(4):
        pose_time = [10., 0., 0., 0.]; pose_time[index] = math.nan
        ns["_BASE_POSES"][:] = [tuple(pose_time)]; expected_failures += 1
        try: ns["choose_command"](.3, 0., (5., 0.))
        except ValueError: failures += 1
    for key in ("vx", "vy", "seen_t", "points"):
        obj = object_fixture(2., 0., 0., 0.)
        obj[key] = [[math.nan, 0.]] if key == "points" else math.nan
        seed(ns, [obj], pose=(0., 0., 0.), now=10.); expected_failures += 1
        try: ns["choose_command"](.3, 0., (5., 0.))
        except ValueError: failures += 1
    record("invalid_inputs_rejected", failures == expected_failures, rejected=failures, cases=expected_failures)

    # Integration cadence after shortening map time; other DBG must not consume it.
    runtime = space(rect=(.42, 2., -1., 1.))
    exec(source[source.index("# ---- DBG 기록 ----"):source.index("# [EDIT REGION] mission controller")], runtime)
    lines = []
    runtime["builtins"] = SimpleNamespace(print=lambda *args, **kwargs: lines.append(" ".join(str(a) for a in args)))
    boundaries, selections = {}, set()
    for frame in range(1202):
        t = frame/60.
        seed(runtime, pose=(0., 0., 0.), now=t)
        runtime["_BASE_DBG_STATE"]["t"] = t
        runtime["print"]("DBG ordinary line consumes global allowance")
        v, w, info = runtime["choose_command"](.3, 0., (5., 0.))
        selections.add((v, w, info["chosen"]))
        if frame in (600, 601, 1200, 1201):
            boundaries[str(frame)] = sum("follow 가 지도에 막힘" in line for line in lines)
    warnings = [line for line in lines if "follow 가 지도에 막힘" in line]
    record("map_warning_and_independent_dbg_cadence_preserved", boundaries == {"600": 0, "601": 1, "1200": 1, "1201": 2}
           and selections == {(0., 0., "stop")} and runtime["_BASE_DBG_STATE"]["dropped"] > 0
           and any("DBG choose t=10.00" in line for line in lines), boundary_counts=boundaries, warning_lines=warnings)
    # Choice changes continue to reset the independent warning timer.
    helper = runtime["_base_note_map_stop"]
    blocked_candidates = {"follow": {"v": .3, "w": 0., "t_hit_map": .5}, "stop": {"v": 0., "w": 0.}}
    reset_ok = True
    for alteration in ("different", "clear", "rewind"):
        runtime["_BASE_MAP_STOP_DBG_STATE"].update(since=None, last_t=None, last_warn_t=None)
        lines.clear(); helper(0., "stop", blocked_candidates); helper(9., "stop", blocked_candidates)
        altered = copy.deepcopy(blocked_candidates)
        if alteration == "different": helper(9.5, "follow", altered)
        elif alteration == "clear":
            altered["follow"]["t_hit_map"] = None; helper(9.5, "stop", altered)
        else: helper(-1., "stop", altered)
        start = 10. if alteration != "rewind" else -1.
        helper(start, "stop", blocked_candidates); helper(start+10., "stop", blocked_candidates)
        reset_ok &= len(lines) == 0
        helper(start+10.+1/60., "stop", blocked_candidates)
        reset_ok &= len(lines) == 1
    record("map_warning_resets_preserved", reset_ok)

    # check_path implementation was AST-preserved; retain boundary/invalid smoke checks.
    path_ns = space(rect=(0., 2., 0., 2.))
    boundary = path_ns["check_path"]([(-.5, .5), (-.5, 1.5)])
    inside = path_ns["check_path"]([(-.499, .5), (-.499, 1.5)])
    record("path_boundary_still_matches_half_width_contract", boundary["first_bad_segment"] is None
           and boundary["min_gap_map"] == .25 and inside["first_bad_segment"] == 0)

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
        row = dict(calls=len(times), objects=10, points_per_object=20, candidates=n, object_horizon_s=3.,
                   map_horizon_range_s=[.8, 1.], median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), target_ms=3.)
        performance[label] = row; record("performance_"+label, row["median_ms"] <= 3., **row)
    v8_source=BUILDER.SRC.read_text()
    record('v9_builder_directly_reproduces_fixed_v8_source',BUILDER.build(v8_source)==source,
           source_v8_sha256=hashlib.sha256(v8_source.encode()).hexdigest())
    class NormalizeDocumentationAndKeep(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='_BASE_OBJ_CFG' for t in node.targets):
                for keyword in node.value.keywords:
                    if keyword.arg=='keep':keyword.value=ast.Constant(1.)
            return self.generic_visit(node)
    normalization=lambda text:ast.dump(NormalizeDocumentationAndKeep().visit(ast.parse(text)))
    record('v8_v9_executable_ast_identical_except_keep_constant',normalization(source)==normalization(v8_source))
    n8,n9=load_ns(BUILDER.SRC),load_ns(BASE)
    c8,c9=copy.deepcopy(n8['_BASE_OBJ_CFG']),copy.deepcopy(n9['_BASE_OBJ_CFG'])
    record('track_config_changes_only_retention_one_to_three',c8.pop('keep')==1. and c9.pop('keep')==3. and c8==c9,
           remaining_config=c9)

    def track_fixture(base=BASE,override=None):
        ns=load_ns(base)
        ns['WALLS']=(-10.,10.,-10.,10.)
        ns['_MAP_RECTS'][:]=[np.array([[50.,51.,50.,51.]])]
        if override is not None:ns['_BASE_OBJ_CFG']['keep']=override
        scans={'front':np.array([[2.,0.],[2.,.2],[2.2,.1]]),'back':np.zeros((0,2))}
        ns['lidar2d_points']=lambda which='front':scans[which]
        def tick(now,points=None):
            if points is not None:scans['front']=np.asarray(points,dtype=float).reshape(-1,2)
            ns['_BASE_POSES'].append((now,0.,0.,0.))
            ns['_base_track_step'](now)
            return ns['observed_objects']()
        return ns,tick

    ns,tick=track_fixture();initial=tick(.1);lost=tick(1.2,[]);boundary=tick(3.1);expired=tick(3.100001)
    record('scan_clock_three_second_boundary_keeps_current_age_three_point_one',
           len(initial)==len(lost)==len(boundary)==1 and boundary[0]['seen_t']==0.
           and abs(boundary[0]['age_s']-3.1)<1e-12 and boundary[0]['id']==initial[0]['id'],
           initial=initial,boundary=boundary,scan_clock_age=3.1-.1-boundary[0]['seen_t'] if boundary else None)
    record('track_expires_immediately_after_three_second_scan_clock_boundary',expired==[],
           current_time=3.100001,scan_clock_time=3.000001)
    ns8,tick8=track_fixture(BUILDER.SRC);tick8(.1);old_boundary=tick8(1.1,[]);old_expired=tick8(1.100001)
    record('v8_retention_boundary_is_unchanged_and_shorter',len(old_boundary)==1 and old_expired==[],
           v8_boundary_age_s=old_boundary[0]['age_s'] if old_boundary else None)

    ns,tick=track_fixture();first=tick(.1);tick(1.2,[])
    reacquired=tick(2.,[[2.1,0.],[2.1,.2],[2.3,.1]])
    record('reacquisition_before_expiry_retains_track_id_and_updates_seen_time',len(reacquired)==1
           and reacquired[0]['id']==first[0]['id'] and abs(reacquired[0]['seen_t']-1.9)<1e-12,
           first_id=first[0]['id'],reacquired=reacquired)
    ns,tick=track_fixture();first=tick(.1);tick(1.2,[]);tick(3.100001)
    fresh=tick(3.2,[[2.1,0.],[2.1,.2],[2.3,.1]])
    record('reacquisition_after_expiry_creates_new_track_id',len(fresh)==1 and fresh[0]['id']!=first[0]['id']
           and abs(fresh[0]['seen_t']-3.1)<1e-12,first_id=first[0]['id'],fresh=fresh)
    copied=ns['observed_objects']();original_point=copied[0]['points'][0][:]
    copied[0]['points'][0][0]=999.;copied[0]['x']=999.;copied.clear()
    again=ns['observed_objects']()
    record('observed_return_remains_deep_copy_after_reacquisition',again[0]['points'][0]==original_point
           and again[0]['x']!=999. and len(again)==1)

    candidate,tc=track_fixture(BUILDER.SRC,override=3.);actual,ta=track_fixture(BASE)
    stream=[(.1,None),(.2,[[2.05,0.],[2.05,.2],[2.25,.1]]),(.4,[[2.1,0.],[2.1,.2],[2.3,.1]]),
            (1.2,[]),(2.,None),(3.4,None),(3.500001,None),(3.6,[[2.2,0.],[2.2,.2],[2.4,.1]])]
    equal=True;steps=[]
    for now,pts in stream:
        co,ao=tc(now,pts),ta(now,pts)
        cv=candidate['choose_command'](.3,.1,(4.,1.));av=actual['choose_command'](.3,.1,(4.,1.))
        same=co==ao and cv==av;equal &= same
        steps.append({'t':now,'observations':len(ao),'same_observed_and_complete_chooser_output':same})
    record('actual_v9_matches_namespace_keep3_candidate_tracker_and_chooser',equal,steps=steps)

    gate_path=ARENA/'results/analysis/b522_gate_failure_v9_gate.json'
    gate=json.loads(gate_path.read_text())
    candidate=json.loads((ARENA/'results/analysis/b522_gate_failure_retention_sim.json').read_text())
    old_gate=next(r for r in candidate['summaries']['gate']['results'] if r['label']=='v1')
    record('actual_v9_simple_gate_v1_matches_tested_candidate',gate['verdict']=='pass'
           and gate['base_sha16']==hashlib.sha256(BASE.read_bytes()).hexdigest()[:16]
           and gate['command_pose_sensor_sha256']==old_gate['command_pose_sensor_sha256']
           and gate['audit']['exactly_once'],result_file=str(gate_path),t=gate['t'],
           command_pose_sensor_sha256=gate['command_pose_sensor_sha256'])
    reference_path=ARENA/'results/analysis/b522_v9_reference_sim.json'
    reference=json.loads(reference_path.read_text());tested={r['label']:r for r in candidate['summaries']['reference']['results']}
    paired=all((r['verdict'],r['t'],r['pose'])==(tested[r['label']]['verdict'],tested[r['label']]['t'],tested[r['label']]['pose'])
               for r in reference['results'])
    record('actual_v9_reference_standard61_complete_and_candidate_outcomes_identical',
           reference['complete'] and len(reference['results'])==61
           and reference['base_sha16']==hashlib.sha256(BASE.read_bytes()).hexdigest()[:16]
           and reference['random_passed']==57 and reference['v1_pass'] and paired,
           random_passed=reference['random_passed'],random_total=reference['random_total'],v1_pass=reference['v1_pass'],
           result_file=str(reference_path),paired_verdict_time_finalpose_identical=paired)

    passed = sum(check["pass"] for check in checks.values())
    output = dict(verdict="pass" if passed == len(checks) else "fail", passed=passed, total=len(checks),
                  base_file=str(BASE), base_sha16=hashlib.sha256(BASE.read_bytes()).hexdigest()[:16],
                  checks=checks, examples=examples, performance=performance,
                  reference_sha256=reference["reference_sha256"],reference_result_file=str(reference_path),
                  source_v8_sha256=BUILDER.SOURCE_SHA256,
                  limitations=["Map horizon is a CPU-acceleration-model heuristic, not a measured braking guarantee.",
                               "Object prediction still assumes constant observed velocity; hidden geometry/acceleration is unknown.",
                               "Synthetic corner rollout and calibrated61-phase CPU sweep are separate results; no GPU/HTTP here."])
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(output), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{output['verdict']} {passed}/{len(checks)} sha={output['base_sha16']} "
          + " ".join(f"{name}={row['median_ms']:.4f}ms" for name, row in performance.items()))
    for name, check in checks.items():
        if not check["pass"]: print("FAIL", name, check)
    return output


if __name__ == "__main__":
    raise SystemExit(0 if run()["verdict"] == "pass" else 1)
