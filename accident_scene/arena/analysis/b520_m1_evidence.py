"""Audit completed B520 a0, B519 a2 tests, and completed simple gates.

No B520 a1 reads, processes, HTTP, GPU, motion simulation, or operational edits.
Only model nominal estimates from saved rounded poses are recomputed here.
"""
import ast
import csv
from datetime import datetime
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A
from b519_a2_submission import command_audit, debug_rows, matching, ref

OUT = ARENA / "results/analysis/b520_m1_evidence.json"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rel(path):
    return str(path.relative_to(ARENA))


def finite_json(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else "-inf" if obj < 0 else "nan"
    if isinstance(obj, dict):
        return {key: finite_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(value) for value in obj]
    return obj


def read_trajectory(path):
    with path.open() as handle:
        return [{**{k: float(v) for k, v in row.items()}, "source_line": i} for i, row in enumerate(csv.DictReader(handle), 2)]


def source_shape(path):
    source = path.read_text()
    tree = ast.parse(A.region_of(source))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    controller = functions["controller"]
    choose_nodes = [n for n in ast.walk(controller) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "choose_command"]
    audit = command_audit(path)
    return {"command_audit": audit, "chooser_call_sites": len(choose_nodes), "has_plan_function": any("plan_path" in name or name == "_astar" for name in functions), "controller_calls_check_path": any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "check_path" for n in ast.walk(controller)), "placeholder_replan": "_maybe_replan" in functions and all(isinstance(n, ast.Pass) for n in functions["_maybe_replan"].body), "choose_refs": matching(path, "= choose_command("), "side_branch_refs": matching(path, "if chosen == 'stop'") + matching(path, "side_target =") + matching(path, "choose_command(v_f_s"), "sensor_branch_refs": matching(path, "_BLOCK_DIST =") + matching(path, "_BLOCK_ANGLE =") + matching(path, "surface_dist =") + matching(path, "if _moving_object_blocking"), "goal_target_refs": matching(path, "choose_command(v_f, w_f, GOAL[:2])")}


def model_record(version, stem, suffix, label, manual_kind):
    folder = ARENA / "results/turns" / f"prompt_b5_{version}" / "isaac_logs"
    source_path = folder / f"{stem}{suffix}_carter_run.py"
    stdout_path = folder / f"{stem}{suffix}.carter-run.stdout.log"
    result_path = folder / f"{stem}{suffix}_result.json"
    trajectory_path = folder / f"{stem}{suffix}_trajectory.csv"
    source, result = source_path.read_text(), json.loads(result_path.read_text())
    assert A.TL._outside(source) == A.TL._outside(BASE.read_text())
    contact = result["metrics"]["first_collision_t"]
    rows = [row for row in debug_rows(stdout_path) if contact - 5 <= row["t"] < contact]
    poses = read_trajectory(trajectory_path)
    for row in rows:
        near = min(poses, key=lambda pose: abs(pose["t"] - row["t"]))
        dx, dy = 8.0 - near["x"], 16.5 - near["y"]
        err = math.atan2(math.sin(math.atan2(dy, dx) - near["yaw"]), math.cos(math.atan2(dy, dx) - near["yaw"]))
        goal_nominal = [0.0 if abs(err) > .35 else min(.3, .5 * math.hypot(dx, dy) + .1), max(-.4, min(.4, 1.5 * err))]
        # The only secondary call occurs after chosen=stop in all audited sources.
        # Every logged first call has GOAL xy; no row in these windows picks stop.
        no_second_call_at_sample = row["pick"] != "stop"
        assert no_second_call_at_sample
        zero_excluded = manual_kind == "stop" and row["f"] != row["s"]
        alternatives = [{"branch": "direct_goal_follow", "nominal_estimate": goal_nominal}]
        if manual_kind == "stop" and not zero_excluded:
            alternatives.append({"branch": "manual_sensor_stop", "nominal": [0.0, 0.0]})
        if manual_kind == "reverse":
            alternatives.append({"branch": "manual_sensor_reverse", "nominal": [-0.2, 0.0]})
        exact_selected = {"back": [-.2, 0.0], "turn_l": [0.0, .4], "turn_r": [0.0, -.4]}.get(row["pick"])
        row["input_audit"] = {"nominal_measured": None, "target_xy_proven": [8.0, 16.5], "target_proof": "All first-call paths use GOAL xy; only secondary call may use another target, and logged pick!=stop excludes that secondary branch at this sample.", "nominal_alternatives": alternatives, "manual_zero_excluded_by_unequal_follow_stop_ttc": zero_excluded, "zero_exclusion_assumption": "Identical zero follow/stop commands use the same unchanged deterministic check at the same tracked snapshot; unequal rounded TTCs therefore exclude the zero-input branch." if zero_excluded else None, "pose_source": {"file": rel(trajectory_path), **near}, "sample_time_difference_s": near["t"] - row["t"], "goal_error_rad_estimate": err, "goal_nominal_estimate_method": "Recomputed algebraically from recorded pose rounded to4decimals; nearest saved time noted. This is not recorded nominal telemetry or sensor replay.", "secondary_call_excluded_at_logged_sample": no_second_call_at_sample, "selected_command_exact_if_fixed_candidate": exact_selected, "returned_command": exact_selected, "returned_command_limit": "Fixed back/turn candidates are exact at this logged call because no post-call override branch executes. Other candidate speeds depend on unlogged nominal; no complete60Hz command history is claimed."}
    return {"label": label, "source_file": rel(source_path), "source_sha256": sha(source), "edit_sha256": sha(A.region_of(source)), "outside_edit_matches_v8": True, "result_file": rel(result_path), "metrics": result["metrics"], "source_shape": source_shape(source_path), "path_debug": matching(stdout_path, "DBG path"), "precontact_5s": rows, "manual_input_kind": manual_kind}


def gate_record(version):
    folder = ARENA / "results/analysis" / f"b{version}_gate_isaac"
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    source_path = folder / f"b{version}_gate_probe_carter_run.py"
    obs_path = folder / "observations.jsonl"
    observations = [{**json.loads(line), "source_line": i} for i, line in enumerate(obs_path.read_text().splitlines(), 1)]
    stdout_rows = debug_rows(folder / "stdout.log")
    selected = []
    for wanted in [i * .5 for i in range(8, 18)]:
        obs = min(observations, key=lambda row: abs(row["t"] - wanted))
        dbg = min(stdout_rows, key=lambda row: abs(row["t"] - wanted))
        info = obs["info"]
        cand = info["candidates"][info["chosen"]]
        selected.append({"requested_comparison_t": wanted, "observation_t": obs["t"], "time_difference_s": obs["t"] - wanted, "observation_ref": {"file": rel(obs_path), "line": obs["source_line"]}, "nominal_recorded": obs["nominal"], "target_xy_recorded": obs["target_xy"], "command_recorded": obs["command"], "pose_recorded": obs["pose"], "chosen": info["chosen"], "reason": info["reason"], "follow_t_hit": info["candidates"]["follow"]["t_hit"], "stop_t_hit": info["candidates"]["stop"]["t_hit"], "selected_min_gap_obj": cand["min_gap_obj"], "selected_robust_min_gap_obj": cand["robust_min_gap_obj"], "selected_min_gap_map": cand["min_gap_map"], "stdout_debug": dbg})
    return {"version": version, "source_file": rel(source_path), "summary_file": rel(summary_path), "base_file": summary["launch"]["base_file"], "base_sha16": summary["launch"]["base_sha16"], "same_v8_as_model": summary["launch"]["base_sha16"] == "d92eed4891798942", "verdict": summary["verdict"], "collision_count": summary["collision_count"], "time_to_goal_s": summary["time_to_goal_s"], "chooser_call_refs": matching(source_path, "= choose_command("), "same_time_rows": selected}


def fmt(value):
    if value is None:
        return "–"
    if isinstance(value, str):
        return value
    return f"{value:.2f}".rstrip("0").rstrip(".")


def main():
    models = [model_record("20", "gpu_gpt_oss_120b_pB520_a0", "", "B520_a0_t1_grading", "none"),
              model_record("20", "gpu_gpt_oss_120b_pB520_a0", "_t2_test01", "B520_a0_t2_test01", "stop"),
              model_record("20", "gpu_gpt_oss_120b_pB520_a0", "_t2", "B520_a0_t2_grading", "stop")]
    models += [model_record("19", "gpu_gpt_oss_120b_pB519_a2", f"_t2_test{i:02d}", f"B519_a2_t2_test{i:02d}", "none" if i == 1 else "reverse") for i in range(1, 7)]
    gates = [gate_record(v) for v in (515, 519, 520)]
    test, grade = models[1], models[2]
    before = A.region_of((ARENA / test["source_file"]).read_text())
    after = A.region_of((ARENA / grade["source_file"]).read_text())
    table = []
    for k, t in enumerate([i * .5 for i in range(8, 18)]):
        a, b = test["precontact_5s"][k], grade["precontact_5s"][k]
        g = gates[-1]["same_time_rows"][k]
        assert a["t"] == b["t"] == t
        table.append({"t": t, "B520_test": {key: a[key] for key in ("pick", "why", "f", "s", "obj_min", "rob", "map")}, "B520_grade": {key: b[key] for key in ("pick", "why", "f", "s", "obj_min", "rob", "map")}, "B520_v8_gate": {key: g[key] for key in ("nominal_recorded", "target_xy_recorded", "command_recorded", "chosen", "reason", "follow_t_hit", "stop_t_hit", "selected_min_gap_obj", "selected_robust_min_gap_obj", "selected_min_gap_map")}})
    def cell(row):
        return f'{row["pick"]}/{row["why"]} {fmt(row["f"])}/{fmt(row["s"])} {fmt(row["obj_min"])}/{fmt(row["rob"])}/{fmt(row["map"])}'
    text_rows = ["t | B520 test pick/reason F/S raw/rob/map | B520 grade | exact-v8 gate"]
    for row in table:
        gate = row["B520_v8_gate"]
        normalized = dict(pick=gate["chosen"], why=gate["reason"], f=gate["follow_t_hit"], s=gate["stop_t_hit"], obj_min=gate["selected_min_gap_obj"], rob=gate["selected_robust_min_gap_obj"], map=gate["selected_min_gap_map"])
        text_rows.append(f'{row["t"]:.1f} | {cell(row["B520_test"])} | {cell(row["B520_grade"])} | {cell(normalized)}')
    output = {
        "scope": "Completed B520 a0 only, completed B519 a2 tests, completed B515/B519/B520 gates, static base. No B520 a1, HTTP, GPU, processes, or motion simulation. Original files read only.",
        "model_base_sha16": sha(BASE.read_text())[:16], "models": models, "gates": gates,
        "b520_test_submission_diff": {"identical": before == after, "unified_edit_diff": "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=test["source_file"], tofile=grade["source_file"])), "substantive_change": "Manual stop detector widens from centroid distance<1.5m/bearing30degrees to max(0,centroid_dist-radius)<3m/bearing45degrees. dist/radius are sensor tracker metadata, not exact true surface geometry. Code is changed after a failing test, not an identical-code grading rerun."},
        "aligned_table": table, "compact_table_text": "\n".join(text_rows),
        "conclusions": ["B520 a0 t2 test01 first contact8.867s/count11; changed final grading first contact8.567s/count33. t1 grading also contact8.867s/count11.", "All executed velocity returns are chooser outputs, but every B520 audited controller can call it twice when first chosenstop and follow.t_hit_map is notNone. The secondary call uses a manually constructed0.6m left-side target; no map path or check_path is computed. _maybe_replan is a pass placeholder.", "No precontact5s logged sample chooses stop. At those samples the post-choice secondary call does not execute, all first-call targets are GOALxy, and fixed back/turn outputs are proven from code. This does not exclude the branch at unlogged ticks.", "Model nominal is not directly recorded. Rounded-pose goal-follow estimates and unlogged sensor-zero/reverse alternatives are kept separate. Unequal follow/stop TTCs rule out zero nominal under the unchanged deterministic snapshot checks.", "B520 test01 nominal goal direction is almost straight at4–5.5s; its6–6.5s selected right turn opposes the positive goal-heading input estimated from the pose. Grading nominal remains near straight at4–7s. Large angular input or in-place nominal rotation is not established as the necessary trigger.", "Exact-v8 B520 simple gate also uses all-unsafe fallback at4/4.5s, selecting fast, and then returns to safe candidates. It follows a different route/pose with roughly.75m selected map gap while model samples approach.2–.5m map gap; same-time comparison is not a matched-state counterfactual.", "Legacy B515 gate used v4; it is retained as historical context and is not mislabeled as the same v8 chooser."],
        "limits": ["Automatic DBG is one first-call record per0.5s window and formats most scalars to1–2decimals. It does not supply every60Hz decision or raw candidate info.", "Model raw lidar points, object tracks/seen_t/velocity and explicit nominal/target state were not logged. No true tracking-error or exact candidate reconstruction can be inferred from those model logs.", "obj_min and rob are selected candidate future-horizon minima across observed objects, not a per-object current physical gap. Negative future gap or robust TTC0 is not the recorded physical first-contact time.", "These artifacts do not establish whether any safe-labelled sample was falsely safe in true geometry; they do establish that all-unsafe predicted-risk selections preceded contact.", "Source branch deductions and rounded-pose algebra are distinguished from direct telemetry. New CPU rollout comparisons and proposed chooser patches belong to the separate runner/base audit."],
    }
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_task18_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(finite_json(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"written": rel(OUT), "models": len(models), "gates": len(gates), "aligned_rows": len(table), "B520_contacts": [m["metrics"]["first_collision_t"] for m in models[:3]], "all_returns_choose": all(m["source_shape"]["command_audit"]["all_motion_returns_are_chooser_outputs"] for m in models)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
