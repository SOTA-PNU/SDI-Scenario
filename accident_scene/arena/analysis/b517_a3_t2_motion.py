"""Read completed B517 a3 t2 tests 06-10; never launch or inspect processes.

Saved DBG commands are observations. Static helper replay is identified separately;
it supplies no invented lidar tracks and cannot validate object prediction accuracy.
"""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from b514_build_base import archive_before_write
from b515_check_offline import load_ns, seed

ARENA = Path(__file__).resolve().parents[1]
LOGS = ARENA / "results/turns/prompt_b5_17/isaac_logs"
PREFIX = "gpu_gpt_oss_120b_pB517_a3_t2_test"
OUT = ARENA / "results/analysis/b517_a3_t2_motion_evidence.json"
CHOOSE = re.compile(r"DBG choose t=([\d.]+) pick=(\w+) why=(\w+) f=(\S+) s=(\S+) obj_min=(\S+) rob=(\S+) map=(\S+)")
CTRL = re.compile(r"DBG ctrl t=([\d.]+) pose=\(([-\d.]+),([-\d.]+),([-\d.]+)\) target=\(([-\d.]+),([-\d.]+)\) v_f=([-\d.]+) w_f=([-\d.]+) chosen=\(([-\d.]+),([-\d.]+)\)")


def counts(rows):
    return dict(collections.Counter(r["pick"] + "/" + r["reason"] for r in rows))


def pose_at(trajectory, t):
    return tuple(float(np.interp(t, trajectory[:, 0], trajectory[:, k])) for k in (1, 2, 3))


def goal_distance(pose):
    return math.hypot(pose[0]-8.0, pose[1]-16.5)


def static_replay(ns, trajectory, command):
    t = command["t"]
    pose = pose_at(trajectory, t)
    tx, ty = command["target"]
    dx, dy = tx-pose[0], ty-pose[1]
    error = math.atan2(dy, dx)-pose[2]
    error = math.atan2(math.sin(error), math.cos(error))
    vf = 0.0 if abs(error) > .35 else min(.3, .5*math.hypot(dx, dy)+.1)
    wf = max(-.4, min(.4, 1.5*error))
    seed(ns, pose=pose, now=t)
    candidates = {name: ns["check_motion"](v, w) for name, v, w in ns["_base_command_candidates"](vf, wf, (tx, ty))}
    return {"t": t, "interpolated_pose": pose, "goal_distance_m": goal_distance(pose),
            "reconstructed_input": [vf, wf], "logged_input": command["input"],
            "target": command["target"], "logged_returned_command": command["chosen"],
            "ctrl_stdout_line": command["line"],
            "map_only_candidates": {name: {"min_gap_map": result["min_gap_map"], "t_hit_map": result["t_hit_map"]}
                                    for name, result in candidates.items()}}


def analyze(number):
    stem = PREFIX + f"{number:02}"
    paths = {kind: LOGS / (stem+suffix) for kind, suffix in
             (("runner", "_carter_run.py"), ("stdout", ".carter-run.stdout.log"),
              ("trajectory", "_trajectory.csv"), ("result", "_result.json"))}
    result = json.loads(paths["result"].read_text())
    metrics = result["metrics"]
    source = paths["runner"].read_text()
    trajectory = np.genfromtxt(paths["trajectory"], delimiter=",", skip_header=1)
    trajectory[:, 3] = np.unwrap(trajectory[:, 3])
    choices, controls, phase_lines = [], {}, []
    for line_number, line in enumerate(paths["stdout"].read_text().splitlines(), 1):
        match = CHOOSE.search(line)
        if match:
            choices.append({"t": float(match[1]), "pick": match[2], "reason": match[3],
                            "follow_t_hit": None if match[4] == "-" else float(match[4]),
                            "stop_t_hit": None if match[5] == "-" else float(match[5]),
                            "chosen_object_horizon_min": float(match[6]),
                            "chosen_robust_object_horizon_min": float(match[7]),
                            "chosen_map_horizon_min_rounded": float(match[8]),
                            "line": line_number, "text": match[0]})
        match = CTRL.search(line)
        if match:
            t = float(match[1])
            controls[t] = {"t": t, "pose_rounded": [float(match[k]) for k in (2, 3, 4)],
                           "target": [float(match[k]) for k in (5, 6)],
                           "input": [float(match[k]) for k in (7, 8)],
                           "chosen": [float(match[k]) for k in (9, 10)], "line": line_number}
        if "TIMEOUT" in line or "aborting " in line:
            phase_lines.append({"line": line_number, "text": line})
    intervals, start = [], None
    for i, row in enumerate(choices):
        if row["pick"] == "stop" and start is None:
            start = i
        if start is not None and (row["pick"] != "stop" or i == len(choices)-1):
            end = i-1 if row["pick"] != "stop" else i
            first, last = choices[start], choices[end]
            p0, p1 = pose_at(trajectory, first["t"]), pose_at(trajectory, last["t"])
            intervals.append({"first_t": first["t"], "last_t": last["t"],
                              "sampled_span_s": last["t"]-first["t"], "sample_count": end-start+1,
                              "first_line": first["line"], "last_line": last["line"],
                              "first_pose": p0, "last_pose": p1,
                              "first_goal_distance_m": goal_distance(p0), "last_goal_distance_m": goal_distance(p1),
                              "net_position_drift_m": math.hypot(p1[0]-p0[0], p1[1]-p0[1]),
                              "input_first": controls[first["t"]]["input"], "input_last": controls[last["t"]]["input"]})
            start = None
    intervals.sort(key=lambda r: r["sample_count"], reverse=True)

    ns = load_ns(paths["runner"])
    ns.update(START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"])
    exec(source[source.index("# [EDIT REGION]"):source.index("# ========================= [END EDIT REGION]")], ns)
    ns["_grid_path"]()
    path = ns["_PATH"]
    distances = [float(ns["segment_clearance"](p, q)) for p, q in zip(path[:-1], path[1:])]
    lengths = [math.hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(path[:-1], path[1:])]
    collision = metrics["first_collision_t"]
    precontact = [r for r in choices if collision is not None and collision-5 <= r["t"] < collision]
    selected = {r["t"] for r in precontact}
    if collision is None:
        selected |= {intervals[0]["first_t"], 100., 200., 400., 450.}
        selected |= {r["t"] for r in choices if r["t"] >= 595.}
    replay = [static_replay(ns, trajectory, controls[t]) for t in sorted(selected) if t in controls]
    dgoal = np.hypot(trajectory[:, 1]-8., trajectory[:, 2]-16.5)
    near = np.flatnonzero(dgoal <= 7.)
    near_time = float(trajectory[near[0], 0]) if near.size else None
    nearest_lines = lambda tokens: [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
                                    if any(token in line for token in tokens)]
    before = [r for r in choices if collision is None or r["t"] < collision]
    return {"test": number, "sources": {k: {"path": str(p.relative_to(ARENA)),
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
            "metrics": metrics, "aborted_after_collision": result.get("aborted_after_collision"),
            "trajectory": {"rows": len(trajectory), "first_t": float(trajectory[0, 0]),
                           "last_t": float(trajectory[-1, 0]), "median_sample_dt_s": float(np.median(np.diff(trajectory[:, 0]))),
                           "first_distance_at_most_7m_t": near_time},
            "choice_sample_counts_all": counts(choices), "choice_samples_all": len(choices),
            "choice_sample_counts_before_collision": counts(before),
            "choice_sample_counts_after_first_distance_at_most_7m": counts([r for r in before if near_time is not None and r["t"] >= near_time]),
            "longest_sampled_stop_intervals": intervals[:3],
            "path_recomputed": {"safety_margin_setting": ns["_SAFETY_MARGIN"], "points": path,
                                "point_count": len(path), "minimum_segment_center_map_gap_m": min(distances),
                                "length_m": sum(lengths), "min_segment_length_m": min(lengths),
                                "median_segment_length_m": float(np.median(lengths))},
            "map_only_replay": replay,
            "precontact_five_second_choices": precontact,
            "precontact_five_second_controls": [controls[r["t"]] for r in precontact],
            "last_choices": choices[-4:], "phase_log_lines": phase_lines,
            "collision_during_controller": collision is not None and any(r["t"] > collision for r in choices),
            "goal_near_object_override_present": "target_xy = (x, y)" in source,
            "goal_near_object_override_reached": bool(dgoal.min() < 1.),
            "source_references": nearest_lines(("_SAFETY_MARGIN =", "def controller", "def _next_target", "if math.hypot(x - GOAL[0]", "target_xy = (x, y)", "choose_command(v_f"))}


def main():
    rows = [analyze(number) for number in range(6, 11)]
    output = {
        "scope": "Completed B517 a3 t2 tests06-10 only. No Isaac, HTTP, process inspection, or B518 running artifact access.",
        "summary": [
            "Tests06-08 exhausted the 600s controller limit (601s result including settle); they did not stand 600s at the final 6.2m goal distance.",
            "Longest sampled continuous stop spans were38.0-453.5,37.5-455.5,37.5-455.0s, initially about22.63m from goal; later the robot resumed and ended6.24-6.29m away.",
            "During the long stop the model requested0.3m/s and near-zero yaw, while the chooser returnedstop_safe. Replayed map-only follow min gap about0.195m is below0.2; stop map clearance remained above0.2 for most of that interval.",
            "Tests09/10 first hit arena_mover_04 at243.15/120.483s during active controller operation, before the post-run settle. Their last5s follow map tests pass, while DBG records object risk and all-unsafe selection.",
            "Goal<1m object overrides are absent in06/07 and never reached in08-10; they do not explain these failures."],
        "method": {"choice_counts": "Counts of0.5s DBG samples, not all60Hz command ticks.",
                   "stop_duration": "Longest consecutive stop entries on the0.5s sampled log; endpoints are not exact command-transition times.",
                   "static_replay": "Saved runner functions and static map, empty observed objects solely to isolate map component. Pose linearly interpolated from rounded CSV with yaw unwrapped; target from actual DBG ctrl.",
                   "input_reconstruction": "Saved follower formula at interpolated pose and logged target. Its goal-only object override was never reached.",
                   "sensor_limit": "DBG contains aggregate predicted object gaps/TTCs, not per-object surface-point/velocity arrays. It supports detected object risk but not a claim about tracking loss or a measured M5 prediction error.",
                   "gap_semantics": "obj_min/rob are minima over the prediction horizon, not current physical separation.",
                   "precision": "CSV sampling varies0.3/0.117/0.05s and pose rounds4decimals. Near command transitions the map replay may differ slightly from the logged live value."},
        "settle_source": {"path": "../../levels/common/carter_env.py", "run_mission_default_settle_s": 1.0,
                          "default_line": 364, "settle_after_controller_loop_line": 453,
                          "timeout_before_controller_line": 418, "abort_after_collision_line": 425},
        "tests": rows,
        "interpretation": {"long_stop": "Static clearance bottleneck, already addressed by the planned B518 path-width check and map-stop diagnostic; no new code was changed here.",
                           "late_goal_area": "Repeated movement/retreat and short waits coincide with predicted object risk; there is no continuous600s stop in the goal area.",
                           "M5_collision": "Map-preserving all-unsafe selection can still select a command with predicted object risk. A safe result is not guaranteed when all candidates are unsafe; saved aggregate diagnostics cannot establish which omitted maneuver or prediction change would prevent contact."}}
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): return "inf" if value > 0 else "-inf"
        if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [finite(v) for v in value]
        return value
    encoded = json.dumps(finite(output), ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    archive_before_write(OUT)
    OUT.write_text(encoded)
    print(OUT)
    for row in rows:
        stop = row["longest_sampled_stop_intervals"][0]
        print(row["test"], "stop", stop["first_t"], stop["last_t"], "initial_goal_dist", stop["first_goal_distance_m"],
              "drift", stop["net_position_drift_m"], "final", row["metrics"]["final_dist_m"],
              "collision", row["metrics"]["first_collision_t"])


if __name__ == "__main__":
    main()
