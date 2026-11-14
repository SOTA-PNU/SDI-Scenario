"""Saved B518 a2 t2 test03/submission evidence; no simulator or stochastic rerun."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from b514_build_base import archive_before_write
from b515_check_offline import load_ns, seed, safe_json

ARENA = Path(__file__).resolve().parents[1]
LOGS = ARENA / "results/turns/prompt_b5_18/isaac_logs"
PREFIX = "gpu_gpt_oss_120b_pB518_a2_t2"
OUT = ARENA / "results/analysis/b518_a2_motion_evidence.json"
CHOOSE = re.compile(r"DBG choose t=([\d.]+) pick=(\w+) why=(\w+) f=(\S+) s=(\S+) obj_min=(\S+) rob=(\S+) map=(\S+)")
WARNING = re.compile(r"DBG follow 가 지도에 막힘: 경로 여유 부족 t=([\d.]+) stop_for=([\d.]+)")
PATH = re.compile(r"DBG path points=(\d+) min_gap=([\d.]+) bad_seg=(\S+)")


def counts(rows):
    return dict(collections.Counter(r["pick"] + "/" + r["reason"] for r in rows))


def pose_at(a, t):
    x, y, yaw = (float(np.interp(t, a[:, 0], a[:, k])) for k in (1, 2, 3))
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw))


def dgoal(p):
    return math.hypot(p[0] - 8., p[1] - 16.5)


def references(source, tokens):
    return [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
            if any(token in line for token in tokens)]


def map_stencil(ns, trajectory, t):
    """Counterfactual command stencil at a saved pose, not the unlogged actual follow."""
    pose = pose_at(trajectory, t)
    seed(ns, pose=pose, now=t)
    rects = ns["map_rects"]()
    gaps = []
    for i, (xmin, xmax, ymin, ymax) in enumerate(rects):
        dx = max(xmin - pose[0], 0., pose[0] - xmax)
        dy = max(ymin - pose[1], 0., pose[1] - ymax)
        gaps.append((math.hypot(dx, dy), i))
    nearest, index = min(gaps)
    commands = [("stop", 0., 0.), ("turn_l", 0., .4), ("turn_r", 0., -.4),
                ("straight", .3, 0.), ("left", .3, .4), ("right", .3, -.4),
                ("back", -.2, 0.)]
    results = {}
    for name, v, w in commands:
        r = ns["check_motion"](v, w)
        results[name] = {"v": v, "w": w, "min_gap_map": r["min_gap_map"], "t_hit_map": r["t_hit_map"]}
    forward = []
    # At a non-goal waypoint, dist>=.35 implies v_follow in [.275,.3]
    # if the follower is translating. This grid is not a continuous proof.
    for v in np.linspace(.275, .3, 3):
        for w in np.linspace(-.4, .4, 17):
            r = ns["check_motion"](float(v), float(w))
            forward.append(r)
    return {"t": t, "pose_interpolated": pose, "goal_distance_m": dgoal(pose),
            "center_map_gap_m": float(ns["map_clearance"](*pose[:2])),
            "nearest_static_rectangle": {"index": index, "label": f"arena_box_{index:02}" if index < 14 else "fixed_obstacle",
                                         "bounds": rects[index].tolist(), "center_gap_m": nearest},
            "counterfactual_commands": results,
            "forward_grid": {"v_samples": 3, "w_samples": 17, "v_range": [.275, .3], "w_range": [-.4, .4],
                             "all_map_unsafe": all(r["t_hit_map"] is not None for r in forward),
                             "max_min_gap_map": max(r["min_gap_map"] for r in forward),
                             "map_hit_time_range": [min(r["t_hit_map"] for r in forward if r["t_hit_map"] is not None),
                                                    max(r["t_hit_map"] for r in forward if r["t_hit_map"] is not None)]}}


def analyze(label, suffix):
    stem = PREFIX + suffix
    paths = {kind: LOGS / (stem + ending) for kind, ending in
             (("runner", "_carter_run.py"), ("stdout", ".carter-run.stdout.log"),
              ("trajectory", "_trajectory.csv"), ("result", "_result.json"))}
    source = paths["runner"].read_text()
    result = json.loads(paths["result"].read_text())
    a = np.genfromtxt(paths["trajectory"], delimiter=",", skip_header=1)
    a[:, 3] = np.unwrap(a[:, 3])
    choices, warnings, path_diag = [], [], []
    for i, line in enumerate(paths["stdout"].read_text().splitlines(), 1):
        m = CHOOSE.search(line)
        if m:
            choices.append({"t": float(m[1]), "pick": m[2], "reason": m[3],
                            "follow_t_hit": None if m[4] == "-" else float(m[4]),
                            "stop_t_hit": None if m[5] == "-" else float(m[5]),
                            "chosen_object_horizon_min": float(m[6]), "chosen_robust_object_horizon_min": float(m[7]),
                            "chosen_map_horizon_min_rounded": float(m[8]), "stdout_line": i, "text": m[0]})
        m = WARNING.search(line)
        if m:
            warnings.append({"t": float(m[1]), "continuous_stop_for_s_rounded": float(m[2]), "stdout_line": i, "text": m[0]})
        m = PATH.search(line)
        if m:
            path_diag.append({"point_count": int(m[1]), "min_gap_map_rounded": float(m[2]),
                              "first_bad_segment": None if m[3] == "None" else m[3], "stdout_line": i, "text": m[0]})
    intervals, start = [], None
    for i, row in enumerate(choices):
        if row["pick"] == "stop" and start is None:
            start = i
        if start is not None and (row["pick"] != "stop" or i == len(choices) - 1):
            end = i - 1 if row["pick"] != "stop" else i
            first, last = choices[start], choices[end]
            p0, p1 = pose_at(a, first["t"]), pose_at(a, last["t"])
            intervals.append({"first_t": first["t"], "last_t": last["t"], "span_s": last["t"] - first["t"],
                              "sample_count": end - start + 1, "first_stdout_line": first["stdout_line"],
                              "last_stdout_line": last["stdout_line"], "first_pose": p0, "last_pose": p1,
                              "first_goal_distance_m": dgoal(p0), "last_goal_distance_m": dgoal(p1),
                              "net_position_displacement_m": math.hypot(p1[0] - p0[0], p1[1] - p0[1])})
            start = None
    intervals.sort(key=lambda r: -r["sample_count"])
    near_index = int(np.argmin(np.hypot(a[:, 1] - 8., a[:, 2] - 16.5)))
    subset = [r for r in choices if r["t"] >= 158.]
    ns = load_ns(paths["runner"])
    sample_times = [0., .3, .5, 1., 100., 138., 150., 157.77, 158., 167.77, 200., 300., 367.77, 395.5, 400., 599.5]
    snapshots = [{"t": t, "pose": pose_at(a, t), "goal_distance_m": dgoal(pose_at(a, t))}
                 for t in sample_times if t <= a[-1, 0]]
    output = {"run": label, "sources": {k: {"path": str(p.relative_to(ARENA)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                                for k, p in paths.items()},
              "result": result, "actual_path_diagnostic": path_diag,
              "trajectory": {"rows": len(a), "first_t": float(a[0, 0]), "last_t": float(a[-1, 0]),
                             "median_sample_dt_s": float(np.median(np.diff(a[:, 0]))),
                             "first_two_samples": a[:2].tolist(), "nearest_goal_sample": a[near_index].tolist(),
                             "snapshots": snapshots},
              "choice_samples": len(choices), "choice_counts": counts(choices),
              "after_158s": {"sample_count": len(subset), "choice_counts": counts(subset),
                             "nonstop_samples": [r for r in subset if r["pick"] != "stop"],
                             "min_chosen_robust_object_gap": min(r["chosen_robust_object_horizon_min"] for r in subset)},
              "longest_sampled_stop_intervals": intervals[:8], "map_blocked_warnings": warnings,
              "first_choices": choices[:8], "last_choices": choices[-12:],
              "approach_and_retreat_choice_counts_130_to158": counts([r for r in choices if 130 <= r["t"] < 158]),
              "near_stall_choices_150_to160": [r for r in choices if 150 <= r["t"] < 160],
              "source_references": references(source, ("import random", "def _plan_path", "random.random", "random.uniform", "if not _PATH_PLANNED",
                                                         "_PATH_PLANNED = True", "if abs(e) >", "v_follow =", "w_follow =", "_dist((x, y)",
                                                         "blocked =", "candidates[\"follow\"][\"t_hit_map\"]", "if not blocked", "elapsed > 10.0", "chosen == \"stop\"")),
              "static_saved_helper_stencil": [map_stencil(ns, a, t) for t in [158., 200., 300., 368., 400., 599.5]] if not suffix else []}
    return output, a


def main():
    passing, a = analyze("test03_pass", "_test03")
    graded, b = analyze("submitted_fail", "")
    common_time = .3
    pa, pb = pose_at(a, common_time), pose_at(b, common_time)
    output = {"scope": "Completed B518 a2 t2 test03 and submitted grading only. No simulator, process query, HTTP, or running-a3 access.",
              "summary": [
                  "Actual planned paths differ: test03 logs99 points/min_gap.351, submitted88/.348, both first_bad_segmentNone.",
                  "The shared EDIT uses Python's global random RRT once. The saved controller and directly used arena/common harness contain no random.seed call; result seed42 is a stored harness field and does not establish a seeded planner.",
                  "The initial pose is identical, but test03 initially turns clockwise and submission counterclockwise; the divergence starts before the later object interactions.",
                  "Submission reaches its nearest recorded goal distance near138s, retreats, and predominantly stops from158s near arena_box_10 southwest at about12m from goal. It times out600s and settles to11.8504m with no collision.",
                  "Twenty-one explicit warnings167.77-367.77s prove continuous chosenstop and follow map rejection, with the logged interval beginning about157.77s. Test03 has no such warnings.",
                  "The submission's longest consecutive0.5s stop samples cover158-395s, but later sparse back choices exist and per-tick continuity cannot be inferred from0.5s logs.",
                  "Late stop map gap is near.2m while selected stop robust object gap stays above.2m. The saved static helper at final pose reproduces immediate map rejection for forward motion; actual target/follow commands are unlogged.",
                  "The path check verifies a centerline polyline, while the local chooser predicts constant-twist body motion. No replanning handles a map-blocked target; a valid initial polyline alone does not guarantee local progress."],
              "method_and_limits": {
                  "counts": "Counts are0.5s chooser DBG samples, not60Hz control counts.",
                  "actual_path": "Only point count and path check summary are logged. We did not rerun stochastic RRT or claim to recover the actual waypoint array/target.",
                  "static_stencil": "Saved runner check_motion and static map at interpolated saved pose, with empty tracks solely to isolate map geometry. Commands are counterfactual unless identified as actual stop. It is not sensor/chooser replay.",
                  "precision": "Trajectory CSV rounds pose to4decimals; sample dt differs. Linear interpolation uses unwrapped yaw. Near.2m threshold replay can disagree with live rounding.",
                  "object_evidence": "Saved stdout includes chosen predicted object minima and combined follow/stop TTCs, without per-track points/velocities/IDs. These are horizon minima, not current physical object distance. No specific mover, missed detection or sensor failure is established.",
                  "warning_end": "No warning after367.77 despite sampled stops means the consecutive blocked-stop predicate ceased between warnings, if recorded stdout is complete. The samples cannot identify the first reset or distinguish a brief different selection from follow-map-clear.",
                  "seed": "Static inspection of saved runner, arena_harness.py and levels/common/carter_env.py found no random.seed. Dependencies were not executed; actual different planned path summaries independently prove path mismatch."},
              "earliest_divergence": {"initial_pose_identical": bool(np.array_equal(a[0, 1:], b[0, 1:])),
                                      "test_first_noninitial_raw_sample": a[1].tolist(), "grade_first_noninitial_raw_sample": b[1].tolist(),
                                      "comparison_t": common_time, "test_pose_interpolated": pa, "grade_pose": pb,
                                      "position_difference_m": math.hypot(pa[0] - pb[0], pa[1] - pb[1]),
                                      "yaw_difference_rad": pb[2] - pa[2]},
              "runs": [passing, graded]}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(OUT)
    print("samples", passing["choice_samples"], graded["choice_samples"], "warnings", len(passing["map_blocked_warnings"]), len(graded["map_blocked_warnings"]))
    print("divergence", output["earliest_divergence"])
    print("stop", graded["longest_sampled_stop_intervals"][0])
    print("after158", graded["after_158s"])


if __name__ == "__main__":
    main()
