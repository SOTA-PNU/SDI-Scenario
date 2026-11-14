"""Replay the recorded static-scene probe scans (claude_ros_probe01, no moving obstacles) through a controller region's
perception and count false moving obstacles: moving capsules within 6 m of the robot that the selection step would see.
Open loop: the recorded poses drive the replay; only _sense and _obstacle_shapes run.

    python3 analysis/claude_ros_replay_perception.py <region.py> [<region.py> ...]
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
cap = pickle.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))
base = (ARENA / "base_carter_run_ros.py").read_text()


def replay(region_file):
    ns = {"np": np, "math": math, "print": lambda *a, **k: None}
    exec(base[base.index("MISSION = dict("):base.index("\n# ---- Carter")], ns)
    ns.update(START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"])
    exec(base[base.index("def to_world"):base.index("# [EDIT REGION]")], ns)
    cur = {"front": np.zeros((0, 2)), "back": np.zeros((0, 2))}
    ns["lidar2d_points"] = lambda which="front": cur[which]
    exec(Path(region_file).read_text(), ns)
    scans = sorted(cap["scans"], key=lambda s: s[0])
    j = 0
    ns["controller"](0.0, tuple(cap["poses"][0][1:]), None)          # first call: map and plan
    false_caps, calls, worst = 0, 0, 9e9
    for t, x, y, yaw in cap["poses"]:
        while j < len(scans) and scans[j][0] <= t + 1e-9:
            if scans[j][4] is not None:
                cur["front"] = scans[j][4].astype(np.float64)
            if scans[j][5] is not None:
                cur["back"] = scans[j][5].astype(np.float64)
            j += 1
        ns["_sense"](t, (x, y, yaw))
        K = int(round(ns["_CFG"]["T"] / ns["_CFG"]["dt"]))
        shapes = ns["_obstacle_shapes"](t, K, ns["_CFG"]["dt"]) if "_obstacle_shapes" in ns else (None, [])
        caps = shapes[1]
        near = [float(np.hypot(*(a[0] - np.array([x, y])))) for a, _, _ in caps]
        near = [d for d in near if d < 6.0]
        false_caps += len(near)
        if near:
            worst = min(worst, min(near))
        calls += 1
    return calls, false_caps, worst


for f in sys.argv[1:]:
    calls, n, worst = replay(f)
    print(f"{Path(f).name}: {calls} control calls, false moving obstacles within 6 m summed over calls {n} "
          f"(per call {n / calls:.3f}), nearest {None if worst > 1e8 else round(worst, 2)} m")
