"""Offline analysis of the static-scene Isaac probe (claude_ros_probe01): no moving obstacles were in the scene, so every
lidar point farther than the margin from the static map is a false 'unmapped' point. Finds the pose delay that best
registers each sensor's scans with the map, the false-point counts by range, and where persistent false points sit.

    python3 analysis/claude_ros_probe_analysis.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
cap = pickle.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))
poses = np.array(cap["poses"])                                   # t, x, y, yaw at every control call
src = (ARENA / "analysis/claude_ros_region.py").read_text()
ns = {"np": np, "math": math}
exec(src[src.index("_WALLS ="):src.index("_BODY_C =")], ns)
mission_src = (ARENA / "base_carter_run_ros.py").read_text()
exec(mission_src[mission_src.index("MISSION = dict("):mission_src.index("\nSTART = ")], ns)
boxes = [(b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2) for b in ns["MISSION"]["boxes"]]
R = np.concatenate([np.array(boxes), ns["_FIXED"]])
WX0, WX1, WY0, WY1 = ns["_WALLS"]


def clear(x, y):
    dx = np.maximum(np.maximum(R[:, 0] - x[:, None], x[:, None] - R[:, 1]), 0)
    dy = np.maximum(np.maximum(R[:, 2] - y[:, None], y[:, None] - R[:, 3]), 0)
    d = np.hypot(dx, dy).min(axis=1)
    return np.maximum(0, np.minimum(d, np.minimum(np.minimum(x - WX0, WX1 - x), np.minimum(y - WY0, WY1 - y))))


def pose_at(t):
    i = np.searchsorted(poses[:, 0], t)
    i = min(max(i, 1), len(poses) - 1)
    a, b = poses[i - 1], poses[i]
    f = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
    f = min(max(f, 0.0), 1.0)
    yaw = a[3] + f * math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), yaw


def world(P, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * P[:, 0] - s * P[:, 1], y + s * P[:, 0] + c * P[:, 1]


bins = [(0, 3), (3, 6), (6, 9), (9, 12)]
for which, col in (("front", 4), ("back", 5)):
    scans = [(s[0], s[col]) for s in cap["scans"] if s[col] is not None]
    if which == "back":
        scans = scans[::6]
    print(f"== {which}: {len(scans)} scans used")
    best = None
    for delay in (0.0, 1 / 60, 0.033, 0.05, 0.067, 0.083, 0.1, 0.133, 0.167):
        cnt = np.zeros(len(bins)); tot = np.zeros(len(bins))
        for t, P in scans:
            P = P.astype(np.float64)
            r = np.hypot(P[:, 0], P[:, 1])
            wx, wy = world(P, pose_at(t - delay))
            un = clear(wx, wy) > 0.25
            for k, (a, b) in enumerate(bins):
                sel = (r >= a) & (r < b)
                cnt[k] += un[sel].sum(); tot[k] += sel.sum()
        frac = cnt / np.maximum(tot, 1)
        score = cnt.sum()
        print(f"  delay {delay:.3f} s: false points per scan by range {[round(c / len(scans), 1) for c in cnt]}  fraction {[round(f, 4) for f in frac]}")
        if best is None or score < best[0]:
            best = (score, delay)
    print(f"  best delay for {which}: {best[1]:.3f} s")
# where do false points persist (front scans, best-ish delay 0.05 s, within 6 m)?
pts = []
for s in cap["scans"]:
    if s[4] is None:
        continue
    P = s[4].astype(np.float64)
    r = np.hypot(P[:, 0], P[:, 1])
    P = P[r < 6]
    wx, wy = world(P, pose_at(s[0] - 0.05))
    un = clear(wx, wy) > 0.25
    pts.append(np.stack([wx[un], wy[un]], axis=1))
F = np.concatenate(pts) if pts else np.zeros((0, 2))
print("persistent false points within 6 m (front, delay 0.05): total", len(F))
if len(F):
    key = np.floor(F / 0.5).astype(int)
    u, cnts = np.unique(key, axis=0, return_counts=True)
    order = np.argsort(-cnts)[:15]
    for k in order:
        cx, cy = (u[k] + 0.5) * 0.5
        print(f"  cell ({cx:.2f}, {cy:.2f}) hits {cnts[k]}  map clearance {float(clear(np.array([cx]), np.array([cy]))[0]):.2f}")
