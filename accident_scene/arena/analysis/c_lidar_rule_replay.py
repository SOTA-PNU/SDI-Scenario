"""Static-scene replay (claude_ros_probe01, no movers): what the current-pose transform does on EVERY control call,
including the calls where the same front scan comes back again (up to 11 steps old), versus the scan-time pose
(6 steps + repetition index). Two numbers: false unmapped points per call by repetition index (front, 0-9 m, margin 0.1),
and the B-5-10 a0 final rule (both 2D lidars, margin 0.05, any unmapped point within 1.5 m of the pose => stop).

    python3 analysis/c_lidar_rule_replay.py
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
cap = pickle.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))
S, P = cap["scans"], np.array(cap["poses"])
src = (ARENA / "analysis/claude_ros_region.py").read_text()
ns = {"np": np, "math": math}
exec(src[src.index("_WALLS ="):src.index("_BODY_C =")], ns)
msrc = (ARENA / "base_carter_run_ros.py").read_text()
exec(msrc[msrc.index("MISSION = dict("):msrc.index("\nSTART = ")], ns)
boxes = [(b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2) for b in ns["MISSION"]["boxes"]]
R = np.concatenate([np.array(boxes), ns["_FIXED"]])
WX0, WX1, WY0, WY1 = ns["_WALLS"]


def clear(x, y):
    dx = np.maximum(np.maximum(R[:, 0] - x[:, None], x[:, None] - R[:, 1]), 0)
    dy = np.maximum(np.maximum(R[:, 2] - y[:, None], y[:, None] - R[:, 3]), 0)
    d = np.hypot(dx, dy).min(axis=1)
    return np.maximum(0, np.minimum(d, np.minimum(np.minimum(x - WX0, WX1 - x), np.minimum(y - WY0, WY1 - y))))


def world(Q, i):
    x, y, yaw = P[max(i, 0), 1], P[max(i, 0), 2], P[max(i, 0), 3]
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * Q[:, 0] - s * Q[:, 1], y + s * Q[:, 0] + c * Q[:, 1]


w = np.zeros(len(P)); w[1:] = np.abs(np.arctan2(np.sin(np.diff(P[:, 3])), np.cos(np.diff(P[:, 3])))) * 60
v = np.zeros(len(P)); v[1:] = np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2])) * 60
rep = {}
for col in (4, 5):
    prev, k, r = None, -1, np.full(len(S), -1)
    for i, s in enumerate(S):
        q = s[col]
        if q is None or len(q) == 0:
            continue
        k = 0 if (prev is None or not (q.shape == prev.shape and np.array_equal(q, prev))) else k + 1
        r[i], prev = k, q
    rep[col] = r
states = {"driving": (v > 0.4) & (w < 0.2), "turning": w > 0.5}
FACTS = {}
# 1. front: false unmapped points per call, current pose, by repetition index
for name, sel in states.items():
    row = []
    for k in range(6):
        idx = np.where((rep[4] == k) & sel)[0]
        idx = idx[::max(1, len(idx) // 200)]
        n = 0
        for i in idx:
            Q = S[i][4]; rr = np.hypot(Q[:, 0], Q[:, 1]); Q = Q[rr < 9]
            x, y = world(Q, i)
            n += int((clear(x, y) > 0.1).sum())
        row.append(round(n / max(1, len(idx)), 1))
    FACTS[f"front_{name}_lag0_by_rep_0_9m"] = row
    print(f"front {name}: false points per call at the current pose, by repetition 0..5: {row}")
# 2. a0 final rule on every call
hit_now, hit_scan = np.zeros(len(S), bool), np.zeros(len(S), bool)
for i in range(len(S)):
    pts_now, pts_scan = [], []
    for col in (4, 5):
        Q = S[i][col]
        if Q is None or len(Q) == 0 or rep[col][i] < 0:
            continue
        rr = np.hypot(Q[:, 0], Q[:, 1]); Q = Q[rr < 3.0]
        if len(Q) == 0:
            continue
        for lag, out in ((0, pts_now), (6 + rep[col][i], pts_scan)):
            x, y = world(Q, i - lag)
            m = clear(x, y) > 0.05
            out.append(np.hypot(x[m] - P[i, 1], y[m] - P[i, 2]))
    hit_now[i] = any((d < 1.5).any() for d in pts_now)
    hit_scan[i] = any((d < 1.5).any() for d in pts_scan)
moving = v >= 0.3
straight = moving & (w < 0.2)
FACTS["a0_rule_calls"] = int(len(S))
FACTS["a0_rule_false_stop_calls_now"] = int(hit_now.sum())
FACTS["a0_rule_false_stop_calls_scan_time"] = int(hit_scan.sum())
FACTS["a0_rule_straight_ge_0.3_share_now"] = round(float(hit_now[straight].mean()), 3)
FACTS["a0_rule_straight_ge_0.3_share_scan_time"] = round(float(hit_scan[straight].mean()), 3)
print(f"a0 rule (both 2D, margin 0.05, < 1.5 m): false stops on {hit_now.sum()} of {len(S)} calls at the current pose, "
      f"{hit_scan.sum()} at the scan-time pose; straight driving >= 0.3 m/s: {hit_now[straight].mean():.1%} vs {hit_scan[straight].mean():.1%}")
out = ARENA / "results/analysis/claude_lidar_rate/rule_replay_facts.json"
out.write_text(json.dumps(FACTS, indent=1)); print("wrote", out)
