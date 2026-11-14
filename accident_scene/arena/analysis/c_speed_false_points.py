"""Does driving slower make the models' current-pose transform harmless? Static-scene replay (claude_ros_probe01, no movers):
every control call (repeated scans included), both 2D lidars transformed with the CURRENT pose as the B-5 models do, binned
by the robot's measured speed v (|w| < 0.1) and turn rate |w| (v < 0.1). Reports false unmapped points per call (0-9 m) at
margins 0.1 / 0.25 and the share of calls with a false point inside a stop radius (the B-5-10 rules: 1.5 m at margin 0.05,
2.0 m at margin 0.2). The pose is the harness pose; the speed is from pose differences.

    python3 analysis/c_speed_false_points.py
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


# speed and turn rate from a centred difference over +-3 steps (0.1 s), so a single-step jitter does not dominate
t = P[:, 0]
v = np.zeros(len(P)); w = np.zeros(len(P))
for i in range(3, len(P) - 3):
    dt = t[i + 3] - t[i - 3]
    v[i] = math.hypot(P[i + 3, 1] - P[i - 3, 1], P[i + 3, 2] - P[i - 3, 2]) / dt
    w[i] = abs(math.atan2(math.sin(P[i + 3, 3] - P[i - 3, 3]), math.cos(P[i + 3, 3] - P[i - 3, 3]))) / dt
rows = []
for i in range(6, len(S) - 3):
    x, y, yaw = P[i, 1], P[i, 2], P[i, 3]
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [q for q in (S[i][4], S[i][5]) if q is not None and len(q)]
    if not pts:
        continue
    Q = np.concatenate(pts)
    r = np.hypot(Q[:, 0], Q[:, 1]); Q = Q[r < 9]
    wx, wy = x + c * Q[:, 0] - s * Q[:, 1], y + s * Q[:, 0] + c * Q[:, 1]
    cl = clear(wx, wy); d = np.hypot(wx - x, wy - y)
    rows.append((v[i], w[i], int((cl > 0.1).sum()), int((cl > 0.25).sum()),
                 bool(((cl > 0.05) & (d < 1.5)).any()), bool(((cl > 0.2) & (d < 2.0)).any())))
A = np.array(rows, dtype=float)
out = {}


def report(label, sel):
    B = A[sel]
    if len(B) < 10:
        print(f"  {label:<26} calls {len(B):>5} (too few)"); return
    res = dict(calls=int(len(B)), false_m010=round(B[:, 2].mean(), 1), false_m025=round(B[:, 3].mean(), 2),
               stop15_share=round(B[:, 4].mean(), 3), stop20_share=round(B[:, 5].mean(), 3))
    out[label] = res
    print(f"  {label:<26} calls {res['calls']:>5} | false pts/call m0.1 {res['false_m010']:>6} m0.25 {res['false_m025']:>6}"
          f" | false stop: 1.5 m@0.05 {res['stop15_share']:>6.1%}  2.0 m@0.2 {res['stop20_share']:>6.1%}")


print("driving straight (|w| < 0.1 rad/s), by speed v:")
for lo, hi in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7)):
    report(f"v {lo:.2f}-{hi:.2f} m/s", (A[:, 0] >= lo) & (A[:, 0] < hi) & (A[:, 1] < 0.1))
print("turning in place (v < 0.1 m/s), by turn rate |w|:")
for lo, hi in ((0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.9)):
    report(f"|w| {lo:.1f}-{hi:.1f} rad/s", (A[:, 1] >= lo) & (A[:, 1] < hi) & (A[:, 0] < 0.1))
(ARENA / "results/analysis/claude_lidar_rate/speed_false_points.json").write_text(json.dumps(out, indent=1))
