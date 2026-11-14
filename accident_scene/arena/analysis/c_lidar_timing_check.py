"""Lidar timing facts for the prompt, from the static-scene Isaac recording (claude_ros_probe01, no movers, the robot
drives the route): how often each 2D lidar's content changes, how old a scan is (in control steps) when it first
reaches controller() and when the same content comes back again, how many false 'unmapped' points the current pose
gives versus the pose at scan time, and where unmapped points remain with the scan-time pose (map residuals).

    python3 analysis/c_lidar_timing_check.py
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
cap = pickle.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))
S, P = cap["scans"], np.array(cap["poses"])                    # P: t, x, y, yaw at every control call
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
    x, y, yaw = P[i, 1], P[i, 2], P[i, 3]
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * Q[:, 0] - s * Q[:, 1], y + s * Q[:, 0] + c * Q[:, 1]


FACTS = {"source": "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "calls": len(S)}
assert len(S) == len(P) and all(abs(S[i][0] - P[i, 0]) < 1e-9 for i in range(len(S)))
w = np.zeros(len(P))
w[1:] = np.abs(np.arctan2(np.sin(np.diff(P[:, 3])), np.cos(np.diff(P[:, 3])))) * 60
v = np.zeros(len(P))
v[1:] = np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2])) * 60
# repetition index: 0 on the call where the content first appears, 1, 2, ... while the same content comes back
rep = {}
for col, name in ((4, "front"), (5, "back")):
    prev, k, r = None, -1, np.full(len(S), -1)
    for i, s in enumerate(S):
        q = s[col]
        if q is None or len(q) == 0:
            continue
        k = 0 if (prev is None or not (q.shape == prev.shape and np.array_equal(q, prev))) else k + 1
        r[i], prev = k, q
    rep[name] = r
    FACTS[f"{name}_rep_counts"] = np.bincount(r[r >= 0]).tolist()
    print(f"{name}: calls with a scan {int((r >= 0).sum())}, first appearances {int((r == 0).sum())}, repetition index counts {np.bincount(r[r >= 0]).tolist()}")


def false_pts(name, idx, lag, rmin, rmax, margin=0.1):
    col = 4 if name == "front" else 5
    n = tot = 0
    for i in idx:
        j = i - lag
        if j < 0:
            continue
        Q = S[i][col]
        rr = np.hypot(Q[:, 0], Q[:, 1])
        Q = Q[(rr >= rmin) & (rr < rmax)]
        if len(Q) == 0:
            continue
        x, y = world(Q, j)
        n += int((clear(x, y) > margin).sum()); tot += len(Q)
    return n / max(1, len(idx)), n / max(1, tot)


print("\nbest lag (control steps) by repetition index, turning |w| > 0.5 rad/s, 3-9 m, margin 0.1 m:")
for name in ("front", "back"):
    for k in range(6):
        idx = [i for i in np.where(rep[name] == k)[0] if w[i] > 0.5]
        if len(idx) < 20:
            continue
        idx = idx[::max(1, len(idx) // 150)]
        res = [(false_pts(name, idx, lag, 3, 9)[0], lag) for lag in range(0, 16)]
        best = min(res)
        FACTS.setdefault("best_lag_steps", {}).setdefault(name, []).append(int(best[1]))
        print(f"  {name} rep {k}: {len(idx)} scans, best lag {best[1]} steps ({best[1] / 60:.3f} s), false/scan at lag 0 {res[0][0]:.1f}, at best {best[0]:.2f}")

print("\nfalse unmapped points per scan, first appearances only, margin 0.1 m (current pose vs pose 6 steps earlier):")
for name in ("front", "back"):
    for label, sel in (("standstill (v<0.05, |w|<0.05)", (v < 0.05) & (w < 0.05)), ("driving (v>0.4, |w|<0.2)", (v > 0.4) & (w < 0.2)),
                       ("turning (|w|>0.5)", w > 0.5)):
        idx = [i for i in np.where((rep[name] == 0) & sel)[0]]
        if len(idx) < 5:
            print(f"  {name} {label}: {len(idx)} scans"); continue
        idx = idx[::max(1, len(idx) // 200)]
        out = []
        for lag in (0, 6):
            out.append([round(false_pts(name, idx, lag, a, b)[0], 1) for a, b in ((0, 3), (3, 6), (6, 9))])
        FACTS.setdefault("false_per_scan_0_9m", {})[f"{name} {label.split(' (')[0]}"] = {"lag0": round(sum(out[0]), 1), "lag6": round(sum(out[1]), 1)}
        print(f"  {name} {label}: {len(idx)} scans | lag 0: {out[0]}  lag 6: {out[1]}  (0-3, 3-6, 6-9 m)")

print("\nresidual unmapped points with the scan-time pose (first appearances, lag 6, within 8 m), by margin:")
for margin in (0.1, 0.15, 0.25):
    pts = []
    for name, col in (("front", 4), ("back", 5)):
        for i in np.where(rep[name] == 0)[0]:
            if i < 6:
                continue
            Q = S[i][col]
            rr = np.hypot(Q[:, 0], Q[:, 1])
            Q = Q[rr < 8]
            x, y = world(Q, i - 6)
            m = clear(x, y) > margin
            pts.append(np.stack([x[m], y[m], clear(x[m], y[m])], axis=1))
    A = np.concatenate(pts)
    FACTS.setdefault("residual_points_8m_lag6", {})[str(margin)] = {"n": int(len(A)), "max_map_dist": round(float(A[:, 2].max()), 3) if len(A) else None,
                                                                   "max_at_xy": [round(float(v), 2) for v in A[A[:, 2].argmax(), :2]] if len(A) else None,
                                                                   "share_near_box08": round(float(np.mean((np.abs(A[:, 0] - 3.4) < 0.75) & (np.abs(A[:, 1] - 13.0) < 0.75))), 3) if len(A) else None}
    print(f"  margin {margin}: {len(A)} points")
    if len(A):
        cells = {}
        for x, y, c in A:
            cells.setdefault((round(x * 2) / 2, round(y * 2) / 2), []).append((x, y, c))
        for (cx, cy), L in sorted(cells.items(), key=lambda kv: -len(kv[1]))[:8]:
            L = np.array(L)
            print(f"     cell ({cx:+.1f}, {cy:+.1f}) hits {len(L):>5}  x {L[:, 0].min():.2f}..{L[:, 0].max():.2f}  y {L[:, 1].min():.2f}..{L[:, 1].max():.2f}  map dist {L[:, 2].min():.2f}..{L[:, 2].max():.2f}")

# content changes of the two lidars on the same call
pf = pb = None; same = fonly = bonly = 0
for s_ in S:
    f, b = s_[4], s_[5]
    cf = f is not None and len(f) > 0 and (pf is None or not (f.shape == pf.shape and np.array_equal(f, pf)))
    cb = b is not None and len(b) > 0 and (pb is None or not (b.shape == pb.shape and np.array_equal(b, pb)))
    same += cf and cb; fonly += cf and not cb; bonly += cb and not cf
    pf, pb = (f if f is not None and len(f) else pf), (b if b is not None and len(b) else pb)
FACTS["content_change_same_call"], FACTS["content_change_front_only"], FACTS["content_change_back_only"] = int(same), int(fonly), int(bonly)
out = ARENA / "results/analysis/claude_lidar_rate/timing_facts.json"
out.write_text(json.dumps(FACTS, indent=1, ensure_ascii=False))
print("wrote", out, "\n", json.dumps(FACTS, ensure_ascii=False))
