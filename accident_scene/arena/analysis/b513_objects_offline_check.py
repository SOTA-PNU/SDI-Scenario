"""Offline check of the B-5-13 base tool observed_objects() (base_carter_run_ros_map_lidartime_objects.py) on the real
Isaac scans recorded in the static-scene probe (claude_ros_probe01: 4,338 control calls, the robot drives the route, no
movers; the arrays exactly as lidar2d_points() returned them, repeats included). (1) static: every reported object is false.
(2) synthetic mover: a 0.6 m square shuttling at 0.5 m/s across the route is drawn into the same scans, visible faces only,
at the pose and time each scan was taken (0.1 s before it first reaches the controller); the tool's nearest object is
compared with the true square. The mover only exists in this check; nothing of it is in the base or the prompt.

    python3 analysis/b513_objects_offline_check.py
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
BASE = (ARENA / "base_carter_run_ros_map_lidartime_objects.py").read_text()
cap = pickle.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))
S, PO = cap["scans"], np.array(cap["poses"])


def load_ns(feed):
    ns = {"np": np, "math": math}
    exec(BASE[BASE.index("MISSION = dict("):BASE.index("\nSTART = ")], ns)
    exec(BASE[BASE.index("def to_world("):BASE.index("# ---- 지도 기본 코드 ----")], ns)
    exec(BASE[BASE.index("# ---- 지도 기본 코드 ----"):BASE.index("# ---- DBG 기록 ----")], ns)
    ns["lidar2d_points"] = feed
    return ns

# v2 (after the review): the recording holds None on calls where that lidar delivered no new array; lidar2d_points() then
# returns the last array, so the feed does too. Visibility in the robot frame around each mount: front |bearing| <= 140 deg,
# back |bearing| >= 40 deg. No occlusion between objects is modelled.


def run(mover=None):
    i_now = {"i": 0}
    cache = {}

    last = {"front": np.zeros((0, 2)), "back": np.zeros((0, 2))}

    def feed(which):
        i = i_now["i"]
        arr = S[i][4 if which == "front" else 5]
        if arr is None:
            arr = last[which]
        last[which] = arr
        if mover is None:
            return arr
        key = (which, id(arr))                      # same array object -> same content -> same synthetic points
        if key not in cache:
            cache[key] = add_mover(arr, i, which)
        return cache[key]

    ns = load_ns(feed)
    rows = []
    for i in range(len(S)):
        i_now["i"] = i
        t, x, y, yaw = PO[i]
        ns["_BASE_POSES"].append((t, x, y, yaw))
        while ns["_BASE_POSES"] and t - ns["_BASE_POSES"][0][0] > 1.0:
            ns["_BASE_POSES"].pop(0)
        ns["_base_track_step"](t)                      # what _robot_only does every step before the controller
        objs = ns["observed_objects"]()
        rows.append((i, t, objs))
    return rows


LANE = ((-7.0, -11.5), (-7.0, -8.6))          # synthetic shuttle across the start of the route (check only)
SPEED, HALF = 0.5, 0.3


def mover_at(t):
    (x0, y0), (x1, y1) = LANE
    L = math.hypot(x1 - x0, y1 - y0)
    s = (SPEED * t) % (2 * L)
    s = s if s <= L else 2 * L - s
    f = s / L
    vx_dir = 1 if (SPEED * t) % (2 * L) <= L else -1
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0), vx_dir * SPEED * (x1 - x0) / L, vx_dir * SPEED * (y1 - y0) / L


def first_index(i, which):
    arr = S[i][4 if which == "front" else 5]
    j = i
    while j > 0 and S[j - 1][4 if which == "front" else 5] is arr:
        j -= 1
    while j > 0 and (S[j - 1][4 if which == "front" else 5] is not None and np.array_equal(S[j - 1][4 if which == "front" else 5], arr)):
        j -= 1
    return j


def add_mover(arr, i, which):
    j = first_index(i, which)                     # the call where this scan content first arrived
    k = max(0, j - 6)                             # taken 6 steps (0.1 s) earlier
    t_s, x, y, yaw = PO[k]
    cx, cy, _, _ = mover_at(t_s)
    mount = np.array([0.026, 0.0]) if which == "front" else np.array([-0.489, 0.0])
    c, s = math.cos(yaw), math.sin(yaw)
    lw = np.array([x + c * mount[0] - s * mount[1], y + s * mount[0] + c * mount[1]])
    pts = []
    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        fc = np.array([cx + nx * HALF, cy + ny * HALF])
        if (lw - fc) @ np.array([nx, ny]) <= 0:
            continue                                # face turned away from this lidar
        for u in np.linspace(-HALF, HALF, 13):
            pts.append(fc + u * np.array([-ny, nx]))
    if not pts or math.hypot(cx - lw[0], cy - lw[1]) > 25:
        return arr
    W = np.array(pts)
    rx, ry = c * (W[:, 0] - x) + s * (W[:, 1] - y), -s * (W[:, 0] - x) + c * (W[:, 1] - y)
    R = np.stack([rx, ry], axis=1)
    ang = np.degrees(np.arctan2(R[:, 1] - mount[1], R[:, 0] - mount[0]))
    vis = np.abs(ang) <= 140 if which == "front" else np.abs(ang) >= 40   # robot-frame bearings each lidar sees
    return np.concatenate([arr, R[vis]]) if vis.any() else arr


out = {}
static = run(None)
false_calls = sum(1 for _, _, o in static if any(ob["dist"] < 8 for ob in o))
ids = {ob["id"] for _, _, o in static for ob in o}
out["static"] = {"calls": len(static), "calls_with_object": false_calls, "distinct_ids": len(ids),
                 "max_points": max([ob["n_points"] for _, _, o in static for ob in o], default=0)}
print("static:", out["static"])
dyn = run("v")
err_p, err_v, false_n, seen = [], [], [], 0
for i, t, objs in dyn:
    if not objs:
        continue
    ob = objs[0]
    mx, my, mvx, mvy = mover_at(ob["seen_t"])
    d_true = math.hypot(mx - PO[i, 1], my - PO[i, 2])
    if d_true > 7.5:
        continue
    near = min(objs, key=lambda o: math.hypot(o["x"] - mx, o["y"] - my))
    e = math.hypot(near["x"] - mx, near["y"] - my)
    if e < 0.8:
        seen += 1
        err_p.append(e)
        err_v.append(math.hypot(near["vx"] - mvx, near["vy"] - mvy))
    false_n.append(sum(1 for o in objs if math.hypot(o["x"] - mx, o["y"] - my) > 1.0 and o["dist"] < 8))
calls_in = sum(1 for i, t, o in dyn if math.hypot(mover_at(PO[i, 0] - 0.1)[0] - PO[i, 1], mover_at(PO[i, 0] - 0.1)[1] - PO[i, 2]) <= 7.5)
out["synthetic_mover"] = {"calls_mover_within_7.5m": calls_in, "calls_tracked": seen,
                          "pos_err_median_m": round(float(np.median(err_p)), 3) if err_p else None,
                          "pos_err_p95_m": round(float(np.percentile(err_p, 95)), 3) if err_p else None,
                          "vel_err_median_mps": round(float(np.median(err_v)), 3) if err_v else None,
                          "vel_err_p95_mps": round(float(np.percentile(err_v, 95)), 3) if err_v else None,
                          "calls_with_false_object": int(sum(1 for f in false_n if f))}
print("synthetic mover:", out["synthetic_mover"])
(ARENA / "results/analysis/b513_objects_offline.json").write_text(json.dumps(out, indent=1))
