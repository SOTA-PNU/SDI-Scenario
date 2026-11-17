"""What the 2D lidars saw around an event, for the analyst (2026-11-15).

Reads <variant>_lidar2d.npz written by arena_harness when CARTER_SENSOR_LOG=1 (both 2D lidars, every 0.1 s, ranges in
cm with the ground-truth pose at that moment). Each return is put in the world with the lidar mounts measured from /tf
(REPORT.md 실험 21; the same values as _LIDAR2D in base_carter_run_ros_noenv.py), points on the robot itself are
dropped, and every other point is called static when it lies on the fixed-structure grid (arena_map.json, 0.25 m) or
on a floor box of MISSION, each grown by STATIC_MARGIN_M. What is left is "not on the map": in this arena that is a
moving obstacle (or noise). For each kept scan in the window the digest gives, in the robot frame (x ahead, y left,
bearing 0 = ahead, + = left), the nearest point of each sector and the nearest not-on-map point.

    python3 sensor_digest.py <variant> [--log-dir DIR] [--t-end T] [--window 5]
"""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOUNTS = {"front": (0.026, 0.0, math.pi), "back": (-0.489, 0.0, 0.0)}   # (x, y, yaw) in base_link, from /tf
SELF_BOX = (-0.70, 0.25, -0.34, 0.34)       # robot-frame box larger than the body (base _BASE_SELF_BOX)
STATIC_MARGIN_M = 0.15
# A scan is put in the world with the ground-truth pose POSE_LAG_S before it was received. Swept on seed 18533188
# (2026-11-15): wall/rack returns land off the map least at 0.133 s for both lidars (front 14,390 -> 3,193 off-map
# returns, back 13,131 -> 4,628); the header stamp alone (0.167 s) is worse. The rest is smear while turning, so a
# not-on-map object needs CLUSTER_MIN returns within CLUSTER_R_M of each other.
POSE_LAG_S = 0.133
CLUSTER_MIN = 5
CLUSTER_R_M = 0.3
# far returns smear more while turning (the back lidar sweeps 360 deg over 0.1 s): a return r metres away also counts
# as static within RANGE_MARGIN_PER_M * r of the map (box 00 returns 3.8 m behind landed 0.04 m off its margin)
RANGE_MARGIN_PER_M = 0.05
MAX_RANGE_M = 8.0
SECTORS = {"front": (-30, 30), "left": (30, 150), "right": (-150, -30)}   # degrees; the rest is "back"


def _v18_literal(name):
    """WALLS / FIXED_OBSTACLES from base v18: arena_map.json misses the wall faces and several racks and pillars at the
    lidar height (first check on seed 18533188: hundreds of false not-on-map returns per scan near x=-10 and y=-12.5)."""
    src = (HERE / "base_carter_run_ros_map_lidartime_objects_check_v18.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    return None


def _mission_boxes():
    src = (HERE / "base_carter_run_ros_noenv.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MISSION":
            return eval(compile(ast.Expression(node.value), "MISSION", "eval"), {"dict": dict})["boxes"]
    return []


class StaticMap:
    def __init__(self):
        m = json.loads((HERE / "arena_map.json").read_text())
        self.x0, self.y0, self.step = m["x0"], m["y0"], m["step"]
        g = np.array(m["grid"], dtype=bool)
        k = int(math.ceil(STATIC_MARGIN_M / self.step))
        grown = g.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                grown |= np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        self.grid = grown
        self.boxes = [(b["x"], b["y"], b["width"] / 2 + STATIC_MARGIN_M, b["depth"] / 2 + STATIC_MARGIN_M)
                      for b in _mission_boxes()]
        self.walls = _v18_literal("WALLS")
        self.fixed = [(x0 - STATIC_MARGIN_M, x1 + STATIC_MARGIN_M, y0 - STATIC_MARGIN_M, y1 + STATIC_MARGIN_M)
                      for _, x0, x1, y0, y1 in (_v18_literal("FIXED_OBSTACLES") or [])]

    def static(self, xy, extra=None):
        """True where a point is on the map; `extra` (per point, m) widens the test by probing 8 directions."""
        out = self._static(xy)
        if extra is not None and len(xy):
            for a in np.radians(np.arange(0, 360, 45)):
                off = np.stack([np.cos(a) * extra, np.sin(a) * extra], axis=1)
                out |= self._static(xy + off)
        return out

    def _static(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.step).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.step).astype(int)
        inb = (ix >= 0) & (ix < self.grid.shape[1]) & (iy >= 0) & (iy < self.grid.shape[0])
        out = np.ones(len(xy), dtype=bool)          # outside the grid: treat as static (beyond the walls)
        out[inb] = self.grid[iy[inb], ix[inb]]
        for bx, by, hw, hd in self.boxes:
            out |= (np.abs(xy[:, 0] - bx) <= hw) & (np.abs(xy[:, 1] - by) <= hd)
        for x0, x1, y0, y1 in self.fixed:
            out |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        if self.walls:
            wx0, wx1, wy0, wy1 = self.walls       # inner wall faces: on or beyond them is wall
            m = STATIC_MARGIN_M
            out |= (xy[:, 0] <= wx0 + m) | (xy[:, 0] >= wx1 - m) | (xy[:, 1] <= wy0 + m) | (xy[:, 1] >= wy1 - m)
        return out


def _scan_points(which, ranges_cm, meta):
    """Robot-frame (x, y) of the valid returns of one scan."""
    r = ranges_cm.astype(np.float32) / 100.0
    ang = meta["angle_min"] + meta["angle_increment"] * np.arange(len(r))
    ok = (r > 0) & (r <= MAX_RANGE_M)
    mx, my, myaw = MOUNTS[which]
    a = ang[ok] + myaw
    pts = np.stack([mx + r[ok] * np.cos(a), my + r[ok] * np.sin(a)], axis=1)
    body = ((pts[:, 0] >= SELF_BOX[0]) & (pts[:, 0] <= SELF_BOX[1]) & (pts[:, 1] >= SELF_BOX[2]) & (pts[:, 1] <= SELF_BOX[3]))
    return pts[~body]


def _to_world(pts, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1)


def _clusters(pts):
    """Robot-frame not-on-map points kept only where CLUSTER_MIN of them lie within CLUSTER_R_M (greedy, nearest first)."""
    if len(pts) < CLUSTER_MIN:
        return pts[:0]
    d = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(d)
    keep = np.zeros(len(pts), dtype=bool)
    for i in order[:400]:
        if keep[i]:
            continue
        near = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]) <= CLUSTER_R_M
        if near.sum() >= CLUSTER_MIN:
            keep |= near
    return pts[keep]


def _nearest(pts):
    if len(pts) == 0:
        return None
    d = np.hypot(pts[:, 0], pts[:, 1])
    i = int(np.argmin(d))
    return {"dist_m": round(float(d[i]), 2), "bearing_deg": round(math.degrees(math.atan2(pts[i, 1], pts[i, 0])), 0)}


def digest(variant, log_dir, t_end=None, window=5.0, after=0.5):
    f = Path(log_dir) / f"{variant}_lidar2d.npz"
    if not f.exists():
        return {"available": False, "note": "센서 기록 없음(CARTER_SENSOR_LOG 꺼짐)"}
    z = np.load(f)
    meta = json.loads(str(z["meta_json"]))
    smap = StaticMap()
    lidars = [w for w in ("front", "back") if w in meta]
    t_last = max(float(z[f"{w}_t"][-1]) for w in lidars)
    t_end = t_last if t_end is None else t_end
    rows = []
    tr = z["traj_full"].astype(np.float64) if "traj_full" in z else None
    for which in lidars:
        T, P, R = z[f"{which}_t"], z[f"{which}_pose"], z[f"{which}_ranges_cm"]
        if tr is not None and f"{which}_t_recv" in z:
            T = z[f"{which}_t_recv"].astype(np.float64) - POSE_LAG_S
            yaw = np.unwrap(tr[:, 3])
            P = np.stack([np.interp(T, tr[:, 0], tr[:, 1]), np.interp(T, tr[:, 0], tr[:, 2]),
                          np.interp(T, tr[:, 0], yaw)], axis=1)
        for k in np.where((T >= t_end - window) & (T <= t_end + after))[0]:
            pts = _scan_points(which, R[k], meta[which])
            wpts = _to_world(pts, P[k])
            rng = np.hypot(pts[:, 0], pts[:, 1])
            unk = _clusters(pts[~smap.static(wpts, RANGE_MARGIN_PER_M * rng)])
            bearing = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            sec = {}
            for name, (lo, hi) in SECTORS.items():
                sec[name] = _nearest(pts[(bearing >= lo) & (bearing < hi)])
            sec["back"] = _nearest(pts[(bearing >= 150) | (bearing < -150)])
            near_unk = _nearest(unk)
            rows.append({"t": round(float(T[k]), 2), "lidar": which, "robot_pose": [round(float(v), 2) for v in P[k]],
                         "nearest_by_sector": {k2: (v["dist_m"] if v else None) for k2, v in sec.items()},
                         "not_on_map": ({**near_unk, "points": int(len(unk)),
                                         "world_xy": [round(float(c), 2) for c in
                                                      _to_world(unk[[int(np.argmin(np.hypot(unk[:, 0], unk[:, 1])))]],
                                                                P[k])[0]]}
                                        if near_unk else None)})
    rows.sort(key=lambda r: (r["t"], r["lidar"]))
    first = {}
    for th in (5.0, 3.0, 2.0, 1.0):
        hit = next((r for r in rows if r["not_on_map"] and r["not_on_map"]["dist_m"] <= th), None)
        first[f"within_{th:g}m"] = ({"t": hit["t"], "lidar": hit["lidar"], **{k: hit["not_on_map"][k]
                                     for k in ("dist_m", "bearing_deg", "world_xy")}} if hit else None)
    return {"available": True, "window": [round(t_end - window, 2), round(t_end + after, 2)],
            "frame": "로봇 기준: 거리는 로봇 중심(base_link)에서, bearing 0 = 정면, + = 왼쪽, - = 오른쪽",
            "timing": f"스캔은 받은 시각보다 {POSE_LAG_S} s 앞의 실제 자세로 월드에 놓음(실측 보정값)",
            "cluster_rule": (f"지도에 없는 점은 {CLUSTER_R_M} m 안에 {CLUSTER_MIN}점 이상 모인 것만 셈. 지도 여유는 "
                             f"{STATIC_MARGIN_M} m + 거리 1 m 당 {RANGE_MARGIN_PER_M} m"),
            "static_rule": (f"벽 안쪽 면, 고정 구조물(격자 arena_map.json 과 FIXED_OBSTACLES), MISSION 박스에서 {STATIC_MARGIN_M} m 안"
                            "이거나 벽 바깥인 점은 지도에 있는 점"),
            "first_not_on_map": first, "scans": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--log-dir", default=str(HERE / "results"))
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--window", type=float, default=5.0)
    a = ap.parse_args()
    print(json.dumps(digest(a.variant, a.log_dir, a.t_end, a.window), ensure_ascii=False, indent=1))
