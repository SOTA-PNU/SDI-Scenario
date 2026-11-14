"""Check the raw all-sensor probe (analysis/c_allsensor_probe.py) against the known map, offline.

For each lidar it transforms the returns to world with a candidate mount and reports the share of
returns that land within 0.15 m of known geometry: arena_map.json occupied cells (walls, racks,
forklifts, extras) plus the MISSION boxes. The /tf mount should score high and a yaw-flipped
mount low. For the 3D lidar it also reports azimuth coverage per message and floor heights.

    python3 analysis/c_allsensor_probe_check.py [snapA snapB]
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
D = ARENA / "results" / "analysis" / "c_allsensor_probe"
M = json.load(open(ARENA / "arena_map.json"))
G = np.array(M["grid"], dtype=np.int32)          # [iy][ix]
X0, Y0, ST = M["x0"], M["y0"], M["step"]
base = (ARENA / "base_carter_run_a2.py").read_text()
BOXES = ast.literal_eval(re.search(r"boxes=(\[.*?\])\)", base, re.S).group(1))
BODY = (-0.70, 0.25, -0.34, 0.34)


def near_known(wx, wy, wz=None, r=0.15):
    """Bool array: within r of an occupied grid cell or a MISSION box (below its top when wz given)."""
    wx, wy = np.asarray(wx), np.asarray(wy)
    hit = np.zeros(wx.shape, bool)
    ix0 = np.floor((wx - r - X0) / ST).astype(int)
    ix1 = np.floor((wx + r - X0) / ST).astype(int)
    iy0 = np.floor((wy - r - Y0) / ST).astype(int)
    iy1 = np.floor((wy + r - Y0) / ST).astype(int)
    for dx in range(0, 3):
        for dy in range(0, 3):
            ix = np.clip(ix0 + dx, 0, G.shape[1] - 1)
            iy = np.clip(iy0 + dy, 0, G.shape[0] - 1)
            ok = (ix0 + dx <= ix1) & (iy0 + dy <= iy1)
            # distance from the point to that cell's rectangle
            cx0, cy0 = X0 + ix * ST, Y0 + iy * ST
            ddx = np.maximum(np.maximum(cx0 - wx, 0), wx - (cx0 + ST))
            ddy = np.maximum(np.maximum(cy0 - wy, 0), wy - (cy0 + ST))
            hit |= ok & (G[iy, ix] != 0) & (np.hypot(ddx, ddy) <= r)
    for b in BOXES:
        x0, x1 = b["x"] - b["width"] / 2, b["x"] + b["width"] / 2
        y0, y1 = b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2
        ddx = np.maximum(np.maximum(x0 - wx, 0), wx - x1)
        ddy = np.maximum(np.maximum(y0 - wy, 0), wy - y1)
        inb = np.hypot(ddx, ddy) <= r
        if wz is not None:
            inb &= np.asarray(wz) <= b["height"] + 0.1
        hit |= inb
    return hit


def scan_points(name, key, mount_xy_yaw):
    meta = json.load(open(D / f"snap{name}.json"))
    s = meta["topics"][key]
    rng = np.load(D / f"snap{name}_{key}.npy")
    a = s["angle_min"] + np.arange(rng.size) * s["angle_increment"]
    ok = np.isfinite(rng) & (rng >= s["range_min"]) & (rng <= s["range_max"])
    mx, my, myaw = mount_xy_yaw
    bx = mx + rng[ok] * np.cos(myaw + a[ok])
    by = my + rng[ok] * np.sin(myaw + a[ok])
    body = (bx >= BODY[0]) & (bx <= BODY[1]) & (by >= BODY[2]) & (by <= BODY[3])
    return meta["pose"], bx[~body], by[~body], int(ok.sum()), int(body.sum())


def to_world(pose, bx, by):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * bx - s * by, y + s * bx + c * by


def main(names):
    out = {}
    for name in names:
        meta = json.load(open(D / f"snap{name}.json"))
        tf = meta["tf"]
        res = {"pose": meta["pose"]}
        # 2D lidars: /tf mount and the yaw-flipped alternative
        for key, frame in (("front_2d", "front_2d_lidar"), ("back_2d", "rear_2d_lidar")):
            tx, ty = tf[frame][1], tf[frame][2]
            qz, qw = tf[frame][6], tf[frame][7]
            yaw_tf = 2 * math.atan2(qz, qw)
            for label, yaw in (("tf", yaw_tf), ("tf+pi", yaw_tf + math.pi)):
                pose, bx, by, n_ok, n_body = scan_points(name, key, (tx, ty, yaw))
                wx, wy = to_world(pose, bx, by)
                share = float(near_known(wx, wy).mean()) if wx.size else None
                res[f"{key}:{label}"] = dict(mount=(round(tx, 4), round(ty, 4), round(yaw, 4)), finite=n_ok,
                                            body_removed=n_body, kept=int(wx.size),
                                            near_known=None if share is None else round(share, 3),
                                            min_kept_range=round(float(np.min(np.hypot(bx - tx, by - ty))), 3) if bx.size else None)
        # 3D lidar: identity rotation at the /tf translation (rotation checked: tf quaternion)
        p = np.load(D / f"snap{name}_lidar3d.npy")
        fr = tf["front_3d_lidar"]
        q = fr[4:8]
        bx, by, bz = p[:, 0] + fr[1], p[:, 1] + fr[2], p[:, 2] + fr[3]
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        hist, _ = np.histogram(az, bins=12, range=(-180, 180))
        body = (bx >= BODY[0]) & (bx <= BODY[1]) & (by >= BODY[2]) & (by <= BODY[3])
        wx, wy = to_world(meta["pose"], bx, by)
        floor = np.abs(bz) < 0.05
        above = (bz > 0.10) & ~body
        res["lidar3d"] = dict(points=int(p.shape[0]), tf_quat=[round(v, 4) for v in q],
                              azimuth_hist_30deg=hist.tolist(),
                              range_m=[round(float(np.min(np.linalg.norm(p, axis=1))), 3),
                                       round(float(np.max(np.linalg.norm(p, axis=1))), 3)],
                              z_floor_frame_minmax=[round(float(bz.min()), 3), round(float(bz.max()), 3)],
                              floor_share=round(float(floor.mean()), 3),
                              floor_z_median=round(float(np.median(bz[floor])), 4) if floor.any() else None,
                              body_points=int(body.sum()),
                              above_near_known=round(float(near_known(wx[above], wy[above], bz[above]).mean()), 3)
                              if above.any() else None, above_points=int(above.sum()))
        out[name] = res
    print(json.dumps(out, indent=1))
    (D / "check_raw.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B"])
