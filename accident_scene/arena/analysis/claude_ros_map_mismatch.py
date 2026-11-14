"""Where does the scene differ from the static map the prompts give? Uses the static-scene probe (claude_ros_probe01, no
moving obstacles) with each scan transformed at its measured age (front 0.133 s, back 0.100 s). Reports (1) the y of lidar
returns along the north wall, (2) clusters of returns farther than 0.25 m from every mapped rect and wall, with extents
and hit counts. Read-only.

    python3 analysis/claude_ros_map_mismatch.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA / "analysis"))
import claude_ros_probe_analysis as PA  # noqa: E402  (loads the capture, map, pose_at, world, clear)

AGE = {4: 0.133, 5: 0.100}
W = []
for s in PA.cap["scans"]:
    for col in (4, 5):
        if s[col] is None:
            continue
        P = s[col].astype(np.float64)
        r = np.hypot(P[:, 0], P[:, 1])
        P = P[r < 8.0]
        wx, wy = PA.world(P, PA.pose_at(s[0] - AGE[col]))
        W.append(np.stack([wx, wy], axis=1))
W = np.concatenate(W)
print("returns within 8 m:", len(W))
nw = W[(W[:, 1] > 16.8) & (W[:, 0] > -9.5) & (W[:, 0] < 8.5)]
if len(nw):
    h, e = np.histogram(nw[:, 1], bins=np.arange(16.8, 18.21, 0.05))
    top = np.argsort(-h)[:5]
    print("north wall returns: y histogram peaks", [(round(float(e[i]), 2), int(h[i])) for i in sorted(top)])
    print("north wall returns: y percentiles 5/50/95 of y > 17.4:", np.percentile(nw[nw[:, 1] > 17.4][:, 1], [5, 50, 95]).round(3).tolist())
F = W[PA.clear(W[:, 0], W[:, 1]) > 0.25]
print("returns > 0.25 m from the map (any wall/rect):", len(F))
if len(F):
    key = np.floor(F / 0.25).astype(int)
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt >= 30
    cells = {tuple(k) for k in u[keep]}
    # connected components of kept cells
    seen, comps = set(), []
    for c in cells:
        if c in seen:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            a = stack.pop(); comp.append(a)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in cells and b not in seen:
                        seen.add(b); stack.append(b)
        comps.append(comp)
    for comp in sorted(comps, key=lambda c: -sum(int(cnt[np.all(u == np.array(k), axis=1)][0]) for k in c)):
        pts = F[np.isin(inv, [int(np.nonzero(np.all(u == np.array(k), axis=1))[0][0]) for k in comp])]
        hits = len(pts)
        x0, x1 = np.percentile(pts[:, 0], [2, 98]); y0, y1 = np.percentile(pts[:, 1], [2, 98])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        print(f"  cluster x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}, hits {hits}, map clearance at centre {float(PA.clear(np.array([cx]), np.array([cy]))[0]):.2f}")
