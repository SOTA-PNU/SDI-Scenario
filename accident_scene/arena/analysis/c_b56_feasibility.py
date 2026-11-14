"""B-5-6 pre-check (실험 31): is a computed route feasible on the exact map that base_carter_run_ros_map.py gives?
Not shown to the model. Reads only; prints:
- the A-3 verified route (실험 12, 1.1 m on the old 0.25 m PhysX sweep map) re-measured with segment_clearance,
- the widest START->GOAL corridor on a 0.05 m grid (largest r such that cells with map_clearance >= r connect them),
- the cost of one map_clearance call (how slow a naive per-cell planner would be)."""
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
MAP = (ARENA / "base_carter_run_ros_map.py").read_text()
ns = {"np": np, "math": math}
exec(MAP[MAP.index("MISSION = dict("):MAP.index("\nSTART = ")], ns)
exec(MAP[MAP.index("# ---- 지도 기본 코드 ----"):MAP.index("# [EDIT REGION] mission controller")], ns)
START, GOAL = ns["MISSION"]["start_xy_yaw"], ns["MISSION"]["goal_xy_yaw"]
sc, mc = ns["segment_clearance"], ns["map_clearance"]

a3 = [tuple(START[:2]), (-4.7, -7.3), (-3.7, 5.1), (0.7, 11.3), (2.3, 15.1), tuple(GOAL[:2])]
print("A-3 route on the exact map:", [round(sc(p, q), 2) for p, q in zip(a3, a3[1:])])

t0 = time.perf_counter(); n = 2000
for i in range(n):
    mc(-5.0 + i * 1e-3, 0.0)
print(f"map_clearance: {1e6 * (time.perf_counter() - t0) / n:.1f} us per call")

wx0, wx1, wy0, wy1 = ns["WALLS"]
h = 0.05
xs, ys = np.arange(wx0, wx1 + 1e-9, h), np.arange(wy0, wy1 + 1e-9, h)
X, Y = np.meshgrid(xs, ys, indexing="ij")
R = ns["map_rects"]()
dx = np.maximum(np.maximum(R[:, 0][None, None] - X[..., None], 0), X[..., None] - R[:, 1][None, None])
dy = np.maximum(np.maximum(R[:, 2][None, None] - Y[..., None], 0), Y[..., None] - R[:, 3][None, None])
C = np.minimum(np.sqrt(dx ** 2 + dy ** 2).min(axis=2), np.minimum.reduce([X - wx0, wx1 - X, Y - wy0, wy1 - Y]))
s = (int(round((START[0] - wx0) / h)), int(round((START[1] - wy0) / h)))
g = (int(round((GOAL[0] - wx0) / h)), int(round((GOAL[1] - wy0) / h)))
print("clearance at START, GOAL:", round(float(C[s]), 2), round(float(C[g]), 2), "grid", C.shape)


def connected(r):
    ok = C >= r
    if not (ok[s] and ok[g]):
        return False
    seen = np.zeros_like(ok); seen[s] = True; q = deque([s])
    while q:
        i, j = q.popleft()
        if (i, j) == g:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < ok.shape[0] and 0 <= b < ok.shape[1] and ok[a, b] and not seen[a, b]:
                seen[a, b] = True; q.append((a, b))
    return False


lo, hi = 0.0, float(min(C[s], C[g]))
for _ in range(20):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if connected(mid) else (lo, mid)
print(f"widest START->GOAL corridor: clearance {lo:.3f} m (pose point to the nearest mapped object or wall)")
for r in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
    print(f"  cells with clearance >= {r}: connected {connected(r)}")
