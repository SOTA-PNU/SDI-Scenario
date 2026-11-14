"""Isaac check of the B-5-12 follow method (forward min(0.3, 0.5*dist + 0.1), turning and alignment clamp(1.5*e, +-0.4),
switch at 0.35 m, rotate in place when |e| > 0.35, align at 0.25 m, done when |e| < 0.05) on the fixed map (no movers),
on base_carter_run_ros_map_dbg_heapq_lidartime.py. Two paths: 'sparse' = the 7 points of Claude's A* path (실험 36),
'dense' = a 0.5 m grid A* that keeps grid points as waypoints (node and edge clearance >= 1.0 m), the shape the models
write. Waits for $WAIT_PID (the video recorder) first: no parallel Isaac runs. Logs to results/analysis/b512_follower_check/.

    WAIT_PID=<pid> python3 analysis/b512_follower_check.py
"""
import os
import shutil
import sys
import time
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/b512_follower_check"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT
base = (ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py").read_text()
B, E = "# [EDIT REGION] mission controller - modify ONLY this block\n", "# ========================= [END EDIT REGION]"
REGION = '''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# B-5-12 follow-method check (experimenter run, not a model answer)
SPARSE = [(-9.0, -10.8), (-4.65, -7.1), (-3.55, 4.0), (-2.15, 6.3), (1.45, 11.5), (2.05, 14.5), (8.0, 16.5)]
_S = {"path": None, "i": 1, "align": False, "last": -9.0}


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _dense():
    res, need = 0.5, 1.0
    x0, x1, y0, y1 = WALLS
    nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
    pt = lambda i, j: (x0 + (i + 0.5) * res, y0 + (j + 0.5) * res)
    near = lambda p: (int((p[0] - x0) / res), int((p[1] - y0) / res))
    s, g = near(START[:2]), near(GOAL[:2])
    ok = {}
    def free(c):
        if c not in ok:
            ok[c] = 0 <= c[0] < nx and 0 <= c[1] < ny and map_clearance(*pt(*c)) >= need
        return ok[c]
    openh, came, cost = [(0.0, s)], {s: None}, {s: 0.0}
    while openh:
        _, c = heapq.heappop(openh)
        if c == g:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (c[0] + di, c[1] + dj)
                if n == c or not free(n) or segment_clearance(pt(*c), pt(*n)) < need:
                    continue
                nc = cost[c] + math.hypot(di, dj) * res
                if nc < cost.get(n, 1e9):
                    cost[n], came[n] = nc, c
                    heapq.heappush(openh, (nc + math.hypot(n[0] - g[0], n[1] - g[1]) * res, n))
    path, c = [], g
    while c is not None:
        path.append(pt(*c)); c = came[c]
    return path[::-1] + [tuple(GOAL[:2])]


def controller(t, pose, env):
    x, y, yaw = pose
    if _S["path"] is None:
        _S["path"] = SPARSE if KIND == "sparse" else _dense()
        print("DBG plan", KIND, len(_S["path"]), "points")
    p = _S["path"]
    if not _S["align"]:
        tx, ty = p[_S["i"]]
        dist, e = math.hypot(tx - x, ty - y), _wrap(math.atan2(ty - y, tx - x) - yaw)
        if _S["i"] < len(p) - 1 and dist < 0.35:
            _S["i"] += 1
            tx, ty = p[_S["i"]]
            dist, e = math.hypot(tx - x, ty - y), _wrap(math.atan2(ty - y, tx - x) - yaw)
        if _S["i"] == len(p) - 1 and dist < 0.25:
            _S["align"] = True
    if _S["align"]:
        e = _wrap(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.4, 0.4), False
    if abs(e) > 0.35:
        v, w = 0.0, _clamp(1.5 * e, -0.4, 0.4)
    else:
        v, w = min(0.3, 0.5 * dist + 0.1), _clamp(1.5 * e, -0.4, 0.4)
    if t - _S["last"] >= 2.0:
        _S["last"] = t
        print("DBG t=%.1f wp=%d/%d pose=(%.2f,%.2f,%.2f) v=%.2f w=%.2f" % (t, _S["i"], len(p) - 1, x, y, yaw, v, w))
    return v, w, False


'''
wait = os.environ.get("WAIT_PID")
if wait:
    while Path(f"/proc/{wait}").exists():
        time.sleep(20)
os.environ.pop("ARENA_MOVERS", None)
for kind in ("sparse", "dense"):
    variant = f"b512_follower_check_{kind}"
    path = ARENA / f"{variant}_carter_run.py"
    assert not path.exists()
    path.write_text(base[:base.index(B)] + REGION.replace("_S = {", f'KIND = "{kind}"\n_S = {{', 1) + base[base.index(E):])
    try:
        r = TL.grade(variant)
    finally:
        shutil.move(str(path), OUT / path.name)
    print(kind, "->", {k: r.get(k) for k in ("verdict",)} if isinstance(r, dict) else r, flush=True)
