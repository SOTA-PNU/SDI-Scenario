"""Waypoint route for prompt A-3 (REPORT.md 실험 12): START -> GOAL with clearance.

Occupancy = arena_map.json (static structures: walls, pillars, west shelf, both
forklifts; PhysX overlap sweep z 0.05-0.55 m, 0.25 m cells) + the 14 MISSION boxes
of base_carter_run_a2.py, rasterised at 0.05 m. A* over the clearance field keeps the
robot centre at least --min-clear from every obstacle cell and prefers more; the cell
path is then cut to straight segments whose every point keeps --min-clear.

    python3 analysis/a3_route.py [--min-clear 0.8] [--prefer 1.3]
"""
import argparse
import heapq
import json
import math
import re
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
RES = 0.05


def mission():
    src = (ARENA / "base_carter_run_a2.py").read_text()
    boxes = [json.loads(b) for b in re.findall(r"(\{\"x\":[^}]*\})", src)]
    return boxes


def build():
    m = json.loads((ARENA / "arena_map.json").read_text())
    x0, y0, step = m["x0"], m["y0"], m["step"]
    occ = []                                   # obstacle rectangles (x0, y0, x1, y1)
    for j, row in enumerate(m["grid"]):
        for i, v in enumerate(row):
            if v:
                occ.append((x0 + i * step, y0 + j * step, x0 + (i + 1) * step, y0 + (j + 1) * step))
    for b in mission():
        occ.append((b["x"] - b["width"] / 2, b["y"] - b["depth"] / 2,
                    b["x"] + b["width"] / 2, b["y"] + b["depth"] / 2))
    return occ


def clearance_fn(occ):
    def clear(x, y):
        best = 1e9
        for (a, b, c, d) in occ:
            dx = max(a - x, 0.0, x - c)
            dy = max(b - y, 0.0, y - d)
            dd = dx * dx + dy * dy
            if dd < best:
                best = dd
        return math.sqrt(best)
    return clear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-clear", type=float, default=0.8)
    ap.add_argument("--prefer", type=float, default=1.3)
    ap.add_argument("--grid", type=float, default=0.2, help="A* cell [m]")
    a = ap.parse_args()
    occ = build()
    clear = clearance_fn(occ)
    sx, sy, gx, gy = -9.0, -10.8, 8.0, 16.5
    X0, X1, Y0, Y1 = -10.5, 9.6, -12.5, 18.2
    G = a.grid
    nx, ny = int((X1 - X0) / G) + 1, int((Y1 - Y0) / G) + 1
    cache = {}

    def c_at(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = clear(X0 + i * G, Y0 + j * G)
        return cache[(i, j)]

    def node(x, y):
        return int(round((x - X0) / G)), int(round((y - Y0) / G))

    s, g = node(sx, sy), node(gx, gy)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        dcur, u = heapq.heappop(pq)
        if u == g:
            break
        if dcur > dist[u] + 1e-9:
            continue
        for di, dj in moves:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny):
                continue
            cv = c_at(*v)
            if cv < a.min_clear and v != g:
                continue
            step = G * math.hypot(di, dj)
            pen = 0.0 if cv >= a.prefer else 4.0 * (a.prefer - cv) / (a.prefer - a.min_clear)
            nd = dcur + step * (1.0 + pen) + 0.0 * math.hypot((v[0] - g[0]) * G, (v[1] - g[1]) * G)
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if g not in prev:
        raise SystemExit("no path")
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells.reverse()
    pts = [(X0 + i * G, Y0 + j * G) for i, j in cells]
    pts[0], pts[-1] = (sx, sy), (gx, gy)

    def seg_ok(p, q):
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05))
        return all(clear(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) >= a.min_clear
                   for k in range(1, n))

    wps = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not seg_ok(pts[i], pts[j]):
            j -= 1
        wps.append(pts[j])
        i = j
    wps = [(round(x, 2), round(y, 2)) for x, y in wps]
    total = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(wps, wps[1:]))
    mins = []
    for p, q in zip(wps, wps[1:]):
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05))
        mins.append(min(clear(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(0, n + 1)))
    print(json.dumps({"min_clear": a.min_clear, "prefer": a.prefer, "length_m": round(total, 2),
                      "waypoints": wps, "segment_min_clearance": [round(c, 2) for c in mins],
                      "start_clearance": round(clear(sx, sy), 2), "goal_clearance": round(clear(gx, gy), 2)},
                     indent=1))


if __name__ == "__main__":
    main()
