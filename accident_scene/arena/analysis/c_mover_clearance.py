"""Minimum gap between the robot and each moving obstacle along a recorded run (project C).

Replays arena_movers.position(m, t) on the trajectory CSV clock (t, x, y, yaw; the mission
clock, the same t the movers use) and measures the 2D gap between two robot outlines and
each 0.6 m mover square:

  chassis  forward -0.59..0.13 m, left +-0.18 m  (only chassis_link carries the contact report)
  outline  forward -0.59..0.14 m, left +-0.25 m  (chassis plus drive wheels)

Outlines from the PhysX ray measurement in arena-isaac-measured-facts. Gap 0 means overlap.
--phase-sweep N re-runs each mover with N start phases spread over its period and reports the
fraction whose chassis gap reaches 0. That is exact only for a trajectory that does not react to
the movers (a controller that reads no sensor), since the robot path is then the same at every phase.
Usage: python3 analysis/c_mover_clearance.py <trajectory.csv> [...] [--scenario v1] [--phase-sweep 200]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import arena_movers  # noqa: E402

OUTLINES = {"chassis": (-0.59, 0.13, -0.18, 0.18), "outline": (-0.59, 0.14, -0.25, 0.25)}


def robot_poly(x, y, yaw, box):
    fx0, fx1, ly0, ly1 = box
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py)
            for px, py in ((fx0, ly0), (fx1, ly0), (fx1, ly1), (fx0, ly1))]


def square(cx, cy, w, d):
    return [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2), (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]


def _seg_pt(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def _inside(p, poly):
    # convex polygon, counter-clockwise or clockwise: all cross products share a sign
    sgn = 0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr != 0:
            if sgn == 0:
                sgn = 1 if cr > 0 else -1
            elif (cr > 0) != (sgn > 0):
                return False
    return True


def _seg_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def gap(P, Q):
    """Distance between two convex polygons; 0 when they overlap."""
    if any(_inside(p, Q) for p in P) or any(_inside(q, P) for q in Q):
        return 0.0
    for i in range(len(P)):
        for j in range(len(Q)):
            if _seg_cross(P[i], P[(i + 1) % len(P)], Q[j], Q[(j + 1) % len(Q)]):
                return 0.0
    best = math.inf
    for poly_a, poly_b in ((P, Q), (Q, P)):
        for p in poly_a:
            for i in range(len(poly_b)):
                best = min(best, _seg_pt(p, poly_b[i], poly_b[(i + 1) % len(poly_b)]))
    return best


def clearance(csv_path, scenario="v1"):
    movers = arena_movers.scenario(scenario)
    rows = [(float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
            for r in csv.DictReader(open(csv_path))]
    out = {}
    for m in movers:
        rec = {}
        for name, box in OUTLINES.items():
            best = (math.inf, None, None, None)
            for t, x, y, yaw in rows:
                mx, my = arena_movers.position(m, t)
                if math.hypot(mx - x, my - y) > 3.0:
                    continue
                g = gap(robot_poly(x, y, yaw, box), square(mx, my, m["size"][0], m["size"][1]))
                if g < best[0]:
                    best = (g, t, (round(x, 3), round(y, 3), round(yaw, 3)), (round(mx, 3), round(my, 3)))
            rec[name] = dict(min_gap_m=None if best[1] is None else round(best[0], 3), t=best[1],
                             robot=best[2], mover=best[3])
        out[m["name"]] = rec
    return dict(trajectory=str(csv_path), scenario=scenario, samples=len(rows),
                t_end=rows[-1][0] if rows else None, movers=out)


def phase_sweep(csv_path, scenario="v1", n=200):
    """Per mover: fraction of start phases (uniform over one period) whose chassis gap reaches 0."""
    movers = arena_movers.scenario(scenario)
    rows = [(float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
            for r in csv.DictReader(open(csv_path))]
    polys = [(t, x, y, robot_poly(x, y, yaw, OUTLINES["chassis"])) for t, x, y, yaw in rows]
    out = {}
    for m in movers:
        (x0, y0), (x1, y1) = m["p0"], m["p1"]
        period = 2.0 * math.hypot(x1 - x0, y1 - y0) / m["speed"]
        # the robot must come within reach of the segment at all for any phase to matter
        seg = [(x0, y0), (x1, y1)]
        near = [(t, P) for t, x, y, P in polys if _seg_pt((x, y), *seg) < 1.2]
        hits = 0
        for k in range(n):
            mm = dict(m, phase=k * period / n)
            for t, P in near:
                mx, my = arena_movers.position(mm, t)
                if gap(P, square(mx, my, m["size"][0], m["size"][1])) == 0.0:
                    hits += 1
                    break
        out[m["name"]] = dict(period_s=round(period, 2), near_samples=len(near), phases=n,
                              hit_fraction=round(hits / n, 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--scenario", default="v1")
    ap.add_argument("--json", help="write all results to this file")
    ap.add_argument("--phase-sweep", type=int, default=0, metavar="N")
    a = ap.parse_args()
    res = [clearance(p, a.scenario) for p in a.csv]
    if a.phase_sweep:
        for r in res:
            r["phase_sweep"] = phase_sweep(r["trajectory"], a.scenario, a.phase_sweep)
    for r in res:
        print(f"== {Path(r['trajectory']).name}  samples {r['samples']}  t_end {r['t_end']}")
        for name, rec in r["movers"].items():
            c, o = rec["chassis"], rec["outline"]
            if c["t"] is None:
                print(f"   {name:14s} never within 3 m")
            else:
                print(f"   {name:14s} chassis gap {c['min_gap_m']:.3f} m at t={c['t']:.2f} robot {c['robot']} mover {c['mover']}"
                      f" | outline gap {o['min_gap_m']:.3f} m at t={o['t']:.2f}")
        for name, ps in r.get("phase_sweep", {}).items():
            print(f"   sweep {name:14s} period {ps['period_s']:6.2f} s  samples near lane {ps['near_samples']:5d}"
                  f"  collide in {ps['hit_fraction']*100:5.1f}% of {ps['phases']} phases")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
