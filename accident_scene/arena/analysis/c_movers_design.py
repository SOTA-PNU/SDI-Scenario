"""Design check for the project C moving obstacles (arena_movers.py), no Isaac.

(a) clearance: each mover's swept rectangle (segment +- half its size) against every collider
    rectangle of the measured scene listing (a21_scene, PhysX occupancy at z 0.03-0.60 m) and
    the wall inner faces, with MARGIN;
(b) the A-4 seed route (graded trajectory of npu_gpt_oss_120b_pA4_a0): when the robot centre
    passes each mover segment, and the phase that puts the mover on that crossing point at that
    time (so a controller that does not look will meet it);
(c) for a robot crossing a segment at a uniformly random time: the share of the mover's
    period during which the crossing point is blocked.

    python3 analysis/c_movers_design.py [scenario]
"""
import csv
import json
import math
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene  # noqa: E402
import arena_movers as AM  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "v1"
MARGIN = 0.15
ROBOT_R = 0.62          # turn-in-place sweep radius of the chassis outline (REPORT, sim2d.py)
L = a21_scene.obstacle_listing(a21_scene.Scene())
rows = L["obstacles"]
W = L["walls_inner_face"]
traj = [(float(r["t"]), float(r["x"]), float(r["y"])) for r in
        csv.DictReader(open(ARENA / "results" / "npu_gpt_oss_120b_pA4_a0_trajectory.csv"))]


def seg_dist(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    u = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + u * vx), py - (ay + u * vy)), u


out = []
for m in AM.scenario(NAME):
    (x0, y0), (x1, y1) = m["p0"], m["p1"]
    sx, sy, _ = m["size"]
    rect = (min(x0, x1) - sx / 2, max(x0, x1) + sx / 2, min(y0, y1) - sy / 2, max(y0, y1) + sy / 2)
    hits = [r[0] for r in rows if r[2] < rect[1] + MARGIN and r[3] > rect[0] - MARGIN
            and r[4] < rect[3] + MARGIN and r[5] > rect[2] - MARGIN]
    walls = []
    if rect[0] - MARGIN < W["x"][0] or rect[1] + MARGIN > W["x"][1]:
        walls.append("x wall")
    if rect[2] - MARGIN < W["y"][0] or rect[3] + MARGIN > W["y"][1]:
        walls.append("y wall")
    Lseg = math.hypot(x1 - x0, y1 - y0)
    period = 2 * Lseg / m["speed"]
    # (b) seed crossing: closest approach of the robot centre to the segment
    best = min(((seg_dist(x, y, m["p0"], m["p1"]), t) for t, x, y in traj), key=lambda v: v[0][0])
    (dmin, u), tc = best
    dc = u * Lseg
    phase_fwd = (dc / m["speed"] - tc) % period          # mover going p0 -> p1 at tc
    # (c) blocked share: the point is blocked while the mover centre is within (half size + robot) of it
    block = 2 * ((max(sx, sy) / 2 + ROBOT_R) / m["speed"]) * (2 if 0 < u < 1 else 1)
    out.append(dict(name=m["name"], rect=[round(v, 2) for v in rect], overlaps=hits, walls=walls,
                    seg_len=round(Lseg, 2), period_s=round(period, 1),
                    seed_closest_m=round(dmin, 2), seed_t=round(tc, 2), seed_u=round(u, 2),
                    phase_to_meet_seed=round(phase_fwd, 2), blocked_share=round(min(1.0, block / period), 2)))
    print(json.dumps(out[-1], ensure_ascii=False))
json.dump(out, open(ARENA / "results" / "analysis" / f"c_movers_design_{NAME}.json", "w"), indent=1)
