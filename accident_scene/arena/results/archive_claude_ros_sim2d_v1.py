"""Fast 2D stand-in for the arena run, to iterate on claude_ros_carter_run.py before Isaac (Claude's own controller).

Runs the EDIT REGION of a run file against: unicycle robot with a first-order cmd_vel lag, the arena_movers.py boxes
(scenario v1 or v1 with other phases), and two synthetic 2D lidars (front: robot-frame bearings within +-140 deg, 10 Hz;
back: outside +-40 deg, 60 Hz; points on the mover squares and on the mapped rects and walls within 8 m, no occlusion;
points inside the base's _BODY box dropped like the base does). Contact = a point of the robot outline
(x -0.59..0.14, y +-0.25) inside a mover square or a mapped rect, or a mover corner inside the robot outline.
Mover positions are used only to draw the synthetic scans and to judge contact; the controller never sees them.

    python3 analysis/claude_ros_sim2d.py [run_file] [--phases v1|random:N[:seed]] [--quiet]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import arena_movers as AM  # noqa: E402

BEGIN, END = "# [EDIT REGION] mission controller - modify ONLY this block\n", "# ========================= [END EDIT REGION]"
OUTLINE = (-0.59, 0.14, -0.25, 0.25)
BODY = (-0.70, 0.25, -0.34, 0.34)


def load_region(run_file):
    src = Path(run_file).read_text()
    head = src[:src.index(BEGIN)]
    ns = {"math": math, "np": np}
    exec(head[head.index("MISSION = dict("):head.index("\n# ---- Carter")], ns)
    return src[src.index(BEGIN):src.index(END)], ns


def rect_perimeter(r, step=0.08):
    x0, x1, y0, y1 = r
    pts = []
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        k = np.linspace(0, 1, n, endpoint=False)
        pts.append(np.stack([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k], axis=1))
    return np.concatenate(pts)


def run(run_file, movers, quiet=True, t_max=600.0, dt=1.0 / 60, tau=0.15):
    region, ns = load_region(run_file)
    mission = ns["MISSION"]
    boxes = [(b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2)
             for b in mission["boxes"]]
    fixed_ns = {"np": np, "math": math}
    msrc = (ARENA / "base_carter_run_ros_map.py").read_text()
    i = msrc.index("FIXED_OBSTACLES = [")
    exec(msrc[i:msrc.index("\n]\n", i) + 3], fixed_ns)
    rects = np.array(boxes + [r[1:] for r in fixed_ns["FIXED_OBSTACLES"]], dtype=float)
    walls = (-10.30, 9.35, -12.15, 17.95)
    static_pts = np.concatenate([rect_perimeter(r) for r in rects] +
                                [rect_perimeter((walls[0] - 0.3, walls[0], walls[2], walls[3])),
                                 rect_perimeter((walls[1], walls[1] + 0.3, walls[2], walls[3])),
                                 rect_perimeter((walls[0], walls[1], walls[2] - 0.3, walls[2])),
                                 rect_perimeter((walls[0], walls[1], walls[3], walls[3] + 0.3))])
    state = {"pose": [*mission["start_xy_yaw"]], "t": 0.0, "front": np.zeros((0, 2)), "back": np.zeros((0, 2)), "k": 0}
    dbg = []

    def mover_squares(t):
        out = []
        for m in movers:
            x, y = AM.position(m, t)
            hx, hy = m["size"][0] / 2, m["size"][1] / 2
            out.append((x - hx, x + hx, y - hy, y + hy))
        return out

    def scan(which):
        x, y, yaw = state["pose"]
        pts = [static_pts] + [rect_perimeter(r, 0.05) for r in mover_squares(state["t"])]
        P = np.concatenate(pts)
        P = P[np.hypot(P[:, 0] - x, P[:, 1] - y) < 8.5]
        c, s = math.cos(yaw), math.sin(yaw)
        rx, ry = c * (P[:, 0] - x) + s * (P[:, 1] - y), -s * (P[:, 0] - x) + c * (P[:, 1] - y)
        R = np.stack([rx, ry], axis=1)
        if which == "front":
            ang = np.degrees(np.arctan2(ry, rx - 0.026))
            R = R[np.abs(ang) <= 140]
        else:
            ang = np.degrees(np.arctan2(ry, rx + 0.489))
            R = R[np.abs(ang) >= 40]
        body = (R[:, 0] >= BODY[0]) & (R[:, 0] <= BODY[1]) & (R[:, 1] >= BODY[2]) & (R[:, 1] <= BODY[3])
        return R[~body]

    def lidar2d_points(which="front"):
        return state[which]

    def to_world(points, pose):
        p = np.array(points, dtype=np.float64)
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * p[:, 0] - s * p[:, 1]
        wy = y + s * p[:, 0] + c * p[:, 1]
        p[:, 0], p[:, 1] = wx, wy
        return p

    def _print(*a, **k):
        line = " ".join(str(v) for v in a)
        dbg.append(line)
        if not quiet:
            print(line)

    ns.update(lidar2d_points=lidar2d_points, to_world=to_world, print=_print,
              START=mission["start_xy_yaw"], GOAL=mission["goal_xy_yaw"])
    exec(region, ns)
    ox = np.concatenate([np.linspace(OUTLINE[0], OUTLINE[1], 15), np.full(11, OUTLINE[1]),
                         np.linspace(OUTLINE[1], OUTLINE[0], 15), np.full(11, OUTLINE[0])])
    oy = np.concatenate([np.full(15, OUTLINE[2]), np.linspace(OUTLINE[2], OUTLINE[3], 11),
                         np.full(15, OUTLINE[3]), np.linspace(OUTLINE[3], OUTLINE[2], 11)])
    va = wa = 0.0
    min_gap = {m["name"]: 9e9 for m in movers}
    result = {"verdict": "timeout", "t": None, "hit": None}
    while state["t"] < t_max:
        t = state["t"]
        if state["k"] % 6 == 0:
            state["front"] = scan("front")
        state["back"] = scan("back")
        v, w, done = ns["controller"](t, tuple(state["pose"]), None)
        if done:
            x, y, yaw = state["pose"]
            gx, gy, gyaw = mission["goal_xy_yaw"]
            ok = math.hypot(x - gx, y - gy) <= 0.75 and abs(math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))) <= 0.26
            result = {"verdict": "pass" if ok else "fail_goal", "t": t, "hit": None}
            break
        va += (v - va) * min(1.0, dt / tau)
        wa += (w - wa) * min(1.0, dt / tau)
        x, y, yaw = state["pose"]
        yaw += wa * dt
        x += va * math.cos(yaw) * dt
        y += va * math.sin(yaw) * dt
        state["pose"] = [x, y, yaw]
        state["t"] = t + dt
        state["k"] += 1
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = x + c * ox - s * oy, y + s * ox + c * oy
        inside = ((wx[:, None] >= rects[:, 0]) & (wx[:, None] <= rects[:, 1]) & (wy[:, None] >= rects[:, 2]) & (wy[:, None] <= rects[:, 3])).any()
        hit = "map" if inside else None
        for m, sq in zip(movers, mover_squares(state["t"])):
            mx, my = (sq[0] + sq[1]) / 2, (sq[2] + sq[3]) / 2
            corners = np.array([[sq[0], sq[2]], [sq[1], sq[2]], [sq[1], sq[3]], [sq[0], sq[3]]])
            rx = c * (corners[:, 0] - x) + s * (corners[:, 1] - y)
            ry = -s * (corners[:, 0] - x) + c * (corners[:, 1] - y)
            pin = ((wx >= sq[0]) & (wx <= sq[1]) & (wy >= sq[2]) & (wy <= sq[3])).any()
            cin = ((rx >= OUTLINE[0]) & (rx <= OUTLINE[1]) & (ry >= OUTLINE[2]) & (ry <= OUTLINE[3])).any()
            dx = np.maximum(np.maximum(sq[0] - wx, wx - sq[1]), 0)
            dy = np.maximum(np.maximum(sq[2] - wy, wy - sq[3]), 0)
            min_gap[m["name"]] = min(min_gap[m["name"]], float(np.hypot(dx, dy).min()))
            if pin or cin:
                hit = m["name"]
        if hit:
            result = {"verdict": "collision", "t": round(state["t"], 2), "hit": hit}
            break
    result["pose"] = [round(v, 2) for v in state["pose"]]
    result["min_gap"] = {k: round(v, 2) for k, v in min_gap.items()}
    result["dbg"] = dbg
    return result


def phase_sets(spec):
    base = AM.scenario("v1")
    if spec == "v1":
        return [base]
    _, n, *seed = spec.split(":")
    rng = np.random.default_rng(int(seed[0]) if seed else 0)
    out = []
    for _ in range(int(n)):
        ms = [dict(m) for m in base]
        for m in ms:
            L = math.hypot(m["p1"][0] - m["p0"][0], m["p1"][1] - m["p0"][1])
            m["phase"] = float(rng.uniform(0, 2 * L / m["speed"]))
        out.append(ms)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_file", nargs="?", default=str(ARENA / "claude_ros_carter_run.py"))
    ap.add_argument("--phases", default="v1")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    sets = phase_sets(a.phases)
    tally = {}
    for k, ms in enumerate(sets):
        r = run(a.run_file, ms, quiet=a.quiet or len(sets) > 1)
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        print(f"[{k}] {r['verdict']} t={r['t']} hit={r['hit']} pose={r['pose']} min_gap={r['min_gap']}"
              + ("" if len(sets) == 1 else f" phases={[round(m['phase'], 2) for m in ms]}"), flush=True)
    print("tally", tally)
