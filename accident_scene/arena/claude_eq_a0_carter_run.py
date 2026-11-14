"""Mission runner - ACCIDENT ARENA (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/arena/nova_carter_warehouse_arena.yaml): the whole warehouse is
the arena; the robot starts in one corner at START and must reach GOAL in the
opposite corner; static boxes stand in between.

PASS criteria are evaluated by the fixed harness (levels/common/carter_env.py):
  reached_goal      final distance <= position tolerance
                    (and |yaw error| <= yaw tolerance when it is set)
  no_collision      zero chassis contact events (when collision is scored)
  max_time_to_goal  first entry into the goal disc within the bound (when set)
  timeout_s         sim-time budget

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default - no control code, the robot never
moves.  Everything outside it is fixed plumbing.
"""

import math  # noqa: F401  (available to the controller)
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "levels" / "common"))
sys.path.insert(0, str(HERE))
from arena_harness import (  # noqa: E402
    ArenaHarness,
    apply_record_quality,
    follow_camera,
    recording,
    setup_topdown_camera,
)

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(HERE / "results")

# scenario givens (from accident_scene/arena/nova_carter_warehouse_arena.yaml - not part of the answer)
# boxes: x, y = centre; width along X, depth along Y; resting on the floor
MISSION = dict(start_xy_yaw=(-9.0, -10.8, 1.0138),
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=120.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.25},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.2},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.18},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.2},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
#
# Deterministic map -> plan -> track controller (only `math` is used).
#
#   MAP     0.1 m occupancy grid seeded with the walled interior (camera note
#           at the end of this file: ~20.0 x 30.6 m incl. walls, centred on
#           (-0.45, 2.85)) and the 14 MISSION boxes.  Four boxes are lower
#           than the default z=0.35 scan plane, so they are known only from
#           MISSION.  Walls, pillars, shelves and the forklift (body and low
#           forks) are learned online from ray fans at nine heights.
#   PLAN    Dijkstra cost-to-go field from GOAL on a 16-connected grid.  A
#           step costs more the closer it is to an obstacle (cells closer than
#           the chassis radius are close to forbidden) and west of x=-7, where
#           the scored west shelf stands.  Re-run when new obstacles touch the
#           current path.
#   TRACK   Steer at the farthest point of that path reachable in a straight
#           line without losing clearance (string pulling).  Speed is limited
#           by heading error, clearance, a swept corridor ahead of the nose
#           and the distance to GOAL.  Every command is checked on the map
#           over the next 0.6 s, so the chassis - tail swing included - never
#           closes in on an obstacle.
#   FINISH  Creep onto GOAL, turn in place to the GOAL yaw, hold until the
#           pose is still, done.

_RES = 0.1                                      # grid cell [m]
_GX0, _GY0, _NX, _NY = -12.2, -14.2, 236, 342   # x -12.2..11.4, y -14.2..20.0
_N = _NX * _NY
_BOUND = (-10.45, 9.55, -12.45, 18.15)   # walled interior incl. walls
_WALL = (-10.25, 9.35, -12.25, 17.95)    # assumed inner wall faces
_R_HARD = 0.45      # centre clearance [m] at which the chassis may touch
_R_LETHAL = 0.65    # preferred minimum centre clearance
_R_SOFT = 1.5       # clearance beyond which a step costs nothing extra
_C_FREE = 10        # cost of a 0.1 m step in the open
_WEST_X = -7.0      # west of this, steps cost extra (scored west shelf)
_INF = 1 << 40
_V_MAX, _W_MAX = 0.9, 1.2                # [m/s], [rad/s]
_ACC, _DEC, _WACC = 0.8, 2.0, 5.0        # [m/s^2], [m/s^2], [rad/s^2]
_NOSE = 0.42                             # centre -> front bumper [m]
# chassis margins: corridor lateral samples, turn-in-place half length and
# width, swept-motion half length and width, gap to keep [m].  The tight set
# (still a bit larger than the chassis) is a last resort once boxed in.
_MARGIN = ((-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4),
           0.45, 0.35, 0.42, 0.31, 0.12)
_MARGIN_TIGHT = ((-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3),
                 0.42, 0.30, 0.40, 0.28, 0.04)
_MAIN = (121, 0.35, 6.0)                 # main fan: beams, z, range
_EXTRA_Z = (0.025, 0.08, 0.14, 0.21, 0.28, 0.42, 0.52, 0.65)  # 2 per step
_EXTRA = (91, 4.0)                       # extra fans: beams, range
# fans are requested all the way round (fov_deg=360) so obstacles beside and
# behind the chassis are mapped too; 180 deg is used if 360 is refused


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _idx(x, y):
    ix = int(math.floor((x - _GX0) / _RES))
    iy = int(math.floor((y - _GY0) / _RES))
    if 0 <= ix < _NX and 0 <= iy < _NY:
        return iy * _NX + ix
    return -1


def _cxy(k):
    return _GX0 + (k % _NX + 0.5) * _RES, _GY0 + (k // _NX + 0.5) * _RES


def _cost(d):
    """Cost of a 0.1 m step into a cell whose clearance is d [m]."""
    if d < _R_HARD:
        return 2000
    if d < _R_LETHAL:
        return 120
    if d < _R_SOFT:
        q = (_R_SOFT - d) / (_R_SOFT - _R_LETHAL)
        return _C_FREE + int(40.0 * q * q + 0.5)
    return _C_FREE


def _dcost(c):
    return int(c * 1.41421356 + 0.5)


def _kcost(c):
    return int(c * 2.23606798 + 0.5)


def _knight_moves():
    # 16-connected grid: besides the 8 neighbours, the (1,2)-type moves.
    # Entry: (move offset, offsets of the two cells the move passes through),
    # all relative to the cell the move starts from.
    out = []
    for a, b in ((1, 2), (-1, 2), (1, -2), (-1, -2),
                 (2, 1), (-2, 1), (2, -1), (-2, -1)):
        if abs(b) == 2:
            m1, m2 = (0, b // 2), (a, b // 2)
        else:
            m1, m2 = (a // 2, 0), (a // 2, b)
        out.append((b * _NX + a, m1[1] * _NX + m1[0], m2[1] * _NX + m2[0]))
    return tuple(out)


_KNIGHT = _knight_moves()
# the same moves seen from the cell they end in (for the search from GOAL)
_KNIGHT_REV = tuple((-d, m1 - d, m2 - d) for d, m1, m2 in _KNIGHT)


def _kernel():
    # (index offset, clearance, cost) around a new obstacle cell; 0.05 m is
    # taken off because the hit can lie anywhere in the cell
    r = int(_R_SOFT / _RES) + 1
    ker = []
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            d = math.hypot(di, dj) * _RES - 0.05
            if (di or dj) and d < _R_SOFT:
                d = max(d, 0.0)
                ker.append((dj * _NX + di, d, _cost(d)))
    return ker


_KER = _kernel()


class _State(object):
    pass


_S = None


def _init_map(S):
    occ = bytearray(_N)
    pri = bytearray(_N)
    clr = [_R_SOFT] * _N
    wx0, wx1, wy0, wy1 = _WALL
    for iy in range(_NY):
        py = _GY0 + (iy + 0.5) * _RES
        dy = min(py - wy0, wy1 - py)
        row = iy * _NX
        for ix in range(_NX):
            px = _GX0 + (ix + 0.5) * _RES
            d = min(px - wx0, wx1 - px, dy)
            if px < _WEST_X:
                pri[row + ix] = int(_C_FREE * min(1.0, (_WEST_X - px) / 2.0) + 0.5)
            if d <= 0.0:
                occ[row + ix] = 1
                clr[row + ix] = 0.0
            elif d < _R_SOFT:
                clr[row + ix] = d
    for b in MISSION["boxes"]:
        x0, x1 = b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"]
        y0, y1 = b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"]
        i0 = max(int(math.floor((x0 - _R_SOFT - _GX0) / _RES)), 0)
        i1 = min(int(math.floor((x1 + _R_SOFT - _GX0) / _RES)), _NX - 1)
        j0 = max(int(math.floor((y0 - _R_SOFT - _GY0) / _RES)), 0)
        j1 = min(int(math.floor((y1 + _R_SOFT - _GY0) / _RES)), _NY - 1)
        for iy in range(j0, j1 + 1):
            py = _GY0 + (iy + 0.5) * _RES
            dy = max(y0 - py, 0.0, py - y1)
            for ix in range(i0, i1 + 1):
                px = _GX0 + (ix + 0.5) * _RES
                dx = max(x0 - px, 0.0, px - x1)
                k = iy * _NX + ix
                if dx < 0.5 * _RES and dy < 0.5 * _RES:
                    occ[k] = 1
                    clr[k] = 0.0
                else:
                    d = math.hypot(dx, dy)
                    if d < clr[k]:
                        clr[k] = d
    cc = [0] * _N
    cd = [0] * _N
    ck = [0] * _N
    for k in range(_N):
        if not occ[k]:
            c = _cost(clr[k]) + pri[k]
            cc[k] = c
            cd[k] = _dcost(c)
            ck[k] = _kcost(c)
    S.occ, S.pri, S.clr, S.cc, S.cd, S.ck = occ, pri, clr, cc, cd, ck


def _stamp(S, k):
    """Mark cell k occupied and lower the clearance/raise the cost around it."""
    occ = S.occ
    if occ[k]:
        return
    occ[k] = 1
    clr, cc, cd, ck, pri, onp = S.clr, S.cc, S.cd, S.ck, S.pri, S.onpath
    clr[k] = 0.0
    cc[k] = cd[k] = ck[k] = 0
    hit = onp[k]
    soft = False
    for off, d, c in _KER:
        j = k + off
        if d < clr[j]:
            clr[j] = d
            c += pri[j]
            if c != cc[j]:
                cc[j] = c
                cd[j] = _dcost(c)
                ck[j] = _kcost(c)
                if onp[j]:
                    soft = True
                    if d < _R_LETHAL:
                        hit = True
    if soft:
        S.dirty = True
    if hit:
        S.urgent = True


def _scan(S, env, n, z, rng, fallback):
    """One fan as [(bearing rad, distance m)]; None if the call fails.  The
    docstring's keyword call is tried first, then the same arguments
    positionally, 360 deg before 180 deg (what works is remembered), then -
    main fan only - the plain default."""
    raw = None
    for mode in (S.call_mode, 1 - S.call_mode):
        for fov in ((S.fov,) if S.fov < 360.0 else (360.0, 180.0)):
            try:
                if mode == 0:
                    raw = env.raycast_scan(n_beams=n, fov_deg=fov, z=z, max_range=rng)
                else:
                    raw = env.raycast_scan(n, fov, z, rng)
                S.call_mode, S.fov = mode, fov
                break
            except Exception:
                raw = None
        if raw is not None:
            break
    if raw is None and fallback:
        try:
            raw = env.raycast_scan()
        except Exception:
            raw = None
    if raw is None:
        return None
    out = []
    try:
        for item in raw:
            b = float(item[0])
            d = float(item[1])
            if not 0.0 < d < rng:          # NaN, inf, max_range: no hit
                d = rng
            out.append((b, d))
    except Exception:
        return None
    if out and max(abs(b) for b, _ in out) > 6.4:   # bearings in degrees
        out = [(math.radians(b), d) for b, d in out]
    return out


def _self_mask(sc):
    # returns closer than 0.5 m at spawn lie inside the chassis outline:
    # the robot's own body, should the harness not filter it
    return [d + 0.05 if d < 0.5 else 0.0 for _, d in sc]


def _march(S, x, y, a, rng):
    c, s = math.cos(a), math.sin(a)
    occ = S.occ
    r = 0.05
    while r < rng:
        k = _idx(x + r * c, y + r * s)
        if k < 0 or occ[k]:
            return r
        r += 0.05
    return rng


def _bearing_sign(S, x, y, yaw, sc):
    """+1 if bearings are counter-clockwise (expected); -1 only if the spawn
    scan clearly matches the known walls/boxes better when mirrored."""
    mask = S.mask_main
    sp = sn = 0.0
    for i, (b, d) in enumerate(sc):
        if mask is not None and i < len(mask) and d < mask[i]:
            continue
        sp += min(1.0, abs(d - _march(S, x, y, yaw + b, _MAIN[2])))
        sn += min(1.0, abs(d - _march(S, x, y, yaw - b, _MAIN[2])))
    return -1.0 if (sn + 4.0 < sp and sn < 0.6 * sp) else 1.0


def _add_main(S, x, y, yaw, sc):
    mask = S.mask_main
    if mask is not None and len(mask) != len(sc):
        mask = None
    occ = S.occ
    rng = _MAIN[2]
    bx0, bx1, by0, by1 = _BOUND
    for i, (b, d) in enumerate(sc):
        if d >= rng - 0.02 or d < 0.3 or (mask is not None and d < mask[i]):
            continue
        a = yaw + S.sign * b
        hx = x + d * math.cos(a)
        hy = y + d * math.sin(a)
        if bx0 < hx < bx1 and by0 < hy < by1:
            k = _idx(hx, hy)
            if not occ[k]:
                _stamp(S, k)


def _add_extra(S, li, x, y, yaw, sc):
    """Low / high fans (forks, shelf beams, pallets).  A cell needs two hits
    (in it or a 4-neighbour) before it becomes an obstacle, and a fan that
    looks like floor or self returns is distrusted."""
    mask = S.mask_extra[li]
    if mask is not None and len(mask) != len(sc):
        mask = None
    occ, cnt, nx = S.occ, S.cnt, _NX
    rng = _EXTRA[1]
    bx0, bx1, by0, by1 = _BOUND
    novel = close = 0
    cand = []
    for i, (b, d) in enumerate(sc):
        if d >= rng - 0.02 or d < 0.45 or (mask is not None and d < mask[i]):
            continue
        a = yaw + S.sign * b
        hx = x + d * math.cos(a)
        hy = y + d * math.sin(a)
        if not (bx0 < hx < bx1 and by0 < hy < by1):
            continue
        k = _idx(hx, hy)
        if occ[k]:
            continue
        if not (occ[k - 1] or occ[k + 1] or occ[k - nx] or occ[k + nx]
                or occ[k - nx - 1] or occ[k - nx + 1]
                or occ[k + nx - 1] or occ[k + nx + 1]):
            novel += 1
            if d < 1.0:
                close += 1
        cand.append(k)
    n = max(len(sc), 1)
    if novel > 0.85 * n or close > 0.5 * n:
        S.strikes[li] += 1
        if S.strikes[li] >= 4:
            S.extra_on[li] = False
        return
    S.strikes[li] = 0
    for k in cand:
        if occ[k]:
            continue
        if cnt[k] < 250:
            cnt[k] += 1
        if cnt[k] + cnt[k - 1] + cnt[k + 1] + cnt[k - nx] + cnt[k + nx] >= 2:
            _stamp(S, k)


def _plan(S, rk):
    """Dijkstra (bucket queue) from GOAL on the 16-connected grid; stops 3000
    cost units past the robot's cell.  A move costs the entered cell's cost
    times its length (knight moves: the worst of the cells they sweep)."""
    nx = _NX
    cc, cd, ck = S.cc, S.cd, S.ck
    ctg = [_INF] * _N
    g = S.goal_k
    ctg[g] = 0
    buckets = {0: [g]}
    c = cmax = 0
    limit = _INF
    knight = _KNIGHT_REV
    while c <= cmax and c <= limit:
        b = buckets.pop(c, None)
        if b is not None:
            for u in b:
                if ctg[u] != c:
                    continue
                if u == rk:
                    limit = c + 3000
                cu = cc[u] or _C_FREE
                s = c + cu
                for v in (u + 1, u - 1, u + nx, u - nx):
                    if s < ctg[v] and cc[v]:
                        ctg[v] = s
                        lst = buckets.get(s)
                        if lst is None:
                            buckets[s] = [v]
                        else:
                            lst.append(v)
                        if s > cmax:
                            cmax = s
                s = c + _dcost(cu)
                for v in (u + nx + 1, u + nx - 1, u - nx + 1, u - nx - 1):
                    if s < ctg[v] and cc[v]:
                        ctg[v] = s
                        lst = buckets.get(s)
                        if lst is None:
                            buckets[s] = [v]
                        else:
                            lst.append(v)
                        if s > cmax:
                            cmax = s
                ku = ck[u] or _kcost(_C_FREE)
                for dv, d1, d2 in knight:
                    v = u + dv
                    k1 = ck[u + d1]
                    k2 = ck[u + d2]
                    if cc[v] and k1 and k2:
                        s = c + max(ku, k1, k2)
                        if s < ctg[v]:
                            ctg[v] = s
                            lst = buckets.get(s)
                            if lst is None:
                                buckets[s] = [v]
                            else:
                                lst.append(v)
                            if s > cmax:
                                cmax = s
        c += 1
    S.ctg = ctg


def _descend(S, k, nmax):
    """Steepest descent of the cost-to-go field from cell k (list of cells)."""
    ctg, cc, cd, ck, nx, g = S.ctg, S.cc, S.cd, S.ck, _NX, S.goal_k
    out = [k]
    for _ in range(nmax):
        if k == g:
            break
        c0 = ctg[k]
        best = -1
        bval = 0
        for v in (k + 1, k - 1, k + nx, k - nx):
            cv = ctg[v]
            if cv < c0 and cc[v]:
                val = cv + cc[v]
                if best < 0 or val < bval:
                    best, bval = v, val
        for v in (k + nx + 1, k + nx - 1, k - nx + 1, k - nx - 1):
            cv = ctg[v]
            if cv < c0 and cc[v]:
                val = cv + cd[v]
                if best < 0 or val < bval:
                    best, bval = v, val
        for dv, d1, d2 in _KNIGHT:
            v = k + dv
            cv = ctg[v]
            if cv < c0 and cc[v] and ck[k + d1] and ck[k + d2]:
                val = cv + max(ck[v], ck[k + d1], ck[k + d2])
                if best < 0 or val < bval:
                    best, bval = v, val
        if best < 0:
            break
        out.append(best)
        k = best
    return out


def _refresh_path(S, rk, t):
    onp = S.onpath
    for k in S.path:
        onp[k] = 0
    p = _descend(S, rk, 6000)
    for k in p:
        onp[k] = 1
    S.path = p
    S.t_path = t
    ex, ey = _cxy(p[-1])
    if math.hypot(ex - GOAL[0], ey - GOAL[1]) > 0.3:
        S.urgent = True


def _ahead(x, y, pts, L):
    """Mean of the path points whose arc length from the robot is L +- 0.2."""
    s = 0.0
    px, py = x, y
    sx = sy = 0.0
    n = 0
    for qx, qy in pts[1:]:
        s += math.hypot(qx - px, qy - py)
        px, py = qx, qy
        if s >= L - 0.2:
            sx += qx
            sy += qy
            n += 1
            if s >= L + 0.2:
                break
    if n == 0:
        return pts[-1]
    return sx / n, sy / n


def _los(S, x0, y0, x1, y1, need):
    """True if the straight segment keeps clearance >= need everywhere."""
    occ, clr = S.occ, S.clr
    L = math.hypot(x1 - x0, y1 - y0)
    n = int(L / 0.1) + 1
    for i in range(1, n + 1):
        u = i / float(n)
        k = _idx(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
        if k < 0 or occ[k] or clr[k] < need:
            return False
    return True


def _pull(S, x, y, path, pts):
    """String pulling: index of the farthest path point (<= 8 m away) that a
    straight line reaches without coming closer to obstacles than the grid
    path itself does over that stretch (never below _R_LETHAL)."""
    clr = S.clr
    need = _R_SOFT
    best = 0
    last = len(path) - 1
    for j in range(1, last + 1):
        if clr[path[j]] < need:
            need = clr[path[j]]
        if j % 3 and j != last:
            continue
        px, py = pts[j]
        if math.hypot(px - x, py - y) > 8.0:
            break
        if not _los(S, x, y, px, py, max(need, _R_LETHAL) - 0.05):
            break
        best = j
    return best


def _corridor(S, x, y, yaw, sgn):
    """Free distance [m] ahead of the nose (sgn=+1) or tail (sgn=-1) along a
    straight corridor of the chassis width, read from the map."""
    c = math.cos(yaw) * sgn
    s = math.sin(yaw) * sgn
    occ = S.occ
    f = _NOSE - 0.1
    while f < 2.6:
        bx, by = x + f * c, y + f * s
        for lat in S.lats:
            k = _idx(bx - lat * s, by + lat * c)
            if k < 0 or occ[k]:
                return max(0.0, f - _NOSE)
        f += 0.07
    return 9.0


def _rot_ok(S, x, y, yaw, ang):
    """True if turning in place by ang [rad] sweeps no mapped obstacle."""
    occ = S.occ
    pts = []
    i0 = int(math.floor((x - 0.65 - _GX0) / _RES))
    j0 = int(math.floor((y - 0.65 - _GY0) / _RES))
    for j in range(j0, j0 + 14):
        for i in range(i0, i0 + 14):
            if occ[j * _NX + i]:
                dx = _GX0 + (i + 0.5) * _RES - x
                dy = _GY0 + (j + 0.5) * _RES - y
                if dx * dx + dy * dy < 0.65 * 0.65:
                    pts.append((dx, dy))
    if not pts:
        return True
    n = int(abs(ang) / 0.05) + 1
    for m in range(n + 1):
        a = yaw + ang * m / n
        c, s = math.cos(a), math.sin(a)
        for dx, dy in pts:
            if (abs(c * dx + s * dy) < S.hl + 0.05
                    and abs(c * dy - s * dx) < S.hw + 0.05):
                return False
    return True


def _near_pts(S, x, y, r):
    """Occupied cell centres within r of (x, y), relative to (x, y)."""
    occ = S.occ
    n = int(r / _RES) + 1
    i0 = int(math.floor((x - _GX0) / _RES)) - n
    j0 = int(math.floor((y - _GY0) / _RES)) - n
    pts = []
    for j in range(j0, j0 + 2 * n + 1):
        row = j * _NX
        dy = _GY0 + (j + 0.5) * _RES - y
        for i in range(i0, i0 + 2 * n + 1):
            if occ[row + i]:
                dx = _GX0 + (i + 0.5) * _RES - x
                if dx * dx + dy * dy <= r * r:
                    pts.append((dx, dy))
    return pts


def _sweep(pts, yaw, v0, w0, v1, w1, T, fl, fw):
    """Smallest distance [m] from a fl x fw half-size chassis to obstacles
    while the command moves from (v0, w0) to (v1, w1) under the acceleration
    limits for T seconds (negative = overlap; 0.07 m off for the cells)."""
    h = 0.05
    px = py = 0.0
    a, v, w = yaw, v0, w0
    m = 9.0
    for i in range(int(T / h) + 1):
        if i:
            v += max(-_DEC * h, min(_ACC * h, v1 - v))
            w += max(-_WACC * h, min(_WACC * h, w1 - w))
            a += w * h
            px += v * h * math.cos(a)
            py += v * h * math.sin(a)
        c, s = math.cos(a), math.sin(a)
        for dx, dy in pts:
            ox, oy = dx - px, dy - py
            qx = abs(c * ox + s * oy) - fl
            qy = abs(c * oy - s * ox) - fw
            if qx > 0.0 or qy > 0.0:
                d = math.hypot(max(qx, 0.0), max(qy, 0.0))
            else:
                d = max(qx, qy)
            if d < m:
                m = d
    return m - 0.07


def _safe_cmd(S, x, y, yaw, rk, v, w):
    """Keep the chassis S.keep (0.12 m) from mapped obstacles over the next
    0.6 s (or at least no closer than now); otherwise fall back to gentler
    commands - this also catches the tail swinging out in a turn."""
    r = 0.6 + abs(v) * 0.6 + 0.2
    if S.clr[rk] > r:
        return v, w
    pts = _near_pts(S, x, y, r)
    if not pts:
        return v, w
    fl, fw = S.fpl, S.fpw
    floor = min(_sweep(pts, yaw, 0.0, 0.0, 0.0, 0.0, 0.0, fl, fw) - 0.01, S.keep)
    for cv, cw in ((v, w), (v, 0.5 * w), (v, 0.0), (0.5 * v, 0.0),
                   (0.0, w), (0.0, 0.5 * w)):
        if _sweep(pts, yaw, S.v, S.w, cv, cw, 0.6, fl, fw) >= floor:
            return cv, cw
    return 0.0, 0.0


def _goal_target(S):
    """GOAL itself, or - if mapped obstacles crowd it - the best-cleared
    point within 0.5 m of it (still well inside the 0.75 m tolerance)."""
    gx, gy = GOAL[0], GOAL[1]
    k = _idx(gx, gy)
    if S.clr[k] >= 0.7 and not S.occ[k]:
        return gx, gy
    best = (gx, gy)
    bval = -1.0
    for dj in range(-5, 6):
        for di in range(-5, 6):
            kk = k + dj * _NX + di
            cx, cy = _cxy(kk)
            d = math.hypot(cx - gx, cy - gy)
            if d <= 0.5 and not S.occ[kk]:
                val = min(S.clr[kk], 0.7) - 0.3 * d
                if val > bval:
                    best, bval = (cx, cy), val
    return best


def _start(t, x, y, yaw, env):
    S = _State()
    _init_map(S)
    S.onpath = bytearray(_N)
    S.cnt = bytearray(_N)
    S.path = []
    S.goal_k = _idx(GOAL[0], GOAL[1])
    S.dirty = S.urgent = S.no_path = False
    S.ctg = None
    S.t_plan = S.t_path = S.t_hist = -1e9
    S.t_last = t
    S.steps = 0
    S.sign = 1.0
    S.v = S.w = 0.0
    S.rot_dir = 0.0
    S.ref, S.t_ref, S.hold_tol = (x, y, yaw), t, 0.1
    S.x0, S.y0 = x, y
    S.mode, S.t_mode = "NAV", t
    S.best_dg, S.t_best = 1e9, t
    S.ey_best, S.t_ey = 9.0, t
    S.hist = []
    S.esc_until = -1.0
    S.li = 0
    S.extra_on = [True] * len(_EXTRA_Z)
    S.strikes = [0] * len(_EXTRA_Z)
    S.mask_main = None
    S.mask_extra = [None] * len(_EXTRA_Z)
    S.call_mode, S.fov = 0, 360.0
    S.lats, S.hl, S.hw, S.fpl, S.fpw, S.keep = _MARGIN
    S.tight = False
    sc = _scan(S, env, _MAIN[0], _MAIN[1], _MAIN[2], True)
    if sc is not None:
        S.mask_main = _self_mask(sc)
        S.sign = _bearing_sign(S, x, y, yaw, sc)
        _add_main(S, x, y, yaw, sc)
    for i, z in enumerate(_EXTRA_Z):
        sc = _scan(S, env, _EXTRA[0], z, _EXTRA[1], False)
        if sc is None:
            S.extra_on[i] = False
        else:
            S.mask_extra[i] = _self_mask(sc)
            _add_extra(S, i, x, y, yaw, sc)
    rk = _idx(x, y)
    _plan(S, rk)
    S.t_plan = t
    _refresh_path(S, rk, t)
    S.no_path = S.urgent
    S.urgent = False
    return S


def controller(t, pose, env):
    """Map, plan and track to GOAL; align to GOAL yaw; stop; done."""
    try:
        return _control(float(t), float(pose[0]), float(pose[1]),
                        float(pose[2]), env)
    except Exception:
        # never take the harness down; stand still this step and replan
        if _S is not None:
            _S.v = _S.w = 0.0
            _S.urgent = True
        return 0.0, 0.0, False


def _control(t, x, y, yaw, env):
    global _S
    if not (abs(x) < 1e3 and abs(y) < 1e3 and abs(yaw) < 1e3):
        return 0.0, 0.0, False          # NaN / garbage pose: hold still
    yaw = _wrap(yaw)
    if _S is None or t < _S.t_last - 1e-6:
        _S = _start(t, x, y, yaw, env)
    S = _S
    dt = t - S.t_last
    if not 1e-4 < dt < 0.2:
        dt = 1.0 / 60.0
    S.t_last = t
    S.steps += 1

    # -- sense: main fan every step, two extra heights per step
    if S.steps > 1:
        sc = _scan(S, env, _MAIN[0], _MAIN[1], _MAIN[2], True)
        if sc is not None:
            _add_main(S, x, y, yaw, sc)
        todo = 2
        for _ in range(len(_EXTRA_Z)):
            if not todo:
                break
            S.li = (S.li + 1) % len(_EXTRA_Z)
            if S.extra_on[S.li]:
                todo -= 1
                sc = _scan(S, env, _EXTRA[0], _EXTRA_Z[S.li], _EXTRA[1], False)
                if sc is None:
                    S.extra_on[S.li] = False
                else:
                    _add_extra(S, S.li, x, y, yaw, sc)

    # -- plan: re-run Dijkstra when new obstacles touch the current path
    rk = _idx(x, y)
    if rk < 0:
        S.v = S.w = 0.0
        return 0.0, 0.0, False
    if S.ctg[rk] >= _INF and not S.occ[rk]:
        S.urgent = True
    since = t - S.t_plan
    if ((S.urgent and since >= (1.0 if S.no_path else 0.15))
            or (S.dirty and since >= 0.6) or since >= 5.0):
        _plan(S, rk)
        S.t_plan = t
        S.dirty = S.urgent = False
        _refresh_path(S, rk, t)
        S.no_path = S.urgent
        S.urgent = False
    elif t - S.t_path >= 0.5:
        _refresh_path(S, rk, t)

    # -- stuck watchdog (position sampled every 0.5 s)
    if t - S.t_hist >= 0.5:
        S.hist.append((x, y))
        S.t_hist = t
        if len(S.hist) > 14:
            S.hist.pop(0)

    gx, gy, gyaw = GOAL
    dg = math.hypot(gx - x, gy - y)
    ey = _wrap(gyaw - yaw)
    if t > 117.5:                      # never run past the 120 s budget
        S.v = S.w = 0.0
        return 0.0, 0.0, True

    if S.mode == "ESCAPE" and t >= S.esc_until:
        S.mode, S.t_mode = "NAV", t
        S.hist = []
        S.urgent = True
    if dg < S.best_dg - 0.02:
        S.best_dg, S.t_best = dg, t
    if not S.tight and t > 60.0 and dg > 1.0 and t - S.t_best > 25.0:
        # no progress for 25 s: the only way left is a squeeze - try it with
        # margins closer to the real chassis rather than time out for sure
        S.tight = True
        S.lats, S.hl, S.hw, S.fpl, S.fpw, S.keep = _MARGIN_TIGHT
        S.urgent = True
    if S.mode == "NAV" and dg < 0.8:
        S.mode, S.t_mode = "APPROACH", t
    if S.mode in ("NAV", "APPROACH", "ESCAPE") and dg < 0.6 and t - S.t_best > 2.0:
        S.mode, S.t_mode = "ALIGN", t   # GOAL disc reached but blocked: settle
    if S.mode in ("APPROACH", "ALIGN", "HOLD") and dg > 1.5:
        S.mode, S.t_mode = "NAV", t
    if (S.mode == "NAV" and t - S.t_mode > 6.0 and len(S.hist) >= 13
            and math.hypot(x - S.hist[-13][0], y - S.hist[-13][1]) < 0.25):
        S.mode, S.t_mode = "ESCAPE", t
        S.esc_until = t + 1.5

    v_des = w_des = 0.0
    turn = 0.0                      # heading change still wanted [rad]
    done = False
    if S.mode == "NAV":
        path = _descend(S, rk, 100)
        if len(path) < 2:
            S.urgent = True
        else:
            pts = [_cxy(k) for k in path]
            L = 0.5 + 0.5 * abs(S.v)
            j = _pull(S, x, y, path, pts)
            tx, ty = pts[j]
            dj = math.hypot(tx - x, ty - y)
            if dj < L:
                tx, ty = _ahead(x, y, pts, L)
            e = _wrap(math.atan2(ty - y, tx - x) - yaw)
            e2 = 0.0                      # turn coming up within 2.2 m?
            if dj < 2.2:
                fx, fy = _ahead(x, y, pts, 2.2)
                e2 = _wrap(math.atan2(fy - y, fx - x) - yaw)
            cmin = min(S.clr[k] for k in path[:16])
            if abs(e) < 0.9:
                ce = math.cos(e)
                v_des = _V_MAX * ce * ce * ce * ce
            v_des = min(v_des,
                        _V_MAX * max(0.35, math.cos(e2)),
                        max(0.15, 0.2 + S.clr[rk] - _R_HARD),
                        max(0.2, 0.25 + cmin - _R_HARD),
                        math.sqrt(2.4 * max(0.0, _corridor(S, x, y, yaw, 1.0) - 0.1)),
                        0.2 + 0.7 * dg)
            w_des = max(-_W_MAX, min(_W_MAX, 2.2 * e))
            turn = abs(e)
    elif S.mode == "ESCAPE":
        if _corridor(S, x, y, yaw, -1.0) > 0.35:
            v_des = -0.15
    if S.mode == "APPROACH":
        ax, ay = _goal_target(S)
        da = math.hypot(ax - x, ay - y)
        e = _wrap(math.atan2(ay - y, ax - x) - yaw)
        if da < 0.12 or (da < 0.4 and abs(e) > 1.0):
            S.mode, S.t_mode = "ALIGN", t
        else:
            if abs(e) < 0.8:
                v_des = min(0.4, 0.15 + 0.6 * da) * math.cos(e) ** 2
            v_des = min(v_des, math.sqrt(2.4 * max(0.0, _corridor(S, x, y, yaw, 1.0) - 0.1)))
            w_des = max(-0.9, min(0.9, 2.0 * e))
            turn = abs(e)
    if S.mode == "ALIGN":
        if S.t_ey < S.t_mode or abs(ey) < S.ey_best - 0.005:
            S.ey_best, S.t_ey = abs(ey), t
        if abs(ey) < 0.03 or (t - S.t_ey > 1.5 and abs(ey) <= 0.2):
            S.mode, S.t_mode = "HOLD", t   # aligned (or as good as it gets)
            S.ref, S.t_ref = (x, y, yaw), t
            S.hold_tol = 0.1 if abs(ey) < 0.03 else 0.2
        else:
            w_min = 0.1 if t - S.t_ey < 1.5 else 0.25   # creeping too slowly?
            w_des = max(-0.6, min(0.6, 1.2 * ey))
            if abs(w_des) < w_min:
                w_des = math.copysign(w_min, ey)
            turn = abs(ey)

    # -- turning in place: never sweep the chassis into a mapped obstacle
    if abs(v_des) < 0.05 and abs(w_des) > 0.05:
        want = 1.0 if w_des > 0.0 else -1.0
        if S.rot_dir and not _rot_ok(S, x, y, yaw, 0.5 * S.rot_dir):
            S.rot_dir = 0.0
        if not S.rot_dir:
            if _rot_ok(S, x, y, yaw, want * min(0.5, turn + 0.1)):
                S.rot_dir = want
            elif _rot_ok(S, x, y, yaw, -0.5 * want):
                S.rot_dir = -want          # the long way round is free
        if S.rot_dir:
            w_des = S.rot_dir * abs(w_des)
        else:
            w_des = 0.0
            if S.mode == "ALIGN" and abs(ey) <= 0.2:
                S.mode, S.t_mode = "HOLD", t   # boxed in, but within tolerance
                S.ref, S.t_ref, S.hold_tol = (x, y, yaw), t, 0.2
            elif _corridor(S, x, y, yaw, -1.0) > 0.3:
                v_des = -0.1               # back off to get room to turn
            elif _corridor(S, x, y, yaw, 1.0) > 0.3:
                v_des = 0.1
    else:
        S.rot_dir = 0.0

    if S.mode == "HOLD":
        # zero command until the pose has really stopped changing, then done
        v_des = w_des = 0.0
        if t - S.t_ref >= 0.25:
            still = (math.hypot(x - S.ref[0], y - S.ref[1]) < 0.01
                     and abs(_wrap(yaw - S.ref[2])) < 0.01)
            S.ref, S.t_ref = (x, y, yaw), t
            if (still and t - S.t_mode >= 0.5) or t - S.t_mode >= 3.0:
                if dg <= 0.62 and abs(ey) <= S.hold_tol:
                    done = True
                else:
                    S.mode, S.t_mode = ("APPROACH" if dg > 0.62 else "ALIGN"), t

    # -- a 180 deg fan never saw beside/behind the spawn pose: drive the
    #    first 0.5 m straight on (START already faces GOAL, and that way was
    #    scanned) rather than swing the tail into the unseen, never reverse
    if S.fov < 360.0 and math.hypot(x - S.x0, y - S.y0) < 1.0:
        if v_des < 0.0:
            v_des = 0.0
        if (S.mode == "NAV" and math.hypot(x - S.x0, y - S.y0) < 0.5
                and _corridor(S, x, y, yaw, 1.0) > 0.5):
            v_des = max(v_des, 0.25)
            w_des = max(-0.15, min(0.15, w_des))

    # -- command with acceleration limits
    v_des, w_des = _safe_cmd(S, x, y, yaw, rk, v_des, w_des)
    v = S.v + max(-_DEC * dt, min(_ACC * dt, v_des - S.v))
    w = S.w + max(-_WACC * dt, min(_WACC * dt, w_des - S.w))
    if done:
        v = w = 0.0
    S.v, S.w = v, w
    return v, w, done


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera
ctrl = controller
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(controller, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
