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

# The spawn yaw is already the straight line to the goal: atan2(16.5 + 10.8,
# 8.0 + 9.0) = 1.0138 rad, and GOAL asks for that very same yaw again.  The
# line itself is unusable - it runs through box 00, through the 5 m barrier
# box 03 and through box 06.  Boxes 06 (h 0.18), 05 (0.20), 13 (0.20) and 04
# (0.25) all stand below the z = 0.35 m scan plane, so the ray fan can never
# report them; box 06 in particular lies right across the last leg into the
# goal corner.  Those four exist only in the MISSION table, so the route has
# to be planned from the table, not sensed.
#
# Plan: a fixed polyline up the middle-west of the hall that keeps >= 0.85 m
# from every mapped box face and stays >= 2.5 m off the west wall (the west
# shelf is a scored collider), crossing the box 03 barrier on its west end and
# passing north of box 08 / box 06 before turning east into the goal.  Pure
# pursuit drives it; the ray fan is layered on top for everything that is NOT
# in the table - walls, pillars, shelves, the forklift - by scoring a fan of
# candidate headings against the measured ranges and the mapped boxes at once.

_WAYPOINTS = (
    (-5.90, -8.80),   # off the west wall, aiming east of box 00
    (-5.00, -5.20),   # ~0.9 m east of box 00
    (-4.00, -1.40),   # west of box 01
    (-3.60, 2.20),    # west of box 02
    (-2.90, 5.20),    # through the gap west of the 5 m box 03
    (-2.90, 9.00),    # ~1.0 m east of box 09
    (-3.00, 12.80),   # ~0.95 m west of box 10
    (-1.00, 15.20),   # north of box 08 and of the invisible box 06
    (4.50, 15.70),    # run east well clear of box 06
    (GOAL[0], GOAL[1]),
)

_BOXES = tuple((b["x"], b["y"], 0.5 * b["width"], 0.5 * b["depth"])
               for b in MISSION["boxes"])

_SCAN_MAX = 6.0      # raycast_scan max_range: a beam at this range is a miss
_HALF_W = 0.55       # half width of the corridor the chassis needs [m]
_WANT_CLEAR = 0.80   # wanted distance to a mapped box face [m]
_V_MAX = 1.05        # [m/s]
_W_MAX = 1.30        # [rad/s]

_S = {"wp": 0, "phase": 0, "hold": 0, "spin": 0.0, "commit": -1.0,
      "align": 0.0, "side": 0.0, "side_t": -1.0, "pivot": 0.0}


def _wrap(a):
    """Fold an angle into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _box_clearance(px, py):
    """Signed distance from (px, py) to the nearest mapped box (<0 = inside)."""
    best = 1.0e9
    for cx, cy, hw, hh in _BOXES:
        dx = abs(px - cx) - hw
        dy = abs(py - cy) - hh
        if dx > 0.0 or dy > 0.0:
            d = math.hypot(dx if dx > 0.0 else 0.0, dy if dy > 0.0 else 0.0)
        else:
            d = dx if dx > dy else dy
        if d < best:
            best = d
    return best


def _corridor(scan, rel):
    """Free run along relative bearing `rel` in a corridor of half width."""
    best = _SCAN_MAX
    for b, r in scan:
        if r >= _SCAN_MAX:
            continue
        db = _wrap(b - rel)
        if db > 1.25 or db < -1.25:
            continue
        lon = r * math.cos(db)
        if 0.0 < lon < best and abs(r * math.sin(db)) <= _HALF_W:
            best = lon
    return best


def _map_range(x, y, head):
    """Range along world heading `head` to the first mapped box, the chassis
    corridor already folded in (same geometry as _corridor).  The ray fan is
    blind to boxes 04, 05, 06 and 13 - they are shorter than the 0.35 m scan
    plane - so this is the only thing that can stop the robot from driving
    straight through one of them."""
    dx = math.cos(head)
    dy = math.sin(head)
    best = _SCAN_MAX
    for cx, cy, hw, hh in _BOXES:
        lox = cx - hw - _HALF_W
        hix = cx + hw + _HALF_W
        loy = cy - hh - _HALF_W
        hiy = cy + hh + _HALF_W
        if lox < x < hix and loy < y < hiy:
            # already inside this box' safety pad: allow only headings that
            # increase the clearance again, block the ones that dig deeper
            ex = abs(x - cx) - hw - _HALF_W
            ey = abs(y - cy) - hh - _HALF_W
            d0 = ex if ex > ey else ey
            ex = abs(x + 0.35 * dx - cx) - hw - _HALF_W
            ey = abs(y + 0.35 * dy - cy) - hh - _HALF_W
            d1 = ex if ex > ey else ey
            if d1 <= d0:
                return 0.0
            continue
        tmin = 0.0
        tmax = best
        hit = True
        for o, dd, lo, hi in ((x, dx, lox, hix), (y, dy, loy, hiy)):
            if -1.0e-9 < dd < 1.0e-9:
                if o < lo or o > hi:
                    hit = False
                    break
            else:
                t1 = (lo - o) / dd
                t2 = (hi - o) / dd
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                if tmin > tmax:
                    hit = False
                    break
        if hit and tmin < best:
            best = tmin
    return best


def _map_cost(x, y, head):
    """Penalty for pushing ~3 m along world heading `head` into a mapped box."""
    cx = math.cos(head)
    cy = math.sin(head)
    cost = 0.0
    for k in (1, 2, 3, 4, 5, 6, 7):
        s = 0.45 * k
        gap = _box_clearance(x + s * cx, y + s * cy)
        if gap < _WANT_CLEAR:
            cost += 2.4 * (_WANT_CLEAR - gap) ** 2 * (3.4 - s)
    return cost


def _choose(t, x, y, yaw, scan, want, ref):
    """Trade the pull towards the waypoint against measured + mapped clutter.

    `ref` is the range that still matters: nothing beyond the target has to be
    driven around, which is what lets the robot park in the goal corner with
    the north/east walls a metre in front of it.
    """
    side = _S["side"] if t < _S["side_t"] else 0.0
    best_cost = None
    best_rel = 0.0
    best_rng = 0.0
    for k in range(-7, 8):
        rel = 0.15 * k
        rng = _corridor(scan, rel)
        mrng = _map_range(x, y, yaw + rel)
        if mrng < rng:
            rng = mrng
        dev = rel - want
        cost = abs(_wrap(dev)) + _map_cost(x, y, yaw + rel)
        if rng < ref:
            cost += 1.4 * (ref - rng) ** 2
        if side * dev < -0.05:
            cost += 0.40          # once dodging one way, do not flip-flop
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_rel = rel
            best_rng = rng
    dev = best_rel - want
    if dev > 0.35 or dev < -0.35:                 # committed to a detour side
        _S["side"] = 1.0 if dev > 0.0 else -1.0
        _S["side_t"] = t + 2.5
    return best_rel, best_rng


def controller(t, pose, env):
    """Waypoint pursuit + ray-fan gap picking; stop on GOAL at the GOAL yaw."""
    x, y, yaw = pose

    if _S["phase"] == 2:                          # already finished
        return 0.0, 0.0, True

    if _S["phase"] == 1:                          # parked on GOAL: align yaw
        yerr = _wrap(GOAL[2] - yaw)
        _S["align"] += 1.0                        # frames spent aligning
        if -0.09 < yerr < 0.09:
            _S["hold"] += 1
            if _S["hold"] >= 20:                  # ~0.33 s standing still
                _S["phase"] = 2
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        _S["hold"] = 0
        if _S["align"] > 720.0:                   # 12 s: never hang here
            _S["phase"] = 2
            return 0.0, 0.0, True
        w = 1.6 * yerr
        if w > 0.8:
            w = 0.8
        elif w < -0.8:
            w = -0.8
        elif w >= 0.0 and w < 0.25:
            w = 0.25
        elif w < 0.0 and w > -0.25:
            w = -0.25
        return 0.0, w, False

    if t < 0.4:                                   # let the chassis settle
        return 0.0, 0.0, False

    # ---- walk the polyline -------------------------------------------------
    i = _S["wp"]
    last = len(_WAYPOINTS) - 1
    tx, ty = _WAYPOINTS[i]
    d = math.hypot(tx - x, ty - y)
    if i < last:
        px, py = _WAYPOINTS[i - 1] if i > 0 else (START[0], START[1])
        passed = (x - tx) * (tx - px) + (y - ty) * (ty - py) > 0.0
        if d < 0.60 or (d < 1.60 and passed):
            i += 1
            _S["wp"] = i
            tx, ty = _WAYPOINTS[i]
            d = math.hypot(tx - x, ty - y)
    if i < last:
        gd = math.hypot(GOAL[0] - x, GOAL[1] - y)
        if gd < 0.70:                             # dodged straight onto GOAL
            i = last
            _S["wp"] = last
            tx, ty = _WAYPOINTS[last]
            d = gd

    want = _wrap(math.atan2(ty - y, tx - x) - yaw)

    try:
        scan = env.raycast_scan()
    except Exception:                             # rather blind than dead
        scan = []
    if not scan:
        scan = []
    front = _corridor(scan, 0.0)
    mfront = _map_range(x, y, yaw)
    if mfront < front:
        front = mfront

    # ---- arrival -----------------------------------------------------------
    if i == last:
        if (d < 0.22
                or (d < 0.55 and (want > 0.9 or want < -0.9))
                or (d < 0.60 and front < 0.85)
                or t > 112.0):
            _S["phase"] = 1
            _S["align"] = 0.0
            return 0.0, 0.0, False

    # ---- pick a heading ----------------------------------------------------
    if t < _S["commit"]:                          # just escaped: keep running
        want = 0.0
    if _S["pivot"] != 0.0:                        # pivoting onto the next leg
        if -0.90 < want < 0.90:
            _S["pivot"] = 0.0
        else:
            return 0.0, _S["pivot"], False
    elif want > 1.25 or want < -1.25:             # leg is abeam: pivot onto it
        _S["pivot"] = 1.0 if want > 0.0 else -1.0
        return 0.0, _S["pivot"], False

    ref = 2.5
    if i == last and d < 2.15:                    # only the goal itself is left
        ref = d + 0.35
        if ref < 0.90:
            ref = 0.90
    rel, rng = _choose(t, x, y, yaw, scan, want, ref)
    if rng > ref and -0.16 < rel - want < 0.16:
        rel = want                                # smooth once the way is open

    # ---- blocked: spin on the spot towards the freer half ------------------
    #      `front` matters as much as `rng`: the chassis drives along `yaw`
    if rng < 0.85 or front < 0.80:
        sp = _S["spin"]
        if sp == 0.0:
            left = 0.0
            right = 0.0
            for k in (3, 5, 7, 9):                # +-0.45 .. +-1.35 rad
                a = 0.15 * k
                lr = _corridor(scan, a)
                mr = _map_range(x, y, yaw + a)
                left += lr if lr < mr else mr
                rr = _corridor(scan, -a)
                mr = _map_range(x, y, yaw - a)
                right += rr if rr < mr else mr
            sp = 1.0 if left >= right else -1.0
            _S["spin"] = sp
        _S["commit"] = t + 1.0
        return 0.0, 0.9 * sp, False
    if t >= _S["commit"]:
        _S["spin"] = 0.0

    # ---- speed -------------------------------------------------------------
    v = _V_MAX
    if rng < 2.6:
        v = 0.28 + 0.5 * (rng - 0.85)
    c = math.cos(rel)
    v *= c if c > 0.0 else 0.0
    if rel > 0.75 or rel < -0.75:
        v = v if v < 0.30 else 0.30
    if i == last:
        cap = 0.18 + 0.35 * d                     # ease into the goal disc
        v = v if v < cap else cap
    elif d < 0.9:
        cap = 0.55 + 0.5 * d                      # ease around the corner
        v = v if v < cap else cap
    if front < 1.05:
        # the chassis translates along `yaw`, not along the heading we steer
        # towards, so the brake has to look straight ahead as well
        cap = (front - 0.62) * 1.4
        v = 0.0 if cap < 0.0 else (v if v < cap else cap)

    w = 2.1 * rel
    if w > _W_MAX:
        w = _W_MAX
    elif w < -_W_MAX:
        w = -_W_MAX
    if v < 0.03 and -0.06 < w < 0.06:             # never freeze: pivot instead
        w = 0.9 if want >= 0.0 else -0.9
    return v, w, False


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
