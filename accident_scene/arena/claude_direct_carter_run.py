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
# Direct-request build (no benchmark prompt, full repo knowledge):
#   * the route is an A* path over this repo's measured occupancy map
#     (sim2d.path_exists, 0.75 m robot-centre keep-out) decimated to 19 points;
#   * a pure-pursuit carrot follows it, with an INTEGRAL term on cross-track
#     error - that is what absorbs a steady steering bias such as the 15 deg
#     wheel-axle fault this build targets;
#   * the ray fan is only a safety net for anything the map does not have.

_WPS = (
    (-8.97, -10.78), (-7.16, -8.97), (-6.28, -8.09), (-4.41, -6.34),
    (-2.59, -4.53), (-1.91, -3.84), (-0.97, -3.28), (0.72, -1.41),
    (2.53, 0.41), (4.34, 2.22), (5.41, 3.28), (5.53, 5.78),
    (5.53, 8.28), (5.53, 10.78), (5.53, 12.28), (5.28, 14.22),
    (7.03, 16.03), (7.78, 16.41), (GOAL[0], GOAL[1]),
)

_LOOKAHEAD = 1.30      # carrot distance along the path [m]
_K_HEAD = 2.10         # heading error -> yaw rate
_K_INT = 0.55          # integral of cross-track error -> yaw rate (bias killer)
_I_CLAMP = 0.45        # rad/s worth of integral authority
_V_MAX = 1.00          # [m/s]
_W_MAX = 1.30          # [rad/s]
_DT = 1.0 / 60.0
_ARRIVE = 0.25         # start the final pivot inside this radius [m]
_YAW_OK = 0.08         # pivot until the yaw error is under this [rad]

_S = {"seg": 0, "int": 0.0, "phase": 0, "hold": 0, "spin": 0.0, "align": 0.0}


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _project(x, y, i):
    """Closest point on segment i and how far along it we are (0..1)."""
    ax, ay = _WPS[i]
    bx, by = _WPS[i + 1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return ax, ay, 1.0, 0.0
    s = ((x - ax) * dx + (y - ay) * dy) / L2
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    px, py = ax + s * dx, ay + s * dy
    cross = (dx * (y - ay) - dy * (x - ax)) / math.sqrt(L2)   # +left of the path
    return px, py, s, cross


def _carrot(x, y):
    """Advance the segment index, then walk _LOOKAHEAD m forward along the path."""
    i = _S["seg"]
    while i < len(_WPS) - 2:
        _, _, s, _ = _project(x, y, i)
        if s < 0.999:
            break
        i += 1
    _S["seg"] = i
    px, py, s, cross = _project(x, y, i)
    need = _LOOKAHEAD
    j = i
    cx, cy = px, py
    while need > 0.0 and j < len(_WPS) - 1:
        bx, by = _WPS[j + 1]
        d = math.hypot(bx - cx, by - cy)
        if d >= need:
            cx += (bx - cx) * need / d
            cy += (by - cy) * need / d
            need = 0.0
        else:
            need -= d
            cx, cy = bx, by
            j += 1
    return cx, cy, cross, i


def _front(scan, half=0.45):
    """Nearest return inside a +-half rad cone around the heading."""
    best = 6.0
    for bearing, dist in scan:
        if -half <= bearing <= half and dist < best:
            best = dist
    return best


def controller(t, pose, env):
    x, y, yaw = pose

    if _S["phase"] == 2:
        return 0.0, 0.0, True

    if _S["phase"] == 1:                                  # parked: align to GOAL yaw
        err = _wrap(GOAL[2] - yaw)
        _S["align"] += _DT
        if -_YAW_OK < err < _YAW_OK:
            _S["hold"] += 1
            if _S["hold"] >= 20:
                _S["phase"] = 2
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        _S["hold"] = 0
        if _S["align"] > 12.0:                            # never hang here
            _S["phase"] = 2
            return 0.0, 0.0, True
        w = 1.8 * err
        w = max(-0.9, min(0.9, w))
        if 0.0 <= w < 0.30:
            w = 0.30
        elif -0.30 < w < 0.0:
            w = -0.30
        return 0.0, w, False

    if t < 0.35:                                          # let the chassis settle
        return 0.0, 0.0, False

    gd = math.hypot(GOAL[0] - x, GOAL[1] - y)
    if gd < _ARRIVE or t > 112.0:
        _S["phase"] = 1
        _S["align"] = 0.0
        return 0.0, 0.0, False

    cx, cy, cross, seg = _carrot(x, y)
    want = _wrap(math.atan2(cy - y, cx - x) - yaw)

    # integral of cross-track error: cancels a steady plant bias (wheel toe)
    _S["int"] += cross * _DT
    _S["int"] = max(-3.0, min(3.0, _S["int"]))
    w = _K_HEAD * want - _K_INT * _S["int"]
    lim = _K_INT * abs(_S["int"])
    if lim > _I_CLAMP:                                    # keep the integral honest
        _S["int"] *= _I_CLAMP / lim
    w = max(-_W_MAX, min(_W_MAX, w))

    # safety net for anything the map does not carry
    try:
        scan = env.raycast_scan()
    except Exception:
        scan = []
    front = _front(scan)
    if front < 0.75:
        sp = _S["spin"]
        if sp == 0.0:
            left = right = 0.0
            for bearing, dist in scan:
                if bearing > 0.25:
                    left += dist
                elif bearing < -0.25:
                    right += dist
            sp = 0.9 if left >= right else -0.9
            _S["spin"] = sp
        return 0.0, sp, False
    _S["spin"] = 0.0

    v = _V_MAX
    if abs(want) > 0.45:                                  # slow into turns
        v *= 0.45
    elif abs(want) > 0.20:
        v *= 0.75
    if front < 2.0:
        v = min(v, 0.25 + 0.45 * (front - 0.75))
    if gd < 2.5:                                          # ease into the goal disc
        v = min(v, 0.20 + 0.42 * gd)
    v = max(0.0, v)
    if v < 0.05 and -0.05 < w < 0.05:                     # never freeze
        w = 0.6 if want >= 0.0 else -0.6
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
