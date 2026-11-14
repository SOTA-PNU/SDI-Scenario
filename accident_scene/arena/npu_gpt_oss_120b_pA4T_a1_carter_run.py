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
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

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
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    """Clamp v to the interval [lo, hi]."""
    return max(lo, min(hi, v))


# Way‑points that keep a safe clearance (>1.1 m) from all boxes and fixed
# structures while using fewer intermediate points than the original solution.
# The last point is the goal (GOAL) itself.
_WAYPOINTS = [
    (-5.0, -10.8),   # start – move east away from the left wall
    (-5.0, 5.8),     # go north, still clear of the left‑side shelf
    (-5.0, 9.0),     # go higher to pass above box 09
    (0.0, 10.0),     # shift east before the cluster around (‑2, 11)
    (1.5, 12.0),     # clear the central obstacle field
    (1.5, 16.5),     # go north past the remaining boxes
    (GOAL[0], GOAL[1])  # final goal position
]

# Tuned control gains – higher than the original but still safe.
_MAX_SPEED = 1.5          # m/s (well below the 2 m/s hardware limit)
_MAX_ANG_SPEED = 1.5      # rad/s (below the 3 rad/s hardware limit)
_KP_ANG = 2.0             # angular gain
_KP_DIST = 0.8            # distance gain for forward speed
_E_THRESH_ROT = 0.35      # rad – rotate‑in‑place threshold
_DIST_THRESH_WP = 0.45    # m – waypoint switch threshold


def controller(t, pose, env):
    """Follow the reduced waypoint list with a faster, still safe controller."""
    # initialise static state on first call
    if not hasattr(controller, "idx"):
        controller.idx = 0          # current waypoint index
        controller.align = False    # flag for goal‑alignment phase

    x, y, yaw = pose

    # ------------------------------------------------------------------
    # 1️⃣ Goal‑alignment phase (after the last waypoint)
    # ------------------------------------------------------------------
    if controller.align:
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(_KP_ANG * e, -_MAX_ANG_SPEED, _MAX_ANG_SPEED)
        if abs(e) < 0.05:
            # finished – report success
            return 0.0, 0.0, True
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 2️⃣ Normal waypoint‑following
    # ------------------------------------------------------------------
    tx, ty = _WAYPOINTS[controller.idx]

    # distance and heading error to the current waypoint
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # advance to the next waypoint if we are close enough (except for the goal)
    if controller.idx < len(_WAYPOINTS) - 1 and dist < _DIST_THRESH_WP:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # 3️⃣ Control law (faster than the original, still respecting limits)
    # ------------------------------------------------------------------
    if abs(e) > _E_THRESH_ROT:
        # rotate on the spot
        v = 0.0
        w = _clamp(_KP_ANG * e, -_MAX_ANG_SPEED, _MAX_ANG_SPEED)
    else:
        # move forward while correcting heading
        v = min(_MAX_SPEED, _KP_DIST * dist + 0.2)
        w = _clamp(_KP_ANG * e, -_MAX_ANG_SPEED, _MAX_ANG_SPEED)

    # when the last waypoint (the goal) is reached, switch to alignment mode
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.align = True

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
