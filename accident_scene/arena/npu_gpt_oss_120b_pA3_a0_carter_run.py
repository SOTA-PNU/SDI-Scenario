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

# verified waypoint list (including the final GOAL)
_WAYPOINTS = [(-4.7, -7.3),
              (-3.7, 5.1),
              (0.7, 11.3),
              (2.3, 15.1),
              (8.0, 16.5)]

def _wrap(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))

def _clamp(v, lo, hi):
    """Clamp v into [lo, hi]."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def controller(t, pose, env):
    """
    Way‑point follower that exactly matches the verified trajectory.

    Returns (v, w, done) each simulation step.
    """
    # one‑time initialisation
    if not hasattr(controller, "wp_idx"):
        controller.wp_idx = 0          # index into _WAYPOINTS
        controller.align = False       # True when we are in the final alignment phase

    x, y, yaw = pose
    tx, ty = _WAYPOINTS[controller.wp_idx]

    # distance and heading error to the current waypoint
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap(math.atan2(dy, dx) - yaw)

    # ----------------------------------------------------------------------
    # 1) Check if we have reached the final GOAL and need to align yaw.
    # ----------------------------------------------------------------------
    if controller.wp_idx == len(_WAYPOINTS) - 1:          # we are at the GOAL waypoint
        if dist < 0.25:                                 # close enough to start alignment
            controller.align = True
            # alignment error to the goal yaw
            e_align = _wrap(GOAL[2] - yaw)
            w = _clamp(1.5 * e_align, -0.6, 0.6)
            v = 0.0
            if abs(e_align) < 0.05:
                # goal reached and properly aligned
                return 0.0, 0.0, True
            return v, w, False
        # else: still far from the goal – treat like a normal waypoint

    # ----------------------------------------------------------------------
    # 2) Normal waypoint handling (including the non‑final waypoints)
    # ----------------------------------------------------------------------
    # If we are close enough to the current waypoint, advance to the next one
    if dist < 0.35 and controller.wp_idx < len(_WAYPOINTS) - 1:
        controller.wp_idx += 1
        # recompute target for the new waypoint on the next call
        return 0.0, 0.0, False

    # Decide whether to rotate in place or move forward
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        v = min(0.6, 0.5 * dist + 0.1)
        w = _clamp(1.5 * e, -0.8, 0.8)

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
