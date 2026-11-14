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

# ----------------------------------------------------------------------
# Way‑point list (all points are (x, y) in world coordinates).
# The last point is the goal position; the robot will align to GOAL[2] there.
# ----------------------------------------------------------------------
_waypoints = [
    (-6.0, -10.8),   # clear the west‑shelf (stay south of it)
    (-4.0, -4.0),    # move north‑east, clear the first three boxes
    (-4.0, -1.0),    # go a little higher before heading east
    (-4.0, 6.0),     # go north, staying left of the big box at (0.5,4.2)
    (2.0, 6.0),      # pass east of that box, still below the upper obstacles
    (2.0, 15.0),     # go straight up, clear boxes 7‑10 and the two north‑west objects
    (8.0, 16.5),     # final goal (same as GOAL position)
]

# index of the current target waypoint
_wp_idx = 0

# ----------------------------------------------------------------------
# Helper utilities (exactly the same as described in the prompt)
# ----------------------------------------------------------------------
def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp *v* to the interval [lo, hi]."""
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------
def controller(t, pose, env):
    """
    Follow the pre‑computed way‑points while keeping a very simple
    obstacle‑avoidance check using the forward ray‑cast.

    The function is called at 60 Hz.  It returns (v, w, done) where
    *v* is forward speed [m s⁻¹], *w* is yaw rate [rad s⁻¹] and *done*
    signals that the mission is finished.
    """
    global _wp_idx

    # ------------------------------------------------------------------
    # 1) Very simple safety check with the forward 180° scan.
    #    If something is closer than 0.6 m in a ±30° cone, stop and turn
    #    slowly to the right.  This demonstrates that we *use* the sensor.
    # ------------------------------------------------------------------
    scans = env.raycast_scan()                     # [(bearing, distance), ...]
    front_dist = min(
        (dist for bearing, dist in scans if abs(bearing) < math.radians(30)),
        default=6.0,
    )
    if front_dist < 0.6:
        # obstacle ahead – rotate in place to the right
        return 0.0, 0.6, False

    # ------------------------------------------------------------------
    # 2) Extract current pose
    # ------------------------------------------------------------------
    x, y, yaw = pose

    # ------------------------------------------------------------------
    # 3) Determine the current target (way‑point or final alignment)
    # ------------------------------------------------------------------
    # Final alignment flag – we are at the last waypoint (the goal)
    at_goal = (_wp_idx >= len(_waypoints) - 1)

    if at_goal:
        # --------------------------------------------------------------
        # Alignment phase (as described in the prompt)
        # --------------------------------------------------------------
        tx, ty = GOAL[0], GOAL[1]          # goal position
        e = _wrap_angle(GOAL[2] - yaw)     # yaw error to goal orientation
        w = _clamp(1.5 * e, -0.6, 0.6)
        v = 0.0

        # If we are already well aligned, finish the mission
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return v, w, False

    # --------------------------------------------------------------
    # Normal waypoint‑following phase
    # --------------------------------------------------------------
    tx, ty = _waypoints[_wp_idx]

    # Distance and heading error to the current waypoint
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    e = _wrap_angle(target_angle - yaw)

    # --------------------------------------------------------------
    # 4) Way‑point switching logic (same thresholds as the prompt)
    # --------------------------------------------------------------
    if dist < 0.35:
        # reached this waypoint – advance to the next one
        _wp_idx += 1
        # make sure we don't step past the list
        if _wp_idx >= len(_waypoints):
            _wp_idx = len(_waypoints) - 1
        # recompute for the new target on the next call
        return 0.0, 0.0, False

    # --------------------------------------------------------------
    # 5) Control law (identical to the proven one in the prompt)
    # --------------------------------------------------------------
    if abs(e) > 0.35:
        # rotate in place until we are roughly facing the waypoint
        v = 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        # drive forward while applying a small heading correction
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
