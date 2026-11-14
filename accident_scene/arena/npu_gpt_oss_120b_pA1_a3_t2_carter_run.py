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
# Tuning constants (purely heuristic – do not import anything else)
# ----------------------------------------------------------------------
_MAX_SPEED = 1.0          # m/s – top forward speed
_MAX_ANG_SPEED = 0.8      # rad/s – top yaw rate
_KP_ANG = 1.5             # proportional gain for yaw error
_POS_TOL_WP = 0.5         # meters – when we consider a waypoint reached
_SAFE_DIST = 1.0          # meters – start obstacle‑avoidance turn when closer
_TURN_SPEED = 0.6         # rad/s – turn rate while avoiding an obstacle

# ----------------------------------------------------------------------
# Waypoint list (hard‑coded, safe corridor around the static boxes)
# ----------------------------------------------------------------------
# start is implicitly the robot’s spawn pose; the first element is the
# first target we head for.
_WAYPOINTS = [
    (-4.0, -10.8),   # east of the start, clear of the western wall
    (-0.5, -10.8),   # pass south of the low‑lying boxes near the origin
    (-0.5,  2.0),    # go north, left of the long box at (0.5,4.2)
    (4.0,   2.0),    # east across the lower aisle
    (4.0,   6.0),    # north, staying clear of the middle‑row boxes
    (7.0,   6.0),    # east toward the centre‑right side
    (7.0,  12.0),    # north, still below the tall box at (7.625,13.4)
    (7.0,  14.5),    # north, now above that box
    (8.0,  16.5),    # final goal position
]

# internal mutable state – kept between calls
_wp_idx = 0               # which waypoint we are currently heading for
_done_reported = False    # ensure we only signal `done` once


def _norm_angle(a: float) -> float:
    """Wrap angle to the interval [-π, +π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple waypoint‑following controller with very light obstacle avoidance.

    * Drives to the next waypoint using a proportional heading controller.
    * When a forward obstacle is closer than ``_SAFE_DIST`` it turns away
      while slowing down.
    * When the final waypoint (the goal) is reached, it aligns the robot’s
      yaw to the goal yaw and signals ``done=True``.
    """
    global _wp_idx, _done_reported

    # ------------------------------------------------------------------
    # unpack current robot state and mission constants
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # ------------------------------------------------------------------
    # If we have already signalled completion, keep returning stop.
    # ------------------------------------------------------------------
    if _done_reported:
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Helper: distance to a point
    # ------------------------------------------------------------------
    def _dist(px, py):
        return math.hypot(px - x, py - y)

    # ------------------------------------------------------------------
    # Check if we are at the final waypoint (the goal) and need only
    # to align yaw.
    # ------------------------------------------------------------------
    if _wp_idx >= len(_WAYPOINTS):
        # We are already at the goal position – just align yaw.
        yaw_error = _norm_angle(goal_yaw - yaw)
        if abs(yaw_error) <= yaw_tol:
            _done_reported = True
            return 0.0, 0.0, True
        # Rotate in place toward the desired yaw.
        w = max(-_MAX_ANG_SPEED,
                min(_MAX_ANG_SPEED, _KP_ANG * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Current target waypoint
    # ------------------------------------------------------------------
    tgt_x, tgt_y = _WAYPOINTS[_wp_idx]

    # If we are close enough to this waypoint, advance to the next one.
    if _dist(tgt_x, tgt_y) <= _POS_TOL_WP:
        _wp_idx += 1
        # Re‑evaluate after advancing (could be the final waypoint).
        return controller(t, pose, env)

    # ------------------------------------------------------------------
    # Desired heading toward the current waypoint
    # ------------------------------------------------------------------
    desired_yaw = math.atan2(tgt_y - y, tgt_x - x)
    yaw_error = _norm_angle(desired_yaw - yaw)

    # Base angular command (proportional controller)
    w = max(-_MAX_ANG_SPEED,
            min(_MAX_ANG_SPEED, _KP_ANG * yaw_error))

    # Base linear speed – slow down when we need to turn sharply.
    v = _MAX_SPEED * max(0.0, math.cos(yaw_error))

    # ------------------------------------------------------------------
    # Very light obstacle avoidance using the forward ray‑fan.
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # default: 61 beams, 180°, max 6 m
    # Look at bearings within ±30° of the robot’s forward direction.
    min_dist = None
    min_bearing = 0.0
    for bearing, dist in scan:
        if -math.radians(30) <= bearing <= math.radians(30):
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_bearing = bearing

    if min_dist is not None and min_dist < _SAFE_DIST:
        # Obstacle too close – turn away from it.
        turn_dir = -1.0 if min_bearing > 0 else 1.0   # >0 ⇒ obstacle left ⇒ turn right
        w = turn_dir * _TURN_SPEED
        v = 0.2 * _MAX_SPEED                         # crawl while turning

    # ------------------------------------------------------------------
    # Return the command for this simulation step.
    # ------------------------------------------------------------------
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
