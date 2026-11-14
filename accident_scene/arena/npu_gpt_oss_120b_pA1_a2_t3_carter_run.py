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
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def _wrap_angle(a):
    """Wrap angle into [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Way‑point controller with very light obstacle avoidance.

    * Way‑points: currently only the final GOAL.
    * Moves toward the current waypoint using a proportional controller.
    * If an obstacle is *very* close (< 0.4 m) inside a ±45° forward cone,
      the robot stops forward motion and rotates left (positive w) until the
      path clears.
    * When both position and yaw are within the tolerances the controller
      reports ``done=True``.
    """
    # ------------------------------------------------------------------
    # Persistent state – initialise on first call
    # ------------------------------------------------------------------
    if not hasattr(controller, "waypoints"):
        # Store only (x, y) – yaw is handled separately at the end.
        controller.waypoints = [(GOAL[0], GOAL[1])]
        controller.wp_idx = 0

    # ------------------------------------------------------------------
    # Current waypoint and geometric errors
    # ------------------------------------------------------------------
    wp = controller.waypoints[controller.wp_idx]
    dx = wp[0] - pose[0]
    dy = wp[1] - pose[1]
    dist_to_wp = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    angle_err = _wrap_angle(target_angle - pose[2])

    # ------------------------------------------------------------------
    # Goal‑reached test (position + yaw)
    # ------------------------------------------------------------------
    if (dist_to_wp <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(pose[2] - GOAL[2])) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Very simple obstacle detection
    # ------------------------------------------------------------------
    # Scan with default parameters (61 beams, 180° FOV).  Bearing = 0 points
    # straight ahead.  We look at a ±45° cone and treat anything closer than
    # 0.4 m as a blocking obstacle.
    scan = env.raycast_scan()
    cone = math.radians(45)          # 45 deg half‑cone
    min_ahead = None
    for bearing, distance in scan:
        if abs(bearing) <= cone:
            if min_ahead is None or distance < min_ahead:
                min_ahead = distance

    # If something is dangerously close, stop forward motion and turn left.
    if min_ahead is not None and min_ahead < 0.4:
        # Fixed turn rate – enough to steer away without overshooting.
        return 0.0, 1.0, False

    # ------------------------------------------------------------------
    # Nominal motion command
    # ------------------------------------------------------------------
    # Speed profile: max 1.2 m/s, linearly reduced when we are within 2 m
    # of the waypoint to avoid overshoot.
    max_speed = 1.2
    speed = max_speed * min(1.0, dist_to_wp / 2.0)

    # Angular velocity: proportional to heading error, limited to ±1.0 rad/s.
    max_w = 1.0
    angular = max(-max_w, min(max_w, 2.0 * angle_err))

    # If we are not roughly facing the waypoint, prioritize turning.
    if abs(angle_err) > 0.5:   # ~30°
        speed = 0.0

    return speed, angular, False


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
