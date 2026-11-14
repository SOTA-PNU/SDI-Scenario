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


def _norm_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple waypoint‑following controller with reactive obstacle avoidance.

    - Waypoint list currently contains only the final GOAL.
    - Moves toward the current waypoint using a proportional controller.
    - If an obstacle is detected within a safety cone ahead, the robot turns
      left (positive w) and stops forward motion until the path clears.
    - When the robot is within position and yaw tolerances of the goal it
      reports `done=True`.
    """
    # ----------------------------------------------------------------------
    # Persistent state (initialized on first call)
    # ----------------------------------------------------------------------
    if not hasattr(controller, "waypoints"):
        # Waypoints are (x, y) tuples.  Here we just head straight for GOAL.
        controller.waypoints = [(GOAL[0], GOAL[1])]
        controller.wp_idx = 0

    # ----------------------------------------------------------------------
    # Helper: distance & bearing to current waypoint
    # ----------------------------------------------------------------------
    wp = controller.waypoints[controller.wp_idx]
    dx = wp[0] - pose[0]
    dy = wp[1] - pose[1]
    distance = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    angle_error = _norm_angle(target_angle - pose[2])

    # ----------------------------------------------------------------------
    # Check if we have arrived at the goal (position & yaw tolerances)
    # ----------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if distance <= pos_tol and abs(_norm_angle(pose[2] - GOAL[2])) <= yaw_tol:
        # Goal reached – stop and signal completion.
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # Obstacle avoidance using raycast
    # ----------------------------------------------------------------------
    # Scan parameters: 61 beams over 180°, bearing = 0 points straight ahead.
    # We consider a forward safety cone of ±30° (≈0.523 rad).
    scan = env.raycast_scan()
    safety_cone = 0.523  # radians
    min_dist_ahead = None
    for bearing, dist in scan:
        if abs(bearing) <= safety_cone:
            if min_dist_ahead is None or dist < min_dist_ahead:
                min_dist_ahead = dist

    # If something is closer than 0.8 m in front, perform a simple avoidance:
    obstacle_threshold = 0.8  # meters
    if min_dist_ahead is not None and min_dist_ahead < obstacle_threshold:
        # Turn left (positive angular velocity) while stopping forward motion.
        # The turn rate is capped to keep motion smooth.
        return 0.0, 1.0, False

    # ----------------------------------------------------------------------
    # Nominal motion towards waypoint
    # ----------------------------------------------------------------------
    # Speed profile: max 1.0 m/s, slower when close to the target.
    max_speed = 1.0
    speed = max_speed * min(1.0, distance / 2.0)  # linear slowdown within 2 m

    # Angular velocity: proportional to angle error, limited to ±1.0 rad/s.
    max_angular = 1.0
    angular = max(-max_angular, min(max_angular, 2.0 * angle_error))

    # If the heading error is large, prioritize turning over forward motion.
    if abs(angle_error) > 0.5:
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
