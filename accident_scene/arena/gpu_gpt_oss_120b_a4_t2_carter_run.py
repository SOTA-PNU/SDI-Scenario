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

def _wrap_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _min_dist_in_range(scan, low, high):
    """
    Return the smallest distance reported by the scan for bearings in
    [low, high].  If no beam falls in the interval, return the max range (6.0).
    """
    best = 6.0
    for bearing, dist in scan:
        if low <= bearing <= high and dist < best:
            best = dist
    return best


def controller(t, pose, env):
    """
    Reactive controller that

    1. Drives toward the goal with a proportional heading controller.
    2. Uses the 180° ray‑scan to avoid obstacles:
       * If something is directly ahead, turn toward the side with more
         clearance.
       * Otherwise adds a gentle repulsive bias proportional to the inverse
         distance of nearby obstacles.
    3. Stops when position and yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable constants (kept deterministic)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0                # m/s
    MAX_TURN = 1.0                 # rad/s
    GOAL_KP = 1.5                  # proportional gain for heading to goal
    AVOID_KP = 2.0                 # gain for obstacle‑avoidance bias
    SAFE_DIST = 1.5                # start reacting when an obstacle is < this
    CRITICAL_DIST = 0.8            # emergency stop distance
    FRONT_ANGLE = 0.2              # rad, half‑width of the “front” sector

    # ----------------------------------------------------------------------
    # Pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    bearing_to_goal = math.atan2(dy, dx)          # world frame
    heading_error = _wrap_angle(bearing_to_goal - yaw)

    # ----------------------------------------------------------------------
    # Termination test
    # ----------------------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached – stop the robot.
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # Sensor processing
    # ----------------------------------------------------------------------
    scan = env.raycast_scan()

    # Distances in three angular sectors
    front_dist = _min_dist_in_range(scan, -FRONT_ANGLE, FRONT_ANGLE)
    left_dist = _min_dist_in_range(scan, FRONT_ANGLE, math.pi)
    right_dist = _min_dist_in_range(scan, -math.pi, -FRONT_ANGLE)

    # ----------------------------------------------------------------------
    # Obstacle avoidance logic
    # ----------------------------------------------------------------------
    # If something is directly ahead, turn toward the clearer side and stop forward motion.
    if front_dist < SAFE_DIST:
        # Choose side with larger clearance.
        turn = MAX_TURN if left_dist > right_dist else -MAX_TURN
        speed = 0.0
    else:
        # Compute a smooth repulsive bias from all nearby obstacles.
        avoidance_bias = 0.0
        min_obstacle_dist = float("inf")
        for bearing, dist in scan:
            if dist >= 6.0:          # no hit on this beam
                continue
            # Track the closest obstacle for possible emergency stop.
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
            if dist < SAFE_DIST:
                # Inverse‑distance repulsion, pushing away from the obstacle.
                repulse = (SAFE_DIST - dist) / SAFE_DIST
                avoidance_bias += -repulse * bearing

        # Combine goal heading and avoidance.
        turn = GOAL_KP * heading_error + AVOID_KP * avoidance_bias
        # Clamp turn rate.
        turn = max(-MAX_TURN, min(MAX_TURN, turn))

        # Speed modulation:
        #   * Slow down when a large turn is required.
        #   * Reduce speed as we approach the goal.
        speed_factor = max(0.0, 1.0 - abs(turn) / (math.pi / 2))
        # Linear ramp near the goal (within 2 m).
        if dist_to_goal < 2.0:
            speed_factor *= dist_to_goal / 2.0
        speed = MAX_SPEED * speed_factor

        # Emergency stop if an obstacle is too close.
        if min_obstacle_dist < CRITICAL_DIST:
            speed = 0.0

    # ----------------------------------------------------------------------
    # Return the command
    # ----------------------------------------------------------------------
    return speed, turn, False


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
