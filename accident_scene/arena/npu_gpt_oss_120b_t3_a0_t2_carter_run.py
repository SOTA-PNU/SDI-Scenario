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

# Simple Bug‑like controller state
_state = "go_to_goal"   # "go_to_goal" or "avoid_obstacle"
_turn_dir = 1           # 1 = turn left (CCW), -1 = turn right (CW)
_avoid_steps = 0        # counter of steps spent in avoidance mode

def _normalize_angle(angle):
    """Wrap angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding obstacles.

    Returns:
        v (float): forward velocity command [m/s]
        w (float): yaw rate command [rad/s]
        done (bool): True when the robot is at the goal with correct yaw.
    """
    global _state, _turn_dir, _avoid_steps

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # --------------------------------------------------------------
    # 1. Goal check
    # --------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    if distance_to_goal <= MISSION["position_tolerance_m"]:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        # Align yaw inside the goal disc
        Kp_yaw = 1.5
        max_w = 1.0
        w = max(-max_w, min(max_w, Kp_yaw * yaw_err))
        return 0.0, w, False

    # --------------------------------------------------------------
    # 2. Sensor processing – front / left / right distances
    # --------------------------------------------------------------
    scan = env.raycast_scan()
    front_dist = 6.0
    left_dist = 6.0
    right_dist = 6.0
    FRONT_ANGLE = math.radians(30)   # +/- 30°
    for bearing, dist in scan:
        if abs(bearing) <= FRONT_ANGLE:
            if dist < front_dist:
                front_dist = dist
        elif bearing > FRONT_ANGLE:
            if dist < left_dist:
                left_dist = dist
        elif bearing < -FRONT_ANGLE:
            if dist < right_dist:
                right_dist = dist

    # --------------------------------------------------------------
    # 3. Parameters
    # --------------------------------------------------------------
    SAFE_FRONT = 0.8          # m
    MAX_SPEED = 1.0           # m/s
    MAX_ANG_SPEED = 1.0       # rad/s
    Kp_angle = 2.0            # proportional gain for heading
    SIDE_AVOID_GAIN = 0.8     # additional yaw rate when side is too close
    Kp_speed = 0.6            # speed gain (m/s per metre to goal)

    # --------------------------------------------------------------
    # 4. Heading to goal
    # --------------------------------------------------------------
    angle_to_goal = math.atan2(dy, dx)
    angle_error = _normalize_angle(angle_to_goal - yaw)

    # --------------------------------------------------------------
    # 5. State machine
    # --------------------------------------------------------------
    if _state == "go_to_goal":
        # Obstacle directly ahead?
        if front_dist < SAFE_FRONT:
            # Choose turning direction based on which side is clearer
            if left_dist > right_dist:
                _turn_dir = 1   # turn left (CCW)
            else:
                _turn_dir = -1  # turn right (CW)
            _state = "avoid_obstacle"
            _avoid_steps = 0
        else:
            # Normal go‑to‑goal controller
            # Base angular command toward the goal
            w = Kp_angle * angle_error
            # Side‑avoidance correction
            if left_dist < SAFE_FRONT:
                w -= SIDE_AVOID_GAIN * (SAFE_FRONT - left_dist)  # turn right
            if right_dist < SAFE_FRONT:
                w += SIDE_AVOID_GAIN * (SAFE_FRONT - right_dist)  # turn left
            # Clamp angular speed
            w = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, w))

            # Forward speed – slower when far from goal or when turning sharply
            speed_factor = max(0.0, math.cos(angle_error))
            v = min(MAX_SPEED, Kp_speed * distance_to_goal) * speed_factor
            return v, w, False

    # --------------------------------------------------------------
    # 6. Avoidance mode
    # --------------------------------------------------------------
    if _state == "avoid_obstacle":
        _avoid_steps += 1

        # Small forward motion helps slide along obstacles
        v = 0.2

        # Rotate at max speed in the chosen direction
        w = _turn_dir * MAX_ANG_SPEED

        # If the way ahead cleared and we are roughly facing the goal, resume normal navigation
        if front_dist >= SAFE_FRONT and abs(angle_error) < math.radians(20):
            _state = "go_to_goal"
            _avoid_steps = 0

        # If we have been turning for a while without progress, flip direction
        if _avoid_steps > 180:   # ~3 s at 60 Hz
            _turn_dir *= -1
            _avoid_steps = 0

        return v, w, False

    # Fallback (should never be reached)
    return 0.0, 0.0, False


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
