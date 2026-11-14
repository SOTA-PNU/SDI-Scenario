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
# Design summary:
# - Deterministic go‑to‑goal using a simple state‑machine with obstacle avoidance.
# - Only the `math` module and the provided `env.raycast_scan()` sensor are used.
# - The robot switches between "go" mode (heading to goal) and "avoid" mode when
#   an obstacle is too close ahead. In "avoid" mode it turns toward the side with
#   more free space while moving slowly.
# - Repulsive influence from obstacles within `repulsion_range` pushes the desired
#   heading away from them (stronger the nearer the obstacle).
# - Forward speed is limited to 0.5 m/s and angular speed to 0.8 rad/s.
# - No randomness, file I/O, threads, or extra imports are used.

def _wrap_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env, _state={"mode": "go", "turn_dir": 1}):
    """Navigate from START to GOAL while avoiding obstacles.

    Args:
        t: simulation time (seconds) – unused.
        pose: (x, y, yaw) ground‑truth robot pose.
        env: arena harness providing `raycast_scan()`.

    Returns:
        (v, w, done): forward speed [m/s], yaw rate [rad/s], termination flag.
    """
    # ------------------------------------------------------------------
    # Unpack pose and mission parameters
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # Vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)                # world‑frame heading to goal
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    # ------------------------------------------------------------------
    # Success test
    # ------------------------------------------------------------------
    if dist_to_goal <= pos_tol and abs(yaw_error_to_goal) <= yaw_tol:
        _state["mode"] = "go"
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Sensor reading (robust wrapper)
    # ------------------------------------------------------------------
    try:
        scan = env.raycast_scan()                    # [(bearing, distance), ...]
    except Exception:
        return 0.0, 0.0, False

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    max_speed = 0.5               # m/s – forward speed limit (conservative)
    max_angular = 0.8             # rad/s – yaw rate limit
    repulsion_range = 5.0         # m – distance within which obstacles repel
    safety_front = 1.0            # m – safety distance directly ahead
    front_angle_limit = math.radians(30)   # rad – front sector width
    max_range = 6.0               # sensor max range

    # ------------------------------------------------------------------
    # Analyse scan: closest obstacle ahead, side free‑space, repulsion
    # ------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    min_front = max_range
    for bearing, distance in scan:
        if distance >= max_range:
            continue
        # Robot‑frame coordinates of the obstacle point
        obs_x = distance * math.cos(bearing)
        obs_y = distance * math.sin(bearing)
        # Repulsion (inverse distance weighting)
        if distance < repulsion_range:
            w = (repulsion_range - distance) / distance
            rep_x -= w * obs_x
            rep_y -= w * obs_y
        # Closest obstacle in front sector
        if abs(bearing) <= front_angle_limit and distance < min_front:
            min_front = distance

    # ------------------------------------------------------------------
    # State machine handling
    # ------------------------------------------------------------------
    if _state["mode"] == "go":
        # If something is too close ahead, switch to avoidance mode
        if min_front < safety_front:
            # Choose turn direction based on summed free space on each side
            left_sum = sum(d for b, d in scan if b > 0 and d < max_range)
            right_sum = sum(d for b, d in scan if b < 0 and d < max_range)
            # Treat missing returns as completely free
            if left_sum == 0.0:
                left_sum = max_range * len([b for b, d in scan if b > 0])
            if right_sum == 0.0:
                right_sum = max_range * len([b for b, d in scan if b < 0])
            _state["turn_dir"] = 1 if left_sum > right_sum else -1
            _state["mode"] = "avoid"
        # Normal go‑to‑goal behaviour
        # Desired heading = goal bearing + repulsive influence
        goal_bearing = _wrap_angle(goal_heading - yaw)
        goal_vec_x = math.cos(goal_bearing)
        goal_vec_y = math.sin(goal_bearing)
        total_x = goal_vec_x + rep_x
        total_y = goal_vec_y + rep_y
        desired_bearing = math.atan2(total_y, total_x)
        angle_error = _wrap_angle(desired_bearing)
        # Angular command (PD‑like)
        w = max(-max_angular, min(max_angular, 2.0 * angle_error))
        # Speed scaling: slower when turning sharply or when near obstacles
        alignment = max(0.0, math.cos(angle_error))
        proximity = max(0.0, (min_front - safety_front) / (max_range - safety_front))
        if dist_to_goal < 2.0:
            goal_factor = dist_to_goal / 2.0
        else:
            goal_factor = 1.0
        v = max_speed * alignment * proximity * goal_factor
        if v < 0.04:
            v = 0.0
        return v, w, False
    else:  # "avoid" mode
        # Turn in place (or with a small forward component) until front clears
        w = _state["turn_dir"] * max_angular
        v = 0.2  # slow forward motion while turning
        # If front is now clear enough, try to go back to "go" mode
        if min_front >= safety_front * 1.5:
            # Also check that the direct bearing to the goal is not blocked
            goal_bearing = _wrap_angle(goal_heading - yaw)
            # Look for any obstacle within a cone of ±30° around the goal bearing
            cone = math.radians(30)
            blocked = any(
                abs(bearing - goal_bearing) <= cone and distance < safety_front * 1.2
                for bearing, distance in scan if distance < max_range
            )
            if not blocked:
                _state["mode"] = "go"
        return v, w, False
# ========================= [END EDIT REGION] ===============================
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
