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

def _normalize_angle(angle):
    """Wrap angle to the interval [-π, π]."""
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
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # 1.  Goal‑related geometry
    # ------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    angle_error = _normalize_angle(angle_to_goal - yaw)

    # ------------------------------------------------------------------
    # 2.  Goal termination condition
    # ------------------------------------------------------------------
    if distance_to_goal <= MISSION["position_tolerance_m"]:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            # Goal reached and correctly oriented
            return 0.0, 0.0, True
        # Inside the goal disc but still need to align yaw
        Kp_yaw = 1.5
        max_w = 1.0
        w = max(-max_w, min(max_w, Kp_yaw * yaw_err))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 3.  Sensor reading
    # ------------------------------------------------------------------
    scan = env.raycast_scan()      # default: 61 beams, 180°, max_range=6.0

    # ------------------------------------------------------------------
    # 4.  Simple sector analysis (front / left / right)
    # ------------------------------------------------------------------
    SAFE_FRONT = 0.6               # meters – distance that is considered unsafe ahead
    front_dist = 6.0
    left_dist = 6.0
    right_dist = 6.0
    for bearing, dist in scan:
        if abs(bearing) <= math.radians(30):
            if dist < front_dist:
                front_dist = dist
        elif bearing > math.radians(30):
            if dist < left_dist:
                left_dist = dist
        elif bearing < -math.radians(30):
            if dist < right_dist:
                right_dist = dist

    # ------------------------------------------------------------------
    # 5.  Immediate obstacle avoidance (if something is too close ahead)
    # ------------------------------------------------------------------
    MAX_ANG_SPEED = 1.0           # rad/s
    if front_dist < SAFE_FRONT:
        # If we are trapped (both sides also blocked), back up a little
        if left_dist < SAFE_FRONT and right_dist < SAFE_FRONT:
            return -0.2, 0.0, False   # reverse slowly
        # Otherwise rotate toward the side that offers more clearance
        if left_dist > right_dist:
            w = MAX_ANG_SPEED
        else:
            w = -MAX_ANG_SPEED
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 6.  Potential‑field navigation (attraction + repulsion)
    # ------------------------------------------------------------------
    # Parameters
    MAX_SPEED = 1.0               # m/s
    Kp_angle = 1.5                # proportional gain for heading
    Kp_speed = 0.8                # speed gain (m/s per metre to goal)
    REPULSION_RANGE = 1.5         # metres – only obstacles nearer than this repel
    K_att = 1.0
    K_rep = 0.8

    # Attractive vector (unit‑length towards the goal, scaled by K_att)
    att_x = K_att * (dx / distance_to_goal)
    att_y = K_att * (dy / distance_to_goal)

    # Repulsive contribution from nearby obstacles
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scan:
        if dist < REPULSION_RANGE and dist > 1e-6:
            # Global angle to the obstacle point
            obs_angle = yaw + bearing
            # Gradient of a 1/d potential (stronger when nearer)
            mag = K_rep * (1.0 / dist - 1.0 / REPULSION_RANGE) / (dist * dist)
            rep_x -= mag * math.cos(obs_angle)
            rep_y -= mag * math.sin(obs_angle)

    # Combine vectors
    tot_x = att_x + rep_x
    tot_y = att_y + rep_y

    # Desired heading derived from the combined vector
    desired_heading = math.atan2(tot_y, tot_x)
    angle_error = _normalize_angle(desired_heading - yaw)

    # Angular velocity (clamped)
    w = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, Kp_angle * angle_error))

    # Forward speed – scaled by distance to goal and by how well we are aligned
    alignment = max(0.0, math.cos(angle_error))
    speed = min(MAX_SPEED, Kp_speed * distance_to_goal) * alignment
    v = speed

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
