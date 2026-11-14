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
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Simple go‑to‑goal with reactive obstacle avoidance.

    The controller runs at 60 Hz.  It uses the forward 180° raycast sensor
    (61 beams) to compute a repulsive field from nearby obstacles and an
    attractive field toward the goal.  The resulting vector determines a
    desired heading.  A proportional controller drives the robot toward that
    heading while respecting a safe distance in front of the robot.

    When the robot is within the position tolerance of the goal it stops
    forward motion and rotates in place until the yaw is within tolerance,
    then reports ``done=True``.
    """
    # Unpack current pose
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------
    # 1.  Goal geometry
    # ------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)            # world frame
    angle_error = _normalize_angle(target_angle - yaw)  # robot frame

    # ------------------------------------------------------------
    # 2.  Check for goal completion
    # ------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    if distance_to_goal <= pos_tol:
        # Within the goal disc – only need to align yaw.
        yaw_error = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_error) <= yaw_tol:
            # Goal satisfied
            return 0.0, 0.0, True
        # Rotate in place to achieve the required yaw.
        max_angular = 1.0          # rad/s
        kp_yaw = 2.0
        w = max_angular * math.tanh(kp_yaw * yaw_error)
        return 0.0, w, False

    # ------------------------------------------------------------
    # 3.  Sensor reading
    # ------------------------------------------------------------
    scans = env.raycast_scan()   # list of (bearing, distance)

    # ------------------------------------------------------------
    # 4.  Reactive obstacle avoidance (potential‑field style)
    # ------------------------------------------------------------
    # Parameters
    repulse_range = 2.0          # meters – range within which obstacles repel
    k_repulse = 1.0              # weight of repulsive component
    max_speed = 1.0              # m/s – nominal forward speed
    max_angular = 1.0            # rad/s – angular speed limit
    kp_angle = 2.0               # proportional gain for heading error
    safe_front_dist = 0.8        # meters – stop if something is closer

    # Attractive vector (unit length) in robot frame
    att_x = math.cos(angle_error)
    att_y = math.sin(angle_error)

    # Repulsive vector (initialized to zero)
    rep_x = 0.0
    rep_y = 0.0

    # Scan each beam
    for bearing, dist in scans:
        # Treat max_range as “no obstacle”
        if dist >= 6.0:
            continue
        # Only consider obstacles inside the repulsive range
        if dist < repulse_range:
            # Strength grows linearly as the obstacle gets closer
            strength = (repulse_range - dist) / repulse_range
            # Repulsion points opposite to the bearing (away from obstacle)
            rep_x -= strength * math.cos(bearing)
            rep_y -= strength * math.sin(bearing)

    # Combine attractive and repulsive components
    vec_x = att_x + k_repulse * rep_x
    vec_y = att_y + k_repulse * rep_y

    # Desired heading in robot frame (relative to current forward direction)
    desired_heading = math.atan2(vec_y, vec_x)

    # ------------------------------------------------------------
    # 5.  Compute control commands
    # ------------------------------------------------------------
    # Angular velocity – proportional to heading error, saturated
    w = max_angular * math.tanh(kp_angle * desired_heading)

    # Forward speed – full speed when heading is roughly aligned,
    # reduced when we need to turn sharply or when something is close ahead.
    # Determine if the front sector (±30°) is blocked.
    front_blocked = False
    min_front_dist = 6.0
    for bearing, dist in scans:
        if -math.pi / 6 <= bearing <= math.pi / 6:   # ±30°
            if dist < min_front_dist:
                min_front_dist = dist
            if dist < safe_front_dist:
                front_blocked = True

    if front_blocked:
        v = 0.0
    else:
        # Speed scales with the cosine of the heading error (i.e. slower when turning)
        speed_factor = max(0.0, math.cos(desired_heading))
        # Reduce speed as we approach the goal to avoid overshoot
        approach_factor = min(1.0, distance_to_goal / 2.0)
        v = max_speed * speed_factor * approach_factor

    # ------------------------------------------------------------
    # 6.  Return command
    # ------------------------------------------------------------
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
