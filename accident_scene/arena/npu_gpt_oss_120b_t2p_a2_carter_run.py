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

"""Controller for ACCIDENT ARENA mission.
Design summary:
- Goal seeking: compute heading to GOAL and apply proportional steering.
- Obstacle avoidance: use raycast scan (61 beams over 180°) to detect nearest
  obstacle in a front sector and steer away proportionally.
- Speed modulation: limit forward speed based on distance to nearest obstacle,
  heading error, and distance to goal.
- Safety: enforce max speed limits, maintain a safety margin from obstacles,
  and stop if too close.
- Termination: when within position and yaw tolerances, set done=True.

Assumptions (constants) are documented inline.
"""

import math

# ---------- Constants (assumptions) ----------
# Robot geometry and safety margin (meters)
ROBOT_RADIUS = 0.3  # assumed circular robot radius
SAFE_MARGIN = 0.2   # additional clearance beyond robot radius
SAFE_DIST = ROBOT_RADIUS + SAFE_MARGIN  # minimum distance to keep from obstacles

# Motion limits
V_MAX = 1.0   # max forward speed (m/s)
W_MAX = 1.0   # max angular speed (rad/s)

# Control gains
KP_HEADING = 2.0   # proportional gain for heading error
KP_AVOID = 1.5     # gain for obstacle avoidance steering

# Obstacle avoidance parameters
FRONT_ANGLE = math.radians(30)   # front sector half-angle for avoidance (rad)
OBSTACLE_THRESHOLD = 1.0         # distance at which avoidance starts (m)

# Goal approach parameters
GOAL_SLOW_DIST = 2.0   # start slowing down within this distance to goal (m)

# Helper: wrap angle to [-pi, pi]
def wrap_angle(a):
    """Wrap angle a to the range [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# The mission constants are defined in the base file and will be imported
# by the harness when this controller is executed. We reference them via
# the global namespace at runtime (START, GOAL, etc.).

def controller(t, pose, env):
    """Return (v, w, done) for the current time step.

    Args:
        t (float): simulation time (seconds) since mission start.
        pose (tuple): (x, y, yaw) robot pose in world coordinates.
        env: harness environment providing raycast_scan().
    Returns:
        v (float): forward linear velocity command [m/s].
        w (float): angular velocity command [rad/s].
        done (bool): True when mission should terminate.
    """
    # Unpack pose
    x, y, yaw = pose

    # Goal pose (provided by base script as GOAL)
    try:
        goal_x, goal_y, goal_yaw = GOAL
    except Exception:
        # If GOAL is not available, fail safe: stop.
        return 0.0, 0.0, False

    # Compute distance and bearing to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    heading_error = wrap_angle(angle_to_goal - yaw)
    yaw_error = wrap_angle(goal_yaw - yaw)

    # Check termination condition (position & yaw tolerance)
    # Tolerances are defined in the base MISSION dict.
    try:
        pos_tol = MISSION["position_tolerance_m"]
        yaw_tol = MISSION["yaw_tolerance_rad"]
    except Exception:
        pos_tol = 0.75
        yaw_tol = 0.26
    if dist_to_goal <= pos_tol and abs(yaw_error) <= yaw_tol:
        # Goal reached and aligned
        return 0.0, 0.0, True

    # Default commands
    v = V_MAX
    w = 0.0

    # ----- Sensor processing -----
    # Get raycast scan; be robust to unexpected failures.
    try:
        scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    except Exception:
        scan = []
    # Ensure scan is iterable of (bearing, distance)
    if not isinstance(scan, (list, tuple)):
        scan = []

    # Find nearest obstacle in the full scan and within front sector.
    min_dist = float('inf')
    min_front_dist = float('inf')
    bearing_of_closest = 0.0
    bearing_of_front_closest = 0.0
    for bearing, distance in scan:
        # Skip invalid entries
        if not isinstance(bearing, (int, float)) or not isinstance(distance, (int, float)):
            continue
        # Treat max_range as no obstacle
        if distance >= 6.0:
            continue
        # Update overall nearest obstacle
        if distance < min_dist:
            min_dist = distance
            bearing_of_closest = bearing
        # Check front sector for avoidance
        if abs(bearing) <= FRONT_ANGLE and distance < min_front_dist:
            min_front_dist = distance
            bearing_of_front_closest = bearing

    # ----- Obstacle avoidance -----
    # If an obstacle is within the avoidance threshold in front, generate a
    # steering command away from it.
    w_avoid = 0.0
    if min_front_dist < OBSTACLE_THRESHOLD:
        # Steering direction: negative bearing => obstacle on right => turn left (positive w)
        # Positive bearing => obstacle on left => turn right (negative w)
        # Scale by how close the obstacle is.
        proximity = (OBSTACLE_THRESHOLD - min_front_dist) / OBSTACLE_THRESHOLD
        w_avoid = -KP_AVOID * bearing_of_front_closest * proximity
        # Reduce forward speed proportionally to proximity.
        v = max(0.0, V_MAX * (1.0 - proximity))
    else:
        # No immediate obstacle; keep full speed but still respect overall nearest.
        v = V_MAX

    # ----- Goal heading control -----
    w_goal = KP_HEADING * heading_error

    # Combine angular commands
    w = w_goal + w_avoid
    # Clamp angular speed
    if w > W_MAX:
        w = W_MAX
    elif w < -W_MAX:
        w = -W_MAX

    # ----- Speed modulation based on heading error -----
    # Reduce speed if we are not well aligned with the goal direction.
    alignment_factor = max(0.0, math.cos(heading_error))  # 1 when aligned, 0 when orthogonal
    v = v * alignment_factor

    # ----- Speed modulation based on distance to goal -----
    if dist_to_goal < GOAL_SLOW_DIST:
        # Linear slowdown as we approach the goal
        v = v * (dist_to_goal / GOAL_SLOW_DIST)

    # ----- Safety: enforce minimum distance from obstacles -----
    if min_dist < SAFE_DIST:
        # Too close to any obstacle: stop and try to turn away.
        v = 0.0
        # If we have a bearing to the nearest obstacle, turn away aggressively.
        if min_dist < 1e-3:
            # Extremely close: spin in place.
            w = W_MAX if bearing_of_closest < 0 else -W_MAX
        else:
            w = -KP_AVOID * bearing_of_closest
        # Clamp angular speed again
        if w > W_MAX:
            w = W_MAX
        elif w < -W_MAX:
            w = -W_MAX

    # Ensure forward speed is within limits
    if v > V_MAX:
        v = V_MAX
    elif v < 0.0:
        v = 0.0

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
