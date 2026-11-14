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

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

"""Controller for ACCIDENT ARENA mission.
Design summary:
- Simple bug-algorithm with state machine: go-to-goal and avoid.
- Uses raycast_scan to detect nearest obstacle.
- If obstacle too close in front sector, rotates away until clear.
- Stuck detection to rotate in place after no progress.
- Speed modulation based on clearance and proximity to goal.
"""

# ---- Assumptions (constants) ----
ROBOT_RADIUS = 0.3          # m, assumed circular robot radius
SAFE_MARGIN = 0.3           # m, extra clearance
SAFE_DIST = ROBOT_RADIUS + SAFE_MARGIN   # minimum distance to any obstacle

V_MAX = 0.8                 # m/s, max forward speed
W_MAX = 1.0                 # rad/s, max angular speed

KP_ANG = 2.0                # proportional gain for angular error (rad/s per rad)
MAX_RANGE = 6.0             # raycast max range (m)

GOAL_SLOW_DIST = 2.0        # start slowing down when within this distance to goal (m)

FRONT_ANGLE = math.radians(45)   # rad, front sector half-angle for avoidance
OBSTACLE_THRESHOLD = SAFE_DIST   # distance to trigger avoidance

STUCK_DIST_THRESH = 0.02    # m, minimal progress per step to consider moving
STUCK_STEPS = 120           # steps (~2 s) without progress before turning
STUCK_TURN_SPEED = 0.6      # rad/s, turn speed when stuck

# ---- Helper functions ----
def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# ---- Global state ----
_state = "go"          # "go" or "avoid"
_avoid_dir = 1          # turn direction during avoidance: 1 = left (positive w), -1 = right
_prev_pose = None      # (x, y) of previous step for stuck detection
_stuck_counter = 0

def controller(t, pose, env):
    """Compute (v, w, done) for the current step.

    t: simulation time (seconds)
    pose: (x, y, yaw) robot pose in world coordinates
    env: harness providing raycast_scan()
    """
    global _state, _avoid_dir, _prev_pose, _stuck_counter

    x, y, yaw = pose

    # Goal pose from base script
    try:
        goal_x, goal_y, goal_yaw = GOAL
    except Exception:
        return 0.0, 0.0, False

    # Compute distance and bearing to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    goal_bearing = wrap_angle(angle_to_goal - yaw)   # relative bearing to goal
    yaw_error = wrap_angle(goal_yaw - yaw)

    # Termination condition
    try:
        pos_tol = MISSION["position_tolerance_m"]
        yaw_tol = MISSION["yaw_tolerance_rad"]
    except Exception:
        pos_tol = 0.75
        yaw_tol = 0.26
    if dist_to_goal <= pos_tol and abs(yaw_error) <= yaw_tol:
        return 0.0, 0.0, True

    # Default commands
    v = V_MAX
    w = 0.0

    # ---- Sensor processing ----
    try:
        scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=MAX_RANGE)
    except Exception:
        scan = []
    if not isinstance(scan, (list, tuple)):
        scan = []

    # Find nearest obstacle and its bearing
    min_dist = float('inf')
    min_bearing = 0.0
    for bearing, distance in scan:
        if not isinstance(bearing, (int, float)) or not isinstance(distance, (int, float)):
            continue
        # Treat max_range as free space
        if distance >= MAX_RANGE:
            distance = MAX_RANGE
        if distance < min_dist:
            min_dist = distance
            min_bearing = bearing

    # ---- State machine ----
    # Check if obstacle is too close in front sector
    obstacle_ahead = (min_dist < OBSTACLE_THRESHOLD) and (abs(min_bearing) < FRONT_ANGLE)

    if _state == "go":
        if obstacle_ahead:
            # Switch to avoidance, decide turn direction away from obstacle
            _state = "avoid"
            _avoid_dir = -1 if min_bearing > 0 else 1  # obstacle on left -> turn right, etc.
    elif _state == "avoid":
        if not obstacle_ahead:
            # Obstacle cleared, go back to go mode
            _state = "go"
        else:
            # Continue rotating away from obstacle
            v = 0.0
            w = _avoid_dir * W_MAX
            # Update previous pose for stuck detection
            _prev_pose = (x, y)
            _stuck_counter = 0
            return v, w, False

    # ---- Goal seeking (when in go mode) ----
    # Angular command towards goal bearing
    w = KP_ANG * goal_bearing
    w = max(-W_MAX, min(W_MAX, w))

    # Speed modulation based on clearance ahead
    # If the nearest obstacle is directly ahead (|min_bearing| < FRONT_ANGLE), reduce speed proportionally
    if abs(min_bearing) < FRONT_ANGLE:
        clearance_factor = max(0.0, (min_dist - SAFE_DIST) / (MAX_RANGE - SAFE_DIST))
        v = V_MAX * clearance_factor
    else:
        v = V_MAX

    # Slow down when approaching goal
    if dist_to_goal < GOAL_SLOW_DIST:
        v = v * (dist_to_goal / GOAL_SLOW_DIST)

    # ---- Stuck detection ----
    if _prev_pose is not None:
        prev_x, prev_y = _prev_pose
        moved = math.hypot(x - prev_x, y - prev_y)
        if moved < STUCK_DIST_THRESH:
            _stuck_counter += 1
        else:
            _stuck_counter = 0
    else:
        _stuck_counter = 0

    if _stuck_counter >= STUCK_STEPS:
        # Force a turn in place to escape
        v = 0.0
        w = STUCK_TURN_SPEED
        _stuck_counter = 0
        _state = "avoid"
        _avoid_dir = 1  # arbitrary turn direction

    # Update previous pose for next step
    _prev_pose = (x, y)

    # Clamp commands
    v = max(0.0, min(V_MAX, v))
    w = max(-W_MAX, min(W_MAX, w))

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
