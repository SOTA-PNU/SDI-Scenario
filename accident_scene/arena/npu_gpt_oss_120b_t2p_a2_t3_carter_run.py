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
- Goal seeking with proportional steering.
- Obstacle avoidance using raycast scan.
- Speed modulation and safety checks.
- Stuck detection to escape local minima.
"""

import math

# ---- Assumptions (constants) ----
ROBOT_RADIUS = 0.3          # m, assumed circular robot radius
SAFE_MARGIN = 0.2           # m, extra clearance
SAFE_DIST = ROBOT_RADIUS + SAFE_MARGIN   # minimum distance to any obstacle

V_MAX = 0.8                 # m/s, max forward speed
W_MAX = 1.0                 # rad/s, max angular speed

KP_HEADING = 2.0            # proportional gain for heading error
LAMBDA_GOAL = 0.5           # weight for deviation from goal direction in gap scoring
MAX_RANGE = 6.0             # raycast max range (m)

FRONT_ANGLE = math.radians(30)   # rad, front sector half-angle for immediate avoidance
OBSTACLE_THRESHOLD = 0.8         # m, distance to start avoidance

GOAL_SLOW_DIST = 2.0        # m, start slowing down when within this distance to goal

STUCK_DIST_THRESHOLD = 0.02 # m, minimal progress per step to consider moving
STUCK_STEPS = 120           # steps (~2 s) without progress before triggering turn
STUCK_TURN_SPEED = 0.6      # rad/s, turn speed when stuck

# ---- Helper functions ----
def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

# ---- Global state for stuck detection ----
_prev_pose = None   # (x, y) of previous step
_stuck_counter = 0

def controller(t, pose, env):
    """Compute (v, w, done) for the current step.

    t: simulation time (s)
    pose: (x, y, yaw)
    env: harness providing raycast_scan()
    """
    global _prev_pose, _stuck_counter

    x, y, yaw = pose

    # Goal pose from base script
    try:
        goal_x, goal_y, goal_yaw = GOAL
    except Exception:
        return 0.0, 0.0, False

    # Distance and bearing to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    heading_error = wrap_angle(angle_to_goal - yaw)
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

    # Find nearest obstacle overall and in front sector; also collect all beams
    min_dist = float('inf')
    min_bearing = 0.0
    front_min_dist = float('inf')
    front_min_bearing = 0.0
    beams = []   # list of (bearing, distance)
    for bearing, distance in scan:
        if not isinstance(bearing, (int, float)) or not isinstance(distance, (int, float)):
            continue
        # Treat max_range as free space
        if distance >= MAX_RANGE:
            distance = MAX_RANGE
        beams.append((bearing, distance))
        if distance < min_dist:
            min_dist = distance
            min_bearing = bearing
        if abs(bearing) <= FRONT_ANGLE and distance < front_min_dist:
            front_min_dist = distance
            front_min_bearing = bearing

    # ---- Immediate safety: too close to any obstacle ----
    if min_dist < SAFE_DIST:
        v = 0.0
        w = -W_MAX if min_bearing > 0 else W_MAX
        _prev_pose = (x, y)
        _stuck_counter = 0
        return v, w, False

    # ---- Immediate avoidance in front sector ----
    if front_min_dist < OBSTACLE_THRESHOLD:
        # Scale speed down proportional to proximity
        v = V_MAX * (front_min_dist / OBSTACLE_THRESHOLD)
        # Turn away from obstacle: positive bearing => obstacle left => turn right (negative w)
        w = -KP_HEADING * front_min_bearing
        w = max(-W_MAX, min(W_MAX, w))
    else:
        v = V_MAX
        w = 0.0

    # ---- Gap navigation: choose best bearing ----
    best_score = -float('inf')
    best_bearing = heading_error   # fallback
    best_dist = MAX_RANGE
    for bearing, distance in beams:
        deviation = abs(wrap_angle(bearing - heading_error))
        score = distance - LAMBDA_GOAL * deviation
        if score > best_score:
            best_score = score
            best_bearing = bearing
            best_dist = distance

    # Compute angular command towards chosen bearing
    w_gap = KP_HEADING * wrap_angle(best_bearing)
    # Combine with any avoidance command already set
    if front_min_dist < OBSTACLE_THRESHOLD:
        w = w + 0.5 * w_gap
    else:
        w = w_gap
    w = max(-W_MAX, min(W_MAX, w))

    # Speed modulation: reduce speed if heading error large
    alignment = max(0.0, math.cos(heading_error))
    v = v * alignment

    # Slow down near goal
    if dist_to_goal < GOAL_SLOW_DIST:
        v = v * (dist_to_goal / GOAL_SLOW_DIST)

    # ---- Stuck detection ----
    if _prev_pose is not None:
        prev_x, prev_y = _prev_pose
        moved = math.hypot(x - prev_x, y - prev_y)
        if moved < STUCK_DIST_THRESHOLD:
            _stuck_counter += 1
        else:
            _stuck_counter = 0
    else:
        _stuck_counter = 0

    if _stuck_counter >= STUCK_STEPS:
        v = 0.0
        w = STUCK_TURN_SPEED
        _stuck_counter = 0

    _prev_pose = (x, y)

    # Clamp commands
    v = max(0.0, min(V_MAX, v))
    w = max(-W_MAX, min(W_MAX, w))

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
