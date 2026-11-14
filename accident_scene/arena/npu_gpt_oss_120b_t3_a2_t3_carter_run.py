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

# ---------------------------------------------------------------------------
# Helper and global state
# ---------------------------------------------------------------------------
def _norm_angle(angle):
    """Normalize an angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

# ---------------------------------------------------------------------------
# Controller parameters (deterministic)
# ---------------------------------------------------------------------------
V_MAX = 0.6            # maximum forward speed [m/s]
W_MAX = 0.8            # maximum yaw rate [rad/s]
KP_ANGLE = 1.5         # proportional gain for heading error
SAFE_FRONT = 1.0       # distance that must be clear ahead before moving forward [m]
SAFE_SIDE = 0.6        # minimum side clearance to allow forward motion [m]
DANGER_ANGLE = 0.2     # angular window around forward direction considered “dangerous” [rad]
DANGER_DIST = 5.0      # distance within the danger window that triggers avoidance [m]
SAFE_MIN_DIST = 0.9    # minimum distance for a bearing to be considered safe for navigation [m]
MAX_RANGE = 6.0        # sensor maximum range (as defined by env.raycast_scan)

# State variables (persist across calls)
_avoid_mode = False    # whether we are currently executing an avoidance turn
_avoid_dir = 0         # -1 for left turn, +1 for right turn

# ---------------------------------------------------------------------------
def controller(t, pose, env):
    """Reactive navigation controller with simple obstacle avoidance.

    Returns:
        v (float): forward velocity command [m/s]
        w (float): yaw rate command [rad/s]
        done (bool): True when the robot has satisfied the goal conditions.
    """
    global _avoid_mode, _avoid_dir

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -----------------------------------------------------------------------
    # Goal geometry
    # -----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Goal bearing in world frame and relative to robot heading
    goal_bearing_world = math.atan2(dy, dx)
    goal_bearing_rel = _norm_angle(goal_bearing_world - yaw)

    # -----------------------------------------------------------------------
    # Completion test
    # -----------------------------------------------------------------------
    yaw_err_to_goal_yaw = _norm_angle(goal_yaw - yaw)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal_yaw) <= MISSION["yaw_tolerance_rad"]):
        # Within both tolerances → stop and signal success
        return 0.0, 0.0, True

    # If we are already inside the position tolerance but not yet oriented,
    # rotate in place until the yaw aligns.
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        w = max(min(KP_ANGLE * yaw_err_to_goal_yaw, W_MAX), -W_MAX)
        return 0.0, w, False

    # -----------------------------------------------------------------------
    # Sensor processing
    # -----------------------------------------------------------------------
    scan = env.raycast_scan()  # list of (bearing, distance) tuples

    # Distances in three sectors: front, left, right
    front_distances = [dist for bearing, dist in scan if abs(bearing) <= DANGER_ANGLE]
    front_dist = min(front_distances) if front_distances else MAX_RANGE

    left_distances = [dist for bearing, dist in scan if bearing > DANGER_ANGLE]
    right_distances = [dist for bearing, dist in scan if bearing < -DANGER_ANGLE]
    left_dist = min(left_distances) if left_distances else MAX_RANGE
    right_dist = min(right_distances) if right_distances else MAX_RANGE

    # -----------------------------------------------------------------------
    # Obstacle avoidance logic (stateful)
    # -----------------------------------------------------------------------
    # Detect a blocking obstacle directly ahead (danger window)
    front_blocked = (front_dist < DANGER_DIST)

    if _avoid_mode:
        # Continue turning until the front is clear and the goal direction is safe
        if not front_blocked and front_dist > SAFE_FRONT:
            # Check if the bearing toward the goal is safe enough
            goal_beam_dist = min(
                [dist for bearing, dist in scan
                 if abs(_norm_angle(bearing - goal_bearing_rel)) < 0.05] or [MAX_RANGE]
            )
            if goal_beam_dist > SAFE_MIN_DIST:
                _avoid_mode = False  # exit avoidance mode
        # If we are still in avoidance mode, keep turning
        if _avoid_mode:
            w = _avoid_dir * W_MAX
            v = 0.0
            return v, w, False

    if not _avoid_mode and front_blocked:
        # Enter avoidance mode: decide turn direction based on side clearance
        _avoid_mode = True
        _avoid_dir = -1 if left_dist > right_dist else 1   # -1 = turn left, 1 = turn right
        w = _avoid_dir * W_MAX
        v = 0.0
        return v, w, False

    # -----------------------------------------------------------------------
    # Normal navigation (goal‑directed) using a simple gap selection
    # -----------------------------------------------------------------------
    # Filter bearings that are at least SAFE_MIN_DIST away
    safe_beams = [(b, d) for (b, d) in scan if d >= SAFE_MIN_DIST]

    if not safe_beams:
        # No safe direction found – rotate slowly to search
        w = W_MAX if left_dist > right_dist else -W_MAX
        return 0.0, w, False

    # Sort by (angular deviation from goal, -distance) to prefer alignment and clearance
    safe_beams.sort(key=lambda bd: (abs(_norm_angle(bd[0] - goal_bearing_rel)), -bd[1]))
    best_bearing, best_dist = safe_beams[0]

    # Desired heading error (relative bearing)
    heading_error = _norm_angle(best_bearing)

    # Angular velocity command (clamped)
    w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

    # Forward speed command
    # Scale by alignment with the chosen heading and by distance to obstacle
    alignment = max(0.0, math.cos(heading_error))
    dist_factor = min(1.0, best_dist / MAX_RANGE)
    v = V_MAX * alignment * dist_factor

    # Additional safety: stop forward motion if side clearance is insufficient
    if left_dist < SAFE_SIDE or right_dist < SAFE_SIDE:
        v = 0.0

    # Ensure non‑negative speed
    v = max(v, 0.0)

    # Continue the mission (not yet done)
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
