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
    pose  (x, y, yaw) ground‑truth world pose of the chassis
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

def _norm_angle(a):
    """Wrap an angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Global state (deterministic)
# ---------------------------------------------------------------------------
_mode = "go"      # "go" → head toward goal, "avoid" → obstacle‑avoidance turn
_avoid_dir = 0    # -1 = turn left, +1 = turn right (valid only in "avoid" mode)

# ---------------------------------------------------------------------------
# Tuning parameters (feel free to tweak if needed)
# ---------------------------------------------------------------------------
V_MAX = 0.8               # max forward speed [m/s]
W_MAX = 0.9               # max yaw rate   [rad/s]
KP_ANGLE = 1.5            # proportional gain for heading error
MAX_RANGE = 6.0           # laser max range (matches env.raycast_scan)

SAFE_FRONT = 0.8          # distance < this in front → enter avoidance
MIN_FRONT_DIST = 0.5      # distance < this → stop forward motion completely
OCCUPIED_THRESH = 0.8     # distance < this at the goal bearing → treat as blocked
GOAL_SLOW_DIST = 5.0      # start slowing down when this close to goal [m]
# ---------------------------------------------------------------------------

def controller(t, pose, env):
    """
    Simple state‑machine controller:

    * “go”   – steer toward the goal while respecting a narrow front‑obstacle
               safety band.
    * “avoid” – turn (left or right) around a blocking obstacle until the
               forward corridor is clear again, then resume “go”.
    """
    global _mode, _avoid_dir

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -------------------------------------------------------------------
    # Goal geometry
    # -------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_world_bearing = math.atan2(dy, dx)          # bearing in world frame
    goal_rel_bearing = _norm_angle(goal_world_bearing - yaw)

    # -------------------------------------------------------------------
    # Completion test
    # -------------------------------------------------------------------
    yaw_err_to_goal_yaw = _norm_angle(goal_yaw - yaw)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal_yaw) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached – stop and signal success
        return 0.0, 0.0, True

    # Inside position tolerance but yaw not yet aligned → rotate in place
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        w = max(min(KP_ANGLE * yaw_err_to_goal_yaw, W_MAX), -W_MAX)
        return 0.0, w, False

    # -------------------------------------------------------------------
    # Sensor processing
    # -------------------------------------------------------------------
    scan = env.raycast_scan()          # list of (bearing, distance) tuples

    # Front distance (narrow cone ≈ ±8.6°)
    front_distances = [dist for bearing, dist in scan if abs(bearing) < 0.15]
    front_dist = min(front_distances) if front_distances else MAX_RANGE

    # Side distances – used to decide which way to turn around an obstacle
    left_distances  = [dist for bearing, dist in scan if 0.3 < bearing < 1.2]   # ~+20° … +70°
    right_distances = [dist for bearing, dist in scan if -1.2 < bearing < -0.3] # ~‑70° … ‑20°
    left_dist  = min(left_distances)  if left_distances  else MAX_RANGE
    right_dist = min(right_distances) if right_distances else MAX_RANGE

    # Distance in the exact goal direction (closest beam to goal bearing)
    goal_beam_dist = MAX_RANGE
    min_angle_diff = float('inf')
    for bearing, dist in scan:
        diff = abs(_norm_angle(bearing - goal_rel_bearing))
        if diff < min_angle_diff:
            min_angle_diff = diff
            goal_beam_dist = dist

    # -------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------
    if _mode == "go":
        # ---------------------------------------------------------------
        # Check for a blocking obstacle directly ahead or on the goal line
        # ---------------------------------------------------------------
        if front_dist < SAFE_FRONT or goal_beam_dist < OCCUPIED_THRESH:
            # Switch to avoidance mode
            _mode = "avoid"
            _avoid_dir = -1 if left_dist > right_dist else 1   # -1 = left, +1 = right
            w = _avoid_dir * W_MAX
            v = V_MAX * 0.2          # move slowly while turning
            return v, w, False

        # ---------------------------------------------------------------
        # Normal “go‑to‑goal” behavior
        # ---------------------------------------------------------------
        heading_error = goal_rel_bearing
        w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

        # Speed scaling factors
        alignment = max(0.0, math.cos(heading_error))          # how well we point to goal
        front_scale = min(1.0, front_dist / SAFE_FRONT)       # more clearance → faster
        goal_scale = 1.0
        if dist_to_goal < GOAL_SLOW_DIST:
            goal_scale = dist_to_goal / GOAL_SLOW_DIST

        v = V_MAX * alignment * front_scale * goal_scale

        # Stop forward motion if something is extremely close ahead
        if front_dist < MIN_FRONT_DIST:
            v = 0.0

        v = max(v, 0.0)
        return v, w, False

    else:   # _mode == "avoid"
        # ---------------------------------------------------------------
        # Continue turning around the obstacle until the forward corridor
        # is clear *and* the direct line to the goal is not blocked.
        # ---------------------------------------------------------------
        if front_dist >= SAFE_FRONT and goal_beam_dist >= OCCUPIED_THRESH:
            # Obstacle cleared – resume normal navigation
            _mode = "go"
            # Re‑compute a fresh heading error (goal may have moved relative to yaw)
            heading_error = goal_rel_bearing
            w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

            alignment = max(0.0, math.cos(heading_error))
            front_scale = min(1.0, front_dist / SAFE_FRONT)
            goal_scale = 1.0
            if dist_to_goal < GOAL_SLOW_DIST:
                goal_scale = dist_to_goal / GOAL_SLOW_DIST

            v = V_MAX * alignment * front_scale * goal_scale
            if front_dist < MIN_FRONT_DIST:
                v = 0.0
            v = max(v, 0.0)
            return v, w, False
        else:
            # Keep rotating (slow forward motion helps “slide” along the obstacle)
            w = _avoid_dir * W_MAX
            v = V_MAX * 0.2
            if front_dist < MIN_FRONT_DIST:
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
