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

def _norm_angle(a):
    """Wrap angle to the interval [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

# Tuning parameters (deterministic)
V_MAX = 0.6               # max forward speed [m/s]
W_MAX = 0.8               # max yaw rate [rad/s]
KP_ANGLE = 2.0            # proportional gain for heading error
MAX_RANGE = 6.0           # laser max range (matches env.raycast_scan)
FREE_DIST = 0.8           # distance >= this considered free (m)
SAFE_FRONT = 0.8          # if obstacle closer than this ahead, treat as blocked
MIN_FRONT_DIST = 0.5      # distance < this => stop forward motion
GOAL_SLOW_DIST = 5.0      # start slowing down when this close to goal (m)

def controller(t, pose, env):
    """
    Gap-following controller with goal bias.
    """
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Goal geometry
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_world_bearing = math.atan2(dy, dx)
    goal_rel_bearing = _norm_angle(goal_world_bearing - yaw)

    # Completion test
    yaw_err_to_goal_yaw = _norm_angle(goal_yaw - yaw)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal_yaw) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # Inside position tolerance but yaw not yet aligned
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        w = max(min(KP_ANGLE * yaw_err_to_goal_yaw, W_MAX), -W_MAX)
        return 0.0, w, False

    # Laser scan
    scan = env.raycast_scan()   # [(bearing, distance), ...]

    # Front distance (narrow cone +/- 0.1 rad)
    front_distances = [dist for bearing, dist in scan if abs(bearing) < 0.1]
    front_dist = min(front_distances) if front_distances else MAX_RANGE

    # Determine free beams
    free_flags = [dist >= FREE_DIST for (_, dist) in scan]

    # Find free intervals (consecutive free beams)
    intervals = []
    start = None
    for i, flag in enumerate(free_flags):
        if flag:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
    if start is not None:
        intervals.append((start, len(free_flags) - 1))

    # If no free interval, rotate toward goal direction
    if not intervals:
        w = max(min(KP_ANGLE * goal_rel_bearing, W_MAX), -W_MAX)
        return 0.0, w, False

    # Choose interval whose centre is closest to goal bearing, prefer wider gaps
    best_center = None
    best_error = float('inf')
    best_width = -1
    for s, e in intervals:
        b_start, _ = scan[s]
        b_end, _ = scan[e]
        centre = _norm_angle((b_start + b_end) / 2.0)
        error = abs(_norm_angle(centre - goal_rel_bearing))
        width = e - s + 1
        if error < best_error or (abs(error - best_error) < 1e-6 and width > best_width):
            best_error = error
            best_width = width
            best_center = centre

    desired_bearing = best_center          # relative bearing we want to follow
    heading_error = _norm_angle(desired_bearing)

    # Angular velocity command
    w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

    # Forward speed command
    alignment = max(0.0, math.cos(heading_error))          # how well we point toward chosen gap
    front_factor = min(1.0, front_dist / SAFE_FRONT)      # slower when something ahead
    goal_factor = 1.0
    if dist_to_goal < GOAL_SLOW_DIST:
        goal_factor = dist_to_goal / GOAL_SLOW_DIST

    v = V_MAX * alignment * front_factor * goal_factor

    # Stop if something is extremely close ahead
    if front_dist < MIN_FRONT_DIST:
        v = 0.0

    v = max(v, 0.0)

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
