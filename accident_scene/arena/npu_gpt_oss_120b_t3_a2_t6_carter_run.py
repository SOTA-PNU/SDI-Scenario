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
    """Wrap an angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Deterministic tuning parameters (feel free to tweak if needed)
# ---------------------------------------------------------------------------
V_MAX = 0.7               # maximum forward speed [m/s]
W_MAX = 1.0               # maximum yaw rate   [rad/s]
KP_ANGLE = 2.0            # proportional gain for heading error
MAX_RANGE = 6.0           # laser max range (matches env.raycast_scan)

# Repulsive‑field parameters
REPULSE_RANGE = 5.0       # consider obstacles within this distance [m]
REPULSE_GAIN = 8.0        # strength of repulsion

# Attractive‑field parameter
ATTRACTION_GAIN = 1.0     # strength of attraction toward the goal

# Safety thresholds
SAFE_FRONT = 1.5          # if something nearer than this straight ahead → avoidance turn
MIN_FRONT_DIST = 0.7      # distance at which forward motion is completely stopped
GOAL_SLOW_DIST = 4.0      # start slowing down when this close to goal [m]
# ---------------------------------------------------------------------------

def controller(t, pose, env):
    """
    Reactive navigation controller.

    * Uses a simple potential‑field (attractive + repulsive) to pick a heading.
    * If an obstacle is too close directly ahead, it turns toward the side with
      the larger clearance.
    * Slows down when approaching the goal and respects a minimum front clearance.
    """
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -----------------------------------------------------------------------
    # Goal geometry
    # -----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Bearing to goal expressed in world frame and relative to robot heading
    goal_world_bearing = math.atan2(dy, dx)
    goal_rel_bearing = _norm_angle(goal_world_bearing - yaw)

    # -----------------------------------------------------------------------
    # Completion test
    # -----------------------------------------------------------------------
    yaw_err_to_goal_yaw = _norm_angle(goal_yaw - yaw)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal_yaw) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached – stop and signal success
        return 0.0, 0.0, True

    # Inside position tolerance but yaw not yet aligned → rotate in place
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        w = max(min(KP_ANGLE * yaw_err_to_goal_yaw, W_MAX), -W_MAX)
        return 0.0, w, False

    # -----------------------------------------------------------------------
    # Laser scan
    # -----------------------------------------------------------------------
    scan = env.raycast_scan()          # list of (bearing, distance) tuples

    # -----------------------------------------------------------------------
    # Front‑obstacle safety check (narrow cone ±0.2 rad ≈ ±11°)
    # -----------------------------------------------------------------------
    front_distances = [dist for bearing, dist in scan if abs(bearing) < 0.2]
    front_dist = min(front_distances) if front_distances else MAX_RANGE

    # If something is too close directly ahead, turn toward the clearer side
    if front_dist < SAFE_FRONT:
        left_distances = [dist for bearing, dist in scan if bearing > 0.5]   # ~+30°
        right_distances = [dist for bearing, dist in scan if bearing < -0.5] # ~‑30°
        left_dist = min(left_distances) if left_distances else MAX_RANGE
        right_dist = min(right_distances) if right_distances else MAX_RANGE

        # Turn toward the side with larger clearance (‑1 = left, +1 = right)
        turn_dir = -1 if left_dist > right_dist else 1
        w = turn_dir * W_MAX
        v = 0.0
        return v, w, False

    # -----------------------------------------------------------------------
    # Potential‑field computation (attractive + repulsive)
    # -----------------------------------------------------------------------
    # Attractive vector (fixed magnitude, points to goal)
    att_x = ATTRACTION_GAIN * math.cos(goal_world_bearing)
    att_y = ATTRACTION_GAIN * math.sin(goal_world_bearing)

    # Repulsive contribution from all obstacles within REPULSE_RANGE
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scan:
        if dist < REPULSE_RANGE:
            # Inverse‑square‑like repulsion that grows quickly as distance shrinks
            mag = REPULSE_GAIN * (REPULSE_RANGE - dist) / (dist * dist + 1e-6)
            world_angle = yaw + bearing
            rep_x += mag * math.cos(world_angle)
            rep_y += mag * math.sin(world_angle)

    # Resultant vector
    total_x = att_x + rep_x
    total_y = att_y + rep_y

    # Desired heading (world frame) and heading error (relative to current heading)
    desired_heading = math.atan2(total_y, total_x)
    heading_error = _norm_angle(desired_heading - yaw)

    # -----------------------------------------------------------------------
    # Angular velocity command
    # -----------------------------------------------------------------------
    w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

    # -----------------------------------------------------------------------
    # Forward speed command
    # -----------------------------------------------------------------------
    # Alignment factor – we move faster when we are pointing toward the chosen direction
    alignment = max(0.0, math.cos(heading_error))

    # Scale speed with how far we can see straight ahead (more clearance → faster)
    front_scale = min(1.0, front_dist / SAFE_FRONT)

    # Slow down when we are close to the goal
    goal_scale = 1.0
    if dist_to_goal < GOAL_SLOW_DIST:
        goal_scale = dist_to_goal / GOAL_SLOW_DIST

    v = V_MAX * alignment * front_scale * goal_scale

    # If something is extremely close directly ahead, halt forward motion
    if front_dist < MIN_FRONT_DIST:
        v = 0.0

    v = max(v, 0.0)   # safety clamp

    # -----------------------------------------------------------------------
    # Return command
    # -----------------------------------------------------------------------
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
