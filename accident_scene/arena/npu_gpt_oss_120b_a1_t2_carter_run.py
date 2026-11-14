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
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    * Attractive force → goal.
    * Repulsive force → any obstacle detected by the 180° ray‑fan that is
      closer than SAFE_DIST.
    * The resulting vector is converted to a heading error, from which a
      proportional yaw‑rate (w) and a speed that is reduced when we are not
      pointing toward the goal or when an obstacle is close are generated.

    The controller stops the robot once it is inside the goal disc and the
    yaw error is within the allowed tolerance.
    """
    # ----------------------------------------------------------------------
    # 1️⃣  Constants (tuned for the arena)
    # ----------------------------------------------------------------------
    MAX_V = 1.0                     # m/s
    MAX_W = 1.0                     # rad/s
    K_W = 1.5                       # proportional gain for heading error
    SAFE_DIST = 1.0                 # m – distance at which repulsion starts
    REPULSION_GAIN = 2.0            # weight of repulsive vectors
    ATTRACTION_GAIN = 1.0           # weight of the goal vector
    MIN_SPEED_FACTOR = 0.2          # never go slower than this fraction of MAX_V

    # ----------------------------------------------------------------------
    # 2️⃣  Unpack current pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # 3️⃣  Goal‑reached test
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    yaw_error_to_goal = _normalize_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_error_to_goal) <= MISSION["yaw_tolerance_rad"]):
        # Goal satisfied – stop and signal completion
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # 4️⃣  Attractive vector (goal) expressed in robot‑frame coordinates
    # ----------------------------------------------------------------------
    # Rotate world‑frame delta into robot frame
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    gx = cos_y * dx - sin_y * dy   # forward component in robot frame
    gy = sin_y * dx + cos_y * dy   # left‑ward component in robot frame

    # Normalise to unit length (unless we are exactly on top of the goal)
    if gx == 0.0 and gy == 0.0:
        attr_x, attr_y = 0.0, 0.0
    else:
        norm = math.hypot(gx, gy)
        attr_x = (gx / norm) * ATTRACTION_GAIN
        attr_y = (gy / norm) * ATTRACTION_GAIN

    # ----------------------------------------------------------------------
    # 5️⃣  Repulsive vectors from the ray‑fan
    # ----------------------------------------------------------------------
    scan = env.raycast_scan()          # [(bearing, distance), ...]
    rep_x, rep_y = 0.0, 0.0
    min_obstacle_dist = None          # keep the closest distance for speed scaling

    for bearing, distance in scan:
        # Only consider obstacles that are nearer than SAFE_DIST
        if distance < SAFE_DIST:
            # Record the closest distance (for later speed reduction)
            if (min_obstacle_dist is None) or (distance < min_obstacle_dist):
                min_obstacle_dist = distance

            # Repulsive direction is opposite to the bearing of the hit
            away_dir = bearing + math.pi
            # Linear magnitude that grows the closer the obstacle is
            mag = (SAFE_DIST - distance) / SAFE_DIST
            rep_x += math.cos(away_dir) * mag * REPULSION_GAIN
            rep_y += math.sin(away_dir) * mag * REPULSION_GAIN

    # ----------------------------------------------------------------------
    # 6️⃣  Combine attractive and repulsive contributions
    # ----------------------------------------------------------------------
    total_x = attr_x + rep_x
    total_y = attr_y + rep_y

    # If the sum vector is zero (unlikely but possible), fall back to pure
    # heading‑to‑goal control.
    if total_x == 0.0 and total_y == 0.0:
        desired_heading = math.atan2(dy, dx)
    else:
        desired_heading = math.atan2(total_y, total_x)  # in robot frame

    # Convert desired heading back to world frame to obtain a heading error
    # (desired_heading is relative to the robot’s forward axis)
    heading_error = _normalize_angle(desired_heading)

    # ----------------------------------------------------------------------
    # 7️⃣  Yaw‑rate command (proportional)
    # ----------------------------------------------------------------------
    w = K_W * heading_error
    w = max(-MAX_W, min(MAX_W, w))

    # ----------------------------------------------------------------------
    # 8️⃣  Forward speed command
    # ----------------------------------------------------------------------
    # Base speed is reduced when we are not pointing toward the goal.
    # The factor (1 - |heading_error|/π) is 1 when heading_error == 0 and 0
    # when we are pointing opposite to the goal.
    speed_factor = 1.0 - min(1.0, abs(heading_error) / math.pi)

    # Further reduce speed if an obstacle is close.
    if min_obstacle_dist is not None:
        speed_factor *= max(MIN_SPEED_FACTOR, min_obstacle_dist / SAFE_DIST)

    v = MAX_V * speed_factor
    v = max(0.0, min(MAX_V, v))

    # ----------------------------------------------------------------------
    # 9️⃣  Return command tuple
    # ----------------------------------------------------------------------
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
