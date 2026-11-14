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


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    * Attractive component pulls the robot toward the goal.
    * Repulsive component pushes it away from any obstacle that appears in the
      180‑degree ray fan and is closer than SAFE_DIST.
    * If an obstacle is inside STOP_DIST we stop forward motion and rotate
      away from the closest one.
    * Completion is reported when position and yaw are within the tolerances.
    """
    # ------------------------------------------------------------------
    # Parameters (all deterministic, no randomness)
    # ------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG_SPEED = 1.0           # rad/s
    Kp_att = 2.0                  # gain for attractive (goal) heading
    Kp_rep = 1.5                  # gain for repulsive contribution
    Kp_ang = 1.5                  # proportional gain for final heading error
    SAFE_DIST = 2.0               # start repelling when obstacle < this (m)
    STOP_DIST = 0.6               # stop forward motion if obstacle < this (m)
    # ------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ---------- Goal reached? ----------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_error_to_goal) <= MISSION["yaw_tolerance_rad"]):
        # Stop and signal success
        return 0.0, 0.0, True

    # ---------- Attractive direction ----------
    desired_heading = math.atan2(dy, dx)

    # ---------- Scan for obstacles ----------
    scan = env.raycast_scan()          # [(bearing, distance), ...]
    # Find the closest obstacle overall (for possible stop) and accumulate
    # repulsive influence from all obstacles inside SAFE_DIST.
    min_dist = None
    min_bearing = None
    repulsive_sum = 0.0

    for bearing, dist in scan:
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_bearing = bearing

        if dist < SAFE_DIST:
            # Repulsive contribution is stronger the nearer the obstacle.
            # Direction is opposite to the bearing (push away).
            weight = (SAFE_DIST - dist) / SAFE_DIST
            repulsive_sum += -weight * bearing   # negative bearing pushes away

    # ---------- Immediate stop if too close ----------
    if min_dist is not None and min_dist < STOP_DIST:
        # Stop forward motion, rotate away from the closest obstacle
        turn_dir = -1.0 if min_bearing > 0 else 1.0
        v = 0.0
        w = turn_dir * MAX_ANG_SPEED
        return v, w, False

    # ---------- Combine attractive and repulsive ----------
    # The combined heading is the attractive heading plus a weighted repulsive term.
    combined_heading = desired_heading + Kp_rep * repulsive_sum
    heading_error = _wrap_angle(combined_heading - yaw)

    # ---------- Command generation ----------
    # Angular velocity from heading error
    w = max(-MAX_ANG_SPEED,
            min(MAX_ANG_SPEED, Kp_ang * heading_error))

    # Forward speed reduced when we need to turn sharply and when obstacles are near
    turn_factor = max(0.0, 1.0 - min(1.0, abs(heading_error) / (math.pi / 2)))
    # Additional slowdown proportional to the nearest obstacle within SAFE_DIST
    if min_dist is not None and min_dist < SAFE_DIST:
        obstacle_factor = max(0.0, (min_dist - STOP_DIST) / (SAFE_DIST - STOP_DIST))
    else:
        obstacle_factor = 1.0

    v = MAX_SPEED * turn_factor * obstacle_factor

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
