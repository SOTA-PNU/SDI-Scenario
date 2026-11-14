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
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Goal‑directed controller with simple frontal‑obstacle avoidance.

    * Move toward the goal with a constant forward speed.
    * If an obstacle is detected within a short frontal cone, turn away and
      reduce speed proportionally.
    * Stop only when both position and yaw are within the tolerances.
    """
    # ------------------------------------------------------------------ #
    # unpack current pose
    x, y, yaw = pose

    # ------------------------------------------------------------------ #
    # goal information
    goal_x, goal_y, goal_yaw = GOAL
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------ #
    # termination test
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    at_pos = dist_to_goal <= pos_tol
    yaw_err_to_goal = _wrap_angle(goal_yaw - yaw)
    at_yaw = abs(yaw_err_to_goal) <= yaw_tol
    if at_pos and at_yaw:
        return 0.0, 0.0, True

    # ------------------------------------------------------------------ #
    # desired heading toward the goal
    desired_yaw = math.atan2(dy, dx)

    # ------------------------------------------------------------------ #
    # sensor scan – look for obstacles in a narrow frontal sector
    scan = env.raycast_scan()
    frontal_half_angle = 0.4          # rad (~23° each side)
    obstacle_close_dist = 0.8         # meters – start reacting inside this range
    min_front_dist = None
    avoidance_turn = 0.0

    for bearing, dist in scan:
        # consider only beams roughly in front of the robot
        if abs(bearing) > frontal_half_angle:
            continue
        if dist >= obstacle_close_dist:
            continue
        # keep the closest distance for speed scaling
        if (min_front_dist is None) or (dist < min_front_dist):
            min_front_dist = dist
        # decide turn direction: obstacle on the left (bearing>0) → turn right
        turn_dir = -1.0 if bearing > 0 else 1.0
        # influence grows as the obstacle gets nearer
        influence = (obstacle_close_dist - dist) / obstacle_close_dist
        avoidance_turn += turn_dir * influence * 1.5   # 1.5 rad/s max turn rate

    # ------------------------------------------------------------------ #
    # angular control (proportional to heading error)
    yaw_error = _wrap_angle(desired_yaw - yaw)
    kp_ang = 2.0
    max_ang = 1.5
    angular = kp_ang * yaw_error
    angular = max(-max_ang, min(max_ang, angular))

    # blend avoidance turn if any obstacle was seen
    if avoidance_turn != 0.0:
        angular = angular + avoidance_turn
        angular = max(-max_ang, min(max_ang, angular))

    # ------------------------------------------------------------------ #
    # speed control – base speed reduced by heading error and obstacles
    base_speed = 1.2                     # m/s
    # slower when we need to turn sharply
    speed = base_speed * (1.0 - min(1.0, abs(yaw_error) / (math.pi / 2)))

    # reduce speed if an obstacle is directly ahead
    if min_front_dist is not None:
        speed *= min_front_dist / obstacle_close_dist
        # never let speed go negative
        speed = max(0.0, speed)

    # slow down when we are close to the goal to allow fine alignment
    if dist_to_goal < 2.0:
        speed *= dist_to_goal / 2.0

    # final safety clamp
    speed = max(0.0, speed)

    # ------------------------------------------------------------------ #
    return speed, angular, False


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
