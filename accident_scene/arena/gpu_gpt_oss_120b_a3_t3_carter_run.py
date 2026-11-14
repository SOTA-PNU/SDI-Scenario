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

def controller(t, pose, env):
    """
    Simple deterministic reactive controller.

    * Drives toward the goal while respecting a modest speed limit.
    * Uses the 180° ray‑scan to generate a repulsive influence when an obstacle
      is closer than ``SAFE_DIST``.  The repulsive influence is blended with the
      attractive heading to the goal.
    * Stops and reports success when both position and yaw are within the
      tolerances defined in ``MISSION``.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    # vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # heading to goal (world frame) and error relative to robot heading
    angle_to_goal = math.atan2(dy, dx)
    angle_error = wrap_angle(angle_to_goal - yaw)          # goal‑centric error

    # ----------------------------------------------------------------------
    # parameters
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0          # m/s
    MAX_ANGULAR = 1.0        # rad/s
    KP_ANGLE = 1.5           # proportional gain for heading error
    SAFE_DIST = 0.8          # meters – start repelling when an obstacle is this close
    MIN_SPEED_NEAR_OBSTACLE = 0.0  # stop if obstacle is too close

    # ----------------------------------------------------------------------
    # ray‑scan processing
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()                     # [(bearing, distance), ...]
    # find the closest obstacle (ignore max_range = 6.0 which means “nothing”)
    valid_scans = [(b, d) for (b, d) in scans if d < 6.0]
    if valid_scans:
        bearing_closest, min_dist = min(valid_scans, key=lambda bd: bd[1])
    else:
        bearing_closest, min_dist = 0.0, 6.0

    # ----------------------------------------------------------------------
    # compute desired heading (blend goal direction with repulsion)
    # ----------------------------------------------------------------------
    if min_dist < SAFE_DIST:
        # strength of repulsion grows linearly as we approach the obstacle
        obs_weight = (SAFE_DIST - min_dist) / SAFE_DIST          # 0 … 1
        # direction away from the obstacle (relative to robot heading)
        avoid_dir = wrap_angle(bearing_closest + math.pi)
        # blend: when far from obstacles obs_weight≈0 → pure goal heading;
        # when very close obs_weight≈1 → turn away from obstacle.
        blended_error = wrap_angle(
            (1.0 - obs_weight) * angle_error + obs_weight * avoid_dir
        )
    else:
        blended_error = angle_error

    # ----------------------------------------------------------------------
    # command generation
    # ----------------------------------------------------------------------
    # angular velocity – proportional to blended heading error, clipped
    w = max(-MAX_ANGULAR, min(MAX_ANGULAR, KP_ANGLE * blended_error))

    # forward speed – reduced when we need to turn sharply
    turn_factor = max(0.0, 1.0 - abs(blended_error) / (math.pi / 2))
    v = MAX_SPEED * turn_factor

    # additional slowdown when we are close to the goal
    if dist_to_goal < 1.5:
        v *= dist_to_goal / 1.5

    # stop completely if an obstacle is extremely close directly ahead
    if min_dist < 0.3:
        v = MIN_SPEED_NEAR_OBSTACLE

    # ----------------------------------------------------------------------
    # goal check
    # ----------------------------------------------------------------------
    done = False
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        v = 0.0
        w = 0.0
        done = True

    return v, w, done


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
