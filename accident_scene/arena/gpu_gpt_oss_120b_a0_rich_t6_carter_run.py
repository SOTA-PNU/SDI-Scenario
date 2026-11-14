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
    """Wrap angle to [-pi, pi)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def _min_distance(scan, low, high):
    """Return the smallest distance for beams whose bearing is in [low, high]."""
    min_d = float("inf")
    for bearing, dist in scan:
        if low <= bearing <= high and dist < min_d:
            min_d = dist
    return min_d


def controller(t, pose, env):
    """
    Simple deterministic controller.

    - Drive toward the goal while the forward sector is clear.
    - If an obstacle is closer than SAFE_FRONT, turn toward the side that
      offers more clearance, moving slowly while turning.
    - When inside the goal disc, align yaw and stop.
    """
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Goal tolerance handling
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_err = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        # Rotate in place to correct yaw
        w = max(-0.8, min(0.8, 2.0 * yaw_err))
        return 0.0, w, False

    # Scan data
    scan = env.raycast_scan()   # list of (bearing, distance)

    # Sector definitions (radians)
    FRONT_HALF = 0.5236          # +/-30 degrees
    LEFT_LOW = 0.5236            # 30 degrees
    LEFT_HIGH = 1.5708           # 90 degrees
    RIGHT_LOW = -1.5708          # -90 degrees
    RIGHT_HIGH = -0.5236         # -30 degrees

    # Minimum distances in sectors
    front_min = _min_distance(scan, -FRONT_HALF, FRONT_HALF)
    left_min = _min_distance(scan, LEFT_LOW, LEFT_HIGH)
    right_min = _min_distance(scan, RIGHT_LOW, RIGHT_HIGH)

    # Parameters
    SAFE_FRONT = 0.85            # meters
    MAX_TURN = 0.9               # rad/s
    BASE_SPEED = 1.2             # m/s

    # Obstacle avoidance
    if front_min < SAFE_FRONT:
        # Choose turn direction: side with larger clearance, tie -> turn right
        turn_dir = -1 if left_min > right_min else 1   # -1 = left, +1 = right

        # Small bias to keep away from very close side obstacles
        bias = 0.0
        if left_min < 0.6:
            bias -= 0.3   # steer right a bit
        if right_min < 0.6:
            bias += 0.3   # steer left a bit

        w = turn_dir * MAX_TURN + bias
        w = max(-MAX_TURN, min(MAX_TURN, w))
        v = 0.25 * BASE_SPEED
        return v, w, False

    # Normal go-to-goal steering
    desired_yaw = math.atan2(dy, dx)
    yaw_err = _wrap_angle(desired_yaw - yaw)

    # Proportional yaw rate
    w = max(-MAX_TURN, min(MAX_TURN, 1.5 * yaw_err))

    # Add side‑clearance bias so we don't scrape walls while moving straight
    if left_min < 0.6:
        w -= 0.2   # steer right
    if right_min < 0.6:
        w += 0.2   # steer left

    # Speed scaling: slower when heading error is large or we are near the goal
    err_factor = max(0.2, 1.0 - min(1.0, abs(yaw_err) / (math.pi / 2)))
    goal_factor = 0.5 if dist_to_goal < 2.0 else 1.0
    v = BASE_SPEED * err_factor * goal_factor

    v = max(0.0, v)
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
