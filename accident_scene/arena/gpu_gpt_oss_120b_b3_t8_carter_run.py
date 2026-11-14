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
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v, lo, hi):
    """Clamp *v* to the range [lo, hi]."""
    return max(lo, min(hi, v))


def controller(t, pose, env):
    """
    Conservative bug‑algorithm controller (Track B).

    * Drive toward the goal while keeping a safe distance from anything
      detected in the laser scan.
    * If something is too close directly ahead, stop forward motion and turn
      away from the closest obstacle.
    * When the robot makes no measurable progress toward the goal for a few
      seconds, spin in place to look for a new opening.
    * Mission ends when both position and yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (feel free to adjust)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.8               # m/s – keep it modest to avoid overshoot
    MAX_ANG_SPEED = 0.8           # rad/s
    SAFE_FRONT = 0.6              # ≤ this distance ahead → treat as immediate obstacle
    FRONT_ANGLE = math.radians(15)   # sector considered “directly ahead”
    SIDE_ANGLE = math.radians(80)    # approximate left/right side bearings
    MIN_SPEED_FACTOR = 0.2        # never go slower than this fraction of MAX_SPEED
    PROG_EPS = 0.02               # minimal progress (metres) considered “moving”
    STAGNATION_STEPS = 180        # ~3 s without progress → spin
    # ----------------------------------------------------------------------

    # ---- persistent state -------------------------------------------------
    if not hasattr(controller, "last_dist"):
        controller.last_dist = None          # distance to goal at previous step
        controller.stagnation_cnt = 0       # consecutive steps without progress
    # ----------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- vector to goal (world frame) ------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_dir_world = math.atan2(dy, dx)
    bearing_to_goal = _norm_angle(goal_dir_world - yaw)   # relative to robot heading

    # ----- termination ------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_norm_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        # reset persistent vars for a clean next run
        controller.last_dist = None
        controller.stagnation_cnt = 0
        return 0.0, 0.0, True

    # ----- laser scan -------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams, 180°, max_range = 6.0

    # ----- analyse scan -----------------------------------------------------
    front_min = 6.0                     # closest distance in the narrow front sector
    front_closest_bearing = 0.0
    left_max = -1.0                     # best (largest) distance on the left side
    right_max = -1.0                    # best distance on the right side

    for bearing, dist in scan:
        # front sector (for emergency stop / obstacle avoidance)
        if abs(bearing) <= FRONT_ANGLE and dist < front_min:
            front_min = dist
            front_closest_bearing = bearing

        # left side (positive bearings up to SIDE_ANGLE)
        if 0.0 < bearing <= SIDE_ANGLE and dist > left_max:
            left_max = dist

        # right side (negative bearings down to -SIDE_ANGLE)
        if -SIDE_ANGLE <= bearing < 0.0 and dist > right_max:
            right_max = dist

    # If no left/right reading was found (unlikely) treat them as clear
    if left_max < 0.0:
        left_max = 6.0
    if right_max < 0.0:
        right_max = 6.0

    # ----------------------------------------------------------------------
    # Stagnation detection – if we are not getting closer to the goal,
    # start spinning in place to re‑orient.
    # ----------------------------------------------------------------------
    if controller.last_dist is None:
        controller.last_dist = dist_to_goal
        controller.stagnation_cnt = 0
    else:
        if controller.last_dist - dist_to_goal < PROG_EPS:
            controller.stagnation_cnt += 1
        else:
            controller.stagnation_cnt = 0
        controller.last_dist = dist_to_goal

    if controller.stagnation_cnt >= STAGNATION_STEPS:
        # Reset counter and spin left at max angular speed.
        controller.stagnation_cnt = 0
        return 0.0, MAX_ANG_SPEED, False

    # ----------------------------------------------------------------------
    # Decide angular velocity
    # ----------------------------------------------------------------------
    if front_min < SAFE_FRONT:
        # Too close in front – turn away from the *closest* obstacle.
        # Desired turn is opposite to the bearing of the closest point.
        desired_yaw = -front_closest_bearing
        w = _clamp(desired_yaw, -MAX_ANG_SPEED, MAX_ANG_SPEED)
        v = 0.0                     # stop forward motion while re‑orienting
        return v, w, False

    # No immediate front obstacle – combine goal heading with a side‑clearance bias.
    # Bias: if left side is clearer than right, add a small left turn, and vice‑versa.
    side_bias = 0.3 * (right_max - left_max)   # positive → turn right, negative → turn left
    steer = bearing_to_goal + side_bias
    w = _clamp(steer, -MAX_ANG_SPEED, MAX_ANG_SPEED)

    # ----------------------------------------------------------------------
    # Decide forward speed
    # ----------------------------------------------------------------------
    # Base speed scales with how clear the front is (but never below a minimum).
    if front_min >= SAFE_FRONT:
        # The farther the nearest front obstacle, the faster we may go.
        speed_factor = min(1.0, front_min / 2.0)   # front_min >= 0.6 → factor ≤ 1
    else:
        speed_factor = 0.0

    # Slow down when we are close to the goal for finer positioning.
    if dist_to_goal < 2.0:
        speed_factor *= (dist_to_goal / 2.0)

    # Ensure we keep at least a minimal crawl speed when we are moving.
    speed_factor = max(MIN_SPEED_FACTOR, speed_factor)

    v = MAX_SPEED * speed_factor
    v = _clamp(v, 0.0, MAX_SPEED)

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
