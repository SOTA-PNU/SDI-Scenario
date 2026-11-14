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
    Simple bug‑algorithm controller (Track B).

    * “go_to_goal” – drive straight toward the goal.
    * When an obstacle appears within SAFE_FRONT in the narrow front sector,
      switch to “wall_follow”.
    * “wall_follow” – keep a left‑hand wall at TARGET_WALL_DIST using the
      left‑most laser beams, move forward slowly, and turn proportionally to the
      distance error.
    * Return to “go_to_goal” when the heading to the goal is clear again and we
      have made progress (distance to goal smaller than when we entered wall
      following).
    * Stop when both position and yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Parameters (feel free to tune if you like)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG_SPEED = 1.0           # rad/s
    SAFE_FRONT = 0.7              # <‑‑ if something this close ahead → start wall‑follow
    FRONT_ANGLE = math.radians(20)  # sector considered “front” for the trigger
    TARGET_WALL_DIST = 0.8        # desired distance to the left wall while following
    KP_WALL = 1.0                 # proportional gain for wall‑following steering
    WALL_SPEED = 0.5              # forward speed while wall‑following
    GOAL_ANGLE_THR = math.radians(15)  # how well we must be aligned to re‑enter go‑to‑goal
    # ----------------------------------------------------------------------

    # ---- persistent state (created on first call) ------------------------
    if not hasattr(controller, "state"):
        controller.state = "go_to_goal"   # "go_to_goal" | "wall_follow"
        controller.entry_dist = None      # distance to goal when we entered wall‑follow
        controller.wall_start = None      # time when wall‑follow began
    # ----------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- vector to goal (world frame) ------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_dir_world = math.atan2(dy, dx)
    angle_to_goal = _norm_angle(goal_dir_world - yaw)

    # ----- termination ------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_norm_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----- laser scan -------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams, 180°, max_range = 6.0

    # front distance (minimum in narrow front sector)
    front_dist = 6.0
    for bearing, d in scan:
        if abs(bearing) <= FRONT_ANGLE and d < front_dist:
            front_dist = d

    # left‑most distance (closest obstacle on the left side, bearing > 0)
    left_dist = 6.0
    for bearing, d in scan:
        if bearing > 0 and d < left_dist:
            left_dist = d

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------
    if controller.state == "go_to_goal":
        # --------------------------------------------------------------
        # GO‑TO‑GOAL behaviour
        # --------------------------------------------------------------
        # Angular command: turn toward the goal
        w = _clamp(angle_to_goal, -MAX_ANG_SPEED, MAX_ANG_SPEED)

        # Speed: full speed, but slow down when we are near the goal
        if dist_to_goal < 2.0:
            v = MAX_SPEED * (dist_to_goal / 2.0)
        else:
            v = MAX_SPEED

        # Obstacle ahead? → switch to wall‑following
        if front_dist < SAFE_FRONT:
            controller.state = "wall_follow"
            controller.entry_dist = dist_to_goal
            controller.wall_start = t
            # In the very first step of wall‑following we just turn left
            w = MAX_ANG_SPEED
            v = 0.0

        return v, w, False

    else:   # controller.state == "wall_follow"
        # --------------------------------------------------------------
        # WALL‑FOLLOWING behaviour (left‑hand rule)
        # --------------------------------------------------------------
        # Steering: keep left distance ≈ TARGET_WALL_DIST
        error = left_dist - TARGET_WALL_DIST
        w = _clamp(KP_WALL * error, -MAX_ANG_SPEED, MAX_ANG_SPEED)

        # Forward motion only if the front is clear enough
        if front_dist > SAFE_FRONT:
            v = WALL_SPEED
        else:
            v = 0.0

        # Condition to leave wall‑following:
        # 1) front is clear,
        # 2) we are roughly pointing at the goal,
        # 3) we have gotten closer to the goal than when we entered.
        if (front_dist > SAFE_FRONT and
                abs(angle_to_goal) < GOAL_ANGLE_THR and
                dist_to_goal < controller.entry_dist):
            controller.state = "go_to_goal"

        # Safety net – if we have been wall‑following for a long time
        # (e.g. > 30 s) give up and try to go to goal anyway.
        if t - controller.wall_start > 30.0:
            controller.state = "go_to_goal"

        v = _clamp(v, 0.0, MAX_SPEED)
        w = _clamp(w, -MAX_ANG_SPEED, MAX_ANG_SPEED)

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
