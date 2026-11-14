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


# ----------------------------------------------------------------------
# Global state for a deterministic “forced‑right‑turn” when the front is
# blocked.  The variables are updated only inside the controller and are
# therefore deterministic across runs.
# ----------------------------------------------------------------------
_TURN_UNTIL = 0.0   # simulation time until which we keep turning right
# ----------------------------------------------------------------------


def _wrap_angle(a: float) -> float:
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic reactive controller.

    * If something is directly ahead within STOP_FRONT_DIST we start a forced
      right turn that lasts TURN_TIME seconds.
    * While the forced turn is active we ignore the goal and simply rotate.
    * Otherwise we try to point toward the goal (heading error threshold) and
      drive forward, but we keep a side‑clearance bias:
        - turn right if an obstacle is too close on the left,
        - turn left  if an obstacle is too close on the right.
    * Success is reported when we are inside the goal disc and yaw is within
      tolerance.
    """
    # ------------------------------------------------------------------
    # Tunable constants (pure numbers – no extra imports)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.9            # m/s
    MAX_ANG = 1.0              # rad/s
    Kp_speed = 0.9             # linear gain
    Kp_ang = 2.0               # angular gain
    HEADING_THRESH = 0.12      # rad – when we consider ourselves “facing” the goal
    SAFE_FRONT = 0.7           # m – front distance that triggers a forced turn
    SIDE_SAFE = 0.6            # m – side distance that triggers a steering bias
    TURN_TIME = 0.8            # s – duration of the forced right turn
    TURN_SPEED = -0.8          # rad/s – forced right‑turn angular speed
    # ------------------------------------------------------------------

    global _TURN_UNTIL

    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # --------------------------------------------------------------
    # Goal distance & success test
    # --------------------------------------------------------------
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        # Reset turn timer for cleanliness
        _TURN_UNTIL = 0.0
        return 0.0, 0.0, True

    # --------------------------------------------------------------
    # Scan processing
    # --------------------------------------------------------------
    scan = env.raycast_scan()          # [(bearing, distance), ...]
    # Front: |bearing| < 0.15 rad
    front_dist = min(
        (d for b, d in scan if abs(b) < 0.15 and d < 6.0),
        default=6.0,
    )
    # Left side: 0.4 .. 1.2 rad
    left_dist = min(
        (d for b, d in scan if 0.4 < b < 1.2 and d < 6.0),
        default=6.0,
    )
    # Right side: -1.2 .. -0.4 rad
    right_dist = min(
        (d for b, d in scan if -1.2 < b < -0.4 and d < 6.0),
        default=6.0,
    )

    # --------------------------------------------------------------
    # Forced right turn when something is too close ahead
    # --------------------------------------------------------------
    if front_dist < SAFE_FRONT and t >= _TURN_UNTIL:
        # start a new forced turn
        _TURN_UNTIL = t + TURN_TIME
        # during the turn we ignore everything else
        return 0.0, TURN_SPEED, False

    if t < _TURN_UNTIL:
        # continue forced turn
        w = TURN_SPEED
        if w < -MAX_ANG:
            w = -MAX_ANG
        return 0.0, w, False

    # --------------------------------------------------------------
    # Normal navigation – first align to goal if needed
    # --------------------------------------------------------------
    goal_angle_world = math.atan2(dy, dx)
    heading_error = _wrap_angle(goal_angle_world - yaw)

    # If we are not roughly facing the goal, rotate in place
    if abs(heading_error) > HEADING_THRESH:
        w = max(-MAX_ANG, min(MAX_ANG, Kp_ang * heading_error))
        return 0.0, w, False

    # --------------------------------------------------------------
    # Compute forward speed (proportional to remaining distance)
    # --------------------------------------------------------------
    v = Kp_speed * min(MAX_SPEED, 0.6 * dist_to_goal)

    # --------------------------------------------------------------
    # Steering bias based on side clearance
    # --------------------------------------------------------------
    w = 0.0
    if left_dist < SIDE_SAFE:
        # obstacle on the left → steer right
        w = -0.5
    elif right_dist < SIDE_SAFE:
        # obstacle on the right → steer left
        w = 0.5

    # Clamp angular speed
    if w > MAX_ANG:
        w = MAX_ANG
    if w < -MAX_ANG:
        w = -MAX_ANG

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
