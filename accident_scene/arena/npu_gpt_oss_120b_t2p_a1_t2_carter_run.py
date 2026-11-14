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
# Design summary:
# - Simple bug‑algorithm style controller with a two‑state machine:
#   "go"  – drive forward while steering toward the goal.
#   "turn" – if an obstacle is detected within a short safety distance directly
#            ahead, rotate left until the front becomes clear, then resume "go".
# - Uses only the provided `env.raycast_scan()` sensor and the global
#   constants `GOAL` and `MISSION` from the base file.
# - No random numbers, file I/O, threads, or imports beyond `math`.
# - Speed limits: max forward speed 0.8 m/s, max angular speed 0.8 rad/s.
# - Success condition: within `position_tolerance_m` of the goal and yaw error
#   ≤ `yaw_tolerance_rad`, then `done=True`.
# - A small forward‑speed jitter guard is applied.
# - State is stored as a function attribute `controller.state`.
#
# Assumptions (documented as constants):
#   * Robot radius ≈ 0.3 m – safety margin is handled by the front distance
#     threshold.
#   * Sensor origin is at the robot centre and bearing is positive to the left
#     (standard ROS convention).
#   * No actuation delay – commands are applied instantly.
#   * The environment's ray‑fan covers 180° front with 61 beams.
# ===========================================================================

import math

# Helper: wrap angle to [-π, π]
def _wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Bug‑algorithm controller.

    Args:
        t: simulation time (seconds) – unused.
        pose: (x, y, yaw) robot ground‑truth pose.
        env: arena harness providing `raycast_scan()`.

    Returns:
        (v, w, done): forward speed [m/s], yaw rate [rad/s], termination flag.
    """
    # Initialise persistent state on first call
    if not hasattr(controller, "state"):
        controller.state = "go"

    # Unpack pose
    x, y, yaw = pose

    # Goal and tolerances (global from base file)
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # Distance and yaw error to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    # Success test
    if dist_to_goal <= pos_tol and abs(yaw_error_to_goal) <= yaw_tol:
        # Reset state for a clean next run
        controller.state = "go"
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Sensor read – safe guard against unexpected failures
    # ------------------------------------------------------------------
    try:
        scan = env.raycast_scan()  # [(bearing, distance), ...]
    except Exception:
        # If sensor call fails, stop the robot but do not terminate.
        return 0.0, 0.0, False

    # ------------------------------------------------------------------
    # Parameters (tuned for the arena)
    # ------------------------------------------------------------------
    max_speed = 0.8            # m/s – conservative forward speed
    max_angular = 0.8          # rad/s – max yaw rate
    min_front_dist = 0.6       # m – safety distance directly ahead
    front_angle = math.radians(15)  # rad – sector considered "front"

    # ------------------------------------------------------------------
    # Front obstacle detection
    # ------------------------------------------------------------------
    front_dist = min(
        (dist for bearing, dist in scan if abs(bearing) <= front_angle),
        default=6.0,
    )

    # ------------------------------------------------------------------
    # Goal bearing (relative to current heading)
    # ------------------------------------------------------------------
    goal_bearing = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    if controller.state == "go":
        # If something is too close in front, switch to turning state
        if front_dist < min_front_dist:
            controller.state = "turn"
        else:
            # Drive forward while steering toward the goal
            v = max_speed
            # Proportional turn toward goal bearing, limited to max_angular
            w = max(-max_angular, min(max_angular, 1.5 * goal_bearing))
            # Small speed guard – avoid jitter when alignment is poor
            if abs(w) > max_angular * 0.9:
                # If we need a sharp turn, reduce forward speed a little
                v *= 0.6
            return v, w, False

    if controller.state == "turn":
        # Keep turning left (positive w) until the front clears
        if front_dist >= min_front_dist:
            controller.state = "go"
            # After clearing, we can immediately issue a forward command
            v = max_speed
            w = max(-max_angular, min(max_angular, 1.5 * goal_bearing))
            return v, w, False
        else:
            # Rotate left on the spot
            return 0.0, max_angular, False

    # Fallback – should never reach here
    return 0.0, 0.0, False
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
