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

# Simple bug‑style finite‑state controller.
# Global state (deterministic, no randomness)
_MODE = "go"          # "go" -> head to goal, "wall" -> follow right wall
_TURN_SPEED = 0.8     # rad/s, used when turning right while wall‑following

def _wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def _min_distance(scan, low, high):
    """Return the smallest distance for beams whose bearing lies in [low, high]."""
    min_d = float("inf")
    for bearing, dist in scan:
        if low <= bearing <= high and dist < min_d:
            min_d = dist
    return min_d


def _distance_at_bearing(scan, target, tol=0.05):
    """Return the distance of the beam nearest to *target* bearing, or inf."""
    best = float("inf")
    for bearing, dist in scan:
        if abs(bearing - target) <= tol and dist < best:
            best = dist
    return best


def controller(t, pose, env):
    """
    Bug‑style controller:
      * In "go" mode drive straight to the goal while the front sector is clear.
      * If something is closer than SAFE_FRONT in front, switch to "wall" mode.
      * In "wall" mode follow the right side of obstacles, turning right.
      * When the path toward the goal becomes clear again, switch back to "go".
      * When inside the goal disc, align yaw and stop.
    """
    global _MODE

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ---------- Goal check ----------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_err = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        w = max(-0.8, min(0.8, 2.0 * yaw_err))
        return 0.0, w, False

    # ---------- Scan ----------
    scan = env.raycast_scan()   # [(bearing, distance), ...]

    # ---------- Parameters ----------
    SAFE_FRONT = 0.8          # m, distance considered "clear" ahead
    SIDE_TARGET = 0.6         # m, desired distance from the right wall
    MAX_TURN = 0.9            # rad/s
    BASE_SPEED = 1.0          # m/s
    TURN_SPEED = 0.3          # m/s while wall‑following

    # Front sector (±30°)
    front_min = _min_distance(scan, -0.5236, 0.5236)

    # Desired heading to goal (world frame)
    desired_yaw = math.atan2(dy, dx)
    heading_error = _wrap_angle(desired_yaw - yaw)   # bearing to goal in robot frame

    # ---------- State machine ----------
    if _MODE == "go":
        # If something blocks the front, start wall following
        if front_min < SAFE_FRONT:
            _MODE = "wall"
        else:
            # Normal go‑to‑goal steering
            w = max(-MAX_TURN, min(MAX_TURN, 1.5 * heading_error))

            # Speed reduction when heading error is large
            err_factor = max(0.2, 1.0 - min(1.0, abs(heading_error) / (math.pi / 2)))
            # Slow down near the goal for precision
            goal_factor = 0.5 if dist_to_goal < 2.0 else 1.0
            v = BASE_SPEED * err_factor * goal_factor
            return v, w, False

    # -------------------------------------------------
    # Wall‑following mode (follow the right side)
    # -------------------------------------------------
    # Right side distance (bearings -90° .. -30°)
    right_min = _min_distance(scan, -1.5708, -0.5236)

    # Compute a small correction to keep roughly SIDE_TARGET from the wall
    side_error = SIDE_TARGET - right_min   # >0 => too far, <0 => too close
    side_correction = 0.5 * side_error     # gain

    # Turn right at a base rate, add side correction (positive side_error -> turn more right)
    w = _TURN_SPEED + side_correction
    w = max(-MAX_TURN, min(MAX_TURN, w))

    v = TURN_SPEED

    # Check if we can return to "go" mode:
    #   - front is clear,
    #   - bearing to goal is roughly within the forward cone,
    #   - distance ahead (along bearing to goal) is not blocked.
    if front_min >= SAFE_FRONT and abs(heading_error) < 0.3:
        # Look up distance in the direction of the goal
        goal_dist_ahead = _distance_at_bearing(scan, heading_error)
        if goal_dist_ahead >= SAFE_FRONT or goal_dist_ahead == float("inf"):
            _MODE = "go"

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
