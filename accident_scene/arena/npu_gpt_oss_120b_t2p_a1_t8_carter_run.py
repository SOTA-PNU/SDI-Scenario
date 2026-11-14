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
    pose  (x, y, yaw) ground‑truth world pose of the chassis
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
# - Deterministic go-to-goal using a simple heading-to-goal controller.
# - Only the `math` module and the provided `env.raycast_scan()` sensor are used.
# - The robot drives toward the goal unless an obstacle is within a short
#   safety distance directly ahead; in that case it stops and turns toward the
#   side with more free space.
# - A tiny global state tracks whether the robot is stuck (distance to goal
#   not decreasing for >2 s) and forces a turn-in-place to escape.
# - Forward speed is limited to 1.0 m/s and is scaled by alignment with the
#   goal bearing and by proximity to the nearest front obstacle.
# - Yaw rate is limited to 0.8 rad/s and is a proportional controller on the
#   bearing error plus a small bias toward the side with more clearance.
# - No randomness, file I/O, threads, or extra imports are used.

# Global state for simple stuck detection
_state = {
    "last_dist": None,   # distance to goal at previous step
    "stuck_time": 0.0,   # accumulated time robot has not made progress
    "last_t": 0.0,       # previous simulation time
}

def _wrap_angle(angle):
    """Wrap angle to the interval [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding obstacles.

    Args:
        t: simulation time (seconds) – unused except for stuck detection.
        pose: (x, y, yaw) ground-truth robot pose.
        env: arena harness providing `raycast_scan()`.

    Returns:
        (v, w, done): forward speed [m/s], yaw rate [rad/s], termination flag.
    """
    global _state

    # ----------------------------------------------------------------------
    # Unpack pose and mission parameters
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # Vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)                # world-frame heading to goal
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    # ----------------------------------------------------------------------
    # Success test
    # ----------------------------------------------------------------------
    if dist_to_goal <= pos_tol and abs(yaw_error_to_goal) <= yaw_tol:
        # Reset state for next run
        _state["last_dist"] = None
        _state["stuck_time"] = 0.0
        _state["last_t"] = 0.0
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # Sensor reading (robust wrapper)
    # ----------------------------------------------------------------------
    try:
        scan = env.raycast_scan()                    # [(bearing, distance), ...]
    except Exception:
        # If the sensor fails we stop moving but do not terminate.
        return 0.0, 0.0, False

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------
    max_speed = 1.0               # m/s – forward speed limit
    max_angular = 0.8             # rad/s – yaw rate limit
    safety_front = 0.5            # m – distance at which we consider the front blocked
    front_angle_limit = math.radians(30)   # rad – front sector width for obstacle check
    max_range = 6.0               # sensor max range (fixed by spec)

    # ----------------------------------------------------------------------
    # Scan processing
    # ----------------------------------------------------------------------
    min_front = max_range          # closest obstacle in the front sector
    left_sum = 0.0
    right_sum = 0.0

    for bearing, distance in scan:
        # Skip beams that see nothing
        if distance >= max_range:
            continue

        # Accumulate clearance sums for bias calculation
        if bearing > 0:
            left_sum += distance
        elif bearing < 0:
            right_sum += distance

        # Track the closest obstacle directly ahead
        if abs(bearing) <= front_angle_limit and distance < min_front:
            min_front = distance

    # ----------------------------------------------------------------------
    # Immediate front-obstacle handling (turn in place)
    # ----------------------------------------------------------------------
    if min_front < safety_front:
        # Turn toward the side with the larger summed clearance
        if left_sum > right_sum:
            w = max_angular      # turn left
        else:
            w = -max_angular     # turn right
        v = 0.0
        # Update stuck-detection state (no forward progress)
        dt = t - _state["last_t"] if _state["last_t"] else 1.0 / 60.0
        _state["stuck_time"] += dt
        _state["last_t"] = t
        _state["last_dist"] = dist_to_goal
        return v, w, False

    # ----------------------------------------------------------------------
    # Heading to goal and bias from side clearance
    # ----------------------------------------------------------------------
    goal_bearing = _wrap_angle(goal_heading - yaw)   # relative bearing to goal
    # Small bias: turn toward the side with more clearance
    bias = 0.0
    if left_sum < right_sum:
        bias = -0.1   # favor right turn
    elif left_sum > right_sum:
        bias = 0.1    # favor left turn

    # ----------------------------------------------------------------------
    # Stuck detection – if distance hasn't decreased for >2 s, force a turn
    # ----------------------------------------------------------------------
    dt = t - _state["last_t"] if _state["last_t"] else 1.0 / 60.0
    if _state["last_dist"] is not None:
        if dist_to_goal < _state["last_dist"] - 0.05:
            # Made progress – reset stuck timer
            _state["stuck_time"] = 0.0
        else:
            _state["stuck_time"] += dt
    _state["last_dist"] = dist_to_goal
    _state["last_t"] = t

    if _state["stuck_time"] > 2.0:
        # Force a slow left turn to escape a possible local minimum
        return 0.0, max_angular, False

    # ----------------------------------------------------------------------
    # Command generation
    # ----------------------------------------------------------------------
    # Angular velocity: proportional to bearing error plus clearance bias
    w = max(-max_angular,
            min(max_angular, 2.0 * goal_bearing + bias))

    # Speed scaling:
    #   - alignment factor (cosine of bearing error)
    #   - proximity factor (slow down when something is close ahead)
    alignment_factor = max(0.0, math.cos(goal_bearing))
    if min_front < max_range:
        proximity_factor = max(0.0,
                               (min_front - safety_front) /
                               (max_range - safety_front))
    else:
        proximity_factor = 1.0

    v = max_speed * alignment_factor * proximity_factor

    # Zero-out tiny speeds to avoid jitter
    if v < 0.02:
        v = 0.0

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
