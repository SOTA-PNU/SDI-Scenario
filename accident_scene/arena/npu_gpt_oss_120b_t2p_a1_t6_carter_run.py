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
# - Deterministic go‑to‑goal using a simple potential‑field approach.
# - Only the `math` module and the provided `env.raycast_scan()` sensor are used.
# - Attractive vector points toward the goal; obstacles within a 3 m radius
#   generate repulsive vectors (scaled down by a factor of 0.5 to keep the goal
#   dominant).
# - If an obstacle is detected directly ahead (≤ 0.6 m) the robot stops forward
#   motion and rotates toward the side with more free space.
# - Forward speed is limited to 0.8 m/s and is scaled by proximity to the
#   nearest front obstacle, alignment with the desired heading, and distance
#   to the goal (linear slowdown inside 2 m).
# - The robot stops when it is within `position_tolerance_m` of the goal and
#   the yaw error is ≤ `yaw_tolerance_rad`, then returns `done=True`.
# - No randomness, file I/O, threads, or extra imports are used.

def _wrap_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding obstacles.

    Args:
        t: simulation time (seconds) – unused.
        pose: (x, y, yaw) ground‑truth robot pose.
        env: arena harness providing `raycast_scan()`.

    Returns:
        (v, w, done): forward speed [m/s], yaw rate [rad/s], termination flag.
    """
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
    goal_heading = math.atan2(dy, dx)                # world‑frame heading to goal
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    # ----------------------------------------------------------------------
    # Success test
    # ----------------------------------------------------------------------
    if dist_to_goal <= pos_tol and abs(yaw_error_to_goal) <= yaw_tol:
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
    # Tunable parameters
    # ----------------------------------------------------------------------
    max_speed = 0.8               # m/s – forward speed limit (conservative)
    max_angular = 0.8             # rad/s – yaw rate limit
    repulsion_range = 3.0         # m – distance within which obstacles repel
    repulsion_factor = 0.5        # scale repulsive strength
    safety_front = 0.6            # m – safety distance directly ahead
    front_angle_limit = math.radians(30)   # rad – front sector width
    max_range = 6.0               # sensor max range (fixed by spec)

    # ----------------------------------------------------------------------
    # Compute repulsive vector and find the closest obstacle ahead
    # ----------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    min_front = max_range          # initialise with max_range (no hit)

    for bearing, distance in scan:
        # Skip beams that see nothing
        if distance >= max_range:
            continue

        # Convert bearing (relative to robot heading) to robot‑frame coordinates
        # Robot frame: +x forward, +y left
        obs_x = distance * math.cos(bearing)
        obs_y = distance * math.sin(bearing)

        # Accumulate repulsion if the obstacle is within the repulsion range
        if distance < repulsion_range:
            # Linear weight, reduced by repulsion_factor
            w = (repulsion_range - distance) / repulsion_range * repulsion_factor
            rep_x -= w * obs_x
            rep_y -= w * obs_y

        # Track the closest obstacle in the narrow front sector
        if abs(bearing) <= front_angle_limit and distance < min_front:
            min_front = distance

    # ----------------------------------------------------------------------
    # Immediate front‑obstacle handling (turn in place)
    # ----------------------------------------------------------------------
    if min_front < safety_front:
        # Estimate free space on each side by summing distances of beams
        left_sum = sum(d for b, d in scan if b > 0 and d < max_range)
        right_sum = sum(d for b, d in scan if b < 0 and d < max_range)

        # Treat a side with no returns as completely free (max_range per beam)
        if left_sum == 0.0:
            left_sum = max_range * len([b for b, d in scan if b > 0])
        if right_sum == 0.0:
            right_sum = max_range * len([b for b, d in scan if b < 0])

        # Turn toward the side with the larger summed distance (more free space)
        if left_sum > right_sum:
            w = max_angular      # positive → turn left
        else:
            w = -max_angular     # negative → turn right
        v = 0.0
        return v, w, False

    # ----------------------------------------------------------------------
    # Combine attractive (goal) and repulsive vectors in robot frame
    # ----------------------------------------------------------------------
    # Goal direction expressed in robot frame (bearing relative to heading)
    goal_bearing = _wrap_angle(goal_heading - yaw)
    goal_vec_x = math.cos(goal_bearing)
    goal_vec_y = math.sin(goal_bearing)

    # Weighted sum – goal has unit weight; repulsion already scaled
    total_x = goal_vec_x + rep_x
    total_y = goal_vec_y + rep_y

    # Desired heading in robot frame
    desired_bearing = math.atan2(total_y, total_x)
    angle_error = _wrap_angle(desired_bearing)   # error relative to current heading

    # ----------------------------------------------------------------------
    # Command generation
    # ----------------------------------------------------------------------
    # Angular velocity: proportional controller, limited to max_angular
    w = max(-max_angular, min(max_angular, 2.0 * angle_error))

    # Speed scaling based on obstacle proximity, alignment, and goal distance
    # 1) Proximity scaling – slower when an obstacle ahead is close
    if min_front < max_range:
        proximity_factor = max(0.0,
                               (min_front - safety_front) /
                               (max_range - safety_front))
    else:
        proximity_factor = 1.0

    # 2) Alignment scaling – slower when we need to turn sharply
    alignment_factor = max(0.0, math.cos(angle_error))

    # 3) Goal‑distance scaling – linear slowdown inside 2 m of the goal
    goal_factor = 1.0
    if dist_to_goal < 2.0:
        goal_factor = dist_to_goal / 2.0

    v = max_speed * proximity_factor * alignment_factor * goal_factor

    # Zero‑out tiny speeds to avoid jitter (but allow a minimal forward nudge)
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
