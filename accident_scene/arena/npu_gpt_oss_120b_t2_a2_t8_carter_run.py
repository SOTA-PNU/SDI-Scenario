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
    """Potential‑field navigation with a simple emergency turn‑away.

    * Attractive vector points to the goal.
    * Repulsive vectors push away from any obstacle closer than REPULSE_RANGE.
    * If something is too close directly ahead (min_front < SAFE_FRONT) the robot
      stops and turns toward the side that has the larger clearance.
    * Speed is scaled by how well the chosen direction aligns with the goal,
      by front clearance, and is reduced near the goal and near side obstacles.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (feel free to adjust)
    MAX_SPEED = 1.0          # m/s – top forward speed
    MAX_TURN = 2.0           # rad/s – top yaw rate
    Kp_turn = 2.5            # proportional gain for heading error
    K_rep = 3.0              # weight of repulsive forces
    REPULSE_RANGE = 2.0      # m – obstacles closer than this push back
    SAFE_FRONT = 0.5         # m – distance that triggers emergency turn
    SAFE_SIDE = 0.4          # m – side clearance that forces speed reduction
    FRONT_ANGLE = math.radians(30)   # half‑cone used for front‑clearance check
    SIDE_ANGLE_MIN = math.radians(30)
    SIDE_ANGLE_MAX = math.radians(90)
    # ------------------------------------------------------------------

    # Helper: wrap angle into [-π, π]
    def wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # Current pose and goal pose
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Goal handling – stop and align yaw when we are close enough
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Desired bearing toward the goal (relative to robot heading)
    desired_heading = math.atan2(dy, dx)          # world‑frame angle to goal
    goal_bearing = wrap_angle(desired_heading - yaw)  # robot‑frame bearing

    # ------------------------------------------------------------------
    # Laser scan
    scans = env.raycast_scan()  # list of (bearing, distance)

    # ------------------------------------------------------------------
    # Front clearance (within +/- FRONT_ANGLE)
    min_front = min(
        (dist for bearing, dist in scans if abs(bearing) <= FRONT_ANGLE),
        default=6.0,
    )

    # ------------------------------------------------------------------
    # Side clearances (30°–90° left and right) – used for emergency turn and speed reduction
    left_min = min(
        (dist for bearing, dist in scans
         if SIDE_ANGLE_MIN <= bearing <= SIDE_ANGLE_MAX),
        default=6.0,
    )
    right_min = min(
        (dist for bearing, dist in scans
         if -SIDE_ANGLE_MAX <= bearing <= -SIDE_ANGLE_MIN),
        default=6.0,
    )

    # ------------------------------------------------------------------
    # Emergency turn‑away if something is too close directly ahead
    if min_front < SAFE_FRONT:
        # Stop forward motion
        v = 0.0
        # Turn toward the side with more clearance
        if left_min > right_min:
            w = MAX_TURN          # turn left
        else:
            w = -MAX_TURN         # turn right
        return v, w, False

    # ------------------------------------------------------------------
    # Repulsive force from obstacles within REPULSE_RANGE
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scans:
        if dist < REPULSE_RANGE:
            # Linear strength that grows as distance shrinks
            strength = (REPULSE_RANGE - dist) / REPULSE_RANGE
            rep_x -= math.cos(bearing) * strength
            rep_y -= math.sin(bearing) * strength

    # ------------------------------------------------------------------
    # Attractive unit vector toward the goal (expressed in robot frame)
    att_x = math.cos(goal_bearing)
    att_y = math.sin(goal_bearing)

    # Combine attraction and repulsion
    total_x = att_x + K_rep * rep_x
    total_y = att_y + K_rep * rep_y

    # Guard against a zero vector (unlikely but safe)
    if total_x == 0.0 and total_y == 0.0:
        total_x, total_y = att_x, att_y

    # Desired direction after considering obstacles (robot‑frame bearing)
    desired_dir = math.atan2(total_y, total_x)
    heading_error = wrap_angle(desired_dir)

    # ------------------------------------------------------------------
    # Angular command
    w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * heading_error))

    # ------------------------------------------------------------------
    # Speed command – several scaling factors
    # 1) alignment with the chosen direction
    alignment = max(0.0, math.cos(heading_error))

    # 2) front clearance scaling
    front_factor = (min_front - SAFE_FRONT) / (6.0 - SAFE_FRONT)
    front_factor = max(0.0, min(1.0, front_factor))

    # 3) side‑clearance scaling (slow down near side obstacles)
    side_factor = 1.0
    if left_min < SAFE_SIDE or right_min < SAFE_SIDE:
        side_factor = 0.3

    v = MAX_SPEED * alignment * front_factor * side_factor

    # ------------------------------------------------------------------
    # Slow down when approaching the goal to avoid overshoot
    if dist_to_goal < 1.0:
        v *= dist_to_goal

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
