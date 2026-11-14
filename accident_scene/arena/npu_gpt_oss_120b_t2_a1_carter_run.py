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
    Simple reactive controller:
      * Drive toward the goal using a proportional heading controller.
      * Use the 180° raycast scan to generate a repulsive turning moment
        when obstacles are within a safety distance.
      * Reduce forward speed if an obstacle is directly ahead.
      * Stop and declare success when within position and yaw tolerances.
    """
    # ----------------------------------------------------------------------
    # Helper: wrap angle to [-π, π]
    # ----------------------------------------------------------------------
    def normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # ----------------------------------------------------------------------
    # Unpack state and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_dir = math.atan2(dy, dx)
    angle_error = normalize_angle(goal_dir - yaw)

    # ----------------------------------------------------------------------
    # Termination condition
    # ----------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        yaw_error = normalize_angle(yaw - goal_yaw)
        if abs(yaw_error) <= yaw_tol:
            # Goal reached and oriented correctly
            return 0.0, 0.0, True
        # Within position tolerance but not yet oriented – rotate in place
        # Simple proportional yaw controller (no forward motion)
        KP_YAW = 1.5
        w_align = KP_YAW * yaw_error
        MAX_TURN = 1.0
        w_align = max(-MAX_TURN, min(MAX_TURN, w_align))
        return 0.0, w_align, False

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.8          # m/s – safe cruising speed
    MAX_TURN = 1.0           # rad/s – max yaw rate
    KP_ANGLE = 1.5           # proportional gain for heading towards goal
    SAFE_DIST = 1.0          # m – distance at which we start repelling
    REPULSION_GAIN = 0.8     # strength of obstacle repulsion
    FRONT_BLOCK_DIST = 0.8   # m – distance at which we stop forward motion

    # ----------------------------------------------------------------------
    # Base angular command towards the goal
    # ----------------------------------------------------------------------
    w_goal = KP_ANGLE * angle_error
    w_goal = max(-MAX_TURN, min(MAX_TURN, w_goal))

    # ----------------------------------------------------------------------
    # Obstacle avoidance using the raycast scan
    # ----------------------------------------------------------------------
    obstacle_w = 0.0
    front_dist = None  # distance of the nearest obstacle in a forward cone

    try:
        scans = env.raycast_scan()  # default: 61 beams, 180°, max_range=6.0
    except Exception:
        scans = []  # safety fallback

    for bearing, distance in scans:
        # distance == max_range means no hit – ignore
        if distance >= 6.0:
            continue

        # Track the nearest obstacle roughly in front (|bearing| < 0.5 rad)
        if abs(bearing) < 0.5:
            if front_dist is None or distance < front_dist:
                front_dist = distance

        # Repulsive turning when an obstacle is within the safety distance
        if distance < SAFE_DIST:
            # Repulsion magnitude (0..1)
            rep = (SAFE_DIST - distance) / SAFE_DIST
            # Determine turning direction: if obstacle on the left (bearing>0),
            # turn right (negative w); if on the right, turn left.
            turn_dir = -math.copysign(1.0, bearing)
            obstacle_w += turn_dir * rep * REPULSION_GAIN

    # Combine goal heading and obstacle avoidance
    w = w_goal + obstacle_w
    w = max(-MAX_TURN, min(MAX_TURN, w))

    # ----------------------------------------------------------------------
    # Speed control
    # ----------------------------------------------------------------------
    # Stop forward motion if something is directly ahead and too close
    if front_dist is not None and front_dist < FRONT_BLOCK_DIST:
        v = 0.0
    else:
        # Scale speed with distance to goal (full speed when far)
        if dist_to_goal > 2.0:
            v = MAX_SPEED
        else:
            # Slow down proportionally when approaching the goal region
            v = MAX_SPEED * (dist_to_goal / 2.0)

        # Reduce speed when turning sharply to improve maneuverability
        if abs(w) > 0.5:
            v *= 0.5

    # Ensure non‑negative speed
    v = max(0.0, v)

    # ----------------------------------------------------------------------
    # Return command
    # ----------------------------------------------------------------------
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
