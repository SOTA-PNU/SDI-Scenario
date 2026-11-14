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
    """Reactive controller with goal attraction, repulsive obstacle forces,
    and simple emergency avoidance when something is too close."""
    # ----------------------------------------------------------------------
    # Tunable parameters (feel free to adjust if needed)
    MAX_SPEED = 1.0            # m/s, top forward speed
    MAX_TURN = 2.0             # rad/s, top yaw rate
    Kp_turn = 3.0              # proportional gain for heading error
    K_avoid = 2.0              # gain for side‑clearance based steering
    SAFE_FRONT = 0.8           # m, distance considered unsafe directly ahead
    SAFE_SIDE = 0.6            # m, distance considered unsafe on the sides
    FRONT_ANGLE = math.radians(30)   # half‑cone used for front clearance
    SIDE_ANGLE_MIN = math.radians(30)
    SIDE_ANGLE_MAX = math.radians(90)
    REPULSE_RANGE = 1.5       # m, range within which obstacles push back
    K_rep = 5.0                # weight of repulsive forces
    # ----------------------------------------------------------------------

    # Helper to wrap angles into [-π, π]
    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # ----------------------------------------------------------------------
    # Current pose and goal
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ----------------------------------------------------------------------
    # Goal check – stop and align yaw when we are close enough
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            # Goal reached with correct orientation
            return 0.0, 0.0, True
        # Rotate in place to achieve the required yaw
        w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * yaw_error))
        return 0.0, w, False

    # ----------------------------------------------------------------------
    # Desired heading toward the goal (world frame) and error in robot frame
    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - yaw)

    # ----------------------------------------------------------------------
    # Laser scan – we get a list of (bearing, distance) pairs.
    scans = env.raycast_scan()

    # Front clearance (±FRONT_ANGLE)
    min_front = min(
        (d for b, d in scans if abs(b) <= FRONT_ANGLE),
        default=6.0,
    )

    # Side clearances (30°–90° left and right)
    left_min = min(
        (d for b, d in scans if SIDE_ANGLE_MIN <= b <= SIDE_ANGLE_MAX),
        default=6.0,
    )
    right_min = min(
        (d for b, d in scans if -SIDE_ANGLE_MAX <= b <= -SIDE_ANGLE_MIN),
        default=6.0,
    )

    # ----------------------------------------------------------------------
    # Repulsive force from nearby obstacles (within REPULSE_RANGE)
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scans:
        if dist < REPULSE_RANGE:
            # Linear weighting: stronger when closer
            w_rep = (REPULSE_RANGE - dist) / REPULSE_RANGE
            rep_x -= math.cos(bearing) * w_rep
            rep_y -= math.sin(bearing) * w_rep

    # Attractive unit vector toward the goal expressed in robot frame
    att_x = math.cos(heading_error)
    att_y = math.sin(heading_error)

    # Combine attraction and repulsion
    total_x = att_x + K_rep * rep_x
    total_y = att_y + K_rep * rep_y

    # Avoid a zero‑vector (unlikely but safe)
    if total_x == 0.0 and total_y == 0.0:
        total_x = att_x
        total_y = att_y

    # Resulting desired direction in robot frame after considering obstacles
    desired_dir = math.atan2(total_y, total_x)
    new_error = wrap_angle(desired_dir)

    # ----------------------------------------------------------------------
    # Angular command: heading correction + side‑clearance bias
    side_bias = (right_min - left_min) / 6.0          # normalised side difference
    w = Kp_turn * new_error + K_avoid * side_bias
    w = max(-MAX_TURN, min(MAX_TURN, w))

    # ----------------------------------------------------------------------
    # Emergency avoidance: if something is too close in front, stop forward
    # motion and turn away from the nearer side.
    if min_front < SAFE_FRONT:
        v = 0.0
        # Choose turn direction based on which side is more open
        if left_min < right_min:
            w = -MAX_TURN   # obstacle more on the left → turn right
        else:
            w = MAX_TURN    # obstacle more on the right → turn left
        return v, w, False

    # ----------------------------------------------------------------------
    # Speed command – scale with heading error, front clearance and side safety
    # Reduce speed when the heading error is large.
    speed_factor = max(0.0, math.cos(new_error))

    # Scale with front clearance (more room → faster)
    front_factor = (min_front - SAFE_FRONT) / (6.0 - SAFE_FRONT)
    front_factor = max(0.0, min(1.0, front_factor))

    v = MAX_SPEED * speed_factor * front_factor

    # Further slow down if we are too close to a side obstacle
    if left_min < SAFE_SIDE or right_min < SAFE_SIDE:
        v = min(v, 0.2)

    # Slow down when approaching the goal to avoid overshoot
    if dist_to_goal < 1.0:
        v *= dist_to_goal / 1.0

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
