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


def _wrap_angle(a: float) -> float:
    """Wrap angle into the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple, deterministic bug‑type controller.

    1. If the robot is already inside the goal tolerance → stop and report done.
    2. Scan the front (±0.2 rad).  If something is closer than SAFE_FRONT,
       rotate in place towards the side that has more free space.
    3. Otherwise drive forward while applying a proportional heading correction
       toward the goal.  A small repulsive term nudges the heading away from
       nearby obstacles (≤ SAFE_DIST).
    """
    # ------------------------------------------------------------------
    # Tunable parameters (all deterministic, no randomness)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.8               # m/s – modest to keep safe distances
    MAX_ANG = 1.0                 # rad/s
    Kp_speed = 0.5                # linear gain (m/s per meter)
    Kp_ang = 1.5                  # angular gain (rad/s per rad)
    SAFE_FRONT = 0.7              # meters – distance that triggers a turn
    FRONT_ANGLE = 0.2             # rad – half‑width of the “front” cone
    SAFE_DIST = 0.8               # meters – distance at which repulsion starts
    REPULSION_GAIN = 0.7          # scaling for obstacle‑induced heading change

    # ------------------------------------------------------------------
    # Unpack current pose and goal pose
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # ------------------------------------------------------------------
    # Goal check – stop as soon as we are inside both tolerances
    # ------------------------------------------------------------------
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Desired heading toward the goal (ignoring obstacles for now)
    # ------------------------------------------------------------------
    goal_angle = math.atan2(dy, dx)
    heading_error = _wrap_angle(goal_angle - yaw)

    # ------------------------------------------------------------------
    # Get a fresh scan
    # ------------------------------------------------------------------
    scan = env.raycast_scan()

    # ------------------------------------------------------------------
    # 1) Front‑obstacle handling – if something is too close straight ahead,
    #    rotate in place towards the clearer side.
    # ------------------------------------------------------------------
    front_blocked = any(
        abs(bearing) <= FRONT_ANGLE and dist < SAFE_FRONT
        for bearing, dist in scan if dist < 6.0
    )
    if front_blocked:
        # Compute aggregate free space on left vs right side
        left_free = sum(
            max(0.0, dist - SAFE_FRONT) for bearing, dist in scan
            if bearing > 0 and dist < 6.0
        )
        right_free = sum(
            max(0.0, dist - SAFE_FRONT) for bearing, dist in scan
            if bearing < 0 and dist < 6.0
        )
        # Turn towards the side with more free space (positive = left turn)
        turn_dir = 1.0 if left_free > right_free else -1.0
        # No forward motion while turning
        return 0.0, turn_dir * MAX_ANG, False

    # ------------------------------------------------------------------
    # 2) Obstacle‑induced repulsion (only for obstacles within SAFE_DIST)
    # ------------------------------------------------------------------
    repulsion = 0.0
    for bearing, dist in scan:
        if dist >= 6.0 or dist >= SAFE_DIST:
            continue
        # The closer the obstacle, the larger the repulsive torque.
        # sin(bearing) > 0 → obstacle on the left → turn right (negative)
        # sin(bearing) < 0 → obstacle on the right → turn left (positive)
        strength = REPULSION_GAIN * (SAFE_DIST - dist) / SAFE_DIST
        repulsion -= strength * math.sin(bearing)

    # ------------------------------------------------------------------
    # 3) Compute final command
    # ------------------------------------------------------------------
    # Linear speed: proportional to distance, reduced when we need to turn a lot
    speed = min(MAX_SPEED, Kp_speed * dist_to_goal)
    if abs(heading_error) > 0.5:   # large heading error → slow down
        speed *= 0.5

    # Angular speed: combine goal heading error and repulsion term
    ang = Kp_ang * heading_error + repulsion

    # Clamp angular velocity
    if ang > MAX_ANG:
        ang = MAX_ANG
    elif ang < -MAX_ANG:
        ang = -MAX_ANG

    # ------------------------------------------------------------------
    # Return the command tuple
    # ------------------------------------------------------------------
    return speed, ang, False


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
