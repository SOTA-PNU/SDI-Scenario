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


def _wrap_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic reactive controller.

    Strategy
    --------
    1. Compute the heading error to the goal (world‑frame).
    2. Scan the 180° front fan and pick the bearing with the largest
       clearance (penalising large turns).
    3. Blend the goal‑heading turn with the “free‑space” turn.
    4. Reduce forward speed when a turn is large, when an obstacle is close,
       and when we are close to the goal.
    5. Stop when position **and** yaw are inside the tolerances.
    """
    # ------------------------------------------------------------------
    # Fixed tuning parameters (deterministic, no randomness)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.9            # m/s – top forward speed
    MAX_TURN = 1.2             # rad/s – top angular speed
    GOAL_KP = 2.0              # gain for pure goal heading
    FREE_KP = 1.0              # gain for free‑space (clearance) heading
    SAFE_DIST = 2.5            # start reacting when an obstacle is nearer than this (m)
    CRITICAL_DIST = 0.5        # emergency stop distance (m)
    TURN_PENALTY = 0.4         # how much we penalise large bearings when picking free space

    # ------------------------------------------------------------------
    # Unpack current pose and goal pose
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # Vector / distance / bearing to the goal (world frame)
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    bearing_to_goal = math.atan2(dy, dx)                # world‑frame bearing
    heading_error = _wrap_angle(bearing_to_goal - yaw)  # relative to robot heading

    # ------------------------------------------------------------------
    # Termination check – must be satisfied *before* we issue any motion.
    # ------------------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Process the ray‑scan
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # [(rel_bearing, distance), ...]
    best_score = -float("inf")
    best_bearing = 0.0                 # bearing of the most open direction
    min_dist = float("inf")            # closest obstacle overall

    for rel_bearing, dist in scan:
        # Treat max_range (6.0) as “no hit” – keep a large distance value.
        if dist >= 6.0:
            dist = 6.0

        # Track the closest obstacle for emergency stop.
        if dist < min_dist:
            min_dist = dist

        # Score = distance - TURN_PENALTY * |bearing|
        #   – encourages large clearances,
        #   – but prefers bearings close to straight ahead.
        score = dist - TURN_PENALTY * abs(rel_bearing)

        if score > best_score:
            best_score = score
            best_bearing = rel_bearing

    # ------------------------------------------------------------------
    # Compute turn commands
    # ------------------------------------------------------------------
    # Turn needed to head directly toward the goal.
    turn_to_goal = GOAL_KP * heading_error

    # Turn needed to point into the clearest free space.
    turn_to_free = FREE_KP * best_bearing

    # Blend the two intents.  The weights (0.6 / 0.4) were chosen to keep
    # progress toward the goal while still yielding a strong avoidance
    # reaction when the free direction deviates noticeably.
    turn = 0.6 * turn_to_goal + 0.4 * turn_to_free

    # Clamp to the physical limits.
    turn = max(-MAX_TURN, min(MAX_TURN, turn))

    # ------------------------------------------------------------------
    # Speed modulation
    # ------------------------------------------------------------------
    # Base speed is the maximum, reduced by:
    #   * how sharply we need to turn (larger turn → slower)
    #   * proximity of the nearest obstacle (closer → slower)
    #   * proximity to the goal (slow down when we are within a few metres)
    turn_factor = max(0.0, 1.0 - abs(turn) / (math.pi / 2))          # 0..1
    obstacle_factor = min(1.0, min_dist / SAFE_DIST)                # 0..1
    goal_factor = 1.0
    if dist_to_goal < 3.0:
        goal_factor = dist_to_goal / 3.0

    speed = MAX_SPEED * turn_factor * obstacle_factor * goal_factor

    # Emergency stop if something is dangerously close.
    if min_dist < CRITICAL_DIST:
        speed = 0.0

    return speed, turn, False


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
