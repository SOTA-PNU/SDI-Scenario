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
    Simple deterministic bug‑style controller.

    * Goal‑heading proportional control.
    * If an obstacle is detected within a safe front distance, the robot
      rotates in place away from the obstacle until the front is clear.
    * A modest wall‑avoidance bias keeps the robot away from the arena edges.
    * Speed is reduced when turning sharply, when near an obstacle, and when
      approaching the goal.
    * Mission ends only when both position and yaw are inside the tolerances.
    """
    # ------------------------------------------------------------------
    # Persistent simple state (mode) – stored as an attribute on the function.
    # ------------------------------------------------------------------
    if not hasattr(controller, "mode"):
        controller.mode = "go"          # other possible value: "avoid"

    # ------------------------------------------------------------------
    # Tunable deterministic constants
    # ------------------------------------------------------------------
    MAX_SPEED = 0.85          # m/s – top forward speed
    MAX_TURN = 1.1           # rad/s – top angular speed
    GOAL_KP = 2.0            # gain for heading to goal
    SAFE_FRONT = 1.6         # start avoidance when obstacle < this (m)
    CRITICAL_FRONT = 0.45    # emergency stop distance (m)
    WALL_MARGIN = 1.0        # distance from arena border where we start turning back
    TURN_PENALTY = 0.4       # penalise large bearing when picking free direction

    # ------------------------------------------------------------------
    # Unpack pose and goal
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # ------------------------------------------------------------------
    # Goal geometry
    # ------------------------------------------------------------------
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    bearing_to_goal = math.atan2(dy, dx)               # world‑frame bearing
    heading_error = _wrap_angle(bearing_to_goal - yaw) # relative to robot heading

    # ------------------------------------------------------------------
    # Termination test (must be checked before issuing any motion)
    # ------------------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Ray‑scan processing
    # ------------------------------------------------------------------
    scan = env.raycast_scan()               # [(rel_bearing, distance), ...]
    front_dist = 6.0                        # initialise as "no hit"
    min_dist = 6.0                          # overall closest obstacle
    left_weight = 0.0
    right_weight = 0.0
    best_score = -float("inf")
    best_bearing = 0.0

    for rel_bearing, dist in scan:
        # Clamp distance to max_range (6.0) – treat as free space.
        if dist >= 6.0:
            dist = 6.0

        # Track closest obstacle for emergency stop.
        if dist < min_dist:
            min_dist = dist

        # Front sector (±30° ≈ 0.52 rad) – used for mode switching.
        if abs(rel_bearing) <= 0.52 and dist < front_dist:
            front_dist = dist

        # Simple left/right repulsion weights (used when moving).
        if dist < SAFE_FRONT:
            w = (SAFE_FRONT - dist) / SAFE_FRONT   # 0..1
            if rel_bearing > 0:
                left_weight += w
            else:
                right_weight += w

        # Score for “most open” direction – encourages straight ahead.
        score = dist - TURN_PENALTY * abs(rel_bearing)
        if score > best_score:
            best_score = score
            best_bearing = rel_bearing

    # ------------------------------------------------------------------
    # Wall‑avoidance bias (very coarse, based only on position)
    # ------------------------------------------------------------------
    turn_wall = 0.0
    # arena extents are roughly x∈[-10,10], y∈[-15,15] (from map size)
    if x < -9.0 + WALL_MARGIN:
        turn_wall = 0.6          # turn right (positive yaw)
    elif x > 9.0 - WALL_MARGIN:
        turn_wall = -0.6         # turn left
    if y < -13.0 + WALL_MARGIN:
        turn_wall = -0.6
    elif y > 13.0 - WALL_MARGIN:
        turn_wall = 0.6

    # ------------------------------------------------------------------
    # Mode decision
    # ------------------------------------------------------------------
    if front_dist < SAFE_FRONT:
        controller.mode = "avoid"
    else:
        controller.mode = "go"

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------
    if controller.mode == "avoid":
        # Rotate away from the obstacle.  The side with the larger repulsion
        # (left_weight vs right_weight) indicates where the obstacle is.
        # Turning opposite to that side pushes us away.
        turn = (right_weight - left_weight) * MAX_TURN
        speed = 0.0
    else:
        # Primary turn towards the goal.
        turn_to_goal = GOAL_KP * heading_error

        # Secondary turn towards the clearest free space (from best_bearing).
        turn_to_free = best_bearing

        # Blend the three influences: goal, free space, wall bias.
        turn = (0.6 * turn_to_goal) + (0.3 * turn_to_free) + (0.1 * turn_wall)

        # Clamp turn rate.
        turn = max(-MAX_TURN, min(MAX_TURN, turn))

        # Speed modulation:
        #   - Reduce speed when we need to turn sharply.
        #   - Reduce speed when an obstacle is relatively close.
        #   - Slow down when we are within a few metres of the goal.
        turn_factor = max(0.0, 1.0 - abs(turn) / (math.pi / 2))   # 0..1
        obstacle_factor = min(1.0, min_dist / SAFE_FRONT)        # 0..1
        goal_factor = 1.0
        if dist_to_goal < 3.0:
            goal_factor = dist_to_goal / 3.0

        speed = MAX_SPEED * turn_factor * obstacle_factor * goal_factor

        # Emergency stop if something is dangerously close.
        if front_dist < CRITICAL_FRONT:
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
