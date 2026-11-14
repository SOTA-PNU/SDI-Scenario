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


def _wrap_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic waypoint‑follow controller with very light obstacle avoidance.

    1. A hard‑coded list of waypoints (derived from the known box layout) guides
       the robot around the cluttered centre of the warehouse.
    2. At every step we steer toward the current waypoint using a proportional
       heading controller.
    3. A simple left/right repulsion derived from the front ray‑scan adds a bias
       that pushes the robot away from any obstacle that gets too close.
    4. Forward speed is reduced when a large turn is required, when an obstacle
       is nearby, and when we are close to the goal.
    5. The mission ends only when both position and yaw are inside the tolerances.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Initialise static data (waypoints, state) on first call.
    # ------------------------------------------------------------------
    if not hasattr(controller, "waypoints"):
        # Waypoints are chosen to stay clear of every static box.
        # They form a rough “Manhattan” corridor from the SW corner to the NE corner.
        controller.waypoints = [
            (-8.0, -5.0),   # climb north a little, clear the bottom‑row boxes
            (-2.0, -5.0),   # move east, still below the central clutter
            (-2.0,  2.0),   # go north, passing left of the three‑box cluster at y≈0
            (-3.0,  6.0),   # shift left to bypass the long horizontal box at y≈4.2
            (-3.0, 12.0),   # head north behind the upper‑row boxes
            ( 2.0, 12.0),   # slide east between the upper‑row boxes
            ( 2.0, 16.0),   # go north close to the goal line
            ( 8.0, 16.5),   # final goal (the harness will check tolerance)
        ]
        controller.wp_idx = 0

    # ------------------------------------------------------------------
    # 2️⃣  Unpack pose and goal for termination checking.
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # ------------------------------------------------------------------
    # 3️⃣  Termination test – must be evaluated before any motion.
    # ------------------------------------------------------------------
    dist_to_goal = math.hypot(gx - x, gy - y)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # 4️⃣  Select the current waypoint.
    # ------------------------------------------------------------------
    wp = controller.waypoints[controller.wp_idx]
    wx, wy = wp
    vec_to_wp = (wx - x, wy - y)
    dist_to_wp = math.hypot(*vec_to_wp)

    # If we are close enough to the current waypoint, advance to the next.
    if dist_to_wp < 0.6 and controller.wp_idx < len(controller.waypoints) - 1:
        controller.wp_idx += 1
        wp = controller.waypoints[controller.wp_idx]
        wx, wy = wp
        vec_to_wp = (wx - x, wy - y)
        dist_to_wp = math.hypot(*vec_to_wp)

    # Desired heading toward the active waypoint.
    desired_heading = math.atan2(vec_to_wp[1], vec_to_wp[0])
    heading_error = _wrap_angle(desired_heading - yaw)

    # ------------------------------------------------------------------
    # 5️⃣  Ray‑scan processing – left/right repulsion and front distance.
    # ------------------------------------------------------------------
    scan = env.raycast_scan()               # [(rel_bearing, distance), ...]
    left_weight = 0.0
    right_weight = 0.0
    front_dist = 6.0                        # initialise as “nothing”
    min_dist = 6.0

    for rel_bearing, dist in scan:
        # Treat max_range as “no hit”.
        if dist >= 6.0:
            continue

        # Track the absolute closest obstacle for emergency stop.
        if dist < min_dist:
            min_dist = dist

        # Front sector (±15° ≈ 0.26 rad) – used for emergency stop.
        if abs(rel_bearing) <= 0.26 and dist < front_dist:
            front_dist = dist

        # Simple left/right repulsion when something is inside SAFE_FRONT.
        SAFE_FRONT = 2.0
        if dist < SAFE_FRONT:
            w = (SAFE_FRONT - dist) / SAFE_FRONT  # 0..1, larger when nearer
            if rel_bearing > 0:
                left_weight += w
            else:
                right_weight += w

    # ------------------------------------------------------------------
    # 6️⃣  Compute turn command.
    # ------------------------------------------------------------------
    # Proportional turn toward waypoint.
    Kp_heading = 2.0
    turn_to_wp = Kp_heading * heading_error

    # Bias away from the side with more repulsion.
    # (right_weight > left_weight) ⇒ obstacle on right ⇒ turn left (positive).
    turn_avoid = (right_weight - left_weight) * 0.9  # scale modestly

    turn = turn_to_wp + turn_avoid

    # Clamp turn rate.
    MAX_TURN = 1.2
    turn = max(-MAX_TURN, min(MAX_TURN, turn))

    # ------------------------------------------------------------------
    # 7️⃣  Speed modulation.
    # ------------------------------------------------------------------
    MAX_SPEED = 0.85

    # Reduce speed when we need to turn sharply.
    turn_factor = max(0.0, 1.0 - abs(turn) / (math.pi / 2))

    # Reduce speed when an obstacle is relatively close (but not yet critical).
    OBSTACLE_INFLUENCE = 2.0
    obstacle_factor = min(1.0, min_dist / OBSTACLE_INFLUENCE)

    # Slow down when we are close to the final goal (smooth stop).
    goal_factor = 1.0
    if dist_to_goal < 3.0:
        goal_factor = dist_to_goal / 3.0

    speed = MAX_SPEED * turn_factor * obstacle_factor * goal_factor

    # Emergency stop if something is directly ahead and too close.
    CRITICAL_FRONT = 0.35
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
