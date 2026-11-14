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

def _wrap_angle(angle):
    """Wrap an angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Reactive navigation controller.

    * Drives toward the goal while keeping a safety margin from obstacles.
    * Uses a simple cost map over the 61‑beam forward ray fan:
        - Beams that report a distance < SAFETY_DIST are considered blocked.
        - For beams within REPEL_RANGE a distance‑based penalty is added.
        - An angular deviation penalty pushes the chosen bearing toward the goal.
    * The selected bearing becomes the desired steering angle.
    * Forward speed is throttled based on front‑clearance, required turn
      magnitude, and proximity to the goal.
    * When the robot reaches the goal disc it aligns its yaw to the goal yaw
      and stops.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – feel free to adjust if needed.
    # ------------------------------------------------------------------
    MAX_SPEED = 0.55            # nominal forward speed [m/s]
    MAX_ANG_SPEED = 1.2         # maximum yaw rate [rad/s]
    Kp_angle = 2.0              # proportional gain for steering
    Kp_yaw = 1.0                # gain for final yaw alignment
    SAFETY_DIST = 0.9           # distance below which a beam is considered blocked
    REPEL_RANGE = 2.5           # distance within which obstacles start to influence cost
    INF_COST = 1e6              # sentinel for blocked directions
    WEIGHT_ANGLE = 1.0          # weight for angular deviation from goal
    WEIGHT_DIST = 2.0           # weight for proximity to obstacles
    FRONT_SAFETY = 0.8          # distance at which we start to throttle forward speed
    CLOSE_GOAL_DIST = 2.0       # start to slow down when this close to the goal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pose & goal information
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Goal bearing expressed in the robot's local frame
    goal_angle = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # Goal reached? (position first, then orientation)
    # ------------------------------------------------------------------
    if distance_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            # Position and orientation are within tolerances – finish.
            return 0.0, 0.0, True
        # Rotate in place to align yaw.
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_yaw * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Normal navigation step
    # ------------------------------------------------------------------
    scan = env.raycast_scan()  # 61 beams over 180°, max_range = 6.0 m

    # -------------------------------------------------
    # 1) Choose a steering direction using a cost map.
    # -------------------------------------------------
    best_cost = INF_COST
    best_bearing = 0.0

    for bearing, dist in scan:
        # Treat a beam that reports a distance below SAFETY_DIST as blocked.
        if dist < SAFETY_DIST:
            cost = INF_COST
        else:
            # Angular deviation penalty (steer toward the goal).
            angle_penalty = abs(bearing - goal_angle) * WEIGHT_ANGLE

            # Distance‑based penalty (repel from nearby obstacles).
            if dist < REPEL_RANGE:
                dist_penalty = (REPEL_RANGE - dist) / REPEL_RANGE * WEIGHT_DIST
            else:
                dist_penalty = 0.0

            cost = angle_penalty + dist_penalty

        if cost < best_cost:
            best_cost = cost
            best_bearing = bearing

    # If every direction is blocked (unlikely), fall back to a modest turn.
    if best_cost >= INF_COST * 0.5:
        # Turn slowly toward the side where the goal lies; if goal is straight,
        # pick a deterministic left turn.
        bias = 0.2 if goal_angle >= 0 else -0.2
        desired_angle = bias
    else:
        desired_angle = best_bearing

    # Add a tiny deterministic bias that flips every 0.5 s to break symmetry.
    # This is based solely on the simulation time `t`.
    if int(t * 2) % 2 == 0:
        desired_angle += 0.05
    else:
        desired_angle -= 0.05

    # Clamp the desired steering angle to what the robot can feasibly achieve.
    max_steer = math.pi / 2
    if desired_angle > max_steer:
        desired_angle = max_steer
    elif desired_angle < -max_steer:
        desired_angle = -max_steer

    # -------------------------------------------------
    # 2) Convert desired angle to angular velocity command.
    # -------------------------------------------------
    w = max(-MAX_ANG_SPEED,
            min(MAX_ANG_SPEED, Kp_angle * desired_angle))

    # -------------------------------------------------
    # 3) Compute forward speed, throttling for safety.
    # -------------------------------------------------
    # Front clearance (beam whose bearing is closest to zero).
    front_dist = 6.0  # default = max range (no obstacle)
    if scan:
        front_beam = min(scan, key=lambda bd: abs(bd[0]))
        front_dist = front_beam[1]

    # Reduce speed if something is close straight ahead.
    if front_dist < FRONT_SAFETY:
        front_factor = front_dist / FRONT_SAFETY
        front_factor = max(0.0, min(1.0, front_factor))
    else:
        front_factor = 1.0

    # Reduce speed when a large turn is required.
    turn_factor = max(0.0, 1.0 - abs(desired_angle) / max_steer)

    # Reduce speed when we are very close to the goal (to avoid overshoot).
    if distance_to_goal < CLOSE_GOAL_DIST:
        goal_factor = distance_to_goal / CLOSE_GOAL_DIST
        goal_factor = max(0.0, min(1.0, goal_factor))
    else:
        goal_factor = 1.0

    speed_factor = front_factor * turn_factor * goal_factor
    v = MAX_SPEED * speed_factor

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
