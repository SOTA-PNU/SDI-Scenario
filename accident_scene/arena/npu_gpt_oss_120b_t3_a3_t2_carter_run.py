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
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _desired_steering_angle(scan, goal_angle):
    """
    Compute a steering angle that blends the attractive goal direction with a
    repulsive term derived from the laser scan.

    The repulsive term pushes the robot away from obstacles that are within
    REPEL_RANGE. The contribution of each obstacle grows the closer it is.
    """
    REPEL_RANGE = 5.0          # metres – the distance at which obstacles start to influence steering
    REPULSION_GAIN = 1.0       # scaling factor for the repulsive contribution

    repulsive = 0.0
    for bearing, dist in scan:
        if dist < REPEL_RANGE:
            # Weight grows linearly as the obstacle gets nearer.
            weight = (REPEL_RANGE - dist) / REPEL_RANGE
            # Positive bearing = obstacle on the left → steer right (negative)
            repulsive -= bearing * weight

    # Blend goal direction and repulsion.
    desired = goal_angle + REPULSION_GAIN * repulsive
    # Clamp to a reasonable range (the robot cannot turn >90° instantaneously)
    max_steer = math.pi / 2
    if desired > max_steer:
        desired = max_steer
    elif desired < -max_steer:
        desired = -max_steer
    return desired


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding static obstacles.

    The controller follows a simple potential‑field‑like scheme:
      * Attractive component – the bearing to the goal expressed in the robot frame.
      * Repulsive component – a sum over laser beams that pushes the robot away
        from nearby obstacles.
    Speed is throttled when something is close ahead and when a large turn is
    required, ensuring a safe, deterministic motion.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (feel free to adjust if needed)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.55            # m/s – nominal forward speed
    MAX_ANG_SPEED = 1.2         # rad/s – maximum yaw rate
    Kp_angle = 2.0              # proportional gain for steering (rad/s per rad)
    Kp_yaw = 1.0                # gain for final yaw alignment
    SAFETY_FRONT = 0.7          # metres – distance at which we start to slow down
    CLOSE_GOAL_DIST = 2.0       # metres – start to reduce speed when we near the goal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pose & goal information
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Goal reached? (position tolerance first)
    # ------------------------------------------------------------------
    if distance_to_goal <= MISSION["position_tolerance_m"]:
        # Within the goal disc – now only need to align orientation.
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            # Both position and orientation are within tolerances → finish.
            return 0.0, 0.0, True
        # Rotate in place to achieve the desired yaw.
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_yaw * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Normal navigation step
    # ------------------------------------------------------------------
    # Angle to the goal expressed in the robot's local frame.
    goal_angle = _wrap_angle(math.atan2(dy, dx) - yaw)

    # Acquire a laser scan.
    scan = env.raycast_scan()  # default: 61 beams over 180°, max_range=6.0

    # Determine a steering direction that respects obstacles.
    desired_angle = _desired_steering_angle(scan, goal_angle)

    # Compute angular velocity command (clamped).
    w = max(-MAX_ANG_SPEED,
            min(MAX_ANG_SPEED, Kp_angle * desired_angle))

    # -------------------------------------------------
    # Forward speed modulation
    # -------------------------------------------------
    # 1) Reduce speed when something is close straight ahead.
    front_dist = 6.0  # default to max range (no obstacle)
    if scan:
        # Find the beam whose bearing is closest to zero (straight ahead).
        front_beam = min(scan, key=lambda bd: abs(bd[0]))
        front_dist = front_beam[1]

    if front_dist < SAFETY_FRONT:
        front_factor = front_dist / SAFETY_FRONT
        front_factor = max(0.0, min(1.0, front_factor))
    else:
        front_factor = 1.0

    # 2) Reduce speed when we are close to the goal (to avoid overshoot).
    if distance_to_goal < CLOSE_GOAL_DIST:
        goal_factor = distance_to_goal / CLOSE_GOAL_DIST
        goal_factor = max(0.0, min(1.0, goal_factor))
    else:
        goal_factor = 1.0

    # 3) Reduce speed when a large turn is required.
    turn_factor = max(0.0, 1.0 - abs(desired_angle) / (math.pi / 2))

    # Combine the factors.
    speed_factor = front_factor * goal_factor * turn_factor
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
