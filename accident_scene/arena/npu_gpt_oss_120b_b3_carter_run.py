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


def _norm_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple reactive controller:

    * Attractive component – head toward the goal.
    * Repulsive component – steer away from obstacles detected by the
      180° ray‑cast fan.
    * Stop when the robot is inside the goal disc and oriented correctly.
    """
    # ----------------------------------------------------------------------
    # Parameters (tuned for the arena size)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG_SPEED = 1.5           # rad/s
    KP_ANGLE = 2.0                # proportional gain for heading error
    REPULSIVE_DIST = 1.2          # distance under which an obstacle repels
    K_REP = 1.0                   # weight of the repulsive angular offset
    CLOSE_STOP_DIST = 0.25        # if something is this close, halt and turn
    # ----------------------------------------------------------------------

    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # 1) Check if we have arrived (position + orientation tolerance)
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_error = _norm_angle(gyaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_error) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached – stop and signal completion
        return 0.0, 0.0, True

    # 2) Desired heading toward the goal (global frame)
    desired_heading = math.atan2(dy, dx)

    # 3) Scan for obstacles
    scans = env.raycast_scan()  # [(bearing_rel, distance), ...]
    # Compute a repulsive angular offset based on nearby obstacles
    repulsive_sum = 0.0
    weight_sum = 0.0
    min_distance_ahead = REPULSIVE_DIST  # track the closest obstacle ahead

    for bearing, distance in scans:
        # distance == env.max_range (6.0) means no hit
        if distance >= 6.0:
            continue

        # Keep track of the closest thing directly ahead (|bearing| < 10°)
        if abs(bearing) < math.radians(10) and distance < min_distance_ahead:
            min_distance_ahead = distance

        if distance < REPULSIVE_DIST:
            # Weight grows as the obstacle gets nearer
            w = (REPULSIVE_DIST - distance) / REPULSIVE_DIST
            repulsive_sum += w * bearing
            weight_sum += w

    # If we are too close to something directly ahead, stop forward motion
    if min_distance_ahead < CLOSE_STOP_DIST:
        v_cmd = 0.0
    else:
        # Base forward speed (will be reduced later if we need to turn a lot)
        v_cmd = MAX_SPEED

    # 4) Combine attractive and repulsive directions
    if weight_sum > 0.0:
        # Average repulsive bearing (negative sign to steer away)
        avg_repulsive = repulsive_sum / weight_sum
        # Apply repulsive offset
        target_heading = desired_heading + K_REP * (-avg_repulsive)
    else:
        target_heading = desired_heading

    # 5) Compute angular error in robot's local frame
    heading_error = _norm_angle(target_heading - yaw)

    # 6) Angular velocity command (PD could be added, but P is enough)
    w_cmd = KP_ANGLE * heading_error
    # Clamp angular speed
    w_cmd = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, w_cmd))

    # 7) Reduce forward speed when we need to turn sharply
    # The larger the heading error, the slower we go.
    turn_factor = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
    v_cmd = v_cmd * turn_factor

    # Ensure we never command negative forward speed
    v_cmd = max(0.0, v_cmd)

    # 8) Return command; not done yet
    return v_cmd, w_cmd, False


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
