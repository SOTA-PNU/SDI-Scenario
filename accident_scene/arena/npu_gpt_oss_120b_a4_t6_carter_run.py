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
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Very small deterministic state‑machine:

    * PHASE_ORIENT  – first rotate to a safe heading (east, yaw = 0 rad) so
      the robot does not drive straight into the first box.
    * PHASE_NAVIGATE – drive toward the goal with simple proportional
      heading control and conservative obstacle avoidance.
    * The robot stops when position and yaw are within the tolerances.
    """
    # ------------------------------------------------------------------
    # Tunable deterministic parameters (no randomness)
    # ------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG_SPEED = 1.0           # rad/s
    Kp_ang = 1.5                  # proportional gain for heading error
    SAFE_DIST = 2.0               # start slowing when obstacle < this (m)
    STOP_DIST = 0.7               # stop forward motion if obstacle < this (m)
    FORWARD_CONE = math.radians(30)   # +/-30° considered forward
    SIDE_CONE = math.radians(90)      # side sectors up to +/-90°
    ORIENT_YAW = 0.0                  # safe heading after start (pointing east)
    ORIENT_TOL = 0.05                 # rad tolerance for orientation phase
    # ------------------------------------------------------------------

    # --------------------------------------------------------------
    # Persistent state initialisation
    # --------------------------------------------------------------
    if not hasattr(controller, "phase"):
        controller.phase = "orient"   # "orient" or "navigate"
        controller.turn_dir = 0      # direction used during avoidance

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # --------------------------------------------------------------
    # Goal reached?
    # --------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_error_to_goal) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # --------------------------------------------------------------
    # Sensor processing (common for both phases)
    # --------------------------------------------------------------
    scan = env.raycast_scan()               # [(bearing, distance), ...]
    min_fwd_dist = None
    min_fwd_bearing = None
    left_max = 0.0
    right_max = 0.0

    for bearing, dist in scan:
        # forward cone
        if abs(bearing) <= FORWARD_CONE:
            if (min_fwd_dist is None) or (dist < min_fwd_dist):
                min_fwd_dist = dist
                min_fwd_bearing = bearing
        # left side (0 to +SIDE_CONE)
        if 0.0 < bearing <= SIDE_CONE:
            if dist > left_max:
                left_max = dist
        # right side (-SIDE_CONE to 0)
        if -SIDE_CONE <= bearing < 0.0:
            if dist > right_max:
                right_max = dist

    # If nothing was seen in front, treat as max range
    if min_fwd_dist is None:
        min_fwd_dist = 6.0
        min_fwd_bearing = 0.0

    # --------------------------------------------------------------
    # Phase: ORIENT – rotate to ORIENT_YAW before any forward motion
    # --------------------------------------------------------------
    if controller.phase == "orient":
        orient_error = _wrap_angle(ORIENT_YAW - yaw)

        # If we are already within tolerance, switch to navigation
        if abs(orient_error) < ORIENT_TOL:
            controller.phase = "navigate"
            # fall through to navigation logic in this same step
        else:
            # Pure rotation, no forward motion
            v = 0.0
            w = max(-MAX_ANG_SPEED,
                    min(MAX_ANG_SPEED, Kp_ang * orient_error))
            return v, w, False

    # --------------------------------------------------------------
    # Phase: NAVIGATE – goal‑directed motion with obstacle avoidance
    # --------------------------------------------------------------

    # Immediate stop if something is dangerously close ahead
    if min_fwd_dist < STOP_DIST:
        # Choose turn direction based on which side offers more clearance
        if left_max > right_max:
            controller.turn_dir = 1   # turn left
        elif right_max > left_max:
            controller.turn_dir = -1  # turn right
        else:
            controller.turn_dir = 1   # default to left

        v = 0.0
        w = controller.turn_dir * MAX_ANG_SPEED
        return v, w, False

    # Desired heading toward the goal
    desired_heading = math.atan2(dy, dx)
    heading_error = _wrap_angle(desired_heading - yaw)

    # Angular command (proportional)
    w = max(-MAX_ANG_SPEED,
            min(MAX_ANG_SPEED, Kp_ang * heading_error))

    # Reduce speed when we need to turn sharply
    turn_factor = max(0.0, 1.0 - min(1.0, abs(heading_error) / (math.pi / 2)))

    # Additional slowdown when an obstacle is in the forward cone but not
    # close enough to trigger the STOP_DIST reaction.
    if min_fwd_dist < SAFE_DIST:
        obstacle_factor = max(0.0, (min_fwd_dist - STOP_DIST) / (SAFE_DIST - STOP_DIST))
    else:
        obstacle_factor = 1.0

    v = MAX_SPEED * turn_factor * obstacle_factor

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
