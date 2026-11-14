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

def _norm_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Reactive controller for Track B.

    * Drives toward the goal while respecting a simple potential‑field
      repulsion from any obstacle detected within SAFE_RANGE.
    * If something is very close in front, the robot stops forward motion
      and rotates toward the direction with the greatest clearance.
    * Stops when both position and yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (all distances are in metres, angles in radians)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0               # maximum forward speed
    MAX_ANG_SPEED = 1.0           # maximum yaw rate
    SAFE_RANGE = 2.0              # distance within which obstacles repel the robot
    CLOSE_FRONT = 0.45            # if something this close ahead → stop forward
    FRONT_ANGLE = math.radians(30)  # sector considered “ahead” for the stop check
    K_REP = 1.5                   # weight of the repulsive vector
    # ----------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- vector to goal (world frame) ------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_dir_world = math.atan2(dy, dx)
    angle_error = _norm_angle(goal_dir_world - yaw)

    # ----- termination ------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_norm_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached – stop the robot
        return 0.0, 0.0, True

    # ----- sensor scan ------------------------------------------------------
    scan = env.raycast_scan()  # 61 beams, 180°, max_range=6.0

    # Minimum distance in the forward sector (used for an emergency stop)
    min_front_dist = None
    for bearing, distance in scan:
        if abs(bearing) <= FRONT_ANGLE:
            if min_front_dist is None or distance < min_front_dist:
                min_front_dist = distance

    # ----- repulsive vector (potential‑field style) -------------------------
    rep_x = 0.0
    rep_y = 0.0
    for bearing, distance in scan:
        if distance < SAFE_RANGE:                     # only consider close obstacles
            weight = (SAFE_RANGE - distance) / SAFE_RANGE
            # Vector pointing *away* from the obstacle, expressed in robot frame
            rep_x += -weight * math.cos(bearing)
            rep_y += -weight * math.sin(bearing)

    # Goal direction expressed in robot frame (unit vector)
    goal_vec_x = math.cos(angle_error)
    goal_vec_y = math.sin(angle_error)

    # Combine attractive (goal) and repulsive vectors
    comb_x = goal_vec_x + K_REP * rep_x
    comb_y = goal_vec_y + K_REP * rep_y
    steer_angle = math.atan2(comb_y, comb_x)          # desired heading change

    # ----- angular velocity -------------------------------------------------
    w = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, steer_angle))

    # ----- forward speed ----------------------------------------------------
    # If something is very close ahead, stop forward motion and rely on rotation.
    if min_front_dist is not None and min_front_dist < CLOSE_FRONT:
        v = 0.0
    else:
        # Base speed proportional to clearance ahead (up to SAFE_RANGE)
        if min_front_dist is None:
            clearance_factor = 1.0
        else:
            clearance_factor = min(1.0, min_front_dist / SAFE_RANGE)

        # Slow down when we are close to the goal to improve final accuracy
        if dist_to_goal < 2.0:
            goal_factor = dist_to_goal / 2.0
        else:
            goal_factor = 1.0

        v = MAX_SPEED * clearance_factor * goal_factor

    # Clamp forward speed to allowed range
    v = max(0.0, min(MAX_SPEED, v))

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
