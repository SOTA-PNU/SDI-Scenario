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


# Global mode for a simple bug‑style behaviour.
# Starts by heading straight to the goal; if an obstacle blocks the way
# the robot will follow the obstacle until a clear path to the goal appears.
_MODE = "go_to_goal"


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """Potential‑field / bug hybrid controller.

    The robot drives toward the goal while repelling from obstacles detected
    by the forward‑facing 180° ray fan.  When an obstacle is directly ahead
    the controller switches to a simple wall‑following mode until the goal
    direction becomes clear again.
    """
    global _MODE

    # ------------------------------------------------------------------
    # Tunable parameters
    # ------------------------------------------------------------------
    MAX_SPEED = 1.2          # nominal forward speed (m/s)
    SAFE_DIST = 1.5          # distance within which obstacles exert repulsion (m)
    MIN_STOP_DIST = 0.2      # stop if something is this close ahead (m)
    FRONT_ANGLE = math.radians(20)   # sector considered “directly ahead”
    K_ATT = 1.0              # attractive weight (goal)
    K_REP = 2.0              # repulsive weight (obstacles)
    K_YAW = 2.0              # proportional gain for angular velocity
    MAX_ANGULAR = 1.5        # hard limit on yaw rate (rad/s)

    # ------------------------------------------------------------------
    # Goal information (global)
    # ------------------------------------------------------------------
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # Current robot state
    # ------------------------------------------------------------------
    x, y, yaw = pose

    # ------------------------------------------------------------------
    # Vector to goal (world frame) and distance
    # ------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Goal‑region handling (stop and align yaw)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        # Inside the goal disc – stop forward motion and rotate to goal yaw.
        v_cmd = 0.0
        yaw_error = _wrap_angle(goal_yaw - yaw)
        w_cmd = K_YAW * yaw_error
        # clamp angular speed
        w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, w_cmd))
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True
        else:
            return v_cmd, w_cmd, False

    # ------------------------------------------------------------------
    # Goal bearing relative to robot heading (range [-pi, pi])
    # ------------------------------------------------------------------
    goal_bearing = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # Sensor processing
    # ------------------------------------------------------------------
    scan = env.raycast_scan()   # defaults: 61 beams, 180°, max_range=6.0

    # Minimum distance in a narrow front sector (for speed scaling / mode switch)
    min_front = float('inf')
    # Accumulate repulsive contribution from all beams inside SAFE_DIST
    rep_x = 0.0
    rep_y = 0.0

    for bearing, dist in scan:
        # front sector distance
        if abs(bearing) <= FRONT_ANGLE and dist < min_front:
            min_front = dist

        # repulsive contribution when obstacle is within SAFE_DIST
        if dist < SAFE_DIST:
            mag = (SAFE_DIST - dist) / SAFE_DIST          # 0 .. 1
            rep_x -= mag * math.cos(bearing)              # push away
            rep_y -= mag * math.sin(bearing)

    # If nothing was seen in the front sector treat it as far away
    if min_front == float('inf'):
        min_front = SAFE_DIST

    # ------------------------------------------------------------------
    # Mode transition logic (bug‑style)
    # ------------------------------------------------------------------
    OBSTACLE_AHEAD = min_front < MIN_STOP_DIST + 0.1   # a little margin

    if _MODE == "go_to_goal":
        if OBSTACLE_AHEAD:
            _MODE = "follow_obstacle"
    else:   # follow_obstacle
        # Exit the obstacle‑following mode when the goal direction looks clear
        # (no obstacle within SAFE_DIST directly ahead and the goal bearing is
        # roughly aligned with the forward direction).
        if min_front > SAFE_DIST and abs(goal_bearing) < math.radians(15):
            _MODE = "go_to_goal"

    # ------------------------------------------------------------------
    # Compute attractive direction (goal) in robot frame
    # ------------------------------------------------------------------
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # Rotate world vector into robot frame
    dx_body = cos_y * dx + sin_y * dy
    dy_body = -sin_y * dx + cos_y * dy
    if dist_to_goal > 0.0:
        att_x = dx_body / dist_to_goal
        att_y = dy_body / dist_to_goal
    else:
        att_x = att_y = 0.0

    # ------------------------------------------------------------------
    # Combine attractive and repulsive vectors
    # ------------------------------------------------------------------
    comb_x = K_ATT * att_x + K_REP * rep_x
    comb_y = K_ATT * att_y + K_REP * rep_y

    # Desired heading in robot frame (relative to forward = 0)
    desired_heading = math.atan2(comb_y, comb_x)

    # ------------------------------------------------------------------
    # Angular velocity command (proportional to heading error)
    # ------------------------------------------------------------------
    w_cmd = K_YAW * desired_heading
    w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, w_cmd))

    # ------------------------------------------------------------------
    # Forward speed command – reduced when turning sharply or obstacle close
    # ------------------------------------------------------------------
    # Reduce speed when the required turn is large
    turn_factor = max(0.1, 1.0 - abs(desired_heading) / (math.pi / 2))

    # Reduce speed when something is close ahead
    if min_front < SAFE_DIST:
        obstacle_factor = min_front / SAFE_DIST
        if min_front < MIN_STOP_DIST:
            obstacle_factor = 0.0
    else:
        obstacle_factor = 1.0

    v_cmd = MAX_SPEED * turn_factor * obstacle_factor

    # ------------------------------------------------------------------
    # In obstacle‑following mode we bias the turn a bit to the left (or right)
    # to keep the robot moving around the obstacle.
    # ------------------------------------------------------------------
    if _MODE == "follow_obstacle":
        # Simple bias: add a small constant turn to the left.
        # The sign could be flipped; left turn works well for the given arena.
        w_cmd += 0.5          # rad/s left bias
        # Clamp again after bias
        w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, w_cmd))
        # Keep a modest forward speed while following
        v_cmd = min(v_cmd, MAX_SPEED * 0.5)

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
