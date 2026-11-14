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
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def controller(t, pose, env):
    """
    Reactive “bug‑algorithm” controller (Track B).

    * State = "go_to_goal": head straight for the goal.
    * If an obstacle appears within SAFE_DIST in front, switch to
      "follow_obstacle" and keep the obstacle on the left side (wall‑following).
    * When the heading to the goal becomes clear again, return to "go_to_goal".
    * Stop when position & yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (all distances in metres, angles in radians)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0          # m /s
    MAX_ANG_SPEED = 1.0      # rad /s
    SAFE_DIST = 0.8          # keep at least this far from anything
    FRONT_ANGLE = math.radians(30)   # sector considered “ahead” for the stop check
    LEFT_ANGLE = math.radians(80)    # approximate bearing for the left‑side ray
    RIGHT_ANGLE = -LEFT_ANGLE        # mirror for the right side (unused here)
    FOLLOW_SPEED = 0.5      # forward speed while wall‑following
    # ----------------------------------------------------------------------

    # ---- initialise persistent state on first call -----------------------
    if not hasattr(controller, "state"):
        controller.state = "go_to_goal"   # "go_to_goal" or "follow_obstacle"
        controller.side = "left"          # which side we keep the obstacle
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
        return 0.0, 0.0, True

    # ----- sensor scan ------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams, 180°, max_range = 6.0

    # Gather a few useful distances
    front_dist = None
    left_dist = None
    for bearing, d in scan:
        # front sector (used for emergency stop & obstacle detection)
        if abs(bearing) <= FRONT_ANGLE:
            front_dist = d if front_dist is None else min(front_dist, d)

        # left‑most beam (approx. +90°) – we use the closest reading on the left side
        if bearing >= LEFT_ANGLE:
            left_dist = d if left_dist is None else min(left_dist, d)

    # If the scan returns max_range (=6.0) we treat it as “no obstacle”
    if front_dist is None:
        front_dist = 6.0
    if left_dist is None:
        left_dist = 6.0

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------
    if controller.state == "go_to_goal":
        # Obstacle directly ahead? → start wall‑following
        if front_dist < SAFE_DIST:
            controller.state = "follow_obstacle"
            # keep the obstacle on the left side (could be right as well)
            controller.side = "left"

        # Normal go‑to‑goal behaviour
        w = _clamp(angle_error, -MAX_ANG_SPEED, MAX_ANG_SPEED)

        # Slow down when something is close in front, otherwise run at full speed
        if front_dist < SAFE_DIST:
            v = 0.0
        elif front_dist < 2.0 * SAFE_DIST:
            v = MAX_SPEED * (front_dist / (2.0 * SAFE_DIST))
        else:
            # Reduce speed when we are close to the goal for better final accuracy
            if dist_to_goal < 2.0:
                v = MAX_SPEED * (dist_to_goal / 2.0)
            else:
                v = MAX_SPEED

        v = _clamp(v, 0.0, MAX_SPEED)

    else:   # controller.state == "follow_obstacle"
        # Simple left‑wall‑following
        # Keep a target distance on the left side
        target_left = SAFE_DIST

        # Determine turning direction
        if left_dist > target_left + 0.2:
            # Too far from left wall → steer left
            w = 0.5
        elif left_dist < target_left - 0.2:
            # Too close to left wall → steer right
            w = -0.5
        else:
            # Good distance → go straight
            w = 0.0

        w = _clamp(w, -MAX_ANG_SPEED, MAX_ANG_SPEED)

        # Forward speed only if front is clear enough
        if front_dist > SAFE_DIST + 0.2:
            v = FOLLOW_SPEED
        else:
            v = 0.0

        # Check whether we can switch back to “go to goal”
        # Condition: heading to goal is roughly aligned and front is clear
        if (abs(angle_error) < 0.25) and (front_dist > SAFE_DIST + 0.5):
            controller.state = "go_to_goal"

        v = _clamp(v, 0.0, MAX_SPEED)

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
