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


def _wrap_angle(a: float) -> float:
    """Wrap angle to the interval [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple state‑based controller with obstacle avoidance.

    1. Turn toward the goal if the heading error is larger than a small threshold.
    2. When roughly facing the goal, check the forward scan:
       * If something is too close ahead, rotate away from it.
       * Otherwise drive forward while keeping a modest speed.
    3. If side obstacles become very close, add a small steering bias
       to keep a safe corridor.
    4. Stop and report success when position and yaw are within the tolerances.
    """
    # ------------------------------------------------------------------
    # Parameters (tuned for safety)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.8          # m/s
    MAX_ANG = 1.0            # rad/s
    Kp_ang = 1.5             # proportional gain for angular control
    HEADING_THRESH = 0.15    # rad – when to start rotating toward goal
    SAFE_FRONT = 0.8         # m – distance considered unsafe straight ahead
    SIDE_SAFE = 0.6          # m – side clearance threshold
    # ------------------------------------------------------------------

    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # Vector to goal in world frame
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    # Success check
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # Desired heading (world frame) and heading error (robot frame)
    goal_angle = math.atan2(dy, dx)
    heading_err = _wrap_angle(goal_angle - yaw)

    # Get laser scan
    scan = env.raycast_scan()  # list of (bearing, distance)

    # ------------------------------------------------------------------
    # 1) Align to goal direction if needed
    # ------------------------------------------------------------------
    if abs(heading_err) > HEADING_THRESH:
        # Rotate in place toward the goal
        w = max(-MAX_ANG, min(MAX_ANG, Kp_ang * heading_err))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 2) Check forward clearance
    # ------------------------------------------------------------------
    # Front distances: bearings within +/- 0.1 rad are considered "ahead"
    front_distances = [d for b, d in scan if abs(b) < 0.1 and d < 6.0]
    front_dist = min(front_distances) if front_distances else 6.0

    if front_dist < SAFE_FRONT:
        # Obstacle too close ahead – decide turn direction based on side clearance
        left_distances = [d for b, d in scan if 0.3 < b < 1.2 and d < 6.0]
        right_distances = [d for b, d in scan if -1.2 < b < -0.3 and d < 6.0]
        left_min = min(left_distances) if left_distances else 6.0
        right_min = min(right_distances) if right_distances else 6.0

        # Turn towards the side with more space
        if left_min > right_min:
            w = MAX_ANG   # turn left
        else:
            w = -MAX_ANG  # turn right
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 3) Drive forward, add minor steering if side obstacles are close
    # ------------------------------------------------------------------
    # Base forward speed proportional to distance, capped
    v = min(MAX_SPEED, 0.6 * dist_to_goal)

    # Side clearance for steering bias
    left_distances = [d for b, d in scan if 0.3 < b < 1.2 and d < 6.0]
    right_distances = [d for b, d in scan if -1.2 < b < -0.3 and d < 6.0]
    left_min = min(left_distances) if left_distances else 6.0
    right_min = min(right_distances) if right_distances else 6.0

    if left_min < SIDE_SAFE:
        w = -0.5   # steer right a bit
    elif right_min < SIDE_SAFE:
        w = 0.5    # steer left a bit
    else:
        w = 0.0

    # Clamp angular speed
    if w > MAX_ANG:
        w = MAX_ANG
    if w < -MAX_ANG:
        w = -MAX_ANG

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
