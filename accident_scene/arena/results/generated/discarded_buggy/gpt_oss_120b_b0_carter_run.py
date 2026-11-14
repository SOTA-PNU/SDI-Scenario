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


def _normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Simple reactive controller for the warehouse arena.

    - Steer toward the goal using a proportional heading controller.
    - Use the forward 180° ray‑scan to avoid obstacles:
        * If something is closer than `SAFE_DIST` in front, turn away.
        * Otherwise drive forward at a speed reduced proportionally to the
          closest obstacle distance.
    - Declare `done` when the robot is inside the goal disc and its yaw is
      within the allowed tolerance.
    """
    # ---- constants ---------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG = 1.0                 # rad/s
    KP_ANGLE = 1.5                # proportional gain for heading error
    SAFE_DIST = 0.8               # meters – start evasive action inside this
    MIN_SPEED = 0.1               # keep a tiny forward motion when possible

    # ---- unpack current state ------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ---- check for termination ---------------------------------------------
    # distance to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # yaw error (desired yaw = goal_yaw)
    yaw_error = _normalize_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_error) <= MISSION["yaw_tolerance_rad"]):
        # Goal reached and correctly oriented
        return 0.0, 0.0, True

    # ---- heading toward goal ------------------------------------------------
    desired_heading = math.atan2(dy, dx)          # world frame angle to goal
    heading_error = _normalize_angle(desired_heading - yaw)

    # base angular command (proportional)
    w = KP_ANGLE * heading_error
    # clamp
    w = max(-MAX_ANG, min(MAX_ANG, w))

    # ---- obstacle sensing ---------------------------------------------------
    # Ray‑cast scan: list of (bearing, distance) where bearing is relative to
    # current heading (yaw). 0 rad is straight ahead.
    scan = env.raycast_scan()  # uses default parameters defined in the harness

    # Find the closest obstacle within the forward 120° (±60°) sector
    min_dist = None
    min_bearing = 0.0
    for bearing, dist in scan:
        # consider only obstacles that are not at max range
        if dist >= 6.0:
            continue
        # focus on a sector around the forward direction
        if abs(bearing) > math.radians(60):
            continue
        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
            min_bearing = bearing

    # ---- reactive avoidance -------------------------------------------------
    if min_dist is not None and min_dist < SAFE_DIST:
        # Obstacle too close: stop forward motion and turn away
        v = 0.0
        # Turn direction opposite to obstacle bearing
        turn_dir = -1.0 if min_bearing > 0 else 1.0
        w = turn_dir * MAX_ANG
        return v, w, False

    # ---- speed modulation based on closest obstacle ------------------------
    if min_dist is None:
        # No obstacle in the considered sector
        speed_factor = 1.0
    else:
        # Linear scaling: full speed at SAFE_DIST, stop at 0 (should have been
        # caught by the previous block)
        speed_factor = min_dist / SAFE_DIST
        speed_factor = max(0.0, min(1.0, speed_factor))

    v = MAX_SPEED * speed_factor
    # Ensure a minimum forward motion so the robot does not stall when far
    if v < MIN_SPEED:
        v = MIN_SPEED

    # Combine goal‑heading and obstacle‑avoidance angular commands.
    # If an obstacle is moderately close (but not within SAFE_DIST) we can
    # bias the turn away a bit.
    if min_dist is not None and min_dist < 2.0 * SAFE_DIST:
        # Repulsive angular term proportional to how close the obstacle is
        repulse = (2.0 * SAFE_DIST - min_dist) / (2.0 * SAFE_DIST)  # 0..1
        # Turn opposite to bearing
        w += -repulse * math.copysign(MAX_ANG, min_bearing)

        # Clamp again
        w = max(-MAX_ANG, min(MAX_ANG, w))

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
