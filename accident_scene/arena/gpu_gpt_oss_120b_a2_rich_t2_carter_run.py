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
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    * Attractive vector points toward the goal.
    * Repulsive vectors are generated from any obstacle detected within
      SAFE_DIST (0.8 m).  The repulsion strength grows quickly the closer the
      obstacle.
    * The resultant vector defines a desired heading; a proportional
      controller produces linear and angular commands.
    * If an obstacle is directly ahead (< 0.5 m, |bearing| < 0.3 rad) the robot
      stops forward motion to avoid collision.
    * Completion is reported when position and yaw are inside the tolerances
      defined in MISSION.
    """
    # ------------------------------------------------------------------
    # Tuning constants (chosen to be safe for the arena size)
    # ------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG = 1.2                 # rad/s
    Kp_speed = 0.6                # linear gain (m/s per meter)
    Kp_ang = 1.8                  # angular gain (rad/s per rad)
    SAFE_DIST = 0.8               # meters – distance at which we start repelling
    REPULSION_GAIN = 0.9          # scaling factor for repulsive force
    STOP_DIST = 0.5               # stop if something this close is in front
    STOP_BEARING = 0.3            # rad – half‑width of the “front” cone

    # ------------------------------------------------------------------
    # Unpack current pose and goal
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # Vector from robot to goal
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    goal_angle = math.atan2(dy, dx)

    # ------------------------------------------------------------------
    # Success test – stop as soon as we are inside both tolerances
    # ------------------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Attractive component (unit vector toward goal, weighted by distance)
    # ------------------------------------------------------------------
    att_weight = min(1.0, dist_to_goal / 10.0)          # fade far away
    att_x = math.cos(goal_angle) * att_weight
    att_y = math.sin(goal_angle) * att_weight

    # ------------------------------------------------------------------
    # Repulsive component from ray‑scan
    # ------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    scan = env.raycast_scan()
    for bearing, dist in scan:
        if dist >= 6.0:               # max_range → no hit
            continue
        if dist >= SAFE_DIST:         # too far to matter
            continue

        # In robot‑frame the obstacle lies at (dist*cos(bearing), dist*sin(bearing))
        # Repulsion points opposite to that direction.
        # Strength grows as (1/dist - 1/SAFE_DIST) → stronger the closer we are.
        strength = REPULSION_GAIN * (1.0 / dist - 1.0 / SAFE_DIST)
        rep_x -= strength * math.cos(bearing)
        rep_y -= strength * math.sin(bearing)

    # ------------------------------------------------------------------
    # Combine attractive and repulsive vectors
    # ------------------------------------------------------------------
    cmd_x = att_x + rep_x
    cmd_y = att_y + rep_y

    # Desired heading from the combined vector
    desired_angle = math.atan2(cmd_y, cmd_x)
    heading_error = _wrap_angle(desired_angle - yaw)

    # ------------------------------------------------------------------
    # Linear speed – proportional to forward component of the combined vector
    # ------------------------------------------------------------------
    forward_component = math.cos(heading_error) * min(MAX_SPEED, Kp_speed * dist_to_goal)
    speed = max(0.0, forward_component)   # never command reverse

    # ------------------------------------------------------------------
    # Angular speed – simple proportional controller
    # ------------------------------------------------------------------
    ang = Kp_ang * heading_error

    # ------------------------------------------------------------------
    # Safety stop: if something is very close directly ahead, halt forward motion
    # ------------------------------------------------------------------
    front_blocked = any(
        abs(bearing) < STOP_BEARING and dist < STOP_DIST
        for bearing, dist in scan if dist < 6.0
    )
    if front_blocked:
        speed = 0.0

    # ------------------------------------------------------------------
    # Clamp commands to allowed limits
    # ------------------------------------------------------------------
    speed = min(speed, MAX_SPEED)
    if ang > MAX_ANG:
        ang = MAX_ANG
    elif ang < -MAX_ANG:
        ang = -MAX_ANG

    # ------------------------------------------------------------------
    # Return the command tuple
    # ------------------------------------------------------------------
    return speed, ang, False


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
