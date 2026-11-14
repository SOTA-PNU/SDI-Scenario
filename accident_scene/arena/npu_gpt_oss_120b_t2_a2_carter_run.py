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


def controller(t, pose, env):
    """Simple reactive controller using goal attraction and obstacle repulsion."""
    # ----- tunable parameters -------------------------------------------------
    MAX_SPEED = 1.2          # m/s, nominal forward speed
    MAX_TURN = 1.5           # rad/s, maximal yaw rate
    Kp_turn = 2.0            # proportional gain for heading correction
    SAFE_FRONT = 0.5         # m, distance considered unsafe directly ahead
    REPULSE_RANGE = 2.0      # m, range within which obstacles exert repulsion
    K_rep = 1.5              # weight of repulsive forces relative to attraction
    # -------------------------------------------------------------------------

    # helper to wrap angles to [-pi, pi]
    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # current robot pose
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # vector to the goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ----- goal handling ------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        # Within the goal disc – now only need to align yaw
        yaw_error = wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            # Goal reached and correctly oriented
            return 0.0, 0.0, True
        # Rotate in place to achieve correct yaw
        w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * yaw_error))
        return 0.0, w, False
    # -------------------------------------------------------------------------

    # desired heading toward the goal (world frame)
    desired_heading = math.atan2(dy, dx)
    # heading error expressed in robot's local frame
    heading_error = wrap_angle(desired_heading - yaw)

    # ----- laser scan ---------------------------------------------------------
    scans = env.raycast_scan()
    # front sector (±45°) used for speed limiting
    front_angle = math.radians(45)
    min_front = min((d for b, d in scans if abs(b) <= front_angle), default=6.0)

    # compute a simple repulsive vector from obstacles
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scans:
        if dist < REPULSE_RANGE:
            # Linear weight that grows as distance shrinks
            w_rep = (REPULSE_RANGE - dist) / REPULSE_RANGE
            rep_x -= math.cos(bearing) * w_rep
            rep_y -= math.sin(bearing) * w_rep
    # -------------------------------------------------------------------------

    # attractive direction expressed in robot frame (unit vector)
    att_x = math.cos(heading_error)
    att_y = math.sin(heading_error)

    # combine attraction and repulsion
    total_x = att_x + K_rep * rep_x
    total_y = att_y + K_rep * rep_y
    # safeguard against a zero vector (unlikely but possible)
    if total_x == 0.0 and total_y == 0.0:
        total_x = att_x
        total_y = att_y

    # resulting desired direction in robot frame
    desired_dir = math.atan2(total_y, total_x)
    # new heading error after considering obstacles
    new_error = wrap_angle(desired_dir)

    # ----- angular command ----------------------------------------------------
    w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * new_error))
    # -------------------------------------------------------------------------

    # ----- forward speed command ----------------------------------------------
    # Reduce speed when heading error is large
    speed_factor = max(0.0, math.cos(new_error))

    # Limit speed based on front clearance
    if min_front > SAFE_FRONT:
        speed_limit = MAX_SPEED * (min_front - SAFE_FRONT) / (6.0 - SAFE_FRONT)
    else:
        speed_limit = 0.0

    v = MAX_SPEED * speed_factor
    v = min(v, speed_limit)

    # Slow down as we approach the goal to avoid overshoot
    if dist_to_goal < 1.0:
        v *= dist_to_goal / 1.0
    # -------------------------------------------------------------------------

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
