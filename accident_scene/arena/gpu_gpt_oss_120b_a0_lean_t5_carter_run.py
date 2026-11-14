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
    """Wrap angle to [-pi, pi)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic go‑to‑goal controller based on a simple potential‑field.
    The robot continuously combines an attractive vector toward the goal with
    repulsive vectors from any obstacle detected by the 180° ray‑fan.  The
    resulting heading is used to compute a bounded yaw‑rate command.  Forward
    speed is scaled down when obstacles are close, and it is halted completely
    if something is directly in front (< 0.3 m) to avoid collisions.
    """
    # ------------------------------------------------------------------
    # 1) unpack current pose and goal
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # Vector from robot to goal (world frame)
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    # Desired heading (world frame) and yaw error
    desired_heading = math.atan2(dy, dx)
    yaw_error = _wrap_angle(desired_heading - yaw)

    # ------------------------------------------------------------------
    # 2) Goal reached?  Align yaw then finish.
    # ------------------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True          # success
        # Rotate in place to correct orientation
        w_align = max(-0.8, min(0.8, 2.0 * yaw_error))
        return 0.0, w_align, False

    # ------------------------------------------------------------------
    # 3) Acquire laser scan and compute repulsive field
    # ------------------------------------------------------------------
    scan = env.raycast_scan()               # [(bearing, distance), ...]
    avoid_range = 1.5                       # start repelling inside this radius
    rep_x, rep_y = 0.0, 0.0                  # robot‑frame repulsive vector

    for bearing, distance in scan:
        if distance < avoid_range:
            # Weight grows the nearer the obstacle
            w = (avoid_range - distance) / avoid_range
            # Direction *away* from the obstacle in robot frame
            rep_x -= w * math.cos(bearing)
            rep_y -= w * math.sin(bearing)

    # ------------------------------------------------------------------
    # 4) Attractive vector toward goal (expressed in robot frame)
    # ------------------------------------------------------------------
    # Transform goal direction into robot frame
    goal_rel_angle = _wrap_angle(desired_heading - yaw)
    att_x = math.cos(goal_rel_angle)
    att_y = math.sin(goal_rel_angle)

    # ------------------------------------------------------------------
    # 5) Blend vectors
    # ------------------------------------------------------------------
    # We give repulsion a slightly higher weight to stay safe
    k_att = 1.0
    k_rep = 1.5
    comb_x = k_att * att_x + k_rep * rep_x
    comb_y = k_att * att_y + k_rep * rep_y

    # If the combined vector is near zero (e.g., trapped), fall back to pure
    # attractive direction so the robot keeps trying to move.
    if math.hypot(comb_x, comb_y) < 1e-3:
        comb_x, comb_y = att_x, att_y

    # Desired steering direction in robot frame
    steer_angle = math.atan2(comb_y, comb_x)   # [-pi, pi]

    # ------------------------------------------------------------------
    # 6) Compute commands
    # ------------------------------------------------------------------
    max_turn = 0.8                     # rad/s
    w = max(-max_turn, min(max_turn, 2.0 * steer_angle))

    # Base forward speed
    base_speed = 1.0

    # Speed scaling factors
    # a) Slow down when heading error is large (to avoid overshoot)
    speed_factor = max(0.2, 1.0 - min(1.0, abs(yaw_error) / (math.pi / 2)))

    # b) Reduce speed according to the closest obstacle in front sector
    #    (±30°) – the closer the obstacle, the slower we go.
    front_sector = [d for b, d in scan if -0.5236 <= b <= 0.5236]
    min_front = min(front_sector) if front_sector else 6.0
    if min_front < avoid_range:
        speed_factor *= max(0.1, min_front / avoid_range)

    # c) If something is extremely close directly ahead, stop completely
    if min_front < 0.3:
        v = 0.0
        w = max(-max_turn, min(max_turn, 2.0 * steer_angle))  # still allow gentle turn
    else:
        v = base_speed * speed_factor

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
