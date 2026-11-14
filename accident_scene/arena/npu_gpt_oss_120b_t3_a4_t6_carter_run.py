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
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Potential‑field navigation with a simple stop‑and‑turn fallback.

    The robot is attracted toward the goal while repelled by obstacles detected
    by the forward 180° ray‑cast.  If an obstacle is detected closer than
    ``SAFE_FRONT`` directly ahead, the robot stops and rotates toward the side
    (left/right) with the larger clearance.  When the robot reaches the goal
    disc it aligns its yaw and reports ``done=True``.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Pose & goal geometry
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # 2️⃣  Goal‑arrival test (position + yaw)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if distance_to_goal <= pos_tol:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= yaw_tol:
            return 0.0, 0.0, True
        # rotate in place to align yaw
        w = 1.0 * math.tanh(2.0 * yaw_err)
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 3️⃣  Sensor scan – front / side distances
    # ------------------------------------------------------------------
    scans = env.raycast_scan()   # [(bearing, distance), ...]

    def min_dist(b_min, b_max):
        d = 6.0
        for bearing, dist in scans:
            if b_min <= bearing <= b_max and dist < d:
                d = dist
        return d

    front_min = min_dist(-math.pi / 6, math.pi / 6)          # ±30°
    left_min  = min_dist(math.pi / 3, math.pi / 2)          # +60° … +90°
    right_min = min_dist(-math.pi / 2, -math.pi / 3)        # -90° … -60°

    # ------------------------------------------------------------------
    # 4️⃣  Parameters
    # ------------------------------------------------------------------
    MAX_SPEED   = 0.9      # [m/s] nominal forward speed
    MAX_ANGULAR = 1.0      # [rad/s] max yaw rate
    SAFE_FRONT  = 0.6      # [m] distance that triggers stop‑and‑turn
    KP_ANGLE    = 2.0      # gain for heading error
    REPULSION_RANGE = 2.0  # [m] distance within which obstacles repel
    K_REP = 0.8            # repulsion strength coefficient
    K_ATT = 1.0            # attraction strength coefficient

    # ------------------------------------------------------------------
    # 5️⃣  Immediate obstacle ahead → stop and rotate toward freer side
    # ------------------------------------------------------------------
    if front_min < SAFE_FRONT:
        turn_dir = 1 if left_min > right_min else -1
        v = 0.0
        w = turn_dir * MAX_ANGULAR
        return v, w, False

    # ------------------------------------------------------------------
    # 6️⃣  Potential‑field computation (world frame)
    # ------------------------------------------------------------------
    # Attractive component (goal direction)
    att_x = K_ATT * dx
    att_y = K_ATT * dy

    # Repulsive component from nearby obstacles
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scans:
        if dist >= REPULSION_RANGE or dist <= 0.0:
            continue
        # world bearing of this beam
        world_bearing = yaw + bearing
        # Simple inverse‑square repulsion scaled by proximity
        rep_strength = K_REP * (REPULSION_RANGE - dist) / (dist * dist)
        rep_x += -rep_strength * math.cos(world_bearing)
        rep_y += -rep_strength * math.sin(world_bearing)

    # Combined vector
    tot_x = att_x + rep_x
    tot_y = att_y + rep_y

    # Desired heading in world frame
    desired_heading = math.atan2(tot_y, tot_x)
    angle_error = _normalize_angle(desired_heading - yaw)

    # ------------------------------------------------------------------
    # 7️⃣  Command generation
    # ------------------------------------------------------------------
    w = MAX_ANGULAR * math.tanh(KP_ANGLE * angle_error)

    # Forward speed – reduced when heading is misaligned or when near goal
    speed_factor = max(0.0, math.cos(angle_error))
    approach_factor = min(1.0, distance_to_goal / 2.0)
    v = MAX_SPEED * speed_factor * approach_factor

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
