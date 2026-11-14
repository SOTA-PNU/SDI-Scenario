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
    pose  (x, y, yaw) ground‑truth world pose of the chassis
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

# Persistent state – used only for deterministic “choose a turn direction”
_last_turn_dir = 0  # -1 = left, 1 = right, 0 = none


def _wrap_angle(a: float) -> float:
    """Wrap angle to the interval [-π, π)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic go‑to‑goal controller with simple potential‑field avoidance.

    The algorithm is deliberately lightweight:
      * Attractive unit vector toward the goal (expressed in robot frame).
      * Repulsive vectors from any obstacle inside a “danger zone”.
      * A proportional yaw controller.
      * Speed is reduced when turning sharply or when obstacles are near.
    """
    global _last_turn_dir

    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # ------------------------------------------------------------------
    # 1) Goal proximity – stop and align if we are inside the tolerance.
    # ------------------------------------------------------------------
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    if dist_to_goal <= MISSION["position_tolerance_m"]:
        # Align to the exact goal yaw.
        yaw_err = _wrap_angle(gyaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True          # success
        # Rotate in‑place until yaw is within tolerance.
        w = max(-0.8, min(0.8, 2.0 * yaw_err))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 2) Sensor reading.
    # ------------------------------------------------------------------
    scan = env.raycast_scan()                     # [(bearing, distance), …]

    # ------------------------------------------------------------------
    # 3) Emergency turn if something is dangerously close straight ahead.
    # ------------------------------------------------------------------
    # Look at a narrow forward cone (±30°) for the *closest* obstacle.
    min_front_dist = float("inf")
    min_front_bearing = 0.0
    for bearing, dist in scan:
        if -0.5236 <= bearing <= 0.5236 and dist < min_front_dist:
            min_front_dist = dist
            min_front_bearing = bearing

    emergency_dist = 0.6   # meters – trigger a hard turn
    if min_front_dist < emergency_dist:
        # Turn away from the obstacle bearing.  Remember direction to avoid flip‑flop.
        turn_dir = -1 if min_front_bearing > 0 else 1
        _last_turn_dir = turn_dir
        w = turn_dir * 1.0            # rad/s – fairly quick turn
        v = 0.2                       # crawl while turning
        return v, w, False

    # ------------------------------------------------------------------
    # 4) Build attractive and repulsive vectors (robot‑frame).
    # ------------------------------------------------------------------
    # Attractive unit vector (world → robot frame)
    goal_world_yaw = math.atan2(dy, dx)
    rel_goal_angle = _wrap_angle(goal_world_yaw - yaw)
    att_x = math.cos(rel_goal_angle)
    att_y = math.sin(rel_goal_angle)

    # Repulsive contribution from any scan point inside the danger radius.
    danger_radius = 2.0                # meters
    rep_x = 0.0
    rep_y = 0.0
    rep_coeff = 0.8                    # scaling factor for repulsion strength

    for bearing, dist in scan:
        if dist >= danger_radius:
            continue
        # Inverse‑square style magnitude, clipped to avoid singularities.
        mag = rep_coeff * (1.0 / max(dist, 0.05) - 1.0 / danger_radius)
        # Vector pointing *away* from the obstacle, expressed in robot frame.
        rep_x += -math.cos(bearing) * mag
        rep_y += -math.sin(bearing) * mag

    # Combine the two vectors.
    comb_x = att_x + rep_x
    comb_y = att_y + rep_y

    # If the sum collapses (e.g., surrounded), fall back to pure attraction.
    if math.hypot(comb_x, comb_y) < 1e-3:
        comb_x, comb_y = att_x, att_y

    # Desired heading relative to current robot orientation.
    desired_rel_angle = math.atan2(comb_y, comb_x)

    # ------------------------------------------------------------------
    # 5) Yaw rate command (simple proportional controller).
    # ------------------------------------------------------------------
    kp_yaw = 1.7
    w = max(-1.0, min(1.0, kp_yaw * desired_rel_angle))

    # ------------------------------------------------------------------
    # 6) Forward speed modulation.
    # ------------------------------------------------------------------
    # Reduce speed when we need to turn sharply.
    turn_factor = max(0.15, 1.0 - min(1.0, abs(desired_rel_angle) / (math.pi / 2)))
    # Reduce speed when any obstacle is relatively close (< 1.5 m).
    close_obstacle = any(d < 1.5 for _, d in scan)
    obstacle_factor = 0.4 if close_obstacle else 1.0

    v_nominal = 1.3                 # nominal free‑space speed (m/s)
    v = v_nominal * turn_factor * obstacle_factor

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
