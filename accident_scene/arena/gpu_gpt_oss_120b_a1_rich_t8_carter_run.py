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
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def controller(t, pose, env):
    """
    Potential‑field controller.
    * Attractive force points to the goal.
    * Repulsive forces push away from any obstacle inside REP_RANGE.
    * Resulting vector determines a desired heading.
    * Yaw rate is proportional to heading error.
    * Forward speed is scaled by alignment, distance to goal and clearance ahead.
    """
    # -------------------- parameters --------------------
    MAX_SPEED = 2.0               # m/s
    KP_ANGLE = 2.0                # rad^-1 (steering gain)
    MAX_TURN = 1.5                # rad/s
    REP_RANGE = 2.0               # m – obstacles farther than this are ignored
    REP_GAIN = 0.8                # strength of repulsion
    ATT_GAIN = 1.0                # strength of attraction
    SAFE_FRONT = 0.3              # m – stop if something this close ahead
    FRONT_ANGLE = math.radians(30)  # rad – half‑width of forward sector
    # --------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- vector to goal (attractive) -----
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal > 0.0:
        att_x = (dx / dist_to_goal) * ATT_GAIN
        att_y = (dy / dist_to_goal) * ATT_GAIN
    else:
        att_x = att_y = 0.0

    # ----- repulsive forces from obstacles -----
    scans = env.raycast_scan()          # [(bearing, distance), ...]
    rep_x = 0.0
    rep_y = 0.0
    for bearing, dist in scans:
        if 0.0 < dist < REP_RANGE:
            # Weight grows as we get nearer (inverse distance)
            w = (REP_RANGE - dist) / REP_RANGE
            # Direction away from obstacle in world frame:
            # obstacle bearing is relative to robot heading
            obs_angle = yaw + bearing
            away_x = -math.cos(obs_angle)
            away_y = -math.sin(obs_angle)
            # Inverse‑square scaling to keep influence limited
            scale = REP_GAIN * w / (dist * dist + 0.01)
            rep_x += away_x * scale
            rep_y += away_y * scale

    # ----- combine attractive and repulsive components -----
    comb_x = att_x + rep_x
    comb_y = att_y + rep_y

    # If the combined vector is near zero (e.g., trapped), fall back to pure goal direction
    if math.hypot(comb_x, comb_y) < 1e-3:
        comb_x, comb_y = att_x, att_y

    desired_heading = math.atan2(comb_y, comb_x)
    angle_error = _wrap_angle(desired_heading - yaw)

    # ----- turn rate (proportional) -----
    w = max(-MAX_TURN, min(MAX_TURN, KP_ANGLE * angle_error))

    # ----- forward speed -----
    # Closest obstacle in the forward sector
    front_distances = [d for b, d in scans if abs(b) <= FRONT_ANGLE]
    min_front = min(front_distances) if front_distances else 6.0

    if min_front < SAFE_FRONT:
        v = 0.0
    else:
        # Alignment factor (how well we are pointing toward the combined direction)
        alignment = max(0.0, math.cos(angle_error))
        # Slow down when we are close to the goal
        distance_factor = min(1.0, dist_to_goal / 5.0)
        # Reduce speed if something is still relatively close ahead
        obstacle_factor = min(1.0, (min_front - SAFE_FRONT) / (REP_RANGE - SAFE_FRONT))
        v = MAX_SPEED * alignment * distance_factor * obstacle_factor

    # ----- termination test (same as above, but after computing controls) -----
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

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
