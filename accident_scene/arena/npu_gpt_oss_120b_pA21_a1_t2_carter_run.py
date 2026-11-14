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
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

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
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# Waypoints that safely avoid all known scored obstacles.
# The robot will travel:
#   1) south‑west corner → east along the far south edge,
#   2) north just west of the forklift (to stay below its y‑range),
#   3) east past the forklift,
#   4) north above the large central boxes,
#   5) east to the goal column,
#   6) north into the goal disc.
_WAYPOINTS = [
    (-9.0, -11.5),   # move a little further south for clearance
    (1.5, -11.5),    # go east along the bottom edge (clear arena_box_13, arena_box_12)
    (1.5, -1.0),     # go north, staying west of the forklift (forklift y ≤ -0.7)
    (6.0, -1.0),     # hop east, now east of the forklift
    (6.0, 14.5),     # go north, staying west of arena_box_06 (x < 5.95)
    (8.0, 14.5),     # east to the goal column (above arena_box_06)
    (8.0, 16.5),     # final goal position
]

_current_wp_idx = 0  # index of the waypoint we are currently targeting

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def controller(t, pose, env):
    """
    Simple deterministic waypoint‑following controller.

    Returns (v, w, done):
        v – forward speed [m/s]
        w – yaw rate [rad/s]
        done – True when the goal has been reached and aligned.
    """
    global _current_wp_idx

    x, y, yaw = pose

    # ------------------------------------------------------------
    # 1. Goal check (final waypoint)
    # ------------------------------------------------------------
    goal_x, goal_y, goal_yaw = GOAL
    dxg = goal_x - x
    dyg = goal_y - y
    dist_to_goal = math.hypot(dxg, dyg)
    yaw_err_goal = _wrap_angle(goal_yaw - yaw)

    if dist_to_goal <= MISSION["position_tolerance_m"] and \
       abs(yaw_err_goal) <= MISSION["yaw_tolerance_rad"]:
        # Goal reached – stop.
        return 0.0, 0.0, True

    # ------------------------------------------------------------
    # 2. Select current waypoint
    # ------------------------------------------------------------
    wp_x, wp_y = _WAYPOINTS[_current_wp_idx]
    dx_wp = wp_x - x
    dy_wp = wp_y - y
    dist_wp = math.hypot(dx_wp, dy_wp)

    # If we are within 0.25 m of the waypoint, advance.
    if dist_wp < 0.25 and _current_wp_idx < len(_WAYPOINTS) - 1:
        _current_wp_idx += 1
        wp_x, wp_y = _WAYPOINTS[_current_wp_idx]
        dx_wp = wp_x - x
        dy_wp = wp_y - y
        dist_wp = math.hypot(dx_wp, dy_wp)

    # Desired heading toward the active waypoint
    desired_yaw = math.atan2(dy_wp, dx_wp)
    yaw_error = _wrap_angle(desired_yaw - yaw)

    # ------------------------------------------------------------
    # 3. Simple proportional controller
    # ------------------------------------------------------------
    K_LINEAR = 0.6   # m/s per metre of distance error
    K_ANGULAR = 2.0  # rad/s per rad of heading error
    MAX_SPEED = 0.8  # m/s
    MAX_TURN = 1.0   # rad/s

    # Turn towards the waypoint; only drive forward when roughly aligned.
    w_cmd = max(-MAX_TURN, min(MAX_TURN, K_ANGULAR * yaw_error))
    if abs(yaw_error) < 0.4:          # within ~23°
        v_cmd = min(MAX_SPEED, K_LINEAR * dist_wp)
    else:
        v_cmd = 0.0

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
