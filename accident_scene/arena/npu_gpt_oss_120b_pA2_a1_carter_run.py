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

# ----------------------------------------------------------------------
#  Global configuration (waypoints, controller gains, limits)
# ----------------------------------------------------------------------
# Waypoints are defined in world (x, y) coordinates.  The robot will
# travel from the start position through each waypoint in order,
# finally stopping at the goal pose with the required yaw.
# The list was chosen to stay clear of every box listed in MISSION.
_WAYPOINTS = [
    (-9.0, -6.0),   # 1 – move north a little, clear the bottom‑row boxes
    (-3.0, -6.0),   # 2 – go east, stay west of the box at (0.2, -9.6)
    (-3.0,  2.0),   # 3 – go north, clear the lower‑left cluster
    (5.0,  2.0),    # 4 – move east, stay clear of the long box at (0.5, 4.2)
    (5.0,  7.0),    # 5 – go north, above the long box
    (5.0, 14.0),    # 6 – go north, clear all mid‑field boxes
    (8.0, 14.0),    # 7 – move east, stay north of box #06
    (8.0, 16.5),    # 8 – final goal position (same as GOAL)
]

# Controller gains and limits
_KP_ANGLE = 2.0          # proportional gain for heading error
_KP_DIST  = 0.6          # proportional gain for distance (speed scaling)
_MAX_SPEED = 0.6         # m/s – safe, slow speed
_MAX_ANG   = 1.0         # rad/s – max yaw rate
_WP_TOL    = 0.35        # m – distance to consider a waypoint reached
_OBS_DIST  = 0.9         # m – start braking if an obstacle is this close
_OBS_ANGLE = 0.5         # rad – consider beams within +/- this bearing

# State kept between controller calls
_current_wp = 0          # index of the waypoint we are currently heading for
_done = False            # becomes True once the goal pose is satisfied


def _normalize_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Waypoint‑following controller.

    Returns (v, w, done) where:
        v – forward speed [m/s]
        w – yaw rate   [rad/s]
        done – True when the robot has reached GOAL with correct yaw.
    """
    global _current_wp, _done

    # ------------------------------------------------------------------
    #  1.  Goal‑check (only after the last waypoint has been reached)
    # ------------------------------------------------------------------
    if _done:
        # already finished – keep robot stopped
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    #  2.  Determine current target waypoint
    # ------------------------------------------------------------------
    # If we have reached the final waypoint, we only need to align yaw.
    if _current_wp >= len(_WAYPOINTS):
        # Align to goal yaw
        _, _, goal_yaw = GOAL
        _, _, cur_yaw = pose
        yaw_err = _normalize_angle(goal_yaw - cur_yaw)

        # Simple P‑controller for yaw
        w = max(-_MAX_ANG, min(_MAX_ANG, _KP_ANGLE * yaw_err))
        v = 0.0

        # Consider the pose “done” when both position and yaw are within tolerance.
        goal_x, goal_y, _ = GOAL
        dx = pose[0] - goal_x
        dy = pose[1] - goal_y
        pos_err = math.hypot(dx, dy)

        if pos_err <= MISSION["position_tolerance_m"] and abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            _done = True
            return 0.0, 0.0, True
        else:
            return v, w, False

    # ------------------------------------------------------------------
    #  3.  Normal waypoint navigation
    # ------------------------------------------------------------------
    wp_x, wp_y = _WAYPOINTS[_current_wp]

    # Vector to waypoint
    dx = wp_x - pose[0]
    dy = wp_y - pose[1]
    dist = math.hypot(dx, dy)

    # Desired heading
    target_yaw = math.atan2(dy, dx)
    yaw_err = _normalize_angle(target_yaw - pose[2])

    # Simple proportional steering
    w = max(-_MAX_ANG, min(_MAX_ANG, _KP_ANGLE * yaw_err))

    # Speed scaling: slower when we need a large turn
    speed_factor = max(0.0, 1.0 - min(1.0, abs(yaw_err) / (math.pi / 2)))
    v = _MAX_SPEED * speed_factor * (dist / (dist + 0.5))  # also slow down when close

    # ------------------------------------------------------------------
    #  4.  Obstacle avoidance (reactive braking / turning)
    # ------------------------------------------------------------------
    # Scan forward 180° with 61 beams (default).  Any obstacle within
    # _OBS_DIST and roughly in front of the robot forces us to stop and
    # turn away.
    scans = env.raycast_scan()
    for bearing, distance in scans:
        if distance < _OBS_DIST and abs(bearing) < _OBS_ANGLE:
            # Obstacle too close – stop forward motion.
            v = 0.0
            # Turn away from the obstacle: positive bearing → turn right (negative w)
            turn_dir = -1.0 if bearing > 0 else 1.0
            w = turn_dir * _MAX_ANG
            break

    # ------------------------------------------------------------------
    #  5.  Waypoint arrival test
    # ------------------------------------------------------------------
    if dist <= _WP_TOL:
        # Snap to waypoint to avoid drift and advance to next one
        _current_wp += 1
        # If the next waypoint is the goal position, we will enter the
        # alignment branch on the next call.
        # Zero velocity for this step to give the robot a tiny pause.
        v = 0.0
        w = 0.0

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
