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

# ---------------------------------------------------------------------------
# Simple waypoint‑following controller with lightweight obstacle avoidance.
# ---------------------------------------------------------------------------
# The robot follows a manually‑crafted list of intermediate way‑points that
# threads safely between all static boxes and stays clear of the forklift that
# appears in the arena.  A proportional heading controller steers toward the
# current waypoint while a proportional distance controller sets forward speed.
# If an obstacle is detected within 1 m in front, the speed is reduced; if an
# obstacle is closer than 0.5 m the robot stops and turns away from it.
# ---------------------------------------------------------------------------

# ---- tunable parameters ----------------------------------------------------
_KP_ANGLE = 2.0          # [rad/(rad)] – heading error → angular speed
_KP_DIST  = 0.6          # [1/s]       – distance error → forward speed
_MAX_V    = 0.8          # [m/s]      – nominal speed limit
_MAX_W    = 1.0          # [rad/s]    – turn‑rate limit
_OBSTACLE_DIST = 1.0     # [m]        – start slowing down
_OBSTACLE_SLOW = 0.3     # [m/s]      – speed when obstacle is very close
_OBSTACLE_STOP = 0.5     # [m]        – distance at which we stop & turn away

# ---------------------------------------------------------------------------
# Way‑points (x, y).  They are ordered from START to GOAL.
# ---------------------------------------------------------------------------
# The chosen corridor goes north along the western side, then east above the
# long box #03, then north‑east under the large box #06 and finally to the goal.
_WAYPOINTS = [
    (-8.0, -5.0),   # clear the start cluster (boxes 13, 00)
    (-6.0,  0.0),   # move north, left of the central box cluster
    (-3.0,  5.0),   # pass between boxes 02 and 03
    ( 1.0,  8.0),   # go under the long box #03 (width 5 m, depth 0.8 m)
    ( 5.0, 12.0),   # stay left of the large box #06
    ( 8.0, 16.5),   # final goal (same as GOAL position)
]

# Persistent state -----------------------------------------------------------
_current_wp_idx = 0   # index of the waypoint we are currently heading for

def _normalize_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def controller(t, pose, env):
    """
    Way‑point follower with simple front‑obstacle avoidance.

    Returns (v, w, done) where:
        v   – forward speed [m/s]
        w   – yaw rate   [rad/s]
        done – True when the robot has reached the goal with the required yaw.
    """
    global _current_wp_idx

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -------------------------------------------------------------------
    # 1)  Goal alignment phase (after the last waypoint has been reached)
    # -------------------------------------------------------------------
    if _current_wp_idx >= len(_WAYPOINTS):
        yaw_err = _normalize_angle(goal_yaw - yaw)
        w = max(-_MAX_W, min(_MAX_W, _KP_ANGLE * yaw_err))
        v = 0.0
        done = abs(yaw_err) <= MISSION["yaw_tolerance_rad"]
        return v, w, done

    # -------------------------------------------------------------------
    # 2)  Current waypoint (the last entry in _WAYPOINTS is the goal position)
    # -------------------------------------------------------------------
    wp_x, wp_y = _WAYPOINTS[_current_wp_idx]

    # -------------------------------------------------------------------
    # 3)  Distance / bearing to waypoint
    # -------------------------------------------------------------------
    dx = wp_x - x
    dy = wp_y - y
    dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    yaw_err = _normalize_angle(target_yaw - yaw)

    # -------------------------------------------------------------------
    # 4)  Way‑point reached?
    # -------------------------------------------------------------------
    if dist <= MISSION["position_tolerance_m"]:
        _current_wp_idx += 1          # advance to next waypoint
        # give the controller a chance to recompute the next target
        return 0.0, 0.0, False

    # -------------------------------------------------------------------
    # 5)  Front‑obstacle scan
    # -------------------------------------------------------------------
    scan = env.raycast_scan()
    # Minimum distance in a forward cone of ±15° (≈0.26 rad)
    min_front_dist = min(
        (d for bearing, d in scan if abs(bearing) < 0.26),
        default=6.0
    )

    # If something is very close, stop and turn away from it
    if min_front_dist < _OBSTACLE_STOP:
        # Turn opposite to the bearing of the nearest obstacle
        nearest = min(
            ((bearing, d) for bearing, d in scan if d == min_front_dist),
            key=lambda bd: abs(bd[0])
        )
        bearing, _ = nearest
        # Positive bearing = obstacle on the left → turn right (negative w)
        w = -_MAX_W if bearing > 0 else _MAX_W
        v = 0.0
        return v, w, False

    # Speed limit based on obstacle distance
    if min_front_dist < _OBSTACLE_DIST:
        v_limit = _OBSTACLE_SLOW * (min_front_dist / _OBSTACLE_DIST)
    else:
        v_limit = _MAX_V

    # -------------------------------------------------------------------
    # 6)  Control law
    # -------------------------------------------------------------------
    # Angular velocity – proportional to heading error
    w = max(-_MAX_W, min(_MAX_W, _KP_ANGLE * yaw_err))

    # Forward speed – proportional to distance and to how well we are facing the target
    # cos(yaw_err) gives a smooth reduction when we are not aligned.
    forward = _KP_DIST * dist * max(0.0, math.cos(yaw_err))
    v = max(0.0, min(v_limit, forward))

    # -------------------------------------------------------------------
    # 7)  Not done yet
    # -------------------------------------------------------------------
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
