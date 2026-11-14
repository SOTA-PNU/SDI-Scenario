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

# ----- helper utilities ----------------------------------------------------
def _angle_normalize(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ----- waypoint definition -------------------------------------------------
# Hard‑coded safe corridor that goes around all known boxes.
# Each waypoint is (x, y).  The final waypoint is the exact GOAL pose.
_WAYPOINTS = [
    (-9.0, -5.0),   # north of the start, left of box 00
    (-5.0, -5.0),   # east, above box 00
    (-5.0,  0.0),   # north, clear of the central cluster
    ( 0.0,  0.0),   # east, still clear
    ( 0.0, 10.0),   # north, passes left of box 07
    ( 8.0, 10.0),   # east, below the tall box cluster
    (10.0, 12.0),   # east of box 06
    (10.0, 16.0),   # north of the cluster
    ( 8.0, 16.5),   # final GOAL position (yaw will be handled separately)
]

# Runtime state (persisted across calls)
_wp_index = 0          # current waypoint we are heading for


# ----- tuning constants ----------------------------------------------------
_MAX_SPEED = 1.2            # m/s – top forward speed
_MAX_ANG = 1.0              # rad/s – top angular speed
_KP_YAW = 1.5               # proportional gain for heading error
_KP_AVOID = 0.9             # gain for reactive avoidance turn
_OBSTACLE_DIST = 1.2        # start avoidance when an obstacle is nearer than this (m)
_OBSTACLE_FOV = math.radians(45)   # consider obstacles within ±45° of heading
_WP_TOL = 0.5               # meters – distance to waypoint considered “reached”


def controller(t, pose, env):
    """
    Waypoint‑following controller with simple proportional steering and
    reactive obstacle avoidance.

    The robot proceeds through the hard‑coded waypoint list.  When the final
    waypoint (the GOAL) is reached, it additionally aligns its yaw to the
    goal yaw and terminates.
    """
    global _wp_index

    x, y, yaw = pose
    gx, gy, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # 1)  Check for mission completion (at final waypoint and correctly aligned)
    # ----------------------------------------------------------------------
    final_wp = _WAYPOINTS[-1]
    dist_to_final = math.hypot(final_wp[0] - x, final_wp[1] - y)
    yaw_error_to_goal = _angle_normalize(goal_yaw - yaw)

    if dist_to_final <= MISSION["position_tolerance_m"] and \
       abs(yaw_error_to_goal) <= MISSION["yaw_tolerance_rad"]:
        # Goal reached and orientation satisfied
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # 2)  Determine the active waypoint (skip to next when close enough)
    # ----------------------------------------------------------------------
    # If we are not yet at the final waypoint, use the waypoint list.
    # When we are within _WP_TOL of the current waypoint, advance.
    while _wp_index < len(_WAYPOINTS) - 1:
        wx, wy = _WAYPOINTS[_wp_index]
        if math.hypot(wx - x, wy - y) <= _WP_TOL:
            _wp_index += 1
        else:
            break

    # Target for this step (either an intermediate waypoint or the final GOAL)
    if _wp_index < len(_WAYPOINTS) - 1:
        target_x, target_y = _WAYPOINTS[_wp_index]
    else:
        # final approach – head for the exact GOAL position
        target_x, target_y = gx, gy

    # ----------------------------------------------------------------------
    # 3)  Pure‑geometric heading / speed towards the target
    # ----------------------------------------------------------------------
    dx = target_x - x
    dy = target_y - y
    dist_to_target = math.hypot(dx, dy)

    # Desired heading to the target
    desired_yaw = math.atan2(dy, dx)
    yaw_error = _angle_normalize(desired_yaw - yaw)

    # Base forward speed – slower when we are close to the target
    speed = min(_MAX_SPEED, 0.3 + 0.7 * (dist_to_target / 10.0))

    # Proportional turning towards the target
    w = max(-_MAX_ANG, min(_MAX_ANG, _KP_YAW * yaw_error))

    # If the heading error is large, back off speed a little to allow a smoother turn
    if abs(yaw_error) > 0.3:
        speed = min(speed, 0.5)

    # ----------------------------------------------------------------------
    # 4)  Reactive obstacle avoidance (using the ray‑scan)
    # ----------------------------------------------------------------------
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)

    closest_dist = None
    closest_bearing = None
    for bearing, dist in scan:
        if dist >= 6.0:          # no hit
            continue
        if abs(bearing) > _OBSTACLE_FOV:
            continue            # ignore obstacles far to the sides/back
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_bearing = bearing

    if closest_dist is not None and closest_dist < _OBSTACLE_DIST:
        # Turn away from the obstacle (opposite sign of bearing)
        avoid_turn = -_KP_AVOID * closest_bearing
        w += avoid_turn
        # Clamp again
        w = max(-_MAX_ANG, min(_MAX_ANG, w))
        # Reduce forward speed when something is close
        speed = min(speed, 0.4)

    # ----------------------------------------------------------------------
    # 5)  Return command
    # ----------------------------------------------------------------------
    return speed, w, False


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
