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

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _wrap_angle(rad):
    """Wrap angle to the interval [-π, π]."""
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


# ----------------------------------------------------------------------
# Waypoint list (hard‑coded)
# ----------------------------------------------------------------------
# The robot starts at START (-9.0, -10.8).  The following way‑points steer it
# around the cluster of static boxes and then head straight to the goal.
# The final entry is the exact GOAL position – the yaw will be handled
# separately once the robot is inside the goal tolerance.
_WAYPOINTS = [
    (-9.0, -5.0),   # move north a bit, clear the low‑lying boxes near the start
    (-2.0, -5.0),   # head east, stay below the central obstacle field
    (-2.0, 5.0),    # go north, pass between the two middle rows of boxes
    (4.0, 5.0),     # move east, clear the large shelf at y≈4.2 (box 03)
    (4.0, 12.0),    # go north, stay left of the far‑right boxes (06,08)
    (8.0, 12.0),    # head east toward the goal corridor
    (8.0, 16.5),    # final goal position (yaw will be aligned later)
]

# Index of the waypoint we are currently heading for.
# Stored as a function attribute so it survives across calls.
def _init_state():
    """Initialise persistent state on the first call."""
    controller.wp_idx = 0          # start with the first waypoint
    controller.done = False       # not finished yet


# ----------------------------------------------------------------------
# Main controller
# ----------------------------------------------------------------------
def controller(t, pose, env):
    """
    Simple waypoint‑following controller with a lightweight reactive
    obstacle‑avoidance layer.

    Parameters
    ----------
    t : float
        Simulation time (seconds).
    pose : tuple(float, float, float)
        Current robot pose (x, y, yaw) in world coordinates.
    env : ArenaHarness
        Harness providing ``raycast_scan``.

    Returns
    -------
    v : float
        Forward linear velocity command [m/s].
    w : float
        Yaw rate command [rad/s].
    done : bool
        True when the robot has reached the goal and is correctly aligned.
    """
    # --------------------------------------------------------------
    # One‑time initialisation
    # --------------------------------------------------------------
    if not hasattr(controller, "wp_idx"):
        _init_state()

    # --------------------------------------------------------------
    # Parameters (tuned for this arena)
    # --------------------------------------------------------------
    MAX_SPEED = 1.2               # m/s
    MAX_TURN = 1.0                # rad/s
    Kp_speed = 0.8                # proportional gain for speed
    Kp_turn = 1.5                 # proportional gain for heading
    WAYPOINT_TOL = 0.5            # meters – distance to consider a waypoint reached
    SAFE_DIST = 0.6               # meters – start reacting to an obstacle
    FRONT_ANGLE = math.radians(30)  # consider rays within ±30° for avoidance

    # --------------------------------------------------------------
    # Unpack pose and goal information
    # --------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # --------------------------------------------------------------
    # Check if we have already finished (should never happen, but safe)
    # --------------------------------------------------------------
    if controller.done:
        return 0.0, 0.0, True

    # --------------------------------------------------------------
    # Goal handling – when the last waypoint (the goal) is reached
    # --------------------------------------------------------------
    if controller.wp_idx >= len(_WAYPOINTS):
        # We are inside the goal tolerance – align yaw then finish.
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        # Rotate in place to correct yaw
        w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * yaw_error))
        return 0.0, w, False

    # --------------------------------------------------------------
    # Current waypoint
    # --------------------------------------------------------------
    wp_x, wp_y = _WAYPOINTS[controller.wp_idx]
    dx_wp = wp_x - x
    dy_wp = wp_y - y
    dist_to_wp = math.hypot(dx_wp, dy_wp)

    # If we are close enough to the current waypoint, advance to the next.
    if dist_to_wp <= WAYPOINT_TOL:
        controller.wp_idx += 1
        # If the next index points past the list, we will handle goal logic
        # on the next call.
        return 0.0, 0.0, False

    # --------------------------------------------------------------
    # Desired heading towards the active waypoint
    # --------------------------------------------------------------
    desired_yaw = math.atan2(dy_wp, dx_wp)
    heading_error = _wrap_angle(desired_yaw - yaw)

    # --------------------------------------------------------------
    # Obstacle avoidance (lightweight)
    # --------------------------------------------------------------
    # Scan the environment.  The default fan (61 beams, 180°) is sufficient.
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)

    # Find the closest obstacle in the forward sector.
    min_front_dist = None
    min_front_bearing = 0.0
    for bearing, dist in scan:
        if abs(bearing) <= FRONT_ANGLE:
            if min_front_dist is None or dist < min_front_dist:
                min_front_dist = dist
                min_front_bearing = bearing

    # If an obstacle is dangerously close, stop forward motion and turn away.
    if min_front_dist is not None and min_front_dist < SAFE_DIST:
        # Turn opposite to the bearing of the obstacle.
        turn_dir = -1.0 if min_front_bearing > 0 else 1.0
        w = turn_dir * MAX_TURN
        return 0.0, w, False

    # --------------------------------------------------------------
    # Nominal control (go to waypoint)
    # --------------------------------------------------------------
    # Speed proportional to distance, limited by MAX_SPEED.
    v = min(MAX_SPEED, Kp_speed * dist_to_wp)

    # Turn rate proportional to heading error, limited by MAX_TURN.
    w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * heading_error))

    # If an obstacle is moderately close, gently reduce speed.
    if min_front_dist is not None and min_front_dist < 2.0:
        # Scale speed down linearly; never drop below 10 % of the nominal speed.
        scale = max(0.1, (min_front_dist - SAFE_DIST) / (2.0 - SAFE_DIST))
        v *= scale

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
