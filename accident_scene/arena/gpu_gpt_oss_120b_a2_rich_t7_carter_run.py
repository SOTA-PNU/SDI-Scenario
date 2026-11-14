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


def _wrap_angle(a: float) -> float:
    """Wrap angle into the interval [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """
    Waypoint‑following controller with simple obstacle repulsion.

    The robot follows a hard‑coded list of waypoints that stay clear of all
    static boxes.  When the final waypoint (the true GOAL) is reached the
    controller reports done.
    """
    # ------------------------------------------------------------------
    # One‑time initialization of waypoints and state
    # ------------------------------------------------------------------
    if not hasattr(controller, "waypoints"):
        # Waypoints are chosen to skirt the dense central cluster.
        # They are expressed in world coordinates (x, y).
        controller.waypoints = [
            (-9.0, -5.0),   # move north a bit
            (-5.0, -5.0),   # go east, staying left of box 00
            (-5.0, 5.0),    # go north, passing left of the large shelf (box 03)
            (8.0, 5.0),     # go east above the shelf
            (8.0, 16.5),    # final goal (matches GOAL)
        ]
        controller.idx = 0

    # ------------------------------------------------------------------
    # Parameters (ASCII only)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.9               # m/s
    MAX_ANG = 1.0                 # rad/s
    Kp_ang = 1.8                  # angular gain toward waypoint
    SAFE_DIST = 0.9               # start repelling inside this radius
    REPULSION_GAIN = 1.0          # scaling for repulsive angular term
    STOP_FRONT_DIST = 0.25        # stop forward motion if something this close ahead
    FRONT_ANGLE = 0.2             # rad, half‑width of the forward cone

    # ------------------------------------------------------------------
    # Current pose
    # ------------------------------------------------------------------
    x, y, yaw = pose

    # ------------------------------------------------------------------
    # Goal test – when we are within the mission tolerances of the final waypoint
    # ------------------------------------------------------------------
    gx, gy, gyaw = GOAL
    dxg = gx - x
    dyg = gy - y
    dist_to_goal = math.hypot(dxg, dyg)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Determine current target waypoint
    # ------------------------------------------------------------------
    wp = controller.waypoints[controller.idx]
    wx, wy = wp
    dwx = wx - x
    dwy = wy - y
    dist_to_wp = math.hypot(dwx, dwy)

    # If we are close enough to the current waypoint, advance to the next one
    if dist_to_wp < 0.5 and controller.idx < len(controller.waypoints) - 1:
        controller.idx += 1
        wp = controller.waypoints[controller.idx]
        wx, wy = wp
        dwx = wx - x
        dwy = wy - y
        dist_to_wp = math.hypot(dwx, dwy)

    # Desired heading toward the active waypoint
    target_angle = math.atan2(dwy, dwx)
    heading_error = _wrap_angle(target_angle - yaw)

    # ------------------------------------------------------------------
    # Scan the environment
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # list of (bearing, distance)

    # ------------------------------------------------------------------
    # Simple front‑obstacle stop: if anything is very close straight ahead,
    # set forward speed to zero.
    # ------------------------------------------------------------------
    front_blocked = any(
        abs(bearing) <= FRONT_ANGLE and dist < STOP_FRONT_DIST
        for bearing, dist in scan if dist < 6.0
    )

    # ------------------------------------------------------------------
    # Repulsive angular term from obstacles inside SAFE_DIST
    # ------------------------------------------------------------------
    repulsion = 0.0
    for bearing, dist in scan:
        if dist >= 6.0 or dist >= SAFE_DIST:
            continue
        strength = REPULSION_GAIN * (SAFE_DIST - dist) / SAFE_DIST
        # sin(bearing) > 0 => obstacle on left => turn right (negative)
        # sin(bearing) < 0 => obstacle on right => turn left (positive)
        repulsion -= strength * math.sin(bearing)

    # ------------------------------------------------------------------
    # Speed command – proportional to distance to waypoint, reduced when we need
    # to turn sharply or when something is directly ahead.
    # ------------------------------------------------------------------
    base_speed = min(MAX_SPEED, 0.6 * dist_to_wp)
    if abs(heading_error) > 0.5:
        base_speed *= 0.5
    if front_blocked:
        base_speed = 0.0

    # ------------------------------------------------------------------
    # Angular command – combine goal heading error and repulsion
    # ------------------------------------------------------------------
    ang = Kp_ang * heading_error + repulsion
    if ang > MAX_ANG:
        ang = MAX_ANG
    elif ang < -MAX_ANG:
        ang = -MAX_ANG

    # ------------------------------------------------------------------
    # Return the command tuple
    # ------------------------------------------------------------------
    return base_speed, ang, False


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
