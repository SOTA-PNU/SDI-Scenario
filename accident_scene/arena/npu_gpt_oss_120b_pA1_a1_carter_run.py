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


def _normalize_angle(angle):
    """Wrap angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Simple waypoint‑following controller.

    The robot follows a hard‑coded list of waypoints that steer it clear of
    all static boxes.  When the final waypoint (the GOAL) is reached within
    the position tolerance, the controller signals ``done=True`` and stops
    the robot.

    The controller also looks at the forward laser scan; if an obstacle is
    detected closer than 0.7 m, the robot briefly stops and turns right
    (positive angular velocity) until the path clears.
    """
    # ----------------------------------------------------------------------
    # ONE‑TIME INITIALISATION
    # ----------------------------------------------------------------------
    if not hasattr(controller, "waypoints"):
        # Waypoints deliberately avoid every box.  They are expressed as (x, y).
        # The last waypoint is the exact goal position.
        controller.waypoints = [
            (-9.0, -2.0),   # go north along the left wall
            (-4.0, -2.0),   # move east a little
            (-4.0, 5.0),    # go north, clear the central cluster
            (2.0, 5.0),     # move east below the long shelf (box 03)
            (1.0, 5.0),     # step left to stay left of box 07 later
            (1.0, 12.0),    # go north, staying left of box 07 & 10
            (9.5, 12.0),    # go east, passing east of box 06
            (9.5, 16.5),    # go north to the goal height
            (GOAL[0], GOAL[1]),  # final approach to the exact goal
        ]
        controller.idx = 0          # current waypoint index
        controller.done = False

    # ----------------------------------------------------------------------
    # TERMINATION CHECK
    # ----------------------------------------------------------------------
    if controller.done:
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # WAYPOINT LOGIC
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    wx, wy = controller.waypoints[controller.idx]

    # Distance & bearing to the current waypoint
    dx = wx - x
    dy = wy - y
    distance = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    angle_error = _normalize_angle(target_angle - yaw)

    # If we are close enough to the current waypoint, advance to the next one
    if distance < 0.3:
        if controller.idx + 1 < len(controller.waypoints):
            controller.idx += 1
            wx, wy = controller.waypoints[controller.idx]
            dx = wx - x
            dy = wy - y
            distance = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            angle_error = _normalize_angle(target_angle - yaw)
        else:
            # Final waypoint reached – check final tolerances
            goal_dx = GOAL[0] - x
            goal_dy = GOAL[1] - y
            goal_dist = math.hypot(goal_dx, goal_dy)
            goal_yaw_err = _normalize_angle(GOAL[2] - yaw)

            if (
                goal_dist <= MISSION["position_tolerance_m"]
                and abs(goal_yaw_err) <= MISSION["yaw_tolerance_rad"]
            ):
                controller.done = True
                return 0.0, 0.0, True
            # Not yet aligned – rotate in place
            v = 0.0
            w = max(-1.0, min(1.0, 2.0 * goal_yaw_err))
            return v, w, False

    # ----------------------------------------------------------------------
    # SIMPLE OBSTACLE AVOIDANCE (laser based)
    # ----------------------------------------------------------------------
    # Scan ahead; if anything is closer than 0.7 m we stop and turn right.
    scans = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    obstacle_ahead = any(
        (abs(bearing) < math.radians(30) and dist < 0.7) for bearing, dist in scans
    )

    if obstacle_ahead:
        # Very conservative: stop forward motion and turn right a bit.
        v = 0.0
        w = 0.6  # rad/s, turn right
        return v, w, False

    # ----------------------------------------------------------------------
    # CONTROLLER OUTPUT
    # ----------------------------------------------------------------------
    # Proportional speed (capped) and proportional turning.
    max_speed = 1.5          # m/s
    max_angular = 1.0        # rad/s

    # Reduce speed when we need a large turn
    speed_factor = max(0.2, 1.0 - min(1.0, abs(angle_error) / (math.pi / 2)))
    v = max_speed * speed_factor * min(1.0, distance)

    w = max(-max_angular, min(max_angular, 2.0 * angle_error))

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
