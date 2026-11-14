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
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ----------------------------------------------------------------------
# Fixed way‑points that steer the robot around every static box.
# The list is expressed in world (x, y) coordinates.
# ----------------------------------------------------------------------
_WAYPOINTS = [
    (-9.0, -2.0),   # north along the western wall
    ( 8.0, -2.0),   # east along the southern floor
    ( 8.0, 10.0),   # north, staying well below the large central box (03)
    ( 6.0, 12.0),   # move left to pass the box at (7.625,13.4)
    ( 8.0, 16.5),   # final goal (x, y) – yaw is handled separately
]
_WP_TOL = 0.5          # meters – distance at which we consider a waypoint reached


def controller(t, pose, env):
    """
    Deterministic waypoint‑following controller with very lightweight
    reactive obstacle avoidance.

    * The robot follows the hard‑coded way‑points in ``_WAYPOINTS``.
    * When an obstacle is closer than ``SAFE_DIST`` directly ahead,
      the robot turns toward the side that offers the larger clearance.
    * If the distance to the current waypoint does not shrink for a few
      seconds, the robot spins in place to get out of a possible dead‑lock.
    * The mission ends when the robot is inside the goal disc *and* its yaw
      error is within the allowed tolerance.
    """
    # ------------------------------------------------------------------
    # unpack current pose
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # termination test (goal disc + yaw tolerance)
    # ------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_err = _wrap_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"]
            and abs(yaw_err) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # persistent controller state (stored on the function object)
    # ------------------------------------------------------------------
    if not hasattr(controller, "wp_idx"):
        controller.wp_idx = 0               # which waypoint we are heading for
        controller.last_dist = dist_to_goal
        controller.last_t = t
        controller.stuck_time = 0.0
        controller.spin_time = 0.0

    dt = t - controller.last_t if t > controller.last_t else 1.0 / 60.0
    controller.last_t = t

    # ------------------------------------------------------------------
    # waypoint handling
    # ------------------------------------------------------------------
    # If we have reached the current waypoint, advance to the next one.
    if controller.wp_idx < len(_WAYPOINTS):
        wx, wy = _WAYPOINTS[controller.wp_idx]
        wp_dist = math.hypot(wx - x, wy - y)
        if wp_dist <= _WP_TOL:
            controller.wp_idx += 1
            # reset progress counters when we switch way‑points
            controller.stuck_time = 0.0
            controller.last_dist = wp_dist
    # Target for the controller is either the current waypoint or the final goal
    if controller.wp_idx < len(_WAYPOINTS):
        target_x, target_y = _WAYPOINTS[controller.wp_idx]
    else:
        target_x, target_y = goal_x, goal_y

    # ------------------------------------------------------------------
    # progress detection – if we are not getting closer, we may be stuck
    # ------------------------------------------------------------------
    cur_target_dist = math.hypot(target_x - x, target_y - y)
    if cur_target_dist < controller.last_dist - 0.01:   # we made progress
        controller.stuck_time = 0.0
    else:
        controller.stuck_time += dt
    controller.last_dist = cur_target_dist

    # ------------------------------------------------------------------
    # sensor processing (ray‑fan)
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # [(bearing, distance), ...]
    MAX_RANGE = 6.0
    SAFE_DIST = 1.0                    # start reacting when < 1 m ahead
    FRONT_ANGLE = 0.30                 # rad ≈ 17°

    front_min = MAX_RANGE
    left_min = MAX_RANGE
    right_min = MAX_RANGE

    for bearing, distance in scan:
        d = distance
        if d >= MAX_RANGE:
            continue

        # front sector (|bearing| <= FRONT_ANGLE)
        if abs(bearing) <= FRONT_ANGLE:
            front_min = min(front_min, d)

        # left side (bearing > 0)
        if bearing > 0.0:
            left_min = min(left_min, d)

        # right side (bearing < 0)
        if bearing < 0.0:
            right_min = min(right_min, d)

    # ------------------------------------------------------------------
    # controller parameters
    # ------------------------------------------------------------------
    MAX_SPEED = 0.45       # m/s – deliberately modest to give avoidance room
    MAX_ANG = 1.2          # rad/s
    KP_ANG = 1.5           # proportional gain for heading toward target
    SPIN_SPEED = 0.9       # rad/s when we decide to spin to get unstuck
    SPIN_DURATION = 1.2    # seconds to spin before resuming normal motion
    PROGRESS_TIMEOUT = 3.0 # seconds without progress → spin mode

    # ------------------------------------------------------------------
    # unstuck spin mode
    # ------------------------------------------------------------------
    if controller.stuck_time > PROGRESS_TIMEOUT:
        if controller.spin_time < SPIN_DURATION:
            controller.spin_time += dt
            return 0.0, SPIN_SPEED, False
        else:
            # reset counters and continue normal navigation
            controller.stuck_time = 0.0
            controller.spin_time = 0.0

    # ------------------------------------------------------------------
    # obstacle avoidance overrides
    # ------------------------------------------------------------------
    if front_min < SAFE_DIST:
        # Choose turn direction based on which side is clearer
        turn_dir = 1.0 if left_min > right_min else -1.0   # +1 → turn left
        w = turn_dir * MAX_ANG
        # Crawl forward a little so we do not stall completely
        v = 0.15 * (front_min / SAFE_DIST)
        return v, w, False

    # ------------------------------------------------------------------
    # normal waypoint‑following behaviour
    # ------------------------------------------------------------------
    # Desired heading toward the current target (world → robot frame)
    desired_yaw = math.atan2(target_y - y, target_x - x)
    heading_err = _wrap_angle(desired_yaw - yaw)

    # Angular command (proportional)
    w = KP_ANG * heading_err
    w = max(-MAX_ANG, min(MAX_ANG, w))

    # Speed scaling:
    #   • slower when we are not pointing toward the target (cosine factor)
    #   • slower when we are already close to the target
    cos_factor = max(0.0, math.cos(heading_err))
    dist_factor = min(1.0, cur_target_dist / 4.0)   # full speed until ~4 m away
    v = MAX_SPEED * cos_factor * dist_factor
    v = max(0.0, min(MAX_SPEED, v))

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
