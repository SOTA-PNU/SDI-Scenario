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

# Persistent simple state for the controller.
# It lives in the module scope and survives across calls.
_controller_state = {"mode": "go_to_goal"}

def _wrap_angle(angle):
    """Wrap an angle into the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Deterministic reactive controller with a tiny bug‑algorithm style state
    machine.

    * In *go_to_goal* mode the robot steers directly toward the goal as long
      as the forward direction is clear.
    * If the forward clearance drops below a safety threshold, or the goal
      bearing is not within a sufficiently wide free angular interval, the
      controller switches to *follow_boundary* mode.
    * In *follow_boundary* mode the robot turns left (counter‑clockwise) while
      keeping a modest forward speed.  As soon as the goal direction becomes
      safe again and the front is clear, it switches back to *go_to_goal*.
    * Upon reaching the goal disc the robot aligns its yaw to the goal yaw
      and stops.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (feel free to adjust)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.5            # nominal forward speed [m/s]
    MIN_SPEED = 0.1            # forward speed while skirting obstacles
    MAX_ANG_SPEED = 1.2        # maximum yaw rate [rad/s]
    Kp_angle = 1.5             # proportional gain for steering
    Kp_yaw = 1.0               # gain for final yaw alignment
    BLOCK_DIST = 0.5           # distance below which a laser beam is considered occupied
    OBSTACLE_DIST = 0.5        # front‑clearance distance that triggers boundary following
    SAFE_FRONT = 0.5           # distance at which forward motion is allowed while turning
    MIN_SAFE_INTERVAL = 0.05   # minimum angular width (rad) of a free interval
    CLOSE_GOAL_DIST = 2.0      # start to slow down when this close to the goal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pose & goal information
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Goal bearing expressed in the robot's local frame
    goal_angle = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # Goal‑reached handling (position first, then orientation)
    # ------------------------------------------------------------------
    if distance_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # success – stop the mission
        # Rotate in place to align yaw.
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_yaw * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams over 180°, max_range = 6.0 m

    # Front clearance (beam whose bearing is closest to zero)
    front_beam = min(scan, key=lambda bd: abs(bd[0]))
    front_dist = front_beam[1]

    # -------------------------------------------------
    # 1) Determine free angular intervals
    # -------------------------------------------------
    occupied = [dist < BLOCK_DIST for _, dist in scan]
    safe_intervals = []
    start = None
    prev_bearing = None

    for i, (bearing, _) in enumerate(scan):
        if not occupied[i]:
            if start is None:
                start = bearing
        else:
            if start is not None:
                safe_intervals.append((start, prev_bearing))
                start = None
        prev_bearing = bearing

    if start is not None:
        safe_intervals.append((start, bearing))

    # Discard intervals that are narrower than the robot can realistically pass
    safe_intervals = [
        (b_start, b_end)
        for (b_start, b_end) in safe_intervals
        if (b_end - b_start) >= MIN_SAFE_INTERVAL
    ]

    # Does any interval already contain the goal bearing?
    goal_in_safe = any(b_start <= goal_angle <= b_end for (b_start, b_end) in safe_intervals)

    # ------------------------------------------------------------------
    # 2) Simple state machine
    # ------------------------------------------------------------------
    # Initialise persistent mode attribute on first call
    if not hasattr(controller, "mode"):
        controller.mode = "go_to_goal"

    mode = controller.mode

    # Transition from go_to_goal to follow_boundary if needed
    if mode == "go_to_goal":
        if front_dist < OBSTACLE_DIST or not goal_in_safe:
            controller.mode = "follow_boundary"
            mode = "follow_boundary"

    # ------------------------------------------------------------------
    # 3) Actuation based on the current mode
    # ------------------------------------------------------------------
    if mode == "go_to_goal":
        # Steer directly toward the goal (we already know the direction is safe)
        desired_angle = goal_angle

        # Angular velocity command (clamped)
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_angle * desired_angle))

        # Forward speed modulation
        #   - Front clearance
        if front_dist < SAFE_FRONT:
            front_factor = front_dist / SAFE_FRONT
            front_factor = max(0.0, min(1.0, front_factor))
        else:
            front_factor = 1.0

        #   - Turning penalty
        turn_factor = max(0.0, 1.0 - abs(desired_angle) / (math.pi / 2))

        #   - Slow down when we are close to the goal
        if distance_to_goal < CLOSE_GOAL_DIST:
            goal_factor = distance_to_goal / CLOSE_GOAL_DIST
            goal_factor = max(0.0, min(1.0, goal_factor))
        else:
            goal_factor = 1.0

        speed_factor = front_factor * turn_factor * goal_factor
        v = MAX_SPEED * speed_factor

        return v, w, False

    else:  # mode == "follow_boundary"
        # Check whether we can safely re‑enter go‑to‑goal mode
        if front_dist >= SAFE_FRONT and goal_in_safe:
            controller.mode = "go_to_goal"
            # Fall through to the go_to_goal branch on the next call
            # (the next call will recompute everything with the new mode)
            # For this step we still need to output a sensible command;
            # we simply rotate left slowly while moving forward a tiny amount.
            w = MAX_ANG_SPEED * 0.5
            v = MIN_SPEED
            return v, w, False
        else:
            # Continue skirting the obstacle: turn left at max yaw rate.
            w = MAX_ANG_SPEED
            # If the front is still too close, stay in place; otherwise
            # advance slowly while turning.
            if front_dist < SAFE_FRONT:
                v = 0.0
            else:
                v = MIN_SPEED
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
