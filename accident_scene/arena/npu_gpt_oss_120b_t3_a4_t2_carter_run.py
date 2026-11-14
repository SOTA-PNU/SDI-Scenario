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

def _normalize_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Reactive go‑to‑goal controller with simple wall‑following avoidance.

    The controller runs at 60 Hz.  It drives straight toward the goal while
    monitoring the forward 180° ray‑cast.  When an obstacle is detected within a
    safe distance in front, the robot switches to a wall‑following mode that
    keeps a modest distance from the chosen side (left or right) and steers
    around the obstacle.  Once the path toward the goal is clear again, the
    controller returns to the go‑to‑goal mode.

    When the robot reaches the goal disc it aligns its yaw to the goal yaw,
    stops, and reports ``done=True``.
    """
    # ------------------------------------------------------------------
    # 1.  Persistent state initialisation
    # ------------------------------------------------------------------
    if not hasattr(controller, "mode"):
        # mode: "go_to_goal" or "avoid"
        controller.mode = "go_to_goal"
        # turn_dir: 1 → follow/turn left, -1 → follow/turn right
        controller.turn_dir = 1

    # ------------------------------------------------------------------
    # 2.  Pose and goal geometry
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)                # world‑frame bearing to goal
    angle_error = _normalize_angle(target_angle - yaw)  # robot‑frame error

    # ------------------------------------------------------------------
    # 3.  Goal completion check
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if distance_to_goal <= pos_tol:
        # Within goal disc – only yaw alignment remains.
        yaw_error = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True   # success
        # Rotate in place to achieve the required yaw.
        max_angular = 1.0
        kp_yaw = 2.0
        w = max_angular * math.tanh(kp_yaw * yaw_error)
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 4.  Sensor scan
    # ------------------------------------------------------------------
    scans = env.raycast_scan()   # [(bearing, distance), ...]

    # Helper to obtain the minimum distance within a bearing interval.
    def min_dist_in_range(b_min, b_max):
        d = 6.0   # max_range → treat as “no obstacle”
        for bearing, dist in scans:
            if b_min <= bearing <= b_max and dist < d:
                d = dist
        return d

    # Front sector (±30°)
    front_min = min_dist_in_range(-math.pi / 6, math.pi / 6)
    # Left side (≈ +60° … +90°)
    left_min = min_dist_in_range(math.pi / 3, math.pi / 2)
    # Right side (≈ -90° … -60°)
    right_min = min_dist_in_range(-math.pi / 2, -math.pi / 3)

    # ------------------------------------------------------------------
    # 5.  Parameters
    # ------------------------------------------------------------------
    safe_front_dist = 0.8          # stop/avoid if something closer than this
    max_speed = 1.0                # nominal forward speed (m/s)
    max_angular = 1.0              # max yaw rate (rad/s)
    kp_angle = 2.0                 # proportional gain for heading error
    kp_wall = 2.0                  # gain for wall‑following distance error
    wall_target = 0.6              # desired distance to the wall we follow (m)

    # ------------------------------------------------------------------
    # 6.  State‑machine logic
    # ------------------------------------------------------------------
    if controller.mode == "go_to_goal":
        # ------------------------------------------------------------------
        # 6a.  Detect imminent obstacle → switch to avoidance
        # ------------------------------------------------------------------
        if front_min < safe_front_dist:
            # Choose the side with the larger clearance.
            if left_min > right_min:
                controller.turn_dir = 1   # prefer left turn
            else:
                controller.turn_dir = -1  # prefer right turn
            controller.mode = "avoid"

        # ------------------------------------------------------------------
        # 6b.  Go‑to‑goal control (pure pursuit style)
        # ------------------------------------------------------------------
        # Angular velocity – proportional to heading error, saturated.
        w = max_angular * math.tanh(kp_angle * angle_error)

        # Forward speed – full speed when heading is aligned, reduced when
        # turning sharply or when we are close to the goal.
        speed_factor = max(0.0, math.cos(angle_error))
        approach_factor = min(1.0, distance_to_goal / 2.0)
        v = max_speed * speed_factor * approach_factor

        return v, w, False

    else:  # controller.mode == "avoid"
        # ------------------------------------------------------------------
        # 6c.  Wall‑following mode
        # ------------------------------------------------------------------
        # Select the distance measurement on the side we are following.
        side_dist = left_min if controller.turn_dir == 1 else right_min

        # Distance error relative to the desired wall distance.
        wall_error = side_dist - wall_target

        # Angular correction to keep the chosen side at the target distance.
        w_wall = kp_wall * wall_error * controller.turn_dir

        # Small bias toward the goal direction so that we do not circle
        # endlessly around the obstacle.
        w_goal = 0.5 * kp_angle * angle_error

        # Combine and saturate.
        w_raw = w_wall + w_goal
        w = max_angular * math.tanh(w_raw)

        # Forward speed – slow but non‑zero as long as the front is not blocked.
        if front_min < safe_front_dist:
            v = 0.0
        else:
            # Slow down when the goal is far to keep the robot controllable.
            v = 0.3 * min(1.0, distance_to_goal / 2.0)

        # ------------------------------------------------------------------
        # 6d.  Check if we can return to go‑to‑goal mode.
        # ------------------------------------------------------------------
        # Conditions: front clear enough and the goal direction is roughly
        # within the forward cone.
        if front_min > 1.2 and abs(angle_error) < 0.4:
            controller.mode = "go_to_goal"

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
