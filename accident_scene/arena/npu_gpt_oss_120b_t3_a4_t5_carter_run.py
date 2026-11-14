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
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Bug‑style navigation with a simple wall‑following fallback.

    The robot drives straight toward the goal while the forward 180° ray‑cast is
    clear.  When something is detected within ``SAFE_FRONT`` it switches to a
    wall‑following mode that hugs the side (left or right) with the larger
    clearance.  Once the direct line to the goal becomes clear again the robot
    returns to the go‑to‑goal mode.  Upon reaching the goal disc it aligns its
    yaw and finishes.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Persistent state (initialise on first call)
    # ------------------------------------------------------------------
    if not hasattr(controller, "mode"):
        controller.mode = "go_to_goal"   # "go_to_goal" | "wall_follow"
        controller.follow_dir = 1        # 1 → keep wall on left, -1 → keep wall on right

    # ------------------------------------------------------------------
    # 2️⃣  Pose & goal geometry
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)                     # world‑frame bearing
    angle_error = _normalize_angle(target_angle - yaw)    # robot‑frame error

    # ------------------------------------------------------------------
    # 3️⃣  Goal‑arrival test (position + yaw)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if distance_to_goal <= pos_tol:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= yaw_tol:
            return 0.0, 0.0, True            # success
        # rotate in place to align yaw
        w = 1.0 * math.tanh(2.0 * yaw_err)
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 4️⃣  Sensor scan processing
    # ------------------------------------------------------------------
    scans = env.raycast_scan()   # [(bearing, distance), ...]

    # minimum distance in a bearing interval
    def min_dist(b_min, b_max):
        d = 6.0
        for bearing, dist in scans:
            if b_min <= bearing <= b_max and dist < d:
                d = dist
        return d

    # distance at a specific bearing (nearest beam)
    def dist_at(bearing_target):
        best = 6.0
        best_diff = float('inf')
        for bearing, dist in scans:
            diff = abs(_normalize_angle(bearing - bearing_target))
            if diff < best_diff:
                best_diff = diff
                best = dist
        return best

    # front sector (±30°)
    front_min = min_dist(-math.pi / 6, math.pi / 6)
    # left side sector (+60° … +90°)
    left_min = min_dist(math.pi / 3, math.pi / 2)
    # right side sector (‑90° … ‑60°)
    right_min = min_dist(-math.pi / 2, -math.pi / 3)

    # distance straight toward the goal (bearing = angle_error)
    goal_dir_dist = dist_at(angle_error)

    # ------------------------------------------------------------------
    # 5️⃣  Tunable parameters (constants)
    # ------------------------------------------------------------------
    MAX_SPEED   = 0.8    # [m/s] nominal forward speed
    MAX_ANGULAR = 1.0    # [rad/s] max yaw rate
    SAFE_FRONT  = 0.9    # [m] distance considered “blocked” ahead
    KP_ANGLE    = 1.5    # gain for heading error
    KP_SIDE     = 2.0    # gain for wall‑following side‑distance error
    TARGET_SIDE = 0.7    # desired distance to the wall we follow (m)
    WALL_SPEED  = 0.3    # forward speed while wall‑following

    # ------------------------------------------------------------------
    # 6️⃣  State machine
    # ------------------------------------------------------------------
    if controller.mode == "go_to_goal":
        # --------------------------------------------------------------
        # Obstacle directly ahead? → switch to wall‑following
        # --------------------------------------------------------------
        if front_min < SAFE_FRONT:
            # Choose the side with the larger clearance.
            if left_min > right_min:
                controller.follow_dir = 1   # keep wall on left, turn left
            else:
                controller.follow_dir = -1  # keep wall on right, turn right
            controller.mode = "wall_follow"
            # Rotate in place to start hugging the chosen side.
            v = 0.0
            w = controller.follow_dir * MAX_ANGULAR
            return v, w, False

        # --------------------------------------------------------------
        # Normal go‑to‑goal motion
        # --------------------------------------------------------------
        # Angular command – proportional to heading error, saturated.
        w = MAX_ANGULAR * math.tanh(KP_ANGLE * angle_error)

        # Forward speed – full speed when heading is aligned,
        # reduced when we need to turn sharply or when we near the goal.
        speed_factor = max(0.0, math.cos(angle_error))
        approach_factor = min(1.0, distance_to_goal / 2.0)
        v = MAX_SPEED * speed_factor * approach_factor

        return v, w, False

    else:   # controller.mode == "wall_follow"
        # --------------------------------------------------------------
        # If the front is still blocked, keep rotating in place.
        # --------------------------------------------------------------
        if front_min < SAFE_FRONT:
            v = 0.0
            w = controller.follow_dir * MAX_ANGULAR
            return v, w, False

        # --------------------------------------------------------------
        # Wall‑following control
        # --------------------------------------------------------------
        side_dist = left_min if controller.follow_dir == 1 else right_min
        side_error = side_dist - TARGET_SIDE

        # Keep the chosen side at the target distance.
        w_wall = KP_SIDE * side_error * controller.follow_dir

        # Small bias toward the goal direction so we eventually leave the wall.
        w_goal = 0.3 * KP_ANGLE * angle_error

        w_raw = w_wall + w_goal
        w = MAX_ANGULAR * math.tanh(w_raw)

        v = WALL_SPEED

        # --------------------------------------------------------------
        # Exit wall‑following: front clear, goal direction clear, and roughly aligned.
        # --------------------------------------------------------------
        if (front_min > SAFE_FRONT * 1.5 and
                goal_dir_dist > SAFE_FRONT * 1.5 and
                abs(angle_error) < 0.4):
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
