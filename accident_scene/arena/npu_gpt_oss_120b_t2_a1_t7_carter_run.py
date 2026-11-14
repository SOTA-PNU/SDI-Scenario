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

# ----------------------------------------------------------------------
# Global state – persists between calls.
# ----------------------------------------------------------------------
_STATE = "go"          # "go" = head to goal, "wall" = follow left wall, "recover" = spin‑in‑place
_LAST_POSE = None      # (x, y) of previous step – for stuck detection
_LAST_T = None         # time of previous step
_STUCK_COUNT = 0       # consecutive frames with negligible motion
_RECOVERY_STEPS = 0    # frames left for recovery spin

def controller(t, pose, env):
    """
    Bug‑style navigation controller:

    * If the robot is inside the positional tolerance → align yaw and stop.
    * In normal “go” mode it heads toward the goal while applying a
      repulsive field from obstacles within SAFE_DIST.
    * When an obstacle is detected very close ahead it switches to a
      left‑hand wall‑following mode (“wall”).  The robot keeps the wall on its
      left side, turning to maintain a target distance.
    * When the front becomes clear and the direct line to the goal is free,
      it returns to “go” mode.
    * If the robot appears stuck (no measurable motion for a few seconds)
      a short spin‑in‑place recovery maneuver is performed.
    """
    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def normalize_angle(a):
        """Wrap angle to the interval [-π, π]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ------------------------------------------------------------------
    # Unpack current pose and goal
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # Tolerances (from the mission definition)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # ------------------------------------------------------------------
    # Goal‑reach handling – stop and align yaw when within tolerance.
    # ------------------------------------------------------------------
    dx_goal = goal_x - x
    dy_goal = goal_y - y
    dist_to_goal = math.hypot(dx_goal, dy_goal)

    if dist_to_goal <= pos_tol:
        # Position acceptable – now only yaw matters.
        yaw_error = normalize_angle(yaw - goal_yaw)
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True          # success
        # Rotate in place until yaw aligns.
        KP_YAW = 1.5
        w_align = KP_YAW * yaw_error
        w_align = clamp(w_align, -1.0, 1.0)
        return 0.0, w_align, False

    # ------------------------------------------------------------------
    # Parameters – tuned for this arena.
    # ------------------------------------------------------------------
    MAX_SPEED = 0.8          # m/s – top forward speed
    MAX_TURN = 1.0           # rad/s – hard limit on yaw rate
    KP_ANGLE = 2.0           # proportional gain for heading control
    KP_WALL = 1.5            # proportional gain for wall‑following
    SAFE_DIST = 1.5          # m – radius for repulsive field
    REPULSION_GAIN = 4.0     # strength of repulsion
    SAFE_FRONT = 0.8         # m – distance at which front is considered blocked
    WALL_TARGET = 0.5        # m – desired distance to the wall on the left side
    STUCK_DIST = 0.02        # m – movement threshold for stuck detection
    STUCK_TIME = 2.0         # s – time window for stuck detection
    RECOVERY_TURN = 1.0      # rad/s – spin speed during recovery
    ANGLE_TOL_GOAL = 0.35    # rad – when we consider the goal direction “clear”

    # ------------------------------------------------------------------
    # Scan processing – build repulsive field, front distance, and side minima.
    # ------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    front_dist = None          # nearest obstacle in a narrow forward cone
    left_min = float("inf")    # closest obstacle on the left side (bearing > 0)
    right_min = float("inf")   # closest obstacle on the right side (bearing < 0)

    try:
        scans = env.raycast_scan()   # 61 beams, 180°, max_range = 6.0
    except Exception:
        scans = []  # safety fallback – treat as no obstacles

    for bearing, distance in scans:
        # distance == max_range (6.0) means “no hit” – ignore.
        if distance >= 6.0:
            continue

        # Front cone (≈±0.15 rad ≈ ±8.5°)
        if abs(bearing) < 0.15:
            if front_dist is None or distance < front_dist:
                front_dist = distance

        # Left / right side minima (ignore the narrow front cone)
        if bearing > 0.15:
            if distance < left_min:
                left_min = distance
        elif bearing < -0.15:
            if distance < right_min:
                right_min = distance

        # Repulsive contribution for any obstacle within SAFE_DIST.
        if distance < SAFE_DIST:
            # Linear repulsion magnitude: 0 at SAFE_DIST, 1 at contact.
            rep_mag = (SAFE_DIST - distance) / SAFE_DIST
            # Unit vector pointing *away* from the obstacle.
            dir_x = -math.cos(bearing)
            dir_y = -math.sin(bearing)
            rep_x += dir_x * rep_mag * REPULSION_GAIN
            rep_y += dir_y * rep_mag * REPULSION_GAIN

    # ------------------------------------------------------------------
    # Simple stuck detection – if the robot has barely moved for a while,
    # trigger a short spin‑in‑place recovery.
    # ------------------------------------------------------------------
    global _STATE, _LAST_POSE, _LAST_T, _STUCK_COUNT, _RECOVERY_STEPS

    if _LAST_POSE is not None and _LAST_T is not None:
        moved = math.hypot(x - _LAST_POSE[0], y - _LAST_POSE[1])
        if t - _LAST_T >= STUCK_TIME:
            if moved < STUCK_DIST:
                _STUCK_COUNT += 1
            else:
                _STUCK_COUNT = 0
    _LAST_POSE = (x, y)
    _LAST_T = t

    # Recovery spin (overrides everything else)
    if _RECOVERY_STEPS > 0:
        _RECOVERY_STEPS -= 1
        return 0.0, RECOVERY_TURN, False

    if _STUCK_COUNT >= 3:
        # Enter recovery: spin left for a short while.
        _RECOVERY_STEPS = 30      # ~0.5 s (30 frames @ 60 Hz)
        _STUCK_COUNT = 0
        return 0.0, RECOVERY_TURN, False

    # ------------------------------------------------------------------
    # Compute heading to the goal (world frame) and robot‑relative error.
    # ------------------------------------------------------------------
    goal_dir_world = math.atan2(dy_goal, dx_goal)
    angle_error = normalize_angle(goal_dir_world - yaw)   # robot‑relative angle

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    if _STATE == "go":
        # --------------------------------------------------------------
        # Obstacle directly ahead? → switch to wall‑following mode.
        # --------------------------------------------------------------
        if front_dist is not None and front_dist < SAFE_FRONT:
            _STATE = "wall"
            # Immediate reaction: start turning left while moving slowly.
            v = 0.2
            w = MAX_TURN          # turn left (CCW)
            return v, w, False

        # --------------------------------------------------------------
        # Normal go‑to‑goal behaviour with attractive + repulsive field.
        # --------------------------------------------------------------
        # Attractive unit vector (robot forward = +x)
        att_x = math.cos(angle_error)
        att_y = math.sin(angle_error)

        # Combine attractive and repulsive vectors.
        total_x = att_x + rep_x
        total_y = att_y + rep_y

        # Desired heading in robot frame (angle of the combined vector).
        desired_heading = math.atan2(total_y, total_x)

        # Angular velocity command.
        w = KP_ANGLE * normalize_angle(desired_heading)
        w = clamp(w, -MAX_TURN, MAX_TURN)

        # Forward speed:
        #   - use the x‑component of the combined direction (forward if >0)
        #   - reduce speed when turning sharply
        #   - reduce speed when approaching the goal
        forward_factor = max(0.0, math.cos(desired_heading))
        turn_factor = max(0.1, 1.0 - abs(w) / MAX_TURN)
        if dist_to_goal < 2.0:
            goal_factor = dist_to_goal / 2.0
        else:
            goal_factor = 1.0
        v = MAX_SPEED * forward_factor * turn_factor * goal_factor

        # If something is still relatively close ahead, scale speed down.
        if front_dist is not None:
            v *= clamp(front_dist / SAFE_FRONT, 0.0, 1.0)

        v = max(0.0, v)
        return v, w, False

    # ------------------------------------------------------------------
    # Wall‑following mode – keep the wall on the left side.
    # ------------------------------------------------------------------
    # We use the left‑side minimum distance (left_min).  If no left wall is
    # detected (left_min == inf) we treat the error as large positive so the
    # robot keeps turning left to search for a wall.
    if left_min == float("inf"):
        left_error = WALL_TARGET - 1e6   # huge negative → turn left strongly
    else:
        left_error = left_min - WALL_TARGET   # >0 → wall too far → turn left

    # Wall‑following angular command (positive w = turn left).
    w = KP_WALL * left_error
    w = clamp(w, -MAX_TURN, MAX_TURN)

    # Forward speed – move slowly while the front is clear.
    if front_dist is not None and front_dist < SAFE_FRONT * 0.9:
        v = 0.0
    else:
        v = 0.3   # modest forward speed while wall‑following

    # --------------------------------------------------------------
    # Exit condition: if the front is clear and the direct line to the
    # goal appears free, go back to “go” mode.
    # --------------------------------------------------------------
    front_clear = (front_dist is None) or (front_dist > SAFE_FRONT * 1.2)

    # Find the scan entry whose bearing is closest to the goal direction.
    # If that distance is large (i.e., no obstacle) we consider the path clear.
    goal_path_clear = False
    if scans:
        # Compute the absolute difference between each bearing and angle_error.
        # Choose the entry with smallest difference.
        best = min(scans, key=lambda bd: abs(bd[0] - angle_error))
        bearing_best, dist_best = best
        if dist_best >= SAFE_FRONT * 1.5 or dist_best >= 6.0:
            goal_path_clear = True
    else:
        goal_path_clear = True

    if front_clear and goal_path_clear and abs(angle_error) < ANGLE_TOL_GOAL:
        _STATE = "go"
        # Reset wall‑following counters (optional)
        # Continue with the current step as a “go” step on the next call.
        return 0.0, 0.0, False

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
