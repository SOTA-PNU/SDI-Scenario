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

# Global state used by the controller.  These variables persist between calls.
_MODE = "go"          # "go" = head toward goal, "avoid" = rotating to bypass obstacle
_AVOID_DIR = 1        # +1 → turn left (CCW), -1 → turn right (CW) when avoiding
_LAST_POSE = None
_LAST_T = None
_STUCK_COUNT = 0      # counts consecutive “stuck” detections


def controller(t, pose, env):
    """
    Simple bug‑style controller with a forward‑go mode and an avoid‑obstacle
    mode.  It steers toward the goal while using the 180° raycast to detect
    imminent collisions.  When an obstacle is too close directly ahead the
    robot switches to a spin‑in‑place mode, turning toward the side with more
    clearance until the front becomes clear again.

    The controller also:
      * Aligns yaw when the robot is inside the goal tolerance.
      * Detects a simple “stuck” condition (no measurable motion for ≥2 s)
        and forces a spin to escape local minima.
    """
    # ----------------------------------------------------------------------
    # Helper utilities
    # ----------------------------------------------------------------------
    def normalize_angle(a):
        """Wrap angle to [-π, π]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ----------------------------------------------------------------------
    # Unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector / distance to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Heading to goal expressed relative to robot heading
    goal_dir_world = math.atan2(dy, dx)
    angle_error = normalize_angle(goal_dir_world - yaw)  # [-π,π]

    # ----------------------------------------------------------------------
    # Goal‑reach / alignment handling
    # ----------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        # Inside the positional tolerance – now only yaw matters.
        yaw_error = normalize_angle(yaw - goal_yaw)
        if abs(yaw_error) <= yaw_tol:
            # Goal satisfied → stop and signal completion.
            return 0.0, 0.0, True
        # Position OK but yaw off – rotate in place.
        KP_YAW = 1.5
        w_align = KP_YAW * yaw_error
        w_align = clamp(w_align, -1.0, 1.0)
        return 0.0, w_align, False

    # ----------------------------------------------------------------------
    # Parameters (tuned for this arena)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.7          # m/s – comfortable cruising speed
    MAX_TURN = 1.0           # rad/s – hard limit on yaw rate
    KP_ANGLE = 2.0           # proportional gain for heading toward goal
    SAFE_FRONT = 0.8         # m – distance at which we consider the front blocked
    SAFE_SIDE = 1.5          # m – distance used for side‑clearance decision
    STUCK_DIST = 0.02        # m – movement threshold to consider “not stuck”
    STUCK_TIME = 2.0         # s – time window for stuck detection

    # ----------------------------------------------------------------------
    # Process raycast scan
    # ----------------------------------------------------------------------
    # Default values if scan fails (should not happen, but be safe)
    front_dist = None          # nearest obstacle within a narrow forward cone
    left_clearance = 0.0       # sum of distances on the left side (bearing > 0)
    right_clearance = 0.0      # sum of distances on the right side (bearing < 0)

    try:
        scans = env.raycast_scan()  # 61 beams, 180°, max_range=6.0
    except Exception:
        scans = []  # treat as no obstacles

    for bearing, distance in scans:
        # distance == max_range (6.0) means no hit → ignore for clearance sums
        if distance >= 6.0:
            continue

        # Front cone: |bearing| < ~0.2 rad (≈±11.5°)
        if abs(bearing) < 0.2:
            if front_dist is None or distance < front_dist:
                front_dist = distance

        # Accumulate side clearances (capped at SAFE_SIDE to avoid huge sums)
        if bearing > 0:
            left_clearance += min(distance, SAFE_SIDE)
        else:
            right_clearance += min(distance, SAFE_SIDE)

    # ----------------------------------------------------------------------
    # Simple stuck detection (optional but helps escape local minima)
    # ----------------------------------------------------------------------
    global _LAST_POSE, _LAST_T, _STUCK_COUNT
    if _LAST_POSE is not None and _LAST_T is not None:
        moved = math.hypot(x - _LAST_POSE[0], y - _LAST_POSE[1])
        if t - _LAST_T >= STUCK_TIME:
            if moved < STUCK_DIST:
                _STUCK_COUNT += 1
            else:
                _STUCK_COUNT = 0
    _LAST_POSE = (x, y)
    _LAST_T = t

    # ----------------------------------------------------------------------
    # Mode handling: "go" → drive toward goal, "avoid" → spin to clear front
    # ----------------------------------------------------------------------
    global _MODE, _AVOID_DIR

    if _MODE == "go":
        # Front blocked? → switch to avoidance mode.
        if front_dist is not None and front_dist < SAFE_FRONT:
            _MODE = "avoid"
            # Choose turn direction based on which side offers more clearance.
            # If left side is more open, turn right (avoid_dir = -1), and vice‑versa.
            if left_clearance > right_clearance:
                _AVOID_DIR = -1   # turn right
            else:
                _AVOID_DIR = 1    # turn left
            # Immediate reaction: spin in place.
            v = 0.0
            w = _AVOID_DIR * MAX_TURN
            return v, w, False

        # Normal go‑to‑goal behavior.
        # Heading control toward the goal.
        w = KP_ANGLE * angle_error
        w = clamp(w, -MAX_TURN, MAX_TURN)

        # Forward speed: full speed when heading is roughly aligned,
        # slower when we need to turn sharply.
        turn_factor = max(0.0, 1.0 - abs(w) / MAX_TURN)  # 1 when straight, 0 when max turn
        # Also slow down when we get close to the goal (helps precise stop).
        if dist_to_goal < 2.0:
            goal_factor = dist_to_goal / 2.0
        else:
            goal_factor = 1.0
        v = MAX_SPEED * turn_factor * goal_factor

        # If something is directly ahead but still beyond SAFE_FRONT,
        # optionally reduce speed proportionally to its distance.
        if front_dist is not None:
            v *= clamp(front_dist / SAFE_FRONT, 0.0, 1.0)

        # Ensure non‑negative speed.
        v = max(0.0, v)

        # Stuck recovery: if we have been stuck for a few consecutive windows,
        # force a spin to break out.
        if _STUCK_COUNT >= 3:
            _MODE = "avoid"
            _AVOID_DIR = 1  # arbitrary turn direction
            _STUCK_COUNT = 0
            v = 0.0
            w = _AVOID_DIR * MAX_TURN
            return v, w, False

        return v, w, False

    else:  # _MODE == "avoid"
        # Continue spinning until the front becomes clear again.
        v = 0.0
        w = _AVOID_DIR * MAX_TURN

        if front_dist is None or front_dist > SAFE_FRONT * 1.2:
            # Front is clear → resume normal navigation.
            _MODE = "go"
            # Reset stuck counter to avoid immediate re‑entry.
            _STUCK_COUNT = 0

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
