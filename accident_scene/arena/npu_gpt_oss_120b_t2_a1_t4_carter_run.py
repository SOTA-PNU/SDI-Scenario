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

# Global state (persisted across calls)
_MODE = "go"            # "go" = head toward current waypoint, "avoid" = spin to clear front
_AVOID_DIR = 1          # +1 → turn left (CCW) while avoiding, -1 → turn right (CW)
_LAST_POSE = None       # last (x, y) to detect being stuck
_LAST_T = None          # last time stamp for stuck detection
_STUCK_COUNT = 0        # consecutive frames with negligible motion
# Waypoints that roughly lead around the central barrier.
# The final entry is the true goal (position only; yaw is handled separately).
_WAYPOINTS = [
    (-8.0, -5.0),   # climb north‑west from the start
    (-5.0, 5.0),    # pass left of the long horizontal box at y≈4.2
    (0.0, 10.0),    # go above the middle cluster of boxes
    (8.0, 16.5),    # final goal position
]

def controller(t, pose, env):
    """
    Waypoint‑following controller with simple obstacle avoidance.
    It drives toward the current waypoint, switches to a spin‑in‑place
    avoidance mode when an obstacle is detected directly ahead, and
    automatically proceeds to the next waypoint when the current one is
    reached.  Upon reaching the final waypoint (the goal), it aligns its
    yaw to the required goal yaw and stops.
    """
    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def normalize_angle(a):
        """Wrap angle to [-π, π]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ------------------------------------------------------------------
    # Unpack pose and goal data
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # Tolerances (from MISSION)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # ------------------------------------------------------------------
    # Waypoint handling
    # ------------------------------------------------------------------
    global _MODE, _AVOID_DIR, _LAST_POSE, _LAST_T, _STUCK_COUNT

    # If we have already visited all waypoints, we are at the goal.
    if _MODE == "finished":
        # Align yaw and stop.
        yaw_error = normalize_angle(yaw - goal_yaw)
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True
        # Rotate in place until yaw is within tolerance.
        KP_YAW = 1.5
        w_align = KP_YAW * yaw_error
        w_align = clamp(w_align, -1.0, 1.0)
        return 0.0, w_align, False

    # Determine which waypoint we are currently targeting.
    # When we reach the last waypoint (the goal position) we switch to the
    # "finished" mode so that only yaw alignment remains.
    wp_idx = getattr(controller, "_wp_idx", 0)
    if wp_idx >= len(_WAYPOINTS):
        # All waypoints visited – enter finished mode.
        _MODE = "finished"
        controller._wp_idx = wp_idx  # keep the attribute for completeness
        return controller(t, pose, env)  # recurse to handle yaw alignment

    target_x, target_y = _WAYPOINTS[wp_idx]

    # Distance and bearing to the current target.
    dx = target_x - x
    dy = target_y - y
    dist_to_target = math.hypot(dx, dy)
    target_dir_world = math.atan2(dy, dx)
    angle_error = normalize_angle(target_dir_world - yaw)  # robot‑relative angle

    # ------------------------------------------------------------------
    # Check if we have arrived at the current waypoint.
    # ------------------------------------------------------------------
    if dist_to_target <= pos_tol:
        # Advance to the next waypoint on the next call.
        controller._wp_idx = wp_idx + 1
        # Zero motion for this frame – the next call will compute a new target.
        return 0.0, 0.0, False

    # ------------------------------------------------------------------
    # Parameters (tuned for this arena)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.7          # m/s – comfortable cruising speed
    MAX_TURN = 1.0           # rad/s – hard limit on yaw rate
    KP_ANGLE = 2.0           # proportional gain for heading toward waypoint
    SAFE_FRONT = 0.8         # m – distance at which the front is considered blocked
    SAFE_SIDE = 1.5          # m – distance used when evaluating left/right clearance
    STUCK_DIST = 0.02        # m – movement threshold for stuck detection
    STUCK_TIME = 2.0         # s – time window for stuck detection

    # ------------------------------------------------------------------
    # Scan processing
    # ------------------------------------------------------------------
    front_dist = None          # nearest obstacle within a narrow forward cone
    left_clearance = 0.0       # summed clearance on the left side (bearing > 0)
    right_clearance = 0.0      # summed clearance on the right side (bearing < 0)

    try:
        scans = env.raycast_scan()  # 61 beams, 180°, max_range=6.0
    except Exception:
        scans = []  # safety fallback – treat as no obstacles

    for bearing, distance in scans:
        # Ignore “no‑hit” readings.
        if distance >= 6.0:
            continue

        # Front cone: |bearing| < ~0.2 rad (≈±11.5°)
        if abs(bearing) < 0.2:
            if front_dist is None or distance < front_dist:
                front_dist = distance

        # Accumulate side clearances (capped to SAFE_SIDE to avoid huge sums).
        if bearing > 0:
            left_clearance += min(distance, SAFE_SIDE)
        else:
            right_clearance += min(distance, SAFE_SIDE)

    # ------------------------------------------------------------------
    # Simple stuck detection (helps escape local minima)
    # ------------------------------------------------------------------
    if _LAST_POSE is not None and _LAST_T is not None:
        moved = math.hypot(x - _LAST_POSE[0], y - _LAST_POSE[1])
        if t - _LAST_T >= STUCK_TIME:
            if moved < STUCK_DIST:
                _STUCK_COUNT += 1
            else:
                _STUCK_COUNT = 0
    _LAST_POSE = (x, y)
    _LAST_T = t

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 1) Avoidance mode – spin in place (with a tiny forward component) until
    #    the front is clear and the heading toward the waypoint is roughly
    #    correct.
    # ------------------------------------------------------------------
    if _MODE == "avoid":
        # Keep moving slowly forward while turning.
        v = 0.2
        w = _AVOID_DIR * MAX_TURN

        # Exit condition: front is clear and we are roughly facing the waypoint.
        front_clear = (front_dist is None) or (front_dist > SAFE_FRONT * 1.2)
        heading_ok = abs(angle_error) < 0.3
        if front_clear and heading_ok:
            _MODE = "go"
            # Reset stuck counter – we have just cleared the obstacle.
            _STUCK_COUNT = 0

        return v, w, False

    # ------------------------------------------------------------------
    # 2) Normal “go” mode – head toward the current waypoint.
    # ------------------------------------------------------------------
    # If something is directly ahead within SAFE_FRONT, switch to avoidance.
    if front_dist is not None and front_dist < SAFE_FRONT:
        _MODE = "avoid"
        # Choose turn direction based on which side offers more clearance.
        if left_clearance > right_clearance:
            _AVOID_DIR = -1   # turn right
        else:
            _AVOID_DIR = 1    # turn left
        # Immediate reaction: spin while moving slowly.
        v = 0.2
        w = _AVOID_DIR * MAX_TURN
        return v, w, False

    # ------------------------------------------------------------------
    # Normal heading control toward the waypoint.
    # ------------------------------------------------------------------
    w = KP_ANGLE * angle_error
    w = clamp(w, -MAX_TURN, MAX_TURN)

    # Speed scaling:
    #   – slower when we need to turn sharply,
    #   – slower when we are close to the waypoint,
    #   – reduced proportionally if an obstacle is near the front (but not
    #     close enough to trigger avoidance).
    turn_factor = max(0.0, 1.0 - abs(w) / MAX_TURN)          # 1 when straight, 0 at max turn
    if dist_to_target < 2.0:
        goal_factor = dist_to_target / 2.0                 # linearly down to 0 at the waypoint
    else:
        goal_factor = 1.0

    v = MAX_SPEED * turn_factor * goal_factor

    if front_dist is not None:
        # Scale speed down as we approach an obstacle in front.
        v *= clamp(front_dist / SAFE_FRONT, 0.0, 1.0)

    # Ensure non‑negative forward speed.
    v = max(0.0, v)

    # ------------------------------------------------------------------
    # Stuck‑recovery: if we have been stuck for a few consecutive windows,
    # force a spin in the opposite direction to try to escape.
    # ------------------------------------------------------------------
    if _STUCK_COUNT >= 3:
        _MODE = "avoid"
        _AVOID_DIR = -_AVOID_DIR  # flip turn direction
        _STUCK_COUNT = 0
        v = 0.0
        w = _AVOID_DIR * MAX_TURN
        return v, w, False

    # ------------------------------------------------------------------
    # Return the command for this time step.
    # ------------------------------------------------------------------
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
