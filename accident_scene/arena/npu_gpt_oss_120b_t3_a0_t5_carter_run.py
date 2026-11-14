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

# Global state for dead‑lock detection and forced recovery turns
_prev_pose = None          # (x, y, yaw) from previous control step
_stuck_counter = 0        # counts consecutive steps with negligible motion
_recovery_steps = 0       # steps remaining in a forced recovery turn


def _normalize_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding collisions.

    Returns:
        v (float): forward velocity command [m/s]
        w (float): yaw rate command [rad/s]
        done (bool): True when the robot is at the goal with correct yaw.
    """
    global _prev_pose, _stuck_counter, _recovery_steps

    # ------------------------------------------------------------------
    # Unpack current pose and compute goal geometry
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)               # absolute angle in world frame
    angle_error = _normalize_angle(angle_to_goal - yaw)  # relative to robot heading

    # ------------------------------------------------------------------
    # 1) Goal test – stop when inside tolerance and yaw aligned
    # ------------------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True                     # success
        # Align yaw inside the goal disc
        w = max(-1.0, min(1.0, 1.5 * yaw_err))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 2) Acquire ray‑cast scan
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # list of (bearing, distance)
    # bearings are already relative to current heading, in radians
    # compute beam step (should be constant, but we derive it just in case)
    if len(scan) >= 2:
        beam_step = (scan[-1][0] - scan[0][0]) / (len(scan) - 1)
    else:
        beam_step = 0.0

    # ------------------------------------------------------------------
    # 3) Dead‑lock detection (robot not making progress)
    # ------------------------------------------------------------------
    if _prev_pose is not None:
        move = math.hypot(x - _prev_pose[0], y - _prev_pose[1])
        if move < 0.01:               # < 1 cm moved this step
            _stuck_counter += 1
        else:
            _stuck_counter = 0
    _prev_pose = (x, y, yaw)

    # If we appear stuck for > 2 s (≈120 steps) trigger a short recovery turn
    if _stuck_counter > 120:
        _recovery_steps = 30   # rotate for ~0.5 s
        _stuck_counter = 0

    # Forced recovery turn (rotate in place until cleared)
    if _recovery_steps > 0:
        _recovery_steps -= 1
        # Choose a turn direction that points toward the side with larger clearance
        # (use the current scan to decide)
        left_clear = max(dist for bearing, dist in scan if bearing > 0)
        right_clear = max(dist for bearing, dist in scan if bearing < 0)
        turn_dir = 1 if left_clear > right_clear else -1
        return 0.0, turn_dir * 1.0, False

    # ------------------------------------------------------------------
    # 4) Simple “follow‑the‑gap” obstacle avoidance
    # ------------------------------------------------------------------
    SAFE_DIST = 0.8          # metres – distance considered safe / free
    MAX_SPEED = 0.6          # metres per second
    MAX_ANG_SPEED = 1.0      # rad/s
    Kp_angle = 1.8           # proportional gain for heading error

    # Build a list of booleans: True = free (distance > SAFE_DIST)
    free = [dist > SAFE_DIST for _, dist in scan]

    # Find the widest contiguous free gap
    best_gap_start = None
    best_gap_end = None
    best_gap_width = -1

    i = 0
    N = len(scan)
    while i < N:
        # skip blocked beams
        if not free[i]:
            i += 1
            continue
        # start of a free segment
        start = i
        while i < N and free[i]:
            i += 1
        end = i - 1  # inclusive
        gap_width = (end - start) * beam_step
        if gap_width > best_gap_width:
            best_gap_width = gap_width
            best_gap_start = start
            best_gap_end = end
    # If no free gap found (all blocked), fall back to a simple turn away
    if best_gap_start is None:
        # Turn away from the closest obstacle (choose side with larger distance)
        left_max = max(dist for bearing, dist in scan if bearing > 0)
        right_max = max(dist for bearing, dist in scan if bearing < 0)
        turn_dir = 1 if left_max > right_max else -1
        return 0.0, turn_dir * MAX_ANG_SPEED, False

    # Within the selected gap, pick the bearing closest to the goal direction
    # Goal bearing relative to heading is angle_error (already normalized)
    # Find the index in the gap whose bearing is nearest to angle_error
    best_idx = None
    best_delta = None
    for idx in range(best_gap_start, best_gap_end + 1):
        bearing = scan[idx][0]
        delta = abs(_normalize_angle(bearing - angle_error))
        if (best_delta is None) or (delta < best_delta):
            best_delta = delta
            best_idx = idx

    # Desired bearing (relative to heading) is the bearing of best_idx
    desired_bearing = scan[best_idx][0]
    # Angular error we need to correct (relative to current heading)
    ang_err = _normalize_angle(desired_bearing)

    # --------------------------------------------------------------
    # 5) Compute commands
    # --------------------------------------------------------------
    # Angular velocity (clamped)
    w = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, Kp_angle * ang_err))

    # If the chosen gap is very narrow, reduce angular speed a bit
    gap_angular_width = (best_gap_end - best_gap_start) * beam_step
    if gap_angular_width < math.radians(20):
        w *= 0.7

    # Forward speed: limited by front clearance, alignment, and distance to goal
    front_dist = scan[N // 2][1]          # centre beam ≈ straight ahead
    front_clearance = max(0.0, (front_dist - SAFE_DIST) / (6.0 - SAFE_DIST))

    alignment = max(0.0, math.cos(ang_err))          # 1 when heading straight to gap centre
    goal_factor = min(1.0, dist_to_goal / 5.0)      # slow down when approaching goal

    v = MAX_SPEED * front_clearance * alignment * goal_factor

    # Very close to an obstacle directly ahead: stop forward motion
    if front_dist < SAFE_DIST:
        v = 0.0

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
