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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _norm_angle(a):
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

# ---------------------------------------------------------------------------
# Tuning parameters (all deterministic)
# ---------------------------------------------------------------------------
V_MAX = 0.7                 # max forward speed [m/s]
W_MAX = 0.8                 # max yaw rate   [rad/s]
KP_ANGLE = 2.0              # proportional gain for heading error
MAX_RANGE = 6.0             # laser max range (matches env.raycast_scan)
MIN_FRONT_DIST = 0.8        # stop forward motion if something nearer than this
GOAL_ALIGN_WEIGHT = 0.5     # how strongly we bias toward the goal direction
DIST_WEIGHT = 1.0           # weight for raw distance score
ALIGN_WEIGHT = 0.5          # weight for angular alignment score
SLOW_NEAR_GOAL = 5.0        # start slowing down when < this many metres from goal
# ---------------------------------------------------------------------------

def controller(t, pose, env):
    """
    Reactive “follow‑the‑gap” controller.

    It scores every laser beam with a combination of distance and alignment
    to the goal, picks the best beam, and drives toward it while respecting
    a simple front‑obstacle safety rule.
    """
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -----------------------------------------------------------------------
    # Goal geometry
    # -----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_world_bearing = math.atan2(dy, dx)          # bearing in world frame
    goal_rel_bearing = _norm_angle(goal_world_bearing - yaw)

    # -----------------------------------------------------------------------
    # Completion test
    # -----------------------------------------------------------------------
    yaw_err_to_goal_yaw = _norm_angle(goal_yaw - yaw)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal_yaw) <= MISSION["yaw_tolerance_rad"]):
        # reached goal within tolerances → stop
        return 0.0, 0.0, True

    # Inside position tolerance but not yet oriented → rotate in place
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        w = max(min(KP_ANGLE * yaw_err_to_goal_yaw, W_MAX), -W_MAX)
        return 0.0, w, False

    # -----------------------------------------------------------------------
    # Laser scan
    # -----------------------------------------------------------------------
    scan = env.raycast_scan()          # list of (bearing, distance) tuples

    # -----------------------------------------------------------------------
    # Score every beam
    #   score = DIST_WEIGHT * (dist / MAX_RANGE) + ALIGN_WEIGHT * cos(bearing‑goal)
    #   Beams that are too close in front are forced to a very low score.
    # -----------------------------------------------------------------------
    best_score = -float('inf')
    best_idx = None
    for i, (bearing, dist) in enumerate(scan):
        # Safety: if something is directly ahead and closer than MIN_FRONT_DIST,
        # we still keep the beam but will later zero forward speed.
        # Compute distance component (0 … 1)
        dist_score = DIST_WEIGHT * (dist / MAX_RANGE)

        # Angular alignment component (0 … 1)
        align_score = ALIGN_WEIGHT * max(0.0, math.cos(bearing - goal_rel_bearing))

        # Combine
        score = dist_score + align_score

        # Slight extra penalty for very close obstacles (helps avoid collisions)
        if dist < 1.0:                     # 1 m is considered “close”
            score *= 0.2

        if score > best_score:
            best_score = score
            best_idx = i

    # Fallback: if for some reason we didn't find a beam (should never happen)
    if best_idx is None:
        # rotate slowly to the side with larger clearance
        left = min([d for b, d in scan if b > 0] or [MAX_RANGE])
        right = min([d for b, d in scan if b < 0] or [MAX_RANGE])
        w = W_MAX if left > right else -W_MAX
        return 0.0, w, False

    # -----------------------------------------------------------------------
    # Desired heading and angular velocity
    # -----------------------------------------------------------------------
    desired_bearing = scan[best_idx][0]          # relative bearing of best beam
    heading_error = _norm_angle(desired_bearing)  # error w.r.t. current heading
    w = max(min(KP_ANGLE * heading_error, W_MAX), -W_MAX)

    # -----------------------------------------------------------------------
    # Forward speed decision
    # -----------------------------------------------------------------------
    # Front‑obstacle check (narrow cone ±0.1 rad ≈ ±5.7°)
    front_distances = [d for b, d in scan if abs(b) < 0.1]
    front_dist = min(front_distances) if front_distances else MAX_RANGE

    # Distance measured along the chosen beam
    dist_along_best = scan[best_idx][1]

    # Basic speed proportional to how far we can see ahead and how well we are aligned
    alignment_factor = max(0.0, math.cos(heading_error))
    v = V_MAX * alignment_factor * (dist_along_best / MAX_RANGE)

    # Slow down when we are close to the goal (helps satisfy the yaw tolerance)
    if dist_to_goal < SLOW_NEAR_GOAL:
        v = min(v, V_MAX * (dist_to_goal / SLOW_NEAR_GOAL))

    # If something is too close directly ahead, stop forward motion (but keep turning)
    if front_dist < MIN_FRONT_DIST:
        v = 0.0

    # Clamp speed to non‑negative
    v = max(v, 0.0)

    # -----------------------------------------------------------------------
    # Return command
    # -----------------------------------------------------------------------
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
