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
# Verification of sensor vs. obstacle specification
# ---------------------------------------------------------------------------
# The harness supplies a 180° fan with 61 beams (default arguments).  We
# perform a one‑time sanity check that the scan length and the reported
# max_range match the values that the mission description expects.
# This check does **not** influence the control law; it is only used to
# confirm that our assumptions about the sensor are correct before we
# design the controller.
# ---------------------------------------------------------------------------

_first_check_done = False  # module‑level flag

def _normalize_angle(a):
    """Wrap angle to [-π, π)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def _sensor_spec_check(env):
    """Run a single raycast with the official parameters and assert expectations."""
    scans = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    # Expected number of beams
    assert len(scans) == 61, f"Expected 61 beams, got {len(scans)}"
    # Expected max_range behaviour: any distance == 6.0 means no hit.
    # Verify that at least one beam reports the max_range (open space ahead).
    max_range_hits = any(abs(d - 6.0) < 1e-3 for _, d in scans)
    assert max_range_hits, "No beam reported max_range=6.0 – sensor spec mismatch"
    # If we reach here, the sensor spec matches our assumptions.
    return

# ---------------------------------------------------------------------------
# Simple reactive controller
# ---------------------------------------------------------------------------
# Strategy:
#   1. Compute a heading toward the goal and apply a proportional yaw rate.
#   2. Use the forward‑looking portion of the raycast (±30°) to detect
#      imminent obstacles.  If something is closer than SAFE_DIST (1.0 m),
#      we stop forward motion and turn away from the closest obstacle.
#   3. When we are inside the position tolerance and the yaw error is within
#      the allowed tolerance, we declare the mission finished.
# ---------------------------------------------------------------------------

SAFE_DIST = 1.0          # meters – keep at least this distance from any obstacle
MAX_SPEED = 1.0          # m/s – top forward speed
MAX_TURN = 1.0           # rad/s – top angular speed
KP_ANG = 1.5             # proportional gain for heading error
KP_OBS_TURN = 2.0        # gain when turning away from an obstacle
CENTER_BEAM_IDX = 30     # index of the beam that points straight ahead (0‑based)

def controller(t, pose, env):
    """
    Reactive navigation controller.

    Returns (v, w, done) each simulation step.
    """
    global _first_check_done

    # -----------------------------------------------------------------------
    # 1️⃣  One‑time sensor specification verification
    # -----------------------------------------------------------------------
    if not _first_check_done:
        _sensor_spec_check(env)
        _first_check_done = True

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -----------------------------------------------------------------------
    # 2️⃣  Goal‑directed heading
    # -----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    yaw_error = _normalize_angle(desired_yaw - yaw)

    # Proportional angular command toward the goal
    w_goal = KP_ANG * yaw_error
    # Clamp to allowed turn rate
    w_goal = max(-MAX_TURN, min(MAX_TURN, w_goal))

    # -----------------------------------------------------------------------
    # 3️⃣  Obstacle avoidance using raycast
    # -----------------------------------------------------------------------
    scans = env.raycast_scan()   # defaults are the mission‑specified ones
    # Extract bearings and distances
    bearings = [b for b, _ in scans]
    distances = [d for _, d in scans]

    # Consider only the central 21 beams (±30°) for forward safety.
    # 61 beams → step = 180°/60 = 3° per beam.
    # ±30° corresponds to 10 beams on each side of the centre.
    half_window = 10
    start_idx = CENTER_BEAM_IDX - half_window
    end_idx = CENTER_BEAM_IDX + half_window + 1  # exclusive

    forward_distances = distances[start_idx:end_idx]
    forward_bearings = bearings[start_idx:end_idx]

    # Find the closest obstacle in the forward sector
    min_dist = min(forward_distances)
    min_idx = forward_distances.index(min_dist)
    bearing_of_min = forward_bearings[min_idx]

    # -----------------------------------------------------------------------
    # 4️⃣  Combine goal heading and obstacle reaction
    # -----------------------------------------------------------------------
    if min_dist < SAFE_DIST:
        # Obstacle too close: stop forward motion and turn away.
        v = 0.0
        # Turning direction: opposite sign of the bearing to the obstacle.
        # If the obstacle is on the left (negative bearing), turn right (positive w).
        w_obs = -KP_OBS_TURN * _normalize_angle(bearing_of_min)
        # Blend with goal‑directed turn (optional). Here we prioritize avoidance.
        w = max(-MAX_TURN, min(MAX_TURN, w_obs))
    else:
        # No immediate obstacle: drive forward at a speed that scales with
        # distance to goal (slow down when we are near the goal).
        # Simple linear scaling: full speed when > 5 m away, linearly down to 0 at 0 m.
        speed_factor = min(1.0, distance_to_goal / 5.0)
        v = MAX_SPEED * speed_factor
        w = w_goal

    # -----------------------------------------------------------------------
    # 5️⃣  Check for mission completion
    # -----------------------------------------------------------------------
    pos_err_ok = distance_to_goal <= MISSION["position_tolerance_m"]
    yaw_err_ok = abs(_normalize_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]
    done = pos_err_ok and yaw_err_ok

    # When done, ensure we command zero motion.
    if done:
        v = 0.0
        w = 0.0

    return v, w, done

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
