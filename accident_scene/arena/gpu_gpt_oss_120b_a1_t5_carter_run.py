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
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def controller(t, pose, env):
    """
    Small state‑machine controller:

    STATE 0 – rotate in place until the robot faces the goal (within a small
              angular tolerance).
    STATE 1 – drive forward while steering toward the goal.  If an obstacle is
              detected inside a safety cone, stop and turn toward the side that
              appears more open.  If the obstacle is extremely close, back up a
              little before turning.

    The controller never commands a speed larger than MAX_SPEED and never
    exceeds MAX_TURN for the yaw rate.
    """
    # ------------------------------------------------------------------
    # Tunable parameters
    MAX_SPEED = 1.5          # m/s
    MAX_TURN = 1.0           # rad/s
    KP_ANGLE = 2.0           # proportional gain for heading error
    ALIGN_TOL = 0.05         # rad – when we consider ourselves aligned
    SAFE_DIST = 0.8          # m – distance that triggers avoidance
    CRITICAL_DIST = 0.4      # m – distance that triggers a brief reverse
    FRONT_ANGLE = math.radians(30)   # ±30° front cone for avoidance
    # ------------------------------------------------------------------

    # --------------------------------------------------------------
    # Persistent state stored on the function object
    if not hasattr(controller, "state"):
        controller.state = 0          # 0 = aligning, 1 = cruising
    # --------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- termination test ------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_err_to_goal = _wrap_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----- compute angle to goal -------------------------------------------
    target_angle = math.atan2(dy, dx)
    angle_error = _wrap_angle(target_angle - yaw)

    # --------------------------------------------------------------
    # State 0 – align with goal direction
    if controller.state == 0:
        if abs(angle_error) > ALIGN_TOL:
            # rotate in place, no forward motion
            w = max(-MAX_TURN, min(MAX_TURN, KP_ANGLE * angle_error))
            return 0.0, w, False
        else:
            controller.state = 1   # aligned, switch to cruising
            # fall through to cruising logic on this same step
    # --------------------------------------------------------------

    # ----- ray‑cast scan ----------------------------------------------------
    scans = env.raycast_scan()   # list of (bearing, distance)

    # Look for obstacles in the forward sector
    front_hits = [(b, d) for (b, d) in scans
                  if abs(b) <= FRONT_ANGLE and d < SAFE_DIST]

    # Critical obstacle directly ahead – back up a little
    critical_hits = [(b, d) for (b, d) in scans
                     if abs(b) <= math.radians(15) and d < CRITICAL_DIST]

    if critical_hits:
        # Back up while turning away from the closest critical obstacle
        closest_bearing, _ = min(critical_hits, key=lambda bd: bd[1])
        w = -math.copysign(MAX_TURN, closest_bearing)   # turn opposite side
        v = -0.5 * MAX_SPEED
        return v, w, False

    if front_hits:
        # Choose turn direction based on which side has more clearance
        left_dist = [d for (b, d) in scans if b > 0]
        right_dist = [d for (b, d) in scans if b < 0]

        avg_left = sum(left_dist) / len(left_dist) if left_dist else 6.0
        avg_right = sum(right_dist) / len(right_dist) if right_dist else 6.0

        turn_dir = 1.0 if avg_left > avg_right else -1.0   # +1 → turn left
        w = turn_dir * MAX_TURN
        v = 0.0
        return v, w, False

    # ----- no immediate obstacle: drive toward goal -------------------------
    # Yaw rate command (clamped)
    w = max(-MAX_TURN, min(MAX_TURN, KP_ANGLE * angle_error))

    # Reduce forward speed when a sharp turn is required
    speed_factor = max(0.0, math.cos(angle_error))
    v = MAX_SPEED * speed_factor

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
