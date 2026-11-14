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
    """Wrap angle to the interval [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def controller(t, pose, env):
    """
    Bug‑type controller:
      * “go”  – drive toward the goal with a proportional heading controller.
      * “wall” – when an obstacle blocks the forward sector, follow the
                 obstacle on the side with more clearance (left‑hand rule).
    The controller keeps its state on the function object itself, so no
    external storage is required.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (chosen to be safe yet fast enough for the arena)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.2          # m/s – top forward speed in “go” mode
    WALL_SPEED = 0.6         # m/s – forward speed while wall‑following
    MAX_TURN = 1.0           # rad/s – absolute limit for angular speed
    KP_ANGLE = 2.0           # proportional gain for heading error
    KP_WALL = 1.0            # proportional gain for side‑distance error
    SAFE_FRONT = 0.8         # m – distance at which the front is considered blocked
    STOP_FRONT = 0.4         # m – stop forward motion if something is very close
    D_TARGET = 0.6           # m – desired distance to the wall while following
    SIDE_ANGLE = math.radians(60)   # consider bearings beyond this as side beams
    # ----------------------------------------------------------------------

    # ---------- initialise persistent state on first call -----------------
    if not hasattr(controller, "mode"):
        controller.mode = "go"          # "go" or "wall"
        controller.side = 1            # +1 = follow left wall, -1 = follow right
    # ---------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ---------- termination test -----------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True
    # ---------------------------------------------------------------------

    # ---------- sensor processing -----------------------------------------
    scans = env.raycast_scan()                     # [(bearing, distance), ...]
    # distance == max_range (6.0) means no hit; treat it as “far”
    # Front sector: |bearing| <= 30°
    front_dist = min(
        (d for b, d in scans if abs(b) <= math.radians(30)),
        default=6.0,
    )
    # Left side sector: bearing > +SIDE_ANGLE
    left_dist = min(
        (d for b, d in scans if b > SIDE_ANGLE),
        default=6.0,
    )
    # Right side sector: bearing < -SIDE_ANGLE
    right_dist = min(
        (d for b, d in scans if b < -SIDE_ANGLE),
        default=6.0,
    )
    # ---------------------------------------------------------------------

    # ---------- compute heading to goal ------------------------------------
    target_angle = math.atan2(dy, dx)
    angle_error = _wrap_angle(target_angle - yaw)
    # ---------------------------------------------------------------------

    # ---------- state machine ---------------------------------------------
    if controller.mode == "go":
        # If something blocks the front, switch to wall‑following
        if front_dist < SAFE_FRONT:
            # Choose the side with more clearance to follow
            if left_dist > right_dist:
                controller.side = 1   # follow left wall
            else:
                controller.side = -1  # follow right wall
            controller.mode = "wall"

            # In wall mode we stop forward motion for the first step
            v = 0.0
            w = controller.side * MAX_TURN
        else:
            # Drive toward the goal
            w = max(-MAX_TURN, min(MAX_TURN, KP_ANGLE * angle_error))

            # Reduce speed when we need to turn sharply
            align_factor = max(0.0, math.cos(angle_error))

            # Slow down as we approach the goal (linear ramp)
            goal_factor = min(1.0, dist_to_goal / 5.0)

            v = MAX_SPEED * align_factor * goal_factor

            # Hard stop if something is extremely close ahead
            if front_dist < STOP_FRONT:
                v = 0.0
    else:  # controller.mode == "wall"
        # Follow the chosen side wall
        side_dist = left_dist if controller.side == 1 else right_dist

        # Error = desired distance – measured distance
        dist_error = D_TARGET - side_dist

        # Turn rate tries to keep the side distance, plus a small bias toward
        # the goal direction so we eventually make progress.
        w_wall = controller.side * KP_WALL * dist_error
        w_goal = KP_ANGLE * angle_error * 0.3   # weaker influence while wall‑following
        w = max(-MAX_TURN, min(MAX_TURN, w_wall + w_goal))

        v = WALL_SPEED

        # If the front is clear again and we are roughly aligned with the goal,
        # go back to “go” mode.
        if front_dist >= SAFE_FRONT and abs(angle_error) < 0.35:
            controller.mode = "go"

        # Safety stop if something is too close ahead
        if front_dist < STOP_FRONT:
            v = 0.0
    # ---------------------------------------------------------------------

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
