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
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple “go‑to‑goal + front‑obstacle avoidance” controller.

    * Drive toward the goal with a proportional heading controller.
    * If any ray in a ±30° front cone reports a distance < SAFE_DIST,
      stop forward motion and rotate away from the obstacle (choose the side
      with the larger clearance).
    * Declare success when both position and yaw are inside the tolerances
      defined in ``MISSION``.
    """
    # ----------------------------------------------------------------------
    # Parameters (tuned for the arena, but remain deterministic)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.2               # m/s
    MAX_TURN = 1.0                # rad/s  (used for both avoidance and heading)
    SAFE_DIST = 0.8               # m – start avoiding when something this close ahead
    KP_HEADING = 2.0              # proportional gain on heading error
    # ----------------------------------------------------------------------

    # unpack current pose and goal
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- distance & heading to goal ---------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_bearing = math.atan2(dy, dx)                 # absolute bearing
    heading_error = _wrap_angle(goal_bearing - yaw)   # error relative to robot heading

    # ----- success test -------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----- ray‑cast scan ------------------------------------------------------
    scans = env.raycast_scan()          # [(bearing, distance), ...]

    # helper to get the minimum distance inside a bearing interval
    def _min_dist_in_range(min_bearing, max_bearing):
        dists = [d for b, d in scans if min_bearing <= b <= max_bearing]
        return min(dists) if dists else 6.0   # 6.0 == max_range (no hit)

    # front cone ±30°
    front_min = _min_dist_in_range(-math.radians(30), math.radians(30))

    # ----------------------------------------------------------------------
    # Obstacle avoidance decision
    # ----------------------------------------------------------------------
    if front_min < SAFE_DIST:
        # obstacle too close in front → turn in place
        # compare clearance on left (30° … 90°) vs right (‑90° … ‑30°)
        left_min = _min_dist_in_range(math.radians(30), math.radians(90))
        right_min = _min_dist_in_range(-math.radians(90), -math.radians(30))

        # turn toward the side with larger clearance
        w = MAX_TURN if left_min > right_min else -MAX_TURN
        v = 0.0                     # stop while rotating
    else:
        # No immediate obstacle → head toward the goal
        # Reduce speed when we need to turn sharply
        speed_factor = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
        v = MAX_SPEED * speed_factor

        # Proportional yaw rate, limited to MAX_TURN
        w = max(-MAX_TURN, min(MAX_TURN, KP_HEADING * heading_error))

        # If we are already well aligned, push to full speed
        if abs(heading_error) < math.radians(10):
            v = MAX_SPEED

    # never command reverse motion
    v = max(0.0, v)

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
