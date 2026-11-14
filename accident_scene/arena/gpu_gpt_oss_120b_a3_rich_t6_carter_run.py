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

def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    * Attractive component – points toward the goal.
    * Repulsive component – for every ray‑cast hit inside a safety distance it
      adds a turning bias away from the obstacle.
    * The resulting heading is used in a simple P‑controller for angular speed.
    * Forward speed is reduced when the robot is poorly aligned, when it is
      close to the goal, or when an obstacle is very near in front.
    * The controller finishes (done=True) when both position and yaw are within
      the tolerances defined in ``MISSION``.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose                     # robot pose in world frame
    goal_x, goal_y, goal_yaw = GOAL       # goal pose in world frame

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def wrap(a):
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    # ----------------------------------------------------------------------
    # geometry to goal
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)          # world heading toward goal

    # ----------------------------------------------------------------------
    # parameters (tuned for the arena)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.8            # m/s – modest to keep a safety margin
    MAX_ANG = 1.0              # rad/s
    KP_ANG = 2.0               # proportional gain for heading error
    SAFETY_DIST = 0.9          # m – distance at which a ray creates repulsion
    CLOSE_DIST = 0.4           # m – stop forward motion when this close
    GOAL_SLOW_DIST = 1.0       # m – start slowing when nearer than this
    REPULSE_GAIN = 1.5         # how strongly a close obstacle bends the heading

    # ----------------------------------------------------------------------
    # ray‑cast processing – compute repulsive heading offset
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()                 # [(bearing, distance), ...]
    heading_offset = 0.0
    min_dist = 6.0                             # track the closest obstacle overall

    for bearing, dist in scans:
        # ignore max‑range readings (no hit)
        if dist >= 6.0:
            continue

        # update global closest distance
        if dist < min_dist:
            min_dist = dist

        # only create a repulsive effect when the obstacle is inside the safety zone
        if dist < SAFETY_DIST:
            # strength grows linearly as the obstacle gets nearer
            strength = (SAFETY_DIST - dist) / SAFETY_DIST   # 0 … 1
            # turning away: opposite sign of the bearing
            heading_offset += -bearing * strength * REPULSE_GAIN

    # ----------------------------------------------------------------------
    # desired world heading = goal direction + repulsive offset
    # ----------------------------------------------------------------------
    desired_world_heading = angle_to_goal + heading_offset
    heading_error = wrap(desired_world_heading - yaw)   # error in robot frame

    # ----------------------------------------------------------------------
    # angular command (P‑controller, clamped)
    # ----------------------------------------------------------------------
    w = max(-MAX_ANG, min(MAX_ANG, KP_ANG * heading_error))

    # ----------------------------------------------------------------------
    # forward speed command
    # ----------------------------------------------------------------------
    # Base speed scaled by how well we are aligned with the desired heading
    align_factor = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
    v = MAX_SPEED * align_factor

    # Slow down when we are near the goal
    if distance_to_goal < GOAL_SLOW_DIST:
        v *= distance_to_goal / GOAL_SLOW_DIST

    # If an obstacle is extremely close in front, stop forward motion
    if min_dist < CLOSE_DIST:
        v = 0.0

    # ----------------------------------------------------------------------
    # termination test (goal reached with correct orientation)
    # ----------------------------------------------------------------------
    done = False
    if (distance_to_goal <= MISSION["position_tolerance_m"] and
            abs(wrap(yaw - goal_yaw)) <= MISSION["yaw_tolerance_rad"]):
        v = 0.0
        w = 0.0
        done = True

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
