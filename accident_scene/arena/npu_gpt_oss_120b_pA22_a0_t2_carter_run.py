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
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

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
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# Waypoints that stay clear of every scored obstacle.
# Each tuple is (x, y) in world coordinates.
_WAYPOINTS = [
    (-10.2, -10.8),   # move a little west to stay left of the forklift column
    (-10.2,  -2.0),   # go north, above the low‑lying forklift and BoxSetA
    (-10.2,  15.0),   # continue north, well above the push‑cart and tall boxes
    (  5.5,  15.0),   # cross the upper‑clear corridor (y = 15 m)
    (  8.0,  15.0),   # east to the goal’s x‑coordinate
    (  8.0,  16.5),   # final goal pose
]

# Persistent controller state
_wp_idx = 0          # index of the current waypoint
_align_done = False  # have we finished the final yaw alignment?

# ----------------------------------------------------------------------
# Helper: normalise an angle to the interval [-π, π]
# ----------------------------------------------------------------------
def _norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def controller(t, pose, env):
    """
    Simple waypoint‑following controller with a very conservative safety check.
    It drives toward the current waypoint, turning in place when the heading
    error is large, and finally aligns to the goal yaw before signalling
    completion.
    """
    global _wp_idx, _align_done

    x, y, yaw = pose

    # --------------------------------------------------------------
    # 1) Goal handling – after the last waypoint we only need to align
    #    to the required yaw and stop.
    # --------------------------------------------------------------
    if _wp_idx >= len(_WAYPOINTS):
        target_yaw = GOAL[2]
        yaw_err = _norm_angle(target_yaw - yaw)
        if abs(yaw_err) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True          # success
        # rotate in place to achieve the final orientation
        w = max(-1.0, min(1.0, 2.0 * yaw_err))
        return 0.0, w, False

    # --------------------------------------------------------------
    # 2) Waypoint navigation
    # --------------------------------------------------------------
    tx, ty = _WAYPOINTS[_wp_idx]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)

    # waypoint reached? (30 cm tolerance – enough for our low speeds)
    if dist < 0.30:
        _wp_idx += 1
        return 0.0, 0.0, False

    # desired heading toward the waypoint
    target_yaw = math.atan2(dy, dx)
    yaw_err = _norm_angle(target_yaw - yaw)

    # --------------------------------------------------------------
    # 3) Very simple reactive safety: if anything is closer than 0.4 m
    #    directly ahead, stop forward motion.
    # --------------------------------------------------------------
    scan = env.raycast_scan()
    min_dist = min(d for _, d in scan)
    safe = min_dist > 0.40

    # --------------------------------------------------------------
    # 4) Motion command generation
    # --------------------------------------------------------------
    # Turn in place when the heading error is large.
    if abs(yaw_err) > 0.70:
        v = 0.0
        w = max(-1.0, min(1.0, 2.0 * yaw_err))
    else:
        # Move forward at a modest constant speed, scaling angular speed
        # to reduce heading error. Stop forward motion if something is too
        # close ahead.
        v = 0.6 if safe else 0.0
        w = max(-0.8, min(0.8, 1.5 * yaw_err))

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
