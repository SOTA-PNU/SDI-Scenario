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

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def controller(t, pose, env):
    """
    Way‑point‑following controller.

    The robot follows a hand‑crafted list of way‑points that stay at least
    1.1 m away from every box and from the fixed structures described in the
    prompt.  The follow‑the‑point logic is exactly the one proven to work in
    earlier experiments (see the prompt for the formulae).
    """
    # ----------------------------------------------------------------------
    # static data – created once on the first call
    # ----------------------------------------------------------------------
    if not hasattr(controller, "wp"):
        # way‑points (x, y) – the last point is the goal position.
        controller.wp = [
            (-9.8, -10.8),   # a little west of the start, away from the left shelf
            (-9.8, -4.0),    # go north, clear the left‑hand shelf
            (-5.5, -4.0),    # move east, pass the forklift_01 on its east side
            (-5.5, 6.0),     # go north, stay west of the large box‑row
            (-3.0, 6.0),     # slip east below box 9
            (-3.0, 13.0),    # head north, staying west of the long box 3
            (1.5, 13.0),     # move east, still west of box 8 (≥ 1.1 m)
            (1.5, 15.5),     # go north, above box 8 and box 6
            (6.0, 15.5),     # go east, clear box 6 (≥ 1.1 m)
            (8.0, 16.5),     # final goal position
        ]
        controller.idx = 0          # index of the current target waypoint
        controller.align = False    # whether we are in the final alignment phase

    # ----------------------------------------------------------------------
    # helper to fetch the current target (or the goal if we are aligning)
    # ----------------------------------------------------------------------
    if controller.align:
        tx, ty, tyaw = GOAL
    else:
        tx, ty = controller.wp[controller.idx]

    x, y, yaw = pose

    # distance and heading error to the current target
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    e = _wrap_angle(target_angle - yaw)

    # ----------------------------------------------------------------------
    # final alignment (only after we have reached the goal waypoint)
    # ----------------------------------------------------------------------
    if controller.align:
        # rotate in place to match GOAL yaw
        e_align = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e_align, -0.6, 0.6)
        v = 0.0
        if abs(e_align) < 0.05:
            # finished – report success
            return 0.0, 0.0, True
        return v, w, False

    # ----------------------------------------------------------------------
    # waypoint‑switching logic
    # ----------------------------------------------------------------------
    # If we are close enough to the current waypoint, advance.
    # The last waypoint (the goal) is handled specially later.
    if controller.idx < len(controller.wp) - 1 and dist < 0.35:
        controller.idx += 1
        # fetch the new target for this step
        tx, ty = controller.wp[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        e = _wrap_angle(target_angle - yaw)

    # ----------------------------------------------------------------------
    # control law (the proven one from the prompt)
    # ----------------------------------------------------------------------
    if abs(e) > 0.35:
        # rotate in place
        v = 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        # move forward while turning a bit
        v = min(0.6, 0.5 * dist + 0.1)
        w = _clamp(1.5 * e, -0.8, 0.8)

    # ----------------------------------------------------------------------
    # check if we have arrived at the final waypoint (the goal position)
    # ----------------------------------------------------------------------
    if controller.idx == len(controller.wp) - 1 and dist < 0.25:
        # enter alignment phase
        controller.align = True

    # optional debug output – harmless for the final run
    # print(f"DBG t={t:.2f} idx={controller.idx} pos=({x:.2f},{y:.2f}) "
    #       f"target=({tx:.2f},{ty:.2f}) dist={dist:.2f} e={e:.2f}")

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
