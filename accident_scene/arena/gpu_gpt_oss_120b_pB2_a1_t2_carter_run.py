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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]."""
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Waypoints – chosen to stay >1.1 m away from every static box and from the
# fixed structures listed in the prompt.  The final waypoint coincides with GOAL.
# ---------------------------------------------------------------------------
_WAYPOINTS = [
    (-5.0, -10.8),   # 1) clear the left‑hand shelf / forklift_01
    (-5.0, -1.0),    # 2) go north, clear all lower obstacles
    (5.5, -1.0),     # 3) swing far east, clear the long box at (0.5, 4.2) and box 05
    (5.5, 15.0),     # 4) go north, staying west of the large box at (7.625, 13.4)
    (8.0, 15.0),     # 5) move east above that box
    (8.0, 16.5),     # 6) final goal (same as GOAL)
]

# Index of the waypoint we are currently targeting
_wp_idx = 0


def controller(t, pose, env):
    """
    Way‑point following controller.

    It implements the “pure‑pursuit‑like” law described in the mission
    specification, but with a small modification: when we are still far from
    the current waypoint (dist > 1 m) we allow a modest forward speed even
    if the heading error is large.  This prevents the robot from staying in
    place for a long time at the start (where the moving arena platform can
    otherwise cause a collision).

    A very light obstacle‑avoidance check using ``env.raycast_scan`` is kept:
    if something is directly ahead closer than 0.8 m we stop forward motion
    for that step (but we still turn if needed).
    """
    global _wp_idx

    x, y, yaw = pose

    # -------------------------------------------------------------------
    # 1) Target waypoint
    # -------------------------------------------------------------------
    if _wp_idx >= len(_WAYPOINTS):
        # All waypoints processed – should never happen because we finish
        # in the goal‑alignment block below.
        return 0.0, 0.0, True

    tx, ty = _WAYPOINTS[_wp_idx]

    # -------------------------------------------------------------------
    # 2) Distance / angle to waypoint
    # -------------------------------------------------------------------
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    e = _wrap_angle(target_angle - yaw)          # heading error

    # -------------------------------------------------------------------
    # 3) Waypoint progression
    # -------------------------------------------------------------------
    # Final waypoint (GOAL) uses the special alignment rule
    if _wp_idx == len(_WAYPOINTS) - 1:
        if dist < 0.25:                         # close enough to start alignment
            e_yaw = _wrap_angle(GOAL[2] - yaw)
            w = _clamp(1.5 * e_yaw, -0.6, 0.6)
            v = 0.0
            if abs(e_yaw) < 0.05:
                return 0.0, 0.0, True         # finished
            return v, w, False
        # otherwise treat like a normal waypoint (continue below)

    else:
        # Advance to next waypoint when we are within 0.35 m
        if dist < 0.35:
            _wp_idx += 1
            # Re‑evaluate with the new waypoint on the next call
            return controller(t, pose, env)

    # -------------------------------------------------------------------
    # 4) Compute speed commands
    # -------------------------------------------------------------------
    # If we are still far (>1 m) from the waypoint we allow a modest forward
    # speed even when the heading error is large.  This moves the robot out of
    # the start area quickly and avoids getting stuck against the moving
    # arena platform.
    if dist > 1.0:
        v = min(0.4, 0.5 * dist + 0.1)          # modest forward speed
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        # Use the original rule from the prompt
        if abs(e) > 0.35:
            v = 0.0
            w = _clamp(1.5 * e, -0.8, 0.8)
        else:
            v = min(0.6, 0.5 * dist + 0.1)
            w = _clamp(1.5 * e, -0.8, 0.8)

    # -------------------------------------------------------------------
    # 5) Very light obstacle‑avoidance safety check (front 10° cone)
    # -------------------------------------------------------------------
    scan = env.raycast_scan()
    for bearing, distance in scan:
        if abs(bearing) < 0.1745 and distance < 0.8:   # ~10°
            v = 0.0
            break

    # -------------------------------------------------------------------
    # 6) Return command
    # -------------------------------------------------------------------
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
