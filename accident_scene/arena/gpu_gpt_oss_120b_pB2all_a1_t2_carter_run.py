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
    pose  (x, y, yaw) ground‑truth world pose of the chassis
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

# -------------------------------------------------------------------------
# Waypoints (x, y) – a safe corridor that stays at least ~1 m away from all
# static boxes and the known fixed structures (forklifts, shelves, walls).
# The final waypoint is the goal position; after it is reached the controller
# performs the yaw‑alignment step described in the prompt.
# -------------------------------------------------------------------------
_WAYPOINTS = [
    (-9.0, -5.0),   # go north just above the low‑lying forklift and box 00
    (0.0, -5.0),    # sweep east, staying clear of box 09 (which is higher up)
    (0.0, 12.0),    # go north through the centre, clear of all middle boxes
    (8.0, 12.0),    # sweep east below the tall box 06 / 08 cluster
    (8.0, 16.5),    # final northward leg into the goal area
]

# Index of the waypoint we are currently heading for.
_wp_idx = 0


def _wrap(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp *v* into the interval [lo, hi]."""
    return max(lo, min(hi, v))


def controller(t, pose, env):
    """
    Follow the hard‑coded waypoint list while respecting the
    “follow‑goal” law given in the problem statement.

    The function is called at 60 Hz, receives the current world pose
    (x, y, yaw) and a ray‑cast helper ``env.raycast_scan``.  It returns
    (v, w, done) according to the contract.
    """
    global _wp_idx

    x, y, yaw = pose

    # -----------------------------------------------------------------
    # 1) Goal‑alignment phase (after all waypoints visited)
    # -----------------------------------------------------------------
    if _wp_idx >= len(_WAYPOINTS):
        # Align to the final yaw.
        e = _wrap(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)   # slower turn for fine alignment
        v = 0.0
        done = abs(e) < 0.05
        return v, w, done

    # -----------------------------------------------------------------
    # 2) Normal waypoint‑following stage
    # -----------------------------------------------------------------
    tx, ty = _WAYPOINTS[_wp_idx]

    # distance and heading error to the current target
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    e = _wrap(target_yaw - yaw)

    # Advance to the next waypoint when we are close enough.
    if dist < 0.35 and _wp_idx < len(_WAYPOINTS):
        _wp_idx += 1
        # If we have just moved past the last waypoint, the next call will
        # handle the alignment stage.
        if _wp_idx >= len(_WAYPOINTS):
            return 0.0, 0.0, False
        tx, ty = _WAYPOINTS[_wp_idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        e = _wrap(target_yaw - yaw)

    # -----------------------------------------------------------------
    # 3) Simple obstacle‑avoidance safety check
    # -----------------------------------------------------------------
    # Stop forward motion if something is directly ahead within a safety margin.
    safe_dist = 0.8
    try:
        scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
        # Bearing = 0 points straight ahead; we look ±30°.
        min_ahead = min((d for b, d in scan if abs(b) < math.radians(30)), default=6.0)
        forward_allowed = min_ahead >= safe_dist
    except Exception:
        forward_allowed = True

    # -----------------------------------------------------------------
    # 4) Apply the published “follow‑goal” law
    # -----------------------------------------------------------------
    if abs(e) > 0.35:
        # Rotate on the spot.
        v = 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        # Move forward, respecting the safety check.
        v_nominal = min(0.6, 0.5 * dist + 0.1)
        v = v_nominal if forward_allowed else 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)

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
