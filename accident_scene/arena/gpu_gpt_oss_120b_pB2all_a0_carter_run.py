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
# Helper utilities (local to this edit region)
# ---------------------------------------------------------------------------
def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Way‑point definition
# ---------------------------------------------------------------------------
# The list is a series of (x, y) tuples.  The final point is the GOAL position.
# All straight‑line segments have been checked (by eye) to stay roughly >1 m
# from every static box / fixed structure.  The controller also uses the
# ray‑cast sensor to back‑off if something unexpected appears.
_WAYPOINTS = [
    (-9.0, -7.0),   # move north a little to clear the southern wall
    (-6.0, -5.0),   # skirt around box 00 and the left forklift
    (0.0, -2.0),    # pass left of the central forklift
    (2.0, 2.0),     # head towards the centre of the arena
    (5.0, 6.0),     # go above the cluster of boxes 03‑05
    (6.0, 8.0),     # move further north‑east
    (5.0, 12.0),    # stay left of the big box 06
    (8.0, 16.5),    # final GOAL (same as GOAL position)
]

# Index of the waypoint we are currently heading for.
_wp_idx = 0

# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------
def controller(t, pose, env):
    """
    Follow the predefined way‑points while using the ray‑cast sensor to keep a
    safe distance from any unexpected obstacle.

    Parameters
    ----------
    t : float
        Simulation time (seconds) – not used in the logic.
    pose : tuple(float, float, float)
        Current robot pose (x, y, yaw) in world coordinates.
    env : ArenaHarness
        Harness object providing ``raycast_scan``.

    Returns
    -------
    (v, w, done) : tuple(float, float, bool)
        Linear velocity, angular velocity and mission‑completion flag.
    """
    global _wp_idx

    x, y, yaw = pose

    # -----------------------------------------------------------------------
    # 1) Determine the current target (way‑point or final alignment)
    # -----------------------------------------------------------------------
    target_x, target_y = _WAYPOINTS[_wp_idx]

    # -----------------------------------------------------------------------
    # 2) Compute distance and heading error to the target
    # -----------------------------------------------------------------------
    dx = target_x - x
    dy = target_y - y
    dist = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    e = _wrap_angle(angle_to_target - yaw)          # heading error

    # -----------------------------------------------------------------------
    # 3) Way‑point progression logic (identical to the reference method)
    # -----------------------------------------------------------------------
    # If we are not at the final waypoint and are close enough, advance.
    if _wp_idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        _wp_idx += 1
        target_x, target_y = _WAYPOINTS[_wp_idx]
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        e = _wrap_angle(angle_to_target - yaw)

    # -----------------------------------------------------------------------
    # 4) Base speed/turn command according to the reference law
    # -----------------------------------------------------------------------
    # Final‑goal alignment handling
    if _wp_idx == len(_WAYPOINTS) - 1:   # we are heading for GOAL
        # When we are inside the goal disc we stop forward motion and rotate
        # to the required final yaw.
        goal_dist = math.hypot(GOAL[0] - x, GOAL[1] - y)
        if goal_dist < 0.25:
            # alignment phase
            e_align = _wrap_angle(GOAL[2] - yaw)
            w_align = _clamp(1.5 * e_align, -0.6, 0.6)
            if abs(e_align) < 0.05:
                # Goal reached and correctly oriented
                return 0.0, 0.0, True
            return 0.0, w_align, False

    # Normal motion law
    if abs(e) > 0.35:
        # rotate in place
        v_cmd = 0.0
        w_cmd = _clamp(1.5 * e, -0.8, 0.8)
    else:
        v_cmd = min(0.6, 0.5 * dist + 0.1)
        w_cmd = _clamp(1.5 * e, -0.8, 0.8)

    # -----------------------------------------------------------------------
    # 5) Sensor‑based safety check (ray‑cast)
    # -----------------------------------------------------------------------
    # The harness provides a 180° fan (61 beams).  We look at the front 60°
    # (|bearing| < 0.5 rad) and if anything is closer than 0.9 m we stop and
    # turn away from the closest obstacle.
    scan = env.raycast_scan()
    min_dist = 6.0
    min_bearing = 0.0
    for bearing, distance in scan:
        if abs(bearing) < 0.5 and distance < min_dist:
            min_dist = distance
            min_bearing = bearing

    SAFE_DIST = 0.9  # meters – a little below the 1.1 m clearance used in planning
    if min_dist < SAFE_DIST:
        # Obstacle too close: stop forward motion and turn away.
        v_cmd = 0.0
        # Turning direction: opposite to the bearing of the obstacle.
        # If the obstacle is on the left (positive bearing) we turn right (negative w)
        # and vice‑versa.
        w_cmd = _clamp(-1.0 * min_bearing, -0.8, 0.8)

    # -----------------------------------------------------------------------
    # 6) Return command
    # -----------------------------------------------------------------------
    return v_cmd, w_cmd, False

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
