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

def _norm_angle(a):
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def controller(t, pose, env):
    """
    Gap‑following controller (Track B).

    * The robot looks at the 180° laser fan (61 beams).
    * Any beam whose distance is ≤ SAFE_DIST is considered “occupied”.
    * Contiguous free beams form a *gap*.  The controller picks the gap that
      best aligns with the goal direction (and is reasonably wide) and steers
      toward its centre.
    * Forward speed is scaled by the angular width of the chosen gap and is
      reduced when we are close to the goal or when something is very close
      directly ahead.
    * The mission ends when position & yaw are inside the tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (distances in metres, angles in radians)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0               # m/s
    MAX_ANG_SPEED = 1.0           # rad/s
    SAFE_DIST = 0.8                # ≤ this → beam is “blocked”
    MIN_FRONT = 0.45               # if something this close ahead → stop forward
    FRONT_ANGLE = math.radians(10) # narrow sector used for emergency stop
    GOAL_WEIGHT = 2.0              # how strongly we favour alignment with goal
    WIDTH_WEIGHT = 1.0             # how strongly we favour wide gaps
    # ----------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- vector to goal (world frame) ------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_dir_world = math.atan2(dy, dx)
    goal_bearing = _norm_angle(goal_dir_world - yaw)   # relative to current heading

    # ----- termination ------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_norm_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----- sensor scan ------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams, 180°, max_range = 6.0
    # scan is already a list of (bearing, distance) where bearing ∈ [-π/2, π/2]

    # ----- emergency front distance (for a hard stop) ----------------------
    front_dist = 6.0
    for bearing, dist in scan:
        if abs(bearing) <= FRONT_ANGLE and dist < front_dist:
            front_dist = dist

    # ----- build list of free gaps -----------------------------------------
    gaps = []               # each entry: (start_bearing, end_bearing)
    in_gap = False
    gap_start = None

    for bearing, dist in scan:
        free = dist > SAFE_DIST
        if free and not in_gap:
            # start a new gap
            in_gap = True
            gap_start = bearing
        elif not free and in_gap:
            # close current gap
            in_gap = False
            gap_end = bearing
            gaps.append((gap_start, gap_end))

    # handle gap that reaches the last beam
    if in_gap:
        gap_end = scan[-1][0]
        gaps.append((gap_start, gap_end))

    # If for some reason we have no gap (should be rare), just stop.
    if not gaps:
        return 0.0, 0.0, False

    # ----- select the best gap ---------------------------------------------
    best_gap = None
    best_score = -1e9

    for start, end in gaps:
        width = end - start                     # angular width of the gap
        center = (start + end) / 2.0            # bearing to the centre of the gap
        # angular difference between gap centre and the direction to the goal
        diff_to_goal = abs(_norm_angle(center - goal_bearing))

        # higher score = better (smaller diff, larger width)
        score = GOAL_WEIGHT * (-diff_to_goal) + WIDTH_WEIGHT * width

        if score > best_score:
            best_score = score
            best_gap = (start, end, center, width)

    # Unpack the chosen gap
    _, _, gap_center, gap_width = best_gap

    # ----- angular velocity -------------------------------------------------
    # steer toward the centre of the selected gap
    w = _clamp(gap_center, -MAX_ANG_SPEED, MAX_ANG_SPEED)

    # ----- forward speed ----------------------------------------------------
    # Base speed proportional to how wide the gap is (max out at 60°)
    max_useful_width = math.radians(60)          # widths larger than this give full speed
    width_factor = min(1.0, gap_width / max_useful_width)

    # Reduce speed when we are close to the goal for finer positioning
    if dist_to_goal < 2.0:
        goal_factor = dist_to_goal / 2.0
    else:
        goal_factor = 1.0

    v = MAX_SPEED * width_factor * goal_factor

    # Emergency stop if something is too close directly ahead
    if front_dist < MIN_FRONT:
        v = 0.0

    v = _clamp(v, 0.0, MAX_SPEED)

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
