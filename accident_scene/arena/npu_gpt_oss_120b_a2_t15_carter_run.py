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
    Deterministic potential‑field controller with a hard emergency‑turn rule.

    1. **Goal attraction** – proportional to the vector from the robot to the
       goal (the “attractive” component).
    2. **Repulsion** – every ray whose distance is below REPEL_RANGE creates a
       repulsive angular offset that pushes the desired heading away from the
       obstacle.  The closer the hit, the stronger the offset.
    3. **Emergency avoidance** – if any ray reports a distance smaller than
       SAFE_DIST the robot stops forward motion and rotates *away* from the
       nearest obstacle (choosing the turn direction that gives the larger
       clearance).  This prevents the robot from getting stuck against a wall
       or a box.
    4. **Speed scaling** – forward speed is reduced when a large turn is
       required or when something is very close straight ahead.
    5. **Success** – when the robot is inside the position‑ and yaw‑tolerances
       defined in ``MISSION`` the controller returns ``done=True``.
    """
    # ----------------------------------------------------------------------
    # Tunable deterministic parameters
    # ----------------------------------------------------------------------
    MAX_SPEED       = 1.0    # m/s – nominal forward speed
    MAX_TURN        = 1.0    # rad/s – absolute yaw‑rate limit
    REPEL_RANGE     = 2.0    # m – start repelling when an obstacle is this close
    SAFE_DIST       = 0.6    # m – distance that forces an immediate stop‑and‑turn
    K_REPULSIVE     = 3.0    # gain for summed repulsive angular offset
    KP_HEADING      = 2.5    # proportional gain on heading error
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Unpack current pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # Success test
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # Scan the environment
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()          # [(bearing, distance), ...] 61 beams

    # Find the overall closest hit (for emergency rule) and also the closest
    # hit in a narrow front cone (±30°) – the latter is used only for speed
    # reduction, not for turning.
    min_dist = 6.0
    min_bearing = 0.0
    front_min = 6.0
    for bearing, dist in scans:
        if dist < min_dist:
            min_dist = dist
            min_bearing = bearing
        if abs(bearing) <= math.radians(30) and dist < front_min:
            front_min = dist

    # ----------------------------------------------------------------------
    # Emergency stop‑and‑turn‑away
    # ----------------------------------------------------------------------
    if min_dist < SAFE_DIST:
        # Turn *away* from the nearest obstacle: opposite sign of its bearing.
        turn_dir = -math.copysign(1.0, min_bearing)   # +1 → left, -1 → right
        v = 0.0
        w = turn_dir * MAX_TURN
        return v, w, False

    # ----------------------------------------------------------------------
    # Attractive heading (goal bearing, robot‑frame)
    # ----------------------------------------------------------------------
    goal_bearing_world = math.atan2(dy, dx)                # absolute bearing
    goal_error = _wrap_angle(goal_bearing_world - yaw)     # relative bearing

    # ----------------------------------------------------------------------
    # Repulsive angular offset from all rays inside REPEL_RANGE
    # ----------------------------------------------------------------------
    repulsive_offset = 0.0
    for bearing, dist in scans:
        if dist >= REPEL_RANGE:
            continue
        # Linear influence: 0 at REPEL_RANGE, 1 at distance == 0
        influence = (REPEL_RANGE - dist) / REPEL_RANGE
        # Positive bearing → obstacle on left → we should turn right (negative offset)
        repulsive_offset += -math.copysign(1.0, bearing) * influence

    # Combine attraction and repulsion
    desired_error = goal_error + K_REPULSIVE * repulsive_offset
    desired_error = _wrap_angle(desired_error)

    # ----------------------------------------------------------------------
    # Yaw‑rate command (proportional, clamped)
    # ----------------------------------------------------------------------
    w = max(-MAX_TURN, min(MAX_TURN, KP_HEADING * desired_error))

    # ----------------------------------------------------------------------
    # Forward speed command
    # ----------------------------------------------------------------------
    # Reduce speed when a large turn is required
    speed_factor = max(0.0, 1.0 - abs(desired_error) / (math.pi / 2))
    v = MAX_SPEED * speed_factor

    # Additional slowdown if something is very close straight ahead
    if front_min < 1.0:
        v = min(v, 0.2)

    # When already well aligned, go full speed
    if abs(desired_error) < math.radians(10):
        v = MAX_SPEED

    # Never command reverse motion
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
