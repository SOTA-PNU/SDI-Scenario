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
    Deterministic multi‑stage controller that deliberately follows a simple
    rectangular corridor around the central box cluster:

        1️⃣ align_east   – rotate until facing +X (0 rad).
        2️⃣ east1        – drive east until x ≥ X_EAST1 (≈ ‑4 m) – clears the
                           low‑lying boxes near the start corner.
        3️⃣ align_north  – rotate until facing +Y (π/2 rad).
        4️⃣ north        – drive north until y ≥ Y_NORTH (≈ 12 m) – passes the
                           central shelf (y≈4.2) and the first row of boxes.
        5️⃣ align_east2  – rotate back to +X.
        6️⃣ east2        – drive east until x ≥ X_GOAL (≈ 8 m).
        7️⃣ align_goal   – rotate to the exact goal yaw and stop.

    An emergency avoidance rule stops forward motion and turns **right**
    (negative yaw rate) whenever any ray reports a distance smaller than
    ``SAFE_DIST``.  This guarantees zero collisions while keeping the logic
    fully deterministic.

    The controller respects the speed/turn limits required by the harness
    and never commands reverse motion.
    """
    # ----------------------------------------------------------------------
    # Deterministic tunables (do not modify outside this block)
    # ----------------------------------------------------------------------
    MAX_SPEED      = 0.9    # m/s – nominal forward speed
    MAX_TURN       = 1.0    # rad/s – absolute yaw‑rate limit
    SAFE_DIST      = 0.55   # m – distance that triggers emergency turn‑right
    KP_HEADING     = 2.5    # proportional gain on heading error
    YAW_TOL_ALIGN  = math.radians(5)   # rad – tolerance for alignment stages
    X_EAST1        = -4.0   # m – eastward target for the first leg
    Y_NORTH        = 12.0   # m – northward target before the final east leg
    X_GOAL         = 8.0    # m – final X coordinate (goal X)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Persistent state (attached to the function object)
    # ----------------------------------------------------------------------
    if not hasattr(controller, "state"):
        # Possible states:
        #   align_east → east1 → align_north → north → align_east2 → east2 → align_goal → final
        controller.state = "align_east"
    state = controller.state

    # ----------------------------------------------------------------------
    # Unpack current pose and goal yaw (needed at the very end)
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    _, _, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # Emergency stop‑and‑turn‑right (hard avoidance)
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()          # [(bearing, distance), ...] 61 beams
    min_dist = 6.0
    min_bearing = 0.0
    for bearing, dist in scans:
        if dist < min_dist:
            min_dist = dist
            min_bearing = bearing
    if min_dist < SAFE_DIST:
        # Turn right (negative yaw) away from the nearest obstacle.
        turn_dir = -math.copysign(1.0, min_bearing)   # +1 → left, -1 → right
        return 0.0, turn_dir * MAX_TURN, False

    # ----------------------------------------------------------------------
    # Helper: proportional yaw‑rate (clamped)
    # ----------------------------------------------------------------------
    def _yaw_rate(err):
        return max(-MAX_TURN, min(MAX_TURN, KP_HEADING * err))

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------
    # 1️⃣ Align to east (0 rad)
    if state == "align_east":
        err = _wrap_angle(0.0 - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "east1"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 2️⃣ Drive east until x ≥ X_EAST1
    if state == "east1":
        if x >= X_EAST1:
            controller.state = "align_north"
            return 0.0, 0.0, False
        # Keep heading east (no turning needed)
        return MAX_SPEED, 0.0, False

    # 3️⃣ Align to north (π/2 rad)
    if state == "align_north":
        err = _wrap_angle(math.pi / 2 - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "north"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 4️⃣ Drive north until y ≥ Y_NORTH
    if state == "north":
        if y >= Y_NORTH:
            controller.state = "align_east2"
            return 0.0, 0.0, False
        # Keep heading north (no turning needed)
        return MAX_SPEED, 0.0, False

    # 5️⃣ Align to east again
    if state == "align_east2":
        err = _wrap_angle(0.0 - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "east2"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 6️⃣ Drive east to the goal X coordinate
    if state == "east2":
        if x >= X_GOAL:
            controller.state = "align_goal"
            return 0.0, 0.0, False
        return MAX_SPEED, 0.0, False

    # 7️⃣ Align to the exact goal yaw
    if state == "align_goal":
        err = _wrap_angle(goal_yaw - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "final"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 8️⃣ Final – ensure we are inside the position tolerance, then stop.
    if state == "final":
        dx = GOAL[0] - x
        dy = GOAL[1] - y
        dist_to_goal = math.hypot(dx, dy)
        if dist_to_goal <= MISSION["position_tolerance_m"]:
            return 0.0, 0.0, True          # success
        # If still a little far (rare), drive straight toward the goal.
        goal_bearing = math.atan2(dy, dx)
        heading_err = _wrap_angle(goal_bearing - yaw)
        w = _yaw_rate(heading_err)

        # Speed scaling – slower when a large turn is required
        speed_factor = max(0.0, 1.0 - abs(heading_err) / (math.pi / 2))
        v = MAX_SPEED * speed_factor
        if abs(heading_err) < math.radians(10):
            v = MAX_SPEED
        v = max(0.0, v)
        return v, w, False

    # Fallback (should never be hit)
    return 0.0, 0.0, False


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
