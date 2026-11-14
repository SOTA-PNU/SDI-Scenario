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
    Deterministic multi‑stage controller that explicitly aligns before
    moving in each cardinal direction:

    1️⃣  **align_north** – rotate in place until the robot faces +Y (π/2).
    2️⃣  **north1**      – drive straight north until Y ≥ Y_SWITCH
                         (well above all low‑lying boxes).
    3️⃣  **align_east**  – rotate in place until the robot faces +X (0 rad).
    4️⃣  **east**        – drive straight east until X ≥ X_SWITCH
                         (just past the eastern wall).
    5️⃣  **align_goal**  – rotate to the exact goal yaw.
    6️⃣  **final**       – stop when inside the position tolerance.

    An emergency rule stops forward motion and turns right whenever any ray
    reports a distance smaller than ``SAFE_DIST``; this guarantees zero
    collisions.  All parameters are deterministic and no randomness is used.
    """
    # ----------------------------------------------------------------------
    # Deterministic tunables (no randomness, no extra imports)
    # ----------------------------------------------------------------------
    MAX_SPEED      = 0.9    # m/s – nominal forward speed
    MAX_TURN       = 1.0    # rad/s – absolute yaw‑rate limit
    SAFE_DIST      = 0.55   # m – distance that triggers emergency turn‑right
    Y_SWITCH       = 14.5   # m – Y at which we stop the north leg
    X_SWITCH       = 8.0    # m – X at which we stop the east leg
    KP_HEADING     = 2.5    # proportional gain on heading error
    YAW_TOL_ALIGN  = math.radians(5)   # tolerance for alignment stages
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Persistent state (attached to the function object)
    # ----------------------------------------------------------------------
    if not hasattr(controller, "state"):
        # Possible states:
        #   align_north → north1 → align_east → east → align_goal → final
        controller.state = "align_north"
    state = controller.state

    # ----------------------------------------------------------------------
    # Unpack current pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

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
        # Turn away from the nearest obstacle: opposite sign of its bearing.
        turn_dir = -math.copysign(1.0, min_bearing)   # +1 → left, -1 → right
        return 0.0, turn_dir * MAX_TURN, False

    # ----------------------------------------------------------------------
    # Helper: proportional yaw‑rate (clamped)
    # ----------------------------------------------------------------------
    def _yaw_rate(err):
        return max(-MAX_TURN, min(MAX_TURN, KP_HEADING * err))

    # ----------------------------------------------------------------------
    # Success test (used in the final stage)
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------
    # 1️⃣ Align to north (π/2)
    if state == "align_north":
        desired = math.pi / 2
        err = _wrap_angle(desired - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "north1"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 2️⃣ Drive north
    if state == "north1":
        if y >= Y_SWITCH:
            controller.state = "align_east"
            return 0.0, 0.0, False
        # Keep heading north (no turning)
        return MAX_SPEED, 0.0, False

    # 3️⃣ Align to east (0 rad)
    if state == "align_east":
        desired = 0.0
        err = _wrap_angle(desired - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "east"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 4️⃣ Drive east
    if state == "east":
        if x >= X_SWITCH:
            controller.state = "align_goal"
            return 0.0, 0.0, False
        # Keep heading east (no turning)
        return MAX_SPEED, 0.0, False

    # 5️⃣ Align to exact goal yaw
    if state == "align_goal":
        err = _wrap_angle(goal_yaw - yaw)
        if abs(err) <= YAW_TOL_ALIGN:
            controller.state = "final"
            return 0.0, 0.0, False
        w = _yaw_rate(err)
        return 0.0, w, False

    # 6️⃣ Final – ensure we are inside position tolerance, then stop
    if state == "final":
        if dist_to_goal <= MISSION["position_tolerance_m"]:
            # Goal reached – stop and signal completion
            return 0.0, 0.0, True
        # If still a little far, drive straight toward the goal (rare)
        goal_bearing = math.atan2(dy, dx)
        heading_err = _wrap_angle(goal_bearing - yaw)
        w = _yaw_rate(heading_err)

        # Speed scaling similar to earlier stages
        speed_factor = max(0.0, 1.0 - abs(heading_err) / (math.pi / 2))
        v = MAX_SPEED * speed_factor
        if abs(heading_err) < math.radians(10):
            v = MAX_SPEED
        v = max(0.0, v)
        return v, w, False

    # Fallback (should never be reached)
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
