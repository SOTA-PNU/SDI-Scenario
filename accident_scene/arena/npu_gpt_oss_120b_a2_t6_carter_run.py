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
    Bug‑algorithm style controller.

    1. **Go‑to‑goal** – proportional heading control while the front is clear.
    2. **Wall‑following** – when an obstacle is detected inside SAFE_DIST in the
       forward ±30° cone, switch to “wall” mode and turn right (negative yaw‑rate)
       while moving slowly.  The robot stays in this mode until the forward cone
       becomes clear *and* the bearing to the goal is roughly aligned (±20°).
    3. Success – robot is inside the position‑ and yaw‑tolerances from MISSION.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (deterministic, no randomness)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.8               # m/s – modest speed to keep a safety margin
    MAX_TURN = 0.8                # rad/s – max yaw rate
    SAFE_DIST = 1.2               # m – start avoidance when something this close ahead
    GOAL_ANGLE_TOL = math.radians(20)   # rad – when we consider the goal direction “clear”
    KP_HEADING = 2.0              # proportional gain for go‑to‑goal heading
    # ----------------------------------------------------------------------

    # -------------------------------------------------------
    # Persistent simple state machine (attached to the function)
    # -------------------------------------------------------
    if not hasattr(controller, "mode"):
        # mode: "go" (default) or "wall"
        controller.mode = "go"
    mode = controller.mode

    # -------------------------------------------------------
    # Unpack pose and goal
    # -------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # -------------------------------------------------------
    # Success test
    # -------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        # reset mode for a clean next run (not required but tidy)
        controller.mode = "go"
        return 0.0, 0.0, True

    # -------------------------------------------------------
    # Helper: minimum distance of rays whose bearing lies in [b_min, b_max]
    # -------------------------------------------------------
    scans = env.raycast_scan()          # [(bearing, distance), ...] 61 beams

    def _min_dist(b_min, b_max):
        dists = [d for b, d in scans if b_min <= b <= b_max]
        return min(dists) if dists else 6.0   # 6.0 == max_range → no hit

    # -------------------------------------------------------
    # Sensor‑derived quantities
    # -------------------------------------------------------
    front_min = _min_dist(-math.radians(30), math.radians(30))

    # Bearing to goal expressed relative to the robot's current heading
    goal_bearing_world = math.atan2(dy, dx)          # absolute bearing
    goal_bearing_rel = _wrap_angle(goal_bearing_world - yaw)

    # -------------------------------------------------------
    # State machine logic
    # -------------------------------------------------------
    if mode == "go":
        # If something is too close ahead, switch to wall‑following mode
        if front_min < SAFE_DIST:
            controller.mode = "wall"
            mode = "wall"
        else:
            # Normal go‑to‑goal behaviour
            heading_error = goal_bearing_rel
            w = max(-MAX_TURN, min(MAX_TURN, KP_HEADING * heading_error))

            # Reduce speed when a large turn is required
            speed_factor = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
            v = MAX_SPEED * speed_factor

            # When already well aligned, go full speed
            if abs(heading_error) < math.radians(10):
                v = MAX_SPEED

            v = max(0.0, v)          # never command reverse
            return v, w, False

    # -------------------------------------------------------
    # Wall‑following mode (right‑hand rule)
    # -------------------------------------------------------
    # Turn right while crawling forward slowly
    w = -MAX_TURN
    v = 0.25

    # Condition to leave wall mode:
    #  * front cone is clear again, and
    #  * the goal direction lies within a modest cone (±GOAL_ANGLE_TOL)
    #    so that we can safely resume goal‑directed travel.
    if front_min >= SAFE_DIST and abs(goal_bearing_rel) <= GOAL_ANGLE_TOL:
        controller.mode = "go"

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
