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

# ---- persistent simple state ------------------------------------------------
# These globals survive across controller invocations.
_mode      = "go"      # "go" = drive toward goal, "turn" = rotate in place
_turn_dir  = 1         # 1 → turn left, -1 → turn right (initial guess)


def controller(t, pose, env):
    """
    Minimal bug‑style controller (Track B).

    * Only the forward 180° ray‑cast sensor (`env.raycast_scan`) is used.
    * While the sector directly ahead is clear beyond SAFE_FRONT the robot
      drives forward, steering proportionally toward the goal.
    * When an obstacle is detected within SAFE_FRONT the robot switches to
      “turn” mode and rotates in place (initially left).  It stays in turn mode
      until the forward sector becomes clear again, then resumes “go” mode.
    * If the robot has been turning for a long time without the front clearing
      the turn direction is flipped – this helps escape dead‑ends.
    * An emergency stop is performed if any beam reports a distance below
      STOP_DIST (collision‑imminent).  The harness will abort after a few
      seconds, but we keep the robot still.
    * The mission ends (`done=True`) when the robot is inside the position
      tolerance **and** its yaw error is within the yaw tolerance.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    global _mode, _turn_dir

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return math.atan2(math.sin(a), math.cos(a))

    # ----------------------------------------------------------------------
    # termination test
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        yaw_err = wrap_angle(goal_yaw - yaw)
        if abs(yaw_err) <= yaw_tol:
            return 0.0, 0.0, True      # success

    # ----------------------------------------------------------------------
    # sensor scan
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()               # list of (bearing, distance)

    # ----------------------------------------------------------------------
    # parameters (chosen for safety and robustness)
    # ----------------------------------------------------------------------
    MAX_SPEED      = 0.55   # m/s – modest forward speed
    MAX_TURN_RATE  = 1.0    # rad/s
    SAFE_FRONT     = 0.85   # meters – must be clear ahead to drive forward
    STOP_DIST      = 0.15   # meters – immediate stop if anything this close
    FRONT_ANGLE    = math.radians(30)   # ±30° front cone for the “clear” test
    TURN_TIMEOUT   = 180    # steps (~3 s) before we flip turn direction

    # ----------------------------------------------------------------------
    # emergency stop if any beam is too close
    # ----------------------------------------------------------------------
    if any(dist < STOP_DIST for _, dist in scans):
        return 0.0, 0.0, False   # stay still; harness will abort after a few seconds

    # ----------------------------------------------------------------------
    # analyse the forward sector
    # ----------------------------------------------------------------------
    front_distances = [dist for bearing, dist in scans if abs(bearing) <= FRONT_ANGLE]
    min_front = min(front_distances) if front_distances else 6.0

    # ----------------------------------------------------------------------
    # decide what to do
    # ----------------------------------------------------------------------
    if _mode == "go":
        if min_front < SAFE_FRONT:
            # obstacle too close → switch to turn mode
            _mode = "turn"
            _turn_steps = 0
        else:
            # ---- GO MODE : drive toward goal --------------------------------
            # speed scales with how far the nearest front obstacle is
            speed_factor = min(1.0, min_front / 2.0)   # 0..1
            v = MAX_SPEED * speed_factor

            # heading toward goal (proportional controller)
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - yaw)
            w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, 1.5 * heading_error))

            # extra slowdown when we are close to the goal for a smoother stop
            if dist_to_goal < 2.0:
                v *= dist_to_goal / 2.0

            # clamp
            v = max(0.0, min(MAX_SPEED, v))
            w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, w))
            return v, w, False

    # ----------------------------------------------------------------------
    # TURN MODE : rotate in place until front clears
    # ----------------------------------------------------------------------
    # we are either already in turn mode or just entered it above
    _mode = "turn"
    # count steps in this turn episode (store in a hidden attribute of the function)
    if not hasattr(controller, "_turn_steps"):
        controller._turn_steps = 0
    controller._turn_steps += 1

    # rotate
    v = 0.0
    w = _turn_dir * MAX_TURN_RATE

    # if front becomes clear, go back to GO mode
    if min_front >= SAFE_FRONT:
        _mode = "go"
        controller._turn_steps = 0
    elif controller._turn_steps > TURN_TIMEOUT:
        # we have been turning for a while without success → flip direction
        _turn_dir = -_turn_dir
        controller._turn_steps = 0

    # clamp turn rate
    w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, w))

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
