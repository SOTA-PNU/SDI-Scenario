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

# ---- simple persistent state -------------------------------------------------
_last_pos = None          # (x, y) of the previous step
_stuck_cnt = 0           # counts steps with negligible motion


def controller(t, pose, env):
    """
    Very conservative reactive controller (Track B).

    * Uses only `env.raycast_scan()` (180° fan, 61 beams).
    * If the forward 30° sector is clear beyond SAFE_FRONT the robot drives
      forward at a low speed while steering proportionally toward the goal.
    * If something is nearer than SAFE_FRONT the robot rotates in place,
      turning away from the nearest obstacle.
    * An emergency stop is issued when any beam reports a distance below
      STOP_DIST (collision‑imminent).  The harness will abort after a few
      seconds, but we keep the robot still.
    * When the robot appears stuck (no appreciable movement for ~1 s) it
      forces a turn in the opposite direction of the previous turn.
    * The mission ends (`done=True`) when the robot is inside the position
      tolerance **and** its yaw error is within the yaw tolerance.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    global _last_pos, _stuck_cnt

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return math.atan2(math.sin(a), math.cos(a))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ----------------------------------------------------------------------
    # termination test (position + yaw)
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
    scans = env.raycast_scan()               # [(bearing, distance), ...]

    # ----------------------------------------------------------------------
    # safety / emergency stop
    # ----------------------------------------------------------------------
    STOP_DIST = 0.12                         # m – immediate stop if anything this close
    if any(dist < STOP_DIST for _, dist in scans):
        return 0.0, 0.0, False               # stay still; harness will abort shortly

    # ----------------------------------------------------------------------
    # parameters (chosen for extreme safety)
    # ----------------------------------------------------------------------
    MAX_SPEED      = 0.30   # m/s – very modest forward speed
    MAX_TURN_RATE  = 0.9    # rad/s
    SAFE_FRONT     = 0.80   # m – front must be clearer than this to drive forward
    FRONT_ANGLE    = math.radians(30)   # ±30° cone for the “clear ahead” test
    Kp_heading     = 2.0    # proportional gain for heading error
    Kp_avoid       = 1.2    # gain for turning away from nearest obstacle
    STUCK_DIST     = 0.04   # m – motion below this is considered “no move”
    STUCK_LIMIT    = 60     # steps (~1 s) before we force a turn

    # ----------------------------------------------------------------------
    # analyse front sector
    # ----------------------------------------------------------------------
    front_distances = [dist for bearing, dist in scans if abs(bearing) <= FRONT_ANGLE]
    min_front = min(front_distances) if front_distances else 6.0

    # ----------------------------------------------------------------------
    # find the closest obstacle overall (bearing, distance)
    # ----------------------------------------------------------------------
    bearing_closest, dist_closest = min(scans, key=lambda bd: bd[1])

    # ----------------------------------------------------------------------
    # stuck detection (based on displacement)
    # ----------------------------------------------------------------------
    if _last_pos is not None:
        moved = math.hypot(x - _last_pos[0], y - _last_pos[1])
        if moved < STUCK_DIST:
            _stuck_cnt += 1
        else:
            _stuck_cnt = 0
    _last_pos = (x, y)

    # ----------------------------------------------------------------------
    # decide motion
    # ----------------------------------------------------------------------
    if min_front >= SAFE_FRONT and _stuck_cnt < STUCK_LIMIT:
        # --------------------------------------------------------------
        # Front is clear and we are not stuck → drive toward goal
        # --------------------------------------------------------------
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - yaw)

        # angular command (proportional)
        w = clamp(Kp_heading * heading_error, -MAX_TURN_RATE, MAX_TURN_RATE)

        # forward speed – scale with how far the nearest front obstacle is
        speed_factor = min(1.0, min_front / 2.0)   # 0 … 1
        v = MAX_SPEED * speed_factor

        # additional slowdown when we are close to the goal
        if dist_to_goal < 2.0:
            v *= dist_to_goal / 2.0

        v = clamp(v, 0.0, MAX_SPEED)
        return v, w, False
    else:
        # --------------------------------------------------------------
        # Obstacle ahead, we are stuck, or front not clear → rotate in place
        # --------------------------------------------------------------
        # Turn away from the nearest obstacle (bearing is relative to heading)
        # Positive bearing → obstacle on the left → turn right (negative w)
        # Negative bearing → obstacle on the right → turn left (positive w)
        w = clamp(-Kp_avoid * bearing_closest, -MAX_TURN_RATE, MAX_TURN_RATE)

        # If we have been stuck for a while, reverse turn direction
        if _stuck_cnt >= STUCK_LIMIT:
            w = -w

        v = 0.0
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
