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

# ---- persistent simple state -------------------------------------------------
_state = "to_goal"      # "to_goal" or "wall" (right‑hand wall following)
_turn_steps = 0        # counts steps while turning in place (fallback)
_turn_dir = 1          # 1 = turn left, -1 = turn right (used only for fallback)


def controller(t, pose, env):
    """
    Bug‑style controller using only the forward 180° ray‑cast.

    * In “to_goal” mode the robot steers toward the goal.  If an obstacle is
      detected within SAFE_FRONT it switches to “wall” mode.
    * In “wall” mode the robot follows the right‑hand wall (keeping a desired
      distance).  When it re‑encounters the straight line from START to GOAL
      (cross‑track error < 0.5 m) and the front is clear, it switches back to
      “to_goal”.
    * An emergency stop is issued if any beam reports a distance below
      STOP_DIST.
    * The mission ends when the robot is inside the position tolerance **and**
      its yaw error is within the yaw tolerance.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    global _state, _turn_steps, _turn_dir

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    start_x, start_y, _ = START

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return math.atan2(math.sin(a), math.cos(a))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def get_distance(target_bearing, tolerance=0.05):
        """
        Return the smallest distance among beams whose bearing is within
        `tolerance` rad of `target_bearing`.  If none are found, returns 6.0
        (the max range).
        """
        best = 6.0
        for bearing, dist in scans:
            if abs(bearing - target_bearing) <= tolerance:
                if dist < best:
                    best = dist
        return best

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
            return 0.0, 0.0, True   # success

    # ----------------------------------------------------------------------
    # sensor scan
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()               # [(bearing, distance), ...]

    # ----------------------------------------------------------------------
    # safety parameters (chosen for robustness)
    # ----------------------------------------------------------------------
    MAX_SPEED      = 0.55   # m/s – modest forward speed
    MAX_TURN_RATE  = 1.0    # rad/s
    SAFE_FRONT     = 0.85   # meters – front must be clearer than this to go straight
    STOP_DIST      = 0.15   # meters – immediate stop if anything this close
    DESIRED_WALL   = 0.55   # meters – target distance to the right wall
    Kp_heading     = 2.0    # proportional gain for heading error
    Kp_wall        = 1.2    # gain for wall‑following error
    TURN_TIMEOUT   = 180    # steps (~3 s) before we flip turn direction in fallback

    # ----------------------------------------------------------------------
    # emergency stop if any beam is too close
    # ----------------------------------------------------------------------
    if any(dist < STOP_DIST for _, dist in scans):
        return 0.0, 0.0, False   # stay still; harness will abort after a few seconds

    # ----------------------------------------------------------------------
    # analyse front sector (±30°) for the nearest obstacle
    # ----------------------------------------------------------------------
    FRONT_ANGLE = math.radians(30)
    front_distances = [dist for bearing, dist in scans if abs(bearing) <= FRONT_ANGLE]
    min_front = min(front_distances) if front_distances else 6.0

    # ----------------------------------------------------------------------
    # state machine
    # ----------------------------------------------------------------------
    if _state == "to_goal":
        # --------------------------------------------------------------
        # TO_GOAL mode – head straight for the goal
        # --------------------------------------------------------------
        if min_front < SAFE_FRONT:
            _state = "wall"          # obstacle ahead → start wall following
        else:
            # drive forward, steering toward the goal
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - yaw)
            w = clamp(Kp_heading * heading_error, -MAX_TURN_RATE, MAX_TURN_RATE)

            # speed scales down when the front obstacle is near
            speed_factor = min(1.0, min_front / 2.0)   # 0 … 1
            v = MAX_SPEED * speed_factor

            # extra slowdown when we are close to the goal for a smoother stop
            if dist_to_goal < 2.0:
                v *= dist_to_goal / 2.0

            v = clamp(v, 0.0, MAX_SPEED)
            return v, w, False

    # ----------------------------------------------------------------------
    # WALL FOLLOWING mode (right‑hand rule)
    # ----------------------------------------------------------------------
    # At this point we are in "wall" mode
    # --------------------------------------------------------------
    # Get distance to the right side (bearing ≈ -π/2)
    # --------------------------------------------------------------
    right_dist = get_distance(-math.pi / 2, tolerance=0.07)   # about ±4°

    # --------------------------------------------------------------
    # If there is no wall on the right (max range), turn right to find one
    # --------------------------------------------------------------
    if right_dist >= 6.0:
        # rotate right in place until we see something on the right
        v = 0.0
        w = -MAX_TURN_RATE
        _turn_steps += 1
        if _turn_steps > TURN_TIMEOUT:
            # fallback: flip direction and try left
            _turn_dir = -_turn_dir
            _turn_steps = 0
        return v, w, False

    # --------------------------------------------------------------
    # Wall‑following control: maintain DESIRED_WALL distance on the right
    # --------------------------------------------------------------
    wall_error = right_dist - DESIRED_WALL          # >0 → too far from wall
    # We want to turn right (negative w) when too far, left (positive w) when too close
    w = clamp(-Kp_wall * wall_error, -MAX_TURN_RATE, MAX_TURN_RATE)

    # modest forward speed while following the wall
    v = clamp(0.6 * MAX_SPEED, 0.0, MAX_SPEED)

    # --------------------------------------------------------------
    # Check if we can leave the wall and head for the goal again
    # --------------------------------------------------------------
    # Compute cross‑track error to the straight line START→GOAL
    line_dx = goal_x - start_x
    line_dy = goal_y - start_y
    line_len2 = line_dx * line_dx + line_dy * line_dy
    if line_len2 > 0:
        proj = ((x - start_x) * line_dx + (y - start_y) * line_dy) / line_len2
        proj = clamp(proj, 0.0, 1.0)   # stay within segment
        line_pt_x = start_x + proj * line_dx
        line_pt_y = start_y + proj * line_dy
        cross_err = math.hypot(x - line_pt_x, y - line_pt_y)
    else:
        cross_err = 6.0

    # If we are close to the m‑line and the front is clear, switch back
    if cross_err < 0.5 and min_front >= SAFE_FRONT:
        _state = "to_goal"
        _turn_steps = 0

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
