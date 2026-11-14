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

# ----- Global deterministic state -------------------------------------------------
_state = {
    "mode": "forward",   # "forward" or "turning"
    "turn_dir": 0,       # -1 = right, 1 = left  (used while turning)
    "turn_timer": 0.0,   # seconds spent turning
}

def _wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Deterministic go‑to‑goal controller with a very simple obstacle‑avoidance
    state machine.  The robot moves forward when the front is clear; otherwise
    it turns in the direction that offers the larger side clearance until the
    front clears, then resumes forward motion.  When the goal disc is entered
    the robot aligns its yaw and stops.
    """
    global _state

    # ----------------------------------------------------------------------
    # 1) unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    # Vector / distance to goal
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    # Desired heading and yaw error (world frame)
    desired_heading = math.atan2(dy, dx)
    yaw_error = _wrap_angle(desired_heading - yaw)

    # ----------------------------------------------------------------------
    # 2) Goal reached?  Align yaw then finish.
    # ----------------------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True          # success
        # Rotate in place to correct yaw
        w_align = max(-0.8, min(0.8, 2.0 * yaw_error))
        return 0.0, w_align, False

    # ----------------------------------------------------------------------
    # 3) Scan processing
    # ----------------------------------------------------------------------
    scan = env.raycast_scan()               # [(bearing, distance), ...]
    # Helper to get minimum distance within a bearing range
    def min_dist_in_range(bearing_min, bearing_max):
        dists = [d for b, d in scan if bearing_min <= b <= bearing_max]
        return min(dists) if dists else 6.0   # 6.0 == max_range (no hit)

    # Front sector (±15°)
    front_min = min_dist_in_range(-0.2618, 0.2618)
    # Side sectors (left: +30°…+90°, right: -90°…-30°)
    left_min = min_dist_in_range(0.5236, 1.5708)
    right_min = min_dist_in_range(-1.5708, -0.5236)

    # ----------------------------------------------------------------------
    # 4) State machine
    # ----------------------------------------------------------------------
    max_turn = 0.8          # rad/s
    base_speed = 1.2        # m/s

    # If something is too close in front, switch to turning mode
    front_threshold = 0.8   # meters
    if front_min < front_threshold:
        # Decide turn direction based on which side has more clearance
        if left_min > right_min:
            turn_dir = 1   # turn left (positive yaw)
        else:
            turn_dir = -1  # turn right (negative yaw)

        _state["mode"] = "turning"
        _state["turn_dir"] = turn_dir
        _state["turn_timer"] = 0.0

    # Execute according to current mode
    if _state["mode"] == "turning":
        # Keep turning until front clears sufficiently
        if front_min >= 1.2:          # safe distance to resume forward motion
            _state["mode"] = "forward"
            _state["turn_dir"] = 0
            _state["turn_timer"] = 0.0
            # fall through to forward handling below
        else:
            # Turn in place, very slow forward motion for safety
            w = _state["turn_dir"] * max_turn
            v = 0.15                 # crawl forward while turning
            _state["turn_timer"] += 1.0 / 60.0
            return v, w, False

    # ----------------------------------------------------------------------
    # 5) Forward mode – steer toward goal while respecting obstacles
    # ----------------------------------------------------------------------
    # Simple proportional yaw control toward goal
    kp_yaw = 1.5
    w = max(-max_turn, min(max_turn, kp_yaw * yaw_error))

    # Reduce speed when yaw error is large or when an obstacle is relatively close
    speed_factor = 1.0
    # Slow down for large heading error
    speed_factor *= max(0.2, 1.0 - min(1.0, abs(yaw_error) / (math.pi / 2)))
    # Slow down if any obstacle within 2 m in the forward 60° sector
    front_60_min = min_dist_in_range(-0.5236, 0.5236)
    if front_60_min < 2.0:
        speed_factor *= max(0.1, front_60_min / 2.0)

    v = base_speed * speed_factor

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
