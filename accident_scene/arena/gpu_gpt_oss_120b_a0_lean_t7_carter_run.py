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

# deterministic state container (module‑level)
_state = {
    "mode": "go",          # "go"  (drive toward goal) or "wall" (follow wall)
    "follow_side": "left", # which side we are following when in wall mode
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
    Bug‑algorithm style controller:
    * Drive straight toward the goal.
    * If an obstacle appears in the front sector, switch to wall‑following
      (left‑hand rule by default, but choose the side with larger clearance).
    * Remain in wall‑following mode until the heading to the goal becomes clear,
      then resume straight driving.
    * Upon entering the goal disc, align yaw and stop.
    """
    global _state

    # ------------------------------------------------------------------
    # 1) unpack pose and compute goal vector
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)            # world‑frame
    yaw_error = _wrap_angle(goal_heading - yaw)  # error in robot frame

    # ------------------------------------------------------------------
    # 2) check for goal completion
    # ------------------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True                # success
        # rotate in place to align yaw
        w_align = max(-0.8, min(0.8, 2.0 * yaw_error))
        return 0.0, w_align, False

    # ------------------------------------------------------------------
    # 3) helper to query minimum distance in a bearing interval
    # ------------------------------------------------------------------
    scan = env.raycast_scan()   # [(bearing, distance), ...]

    def min_dist(b_min, b_max):
        dlist = [d for b, d in scan if b_min <= b <= b_max]
        return min(dlist) if dlist else 6.0   # 6.0 == max_range (no hit)

    # front sector (±30°)
    front_min = min_dist(-0.5236, 0.5236)

    # left side sector (+60° … +90°) – used for wall following distance
    left_min = min_dist(1.0472, 1.5708)

    # right side sector (-90° … -60°)
    right_min = min_dist(-1.5708, -1.0472)

    # ------------------------------------------------------------------
    # 4) state machine logic
    # ------------------------------------------------------------------
    # thresholds (meters)
    front_obstacle_thresh = 0.8   # if something nearer -> need to avoid
    front_clear_thresh    = 1.2   # safe distance to resume go‑to‑goal
    side_target_dist      = 0.5   # desired distance to wall while following

    # decide which side to follow when we first encounter an obstacle
    if _state["mode"] == "go":
        if front_min < front_obstacle_thresh:
            # choose side with larger clearance
            if left_min > right_min:
                _state["follow_side"] = "left"
            else:
                _state["follow_side"] = "right"
            _state["mode"] = "wall"
    else:   # currently wall‑following
        # exit wall mode when front is clear enough AND heading to goal is roughly forward
        if front_min >= front_clear_thresh and abs(yaw_error) < 0.4:
            _state["mode"] = "go"

    # ------------------------------------------------------------------
    # 5) command generation
    # ------------------------------------------------------------------
    max_turn = 0.8          # rad/s
    max_speed = 1.2         # m/s

    # safety stop if something is extremely close directly ahead
    if front_min < 0.3:
        return 0.0, 0.0, False

    if _state["mode"] == "wall":
        # ----- wall‑following mode -----
        # compute side distance and error
        if _state["follow_side"] == "left":
            side_dist = left_min
            side_error = side_target_dist - side_dist   # positive -> too far
            # yaw bias to keep wall on left: turn left if too far, right if too close
            side_bias = 0.6 * side_error
        else:
            side_dist = right_min
            side_error = side_target_dist - side_dist
            side_bias = -0.6 * side_error               # opposite sign for right side

        # also steer a bit away from any very close front obstacle
        front_bias = 0.0
        if front_min < 0.8:
            front_bias = -0.5 if _state["follow_side"] == "left" else 0.5

        w = max(-max_turn, min(max_turn, side_bias + front_bias))

        # move forward slowly while following the wall
        v = 0.5 * max_speed
        return v, w, False

    else:
        # ----- go‑to‑goal mode -----
        # yaw control toward goal
        kp_yaw = 1.5
        w = max(-max_turn, min(max_turn, kp_yaw * yaw_error))

        # speed scaling: slower when heading error large or front obstacle near
        speed_factor = 1.0
        speed_factor *= max(0.2, 1.0 - min(1.0, abs(yaw_error) / (math.pi / 2)))
        if front_min < 2.0:
            speed_factor *= max(0.1, front_min / 2.0)

        v = max_speed * speed_factor
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
