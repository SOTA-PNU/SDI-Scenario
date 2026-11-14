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
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# Design summary:
# - Deterministic go‑to‑goal using a simple state‑machine (go‑to‑goal / follow‑obstacle).
# - Only the `math` module and the provided `env.raycast_scan()` sensor are used.
# - The robot turns away from obstacles detected in a forward cone before they are
#   too close, then follows the obstacle on the chosen side until a clear path
#   to the goal appears.
# - Speed limits: forward ≤ 0.8 m/s, angular ≤ 1.0 rad/s.
# - No randomness, file I/O, threads, or extra imports are used.

# Global state for the controller (deterministic across calls).
_controller_state = {
    "mode": "go",          # "go" or "follow"
    "wall_side": None,      # "left" or "right" when in follow mode
    "hit_point": None,      # (x, y, yaw) where obstacle was first encountered
}


def _wrap_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _choose_wall_side(scan, front_angle_limit=math.radians(15)):
    """Choose which side (left/right) has more free space.

    Returns "left" if the summed distances on the left side are larger,
    otherwise "right".
    """
    left_sum = sum(d for b, d in scan if b > 0 and d < 6.0)
    right_sum = sum(d for b, d in scan if b < 0 and d < 6.0)
    # Treat missing returns as completely free (max_range)
    if left_sum == 0.0:
        left_sum = 6.0 * len([b for b, d in scan if b > 0])
    if right_sum == 0.0:
        right_sum = 6.0 * len([b for b, d in scan if b < 0])
    return "left" if left_sum > right_sum else "right"


def controller(t, pose, env):
    """Navigate from START to GOAL while avoiding obstacles.

    Args:
        t: simulation time (seconds) – unused.
        pose: (x, y, yaw) ground‑truth robot pose.
        env: arena harness providing `raycast_scan()`.

    Returns:
        (v, w, done): forward speed [m/s], yaw rate [rad/s], termination flag.
    """
    # ------------------------------------------------------------------
    # Unpack pose and mission parameters
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # Vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)               # world‑frame heading to goal
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    # ------------------------------------------------------------------
    # Success test
    # ------------------------------------------------------------------
    if dist_to_goal <= pos_tol and abs(yaw_error_to_goal) <= yaw_tol:
        # Reset state for a fresh run if the controller is reused.
        _controller_state["mode"] = "go"
        _controller_state["wall_side"] = None
        _controller_state["hit_point"] = None
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Sensor reading (safe wrapper)
    # ------------------------------------------------------------------
    try:
        scan = env.raycast_scan()  # [(bearing, distance), ...]
    except Exception:
        # If the sensor fails we stop moving but do not terminate.
        return 0.0, 0.0, False

    # ------------------------------------------------------------------
    # Tunable parameters (all deterministic)
    # ------------------------------------------------------------------
    max_speed = 0.8               # m/s – forward speed limit
    max_angular = 1.0             # rad/s – yaw rate limit
    repulsion_range = 3.0         # m – distance within which obstacles repel
    front_dist_thresh = 1.0       # m – distance at which we consider an obstacle "in front"
    front_angle_limit = math.radians(15)  # rad – half‑width of the forward cone

    # ------------------------------------------------------------------
    # Compute repulsive vector from all obstacles within repulsion_range.
    # ------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    for bearing, distance in scan:
        if distance >= 6.0:
            continue  # no hit on this beam
        # Convert bearing to robot‑frame coordinates (x forward, y left)
        obs_x = distance * math.cos(bearing)
        obs_y = distance * math.sin(bearing)
        if distance < repulsion_range:
            # Stronger repulsion the nearer the obstacle (inverse distance weighting)
            w = (repulsion_range - distance) / distance
            rep_x -= w * obs_x
            rep_y -= w * obs_y

    # ------------------------------------------------------------------
    # State machine handling
    # ------------------------------------------------------------------
    mode = _controller_state["mode"]
    if mode == "go":
        # Check for a close obstacle in the forward cone.
        front_obstacle = any(
            distance < front_dist_thresh and abs(bearing) <= front_angle_limit
            for bearing, distance in scan
        )
        if front_obstacle:
            # Switch to follow mode.
            _controller_state["mode"] = "follow"
            _controller_state["hit_point"] = (x, y, yaw)
            _controller_state["wall_side"] = _choose_wall_side(scan, front_angle_limit)
            # Immediately start turning away.
            w = max_angular if _controller_state["wall_side"] == "left" else -max_angular
            v = 0.0
            return v, w, False
        # No immediate obstacle – proceed towards the goal using a combined vector.
        goal_bearing = _wrap_angle(goal_heading - yaw)
        goal_vec_x = math.cos(goal_bearing)
        goal_vec_y = math.sin(goal_bearing)
        total_x = goal_vec_x + rep_x
        total_y = goal_vec_y + rep_y
        desired_bearing = math.atan2(total_y, total_x)
        angle_error = _wrap_angle(desired_bearing)
        w = max(-max_angular, min(max_angular, 2.0 * angle_error))
        # Speed scales with alignment and distance to goal.
        alignment = max(0.0, math.cos(angle_error))
        speed_factor = alignment
        if dist_to_goal < 2.0:
            speed_factor *= (dist_to_goal / 2.0)
        v = max_speed * speed_factor
        if v < 0.05:
            v = 0.0
        return v, w, False
    else:  # mode == "follow"
        # Follow the obstacle on the chosen side.
        side = _controller_state["wall_side"]
        # Desired turning direction.
        turn_dir = 1.0 if side == "left" else -1.0
        # Simple wall‑following: keep a modest forward speed and turn.
        v = 0.4  # modest forward speed while following
        w = turn_dir * max_angular * 0.6  # turn gently while moving
        # Determine if we can leave follow mode: clear forward cone and we are
        # roughly heading toward the goal.
        clear_ahead = all(
            distance >= front_dist_thresh or abs(bearing) > front_angle_limit
            for bearing, distance in scan
        )
        # Compute bearing to goal.
        goal_bearing = _wrap_angle(goal_heading - yaw)
        goal_aligned = abs(goal_bearing) < math.radians(10)
        if clear_ahead and goal_aligned:
            # Switch back to go mode.
            _controller_state["mode"] = "go"
            _controller_state["wall_side"] = None
            _controller_state["hit_point"] = None
            # Re‑compute commands for go mode on the next call.
            return 0.0, 0.0, False
        return v, w, False
# ========================= [END EDIT REGION] ===============================
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
