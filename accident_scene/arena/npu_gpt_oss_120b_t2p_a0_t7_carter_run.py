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
"""Controller for ACCIDENT ARENA mission.

Design:
- Deterministic bug‑algorithm style controller.
- Uses only the provided `math` module (imported in the base file) and the ray‑cast sensor.
- Maintains a simple state (mode "go" or "avoid") across calls.
- In "go" mode the robot steers toward the goal while biasing away from the most open direction.
- When an obstacle is detected within a safety cone, it switches to "avoid" mode, turning toward the side with larger clearance until the front is clear, then resumes "go" mode.
- Stops when the robot is inside both the position (0.75 m) and yaw (0.26 rad) tolerances.
"""

# Global state persisted across controller calls
_state = {"mode": "go", "turn_dir": 1}  # turn_dir: 1 = left, -1 = right


def _wrap_angle(angle):
    """Wrap angle to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """Return (v, w, done) for the robot.

    Args:
        t (float): simulation time in seconds.
        pose (tuple): (x, y, yaw) robot pose in world coordinates.
        env: harness environment providing `raycast_scan`.
    """
    # ------------------------------------------------------------------
    # Unpack pose
    x, y, yaw = pose

    # Goal (hard‑coded to match MISSION)
    goal_x, goal_y, goal_yaw = 8.0, 16.5, 1.0138

    # ------------------------------------------------------------------
    # Termination test – position & yaw tolerances
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal <= 0.75 and abs(_wrap_angle(goal_yaw - yaw)) <= 0.26:
        # Reset state for a fresh run (optional but harmless)
        _state["mode"] = "go"
        _state["turn_dir"] = 1
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # Obtain ray‑cast scan (61 beams, 180° FOV, max range 6 m)
    try:
        scan = env.raycast_scan()
    except Exception:
        scan = []

    # ------------------------------------------------------------------
    # Parameters (deterministic)
    MAX_RANGE = 6.0          # sensor max range (m)
    V_MAX = 1.0              # forward speed limit (m/s)
    MAX_W = 1.0              # angular speed limit (rad/s)
    K_W = 2.0                # proportional gain for angular velocity
    SAFE_DIST = 0.8          # distance considered unsafe (m)
    FRONT_ANGLE = math.radians(30)   # front cone for obstacle detection (±30°)

    # ------------------------------------------------------------------
    # Helper: is there an obstacle within the safety cone?
    obstacle_ahead = False
    for bearing, d in scan:
        if d is None:
            continue
        if abs(bearing) <= FRONT_ANGLE and d < SAFE_DIST:
            obstacle_ahead = True
            break

    # ------------------------------------------------------------------
    # State machine
    if _state["mode"] == "go":
        if obstacle_ahead:
            # Choose turn direction based on side clearance
            left_sum = right_sum = 0.0
            left_cnt = right_cnt = 0
            for bearing, d in scan:
                if d is None:
                    continue
                if bearing > 0:
                    left_sum += d
                    left_cnt += 1
                elif bearing < 0:
                    right_sum += d
                    right_cnt += 1
            left_avg = left_sum / left_cnt if left_cnt else MAX_RANGE
            right_avg = right_sum / right_cnt if right_cnt else MAX_RANGE
            _state["turn_dir"] = 1 if left_avg > right_avg else -1
            _state["mode"] = "avoid"
        else:
            # Normal goal‑directed steering with clearance bias
            # Find most open bearing (largest distance)
            best_bearing = 0.0
            best_dist = 0.0
            for bearing, d in scan:
                if d is None or (isinstance(d, float) and math.isnan(d)):
                    d = MAX_RANGE
                if d > best_dist:
                    best_dist = d
                    best_bearing = bearing
            if best_dist == 0.0:
                best_bearing = 0.0
                best_dist = MAX_RANGE
            clearance_angle = yaw + best_bearing

            # Blend goal direction and clearance direction
            angle_to_goal = math.atan2(dy, dx)
            gvx = math.cos(angle_to_goal)
            gvy = math.sin(angle_to_goal)
            cvx = math.cos(clearance_angle)
            cvy = math.sin(clearance_angle)
            comb_x = 1.0 * gvx + 0.5 * cvx
            comb_y = 1.0 * gvy + 0.5 * cvy
            desired_heading = math.atan2(comb_y, comb_x)
            heading_err = _wrap_angle(desired_heading - yaw)

            # Angular velocity (clamped)
            w = max(-MAX_W, min(MAX_W, K_W * heading_err))

            # Forward speed – scale with alignment to desired heading
            if abs(heading_err) > math.pi / 2:
                v = 0.0
            else:
                v = V_MAX * max(0.0, math.cos(heading_err))
            return v, w, False
    else:  # "avoid" mode
        # Turn in the chosen direction while moving slowly
        w = _state["turn_dir"] * MAX_W
        v = 0.5 * V_MAX
        # Check if front is now clear
        front_clear = True
        for bearing, d in scan:
            if d is None:
                continue
            if abs(bearing) <= FRONT_ANGLE and d < SAFE_DIST:
                front_clear = False
                break
        if front_clear:
            _state["mode"] = "go"
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
