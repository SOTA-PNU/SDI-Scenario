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

# -------------------------------------------------------------------------
# Persistent simple state (kept between calls)
# -------------------------------------------------------------------------
_STATE = {
    "mode": "go",        # "go" = head for goal, "wall" = follow obstacle
    "turn_dir": 1,       # 1 = turn left, -1 = turn right (chosen when we hit)
}


def _normalize_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _line_to_goal_clear(scan, rel_goal_heading, cone_half, safe_dist):
    """
    Return True if no ray within ±cone_half of *rel_goal_heading* reports a
    distance smaller than *safe_dist*.
    """
    for bearing, dist in scan:
        if abs(_normalize_angle(bearing - rel_goal_heading)) <= cone_half:
            if dist < safe_dist:
                return False
    return True


def controller(t, pose, env):
    """
    Bug‑2 style “go‑to‑goal + wall‑follow” controller.

    1️⃣  If we are inside the goal disc **and** yaw is within tolerance → stop.
    2️⃣  Otherwise we are either in “go” mode (drive straight to the goal) or
        in “wall” mode (follow the obstacle on the left side).
    3️⃣  In “go” mode we look a short distance ahead (SAFE_DIST).  If something
        is closer we switch to “wall” mode and start turning left.
    4️⃣  In “wall” mode we keep a low forward speed while turning left.
        As soon as the direct line to the goal becomes clear we switch back
        to “go” mode.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Tunable constants
    # ------------------------------------------------------------------
    MAX_V = 1.0                     # m/s (forward speed limit)
    MAX_W = 1.0                     # rad/s (yaw‑rate limit)
    SAFE_DIST = 2.0                 # start avoiding when an obstacle is this close
    FRONT_CONE = math.radians(30)   # half‑width of the forward‑obstacle cone
    GOAL_CLEAR_CONE = math.radians(15)  # cone we require to be clear to resume GO
    MIN_SPEED_GO = 0.2              # minimum forward speed while following wall
    GO_SPEED_FACTOR = 0.8           # factor for speed when heading roughly to goal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 2️⃣  Current pose and goal
    # ------------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # 3️⃣  Goal‑reached test
    # ------------------------------------------------------------------
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_err_to_goal = _normalize_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal) <= MISSION["yaw_tolerance_rad"]):
        # Goal satisfied – stop and signal completion
        return 0.0, 0.0, True

    # ------------------------------------------------------------------
    # 4️⃣  Scan the environment (61 beams, 180° fan)
    # ------------------------------------------------------------------
    scan = env.raycast_scan()   # [(bearing, distance), ...]

    # ------------------------------------------------------------------
    # 5️⃣  Heading to goal expressed relative to robot heading
    # ------------------------------------------------------------------
    world_goal_heading = math.atan2(dy, dx)               # world‑frame angle
    rel_goal_heading = _normalize_angle(world_goal_heading - yaw)

    # ------------------------------------------------------------------
    # 6️⃣  State machine
    # ------------------------------------------------------------------
    mode = _STATE["mode"]

    # --------------------------------------------------------------
    # GO mode – try to drive straight to the goal
    # --------------------------------------------------------------
    if mode == "go":
        # Look for any obstacle in the forward cone that is closer than SAFE_DIST
        obstacle_ahead = None
        for bearing, dist in scan:
            if abs(_normalize_angle(bearing)) <= FRONT_CONE and dist < SAFE_DIST:
                if obstacle_ahead is None or dist < obstacle_ahead[1]:
                    obstacle_ahead = (bearing, dist)

        if obstacle_ahead:
            # Obstacle too close → switch to wall‑follow mode, turn left
            _STATE["mode"] = "wall"
            _STATE["turn_dir"] = 1.0          # left turn
            mode = "wall"                     # handle wall mode in this step

    # --------------------------------------------------------------
    # WALL mode – follow the obstacle on the left side
    # --------------------------------------------------------------
    if mode == "wall":
        # Turn left at max rate while moving slowly
        w = _STATE["turn_dir"] * MAX_W
        v = MAX_V * MIN_SPEED_GO

        # Check whether the direct line to the goal is now clear.
        # If no obstacle within SAFE_DIST in a narrow cone around the goal direction,
        # we can go back to GO mode.
        if _line_to_goal_clear(scan, rel_goal_heading, GOAL_CLEAR_CONE, SAFE_DIST):
            _STATE["mode"] = "go"
            # Re‑compute heading error for this step (now we are leaving wall mode)
            heading_error = rel_goal_heading
            w = max(-MAX_W, min(MAX_W, K_W * heading_error))
            # Speed proportional to how well we are aligned
            speed_factor = 1.0 - min(1.0, abs(heading_error) / math.pi)
            v = MAX_V * max(MIN_SPEED_GO, speed_factor)
        return v, w, False

    # ------------------------------------------------------------------
    # 7️⃣  GO mode – compute normal go‑to‑goal commands (no immediate obstacle)
    # ------------------------------------------------------------------
    # Proportional yaw‑rate toward the goal
    K_W = 1.5
    w = K_W * rel_goal_heading
    w = max(-MAX_W, min(MAX_W, w))

    # Forward speed: slower when we are not pointing toward the goal
    speed_factor = 1.0 - min(1.0, abs(rel_goal_heading) / math.pi)
    # Give a small boost when we are roughly aligned
    speed_factor = max(GO_SPEED_FACTOR * speed_factor, 0.2)
    v = MAX_V * speed_factor
    v = max(0.0, min(MAX_V, v))

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
