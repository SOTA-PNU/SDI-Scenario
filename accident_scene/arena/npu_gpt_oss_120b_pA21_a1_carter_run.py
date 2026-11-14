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
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

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
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# Waypoints that safely navigate around the known boxes.
# Each tuple is (x, y) in world coordinates.
_WAYPOINTS = [
    (-9.0, -5.0),   # move north along the western wall
    (-9.0, 0.0),    # clear the first cluster of boxes
    (-2.0, 5.0),    # go east above the long central box (id 03)
    (5.0, 5.0),     # pass to the east side of the central corridor
    (5.0, 14.0),    # go north, staying left of the large box near the goal (id 06)
    (8.0, 16.5),    # final goal position
]

# Index of the current waypoint we are heading for.
_current_wp_idx = 0

def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def controller(t, pose, env):
    """
    Simple waypoint‑following controller with minimal obstacle avoidance.

    Returns (v, w, done):
        v – forward speed (m/s)
        w – yaw rate (rad/s)
        done – True when the goal has been reached and aligned.
    """
    global _current_wp_idx

    x, y, yaw = pose

    # ------------------------------------------------------------
    # 1. Check if we have reached the final waypoint (the goal)
    # ------------------------------------------------------------
    goal_x, goal_y, goal_yaw = GOAL
    dxg = goal_x - x
    dyg = goal_y - y
    dist_to_goal = math.hypot(dxg, dyg)
    yaw_error_to_goal = _wrap_angle(goal_yaw - yaw)

    if dist_to_goal <= MISSION["position_tolerance_m"] and \
       abs(yaw_error_to_goal) <= MISSION["yaw_tolerance_rad"]:
        # Goal satisfied – stop the robot.
        return 0.0, 0.0, True

    # ------------------------------------------------------------
    # 2. Determine the current target waypoint
    # ------------------------------------------------------------
    # If we are close enough to the current waypoint, advance to the next.
    wp_x, wp_y = _WAYPOINTS[_current_wp_idx]
    dx_wp = wp_x - x
    dy_wp = wp_y - y
    dist_wp = math.hypot(dx_wp, dy_wp)

    if dist_wp < 0.25:  # within 25 cm → consider waypoint reached
        if _current_wp_idx < len(_WAYPOINTS) - 1:
            _current_wp_idx += 1
            wp_x, wp_y = _WAYPOINTS[_current_wp_idx]
            dx_wp = wp_x - x
            dy_wp = wp_y - y
            dist_wp = math.hypot(dx_wp, dy_wp)

    # Desired heading toward the active waypoint
    desired_yaw = math.atan2(dy_wp, dx_wp)
    yaw_error = _wrap_angle(desired_yaw - yaw)

    # ------------------------------------------------------------
    # 3. Very simple front‑obstacle avoidance using the raycast scan
    # ------------------------------------------------------------
    # Scan for obstacles in a ±30° cone ahead of the robot.
    scan = env.raycast_scan()
    min_front_dist = 6.0  # default: no obstacle
    for bearing, dist in scan:
        if -0.5236 <= bearing <= 0.5236:  # ±30°
            if dist < min_front_dist:
                min_front_dist = dist

    # If something is closer than 0.6 m, we stop forward motion and turn
    # away from the obstacle (simple heuristic: turn left).
    obstacle_avoid = False
    if min_front_dist < 0.6:
        obstacle_avoid = True

    # ------------------------------------------------------------
    # 4. Compute command velocities
    # ------------------------------------------------------------
    # Proportional gains
    K_LINEAR = 0.6   # m/s per meter of error
    K_ANGULAR = 2.0  # rad/s per rad of heading error
    MAX_SPEED = 0.8  # m/s
    MAX_TURN = 1.0   # rad/s

    # If we need to avoid an obstacle, prioritize turning.
    if obstacle_avoid:
        v_cmd = 0.0
        w_cmd = MAX_TURN
    else:
        # Turn towards the waypoint first; only drive forward when roughly aligned.
        w_cmd = max(-MAX_TURN, min(MAX_TURN, K_ANGULAR * yaw_error))
        if abs(yaw_error) < 0.4:  # within ~23°
            v_cmd = min(MAX_SPEED, K_LINEAR * dist_wp)
        else:
            v_cmd = 0.0

    # ------------------------------------------------------------
    # 5. Return the command
    # ------------------------------------------------------------
    return v_cmd, w_cmd, False

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
