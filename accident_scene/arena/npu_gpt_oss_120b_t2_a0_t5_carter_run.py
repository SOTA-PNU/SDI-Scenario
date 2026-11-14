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


def controller(t, pose, env):
    """Reactive gap‑following controller with goal bias.

    Uses the 180° ray fan (61 beams) to find the widest safe gap and steers
    toward the centre of that gap, biasing the choice toward the direction of
    the goal.  When the robot reaches the goal disc it stops and aligns its
    yaw to the goal orientation.
    """
    # ------------------------------------------------------------------
    # Parameters (tune if needed)
    # ------------------------------------------------------------------
    MAX_SPEED = 1.0          # m/s, nominal forward speed
    SAFE_DIST = 1.2          # m, distance considered free of obstacles
    MIN_STOP_DIST = 0.2      # m, stop if something is this close ahead
    K_ANG = 2.0              # proportional gain for angular velocity
    MAX_ANGULAR = 1.5        # rad/s, hard limit on yaw rate
    GAP_SCORE_WEIGHT = 0.8   # weight for alignment with goal bearing (0..1)

    # ------------------------------------------------------------------
    # Goal information (global)
    # ------------------------------------------------------------------
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # Current robot state
    # ------------------------------------------------------------------
    x, y, yaw = pose

    # ------------------------------------------------------------------
    # Vector to goal expressed in robot frame
    # ------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    # Goal bearing relative to robot heading (wrap to [-pi, pi])
    goal_bearing = math.atan2(dy, dx) - yaw
    while goal_bearing > math.pi:
        goal_bearing -= 2.0 * math.pi
    while goal_bearing < -math.pi:
        goal_bearing += 2.0 * math.pi

    # ------------------------------------------------------------------
    # Goal‑region handling (stop and align yaw)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal <= pos_tol:
        # Inside goal disc – stop forward motion and rotate to goal yaw.
        v_cmd = 0.0
        yaw_error = goal_yaw - yaw
        # wrap yaw_error to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2.0 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2.0 * math.pi
        w_cmd = K_ANG * yaw_error
        # clamp angular speed
        if w_cmd > MAX_ANGULAR:
            w_cmd = MAX_ANGULAR
        elif w_cmd < -MAX_ANGULAR:
            w_cmd = -MAX_ANGULAR
        # finish when orientation is within tolerance
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True
        else:
            return v_cmd, w_cmd, False

    # ------------------------------------------------------------------
    # Get raycast data
    # ------------------------------------------------------------------
    scan = env.raycast_scan()   # 61 beams, 180°, max_range=6.0

    # ------------------------------------------------------------------
    # Identify free gaps (distance > SAFE_DIST)
    # ------------------------------------------------------------------
    gaps = []   # list of (start_bearing, end_bearing)
    in_gap = False
    gap_start = 0.0

    for bearing, dist in scan:
        free = dist > SAFE_DIST
        if free and not in_gap:
            # start a new gap
            in_gap = True
            gap_start = bearing
        elif not free and in_gap:
            # close current gap
            in_gap = False
            gap_end = bearing
            gaps.append((gap_start, gap_end))

    # If the last beam was free, close the final gap at the end of FOV
    if in_gap:
        gap_end = scan[-1][0]
        gaps.append((gap_start, gap_end))

    # ------------------------------------------------------------------
    # If no gap found (everything too close), fall back to rotating away
    # ------------------------------------------------------------------
    if not gaps:
        # Find the closest obstacle bearing (minimum distance)
        min_dist = float('inf')
        min_bearing = 0.0
        for bearing, dist in scan:
            if dist < min_dist:
                min_dist = dist
                min_bearing = bearing
        # Turn opposite to the closest obstacle
        w_cmd = -K_ANG * (1.0 if min_bearing >= 0.0 else -1.0)
        w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, w_cmd))
        return 0.0, w_cmd, False

    # ------------------------------------------------------------------
    # Score gaps: prefer wide gaps and those whose centre aligns with goal bearing
    # ------------------------------------------------------------------
    best_gap = None
    best_score = -float('inf')
    for start, end in gaps:
        width = end - start
        centre = (start + end) / 2.0
        # Alignment term: smaller angular deviation from goal bearing is better
        align_score = -abs(centre - goal_bearing)
        # Overall score combines width and alignment
        score = width + GAP_SCORE_WEIGHT * align_score
        if score > best_score:
            best_score = score
            best_gap = (start, end, centre, width)

    # ------------------------------------------------------------------
    # Desired steering direction is the centre of the best gap
    # ------------------------------------------------------------------
    _, _, gap_centre, _ = best_gap
    desired_bearing = gap_centre

    # ------------------------------------------------------------------
    # Angular velocity command (proportional to desired bearing)
    # ------------------------------------------------------------------
    w_cmd = K_ANG * desired_bearing
    # Clamp angular speed
    if w_cmd > MAX_ANGULAR:
        w_cmd = MAX_ANGULAR
    elif w_cmd < -MAX_ANGULAR:
        w_cmd = -MAX_ANGULAR

    # ------------------------------------------------------------------
    # Forward speed command – reduce speed when turning sharply or obstacle close
    # ------------------------------------------------------------------
    # Turning factor (more turn -> slower)
    turn_factor = max(0.1, 1.0 - abs(desired_bearing) / (math.pi / 2))

    # Obstacle factor: use the minimum distance in the front sector
    FRONT_SECTOR = math.radians(30)   # +/- 30°
    min_front = float('inf')
    for bearing, dist in scan:
        if abs(bearing) <= FRONT_SECTOR and dist < min_front:
            min_front = dist
    if min_front == float('inf'):
        obstacle_factor = 1.0
    elif min_front < MIN_STOP_DIST:
        obstacle_factor = 0.0
    else:
        obstacle_factor = min(1.0, min_front / SAFE_DIST)

    v_cmd = MAX_SPEED * turn_factor * obstacle_factor

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
