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
    """Reactive go‑to‑goal controller with simple gap‑finding avoidance.

    The controller runs at 60 Hz.  It uses the forward 180° ray‑fan
    (61 beams) to detect obstacles, chooses a safe steering direction,
    and drives toward the goal.  When the robot reaches the goal disc it
    only rotates in place to align the final yaw.
    """
    # ----------------------------------------------------------------------
    # unpack current pose and goal pose
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # geometric information relative to the robot
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # bearing to the goal expressed in the robot's frame
    angle_to_goal = math.atan2(dy, dx) - yaw
    while angle_to_goal > math.pi:
        angle_to_goal -= 2.0 * math.pi
    while angle_to_goal < -math.pi:
        angle_to_goal += 2.0 * math.pi

    # ----------------------------------------------------------------------
    # If we are already inside the goal disc, stop forward motion and only
    # rotate to achieve the required yaw tolerance.
    # ----------------------------------------------------------------------
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = goal_yaw - yaw
        while yaw_error > math.pi:
            yaw_error -= 2.0 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2.0 * math.pi

        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            # Goal reached and correctly oriented
            return 0.0, 0.0, True

        # Rotate in place to correct yaw
        KP_YAW = 2.0
        MAX_W = 1.5
        w = KP_YAW * yaw_error
        if w > MAX_W:
            w = MAX_W
        elif w < -MAX_W:
            w = -MAX_W
        return 0.0, w, False

    # ----------------------------------------------------------------------
    # Sensor processing
    # ----------------------------------------------------------------------
    scan = env.raycast_scan()               # [(bearing, distance), ...]
    # parameters
    SAFETY_DIST = 0.7                        # stop distance for obstacles
    SAFE_FRONT_DIST = 1.5                    # distance where we start to slow down
    MAX_SPEED = 1.0
    MAX_W = 1.5
    KP_ANGLE = 2.0
    FRONT_CONE = math.radians(30)            # +/- 30° for forward clearance

    # minimum distance in the forward cone (used for speed scaling)
    min_front = 6.0
    for bearing, dist in scan:
        if -FRONT_CONE <= bearing <= FRONT_CONE:
            if dist < min_front:
                min_front = dist

    # ----------------------------------------------------------------------
    # Obstacle avoidance / gap finding
    # ----------------------------------------------------------------------
    if min_front < SAFETY_DIST:
        # ------- gap detection ------------------------------------------------
        gaps = []  # list of (start_bearing, end_bearing, width)
        cur_start = None
        for bearing, dist in scan:
            free = dist > SAFETY_DIST
            if free:
                if cur_start is None:
                    cur_start = bearing
                # still inside a free interval
            else:
                if cur_start is not None:
                    # free interval ends just before this occupied beam
                    end_bearing = bearing
                    width = end_bearing - cur_start
                    if width > 0.0:
                        gaps.append((cur_start, end_bearing, width))
                    cur_start = None
        # handle a free interval that reaches the last beam
        if cur_start is not None:
            end_bearing = scan[-1][0]
            width = end_bearing - cur_start
            if width > 0.0:
                gaps.append((cur_start, end_bearing, width))

        # Choose the most suitable gap
        if gaps:
            best_gap = None
            best_score = -1e9
            for start_b, end_b, width in gaps:
                center = (start_b + end_b) / 2.0
                # Prefer wide gaps, but also bias toward the goal direction
                score = width - 0.5 * abs(center - angle_to_goal)
                if score > best_score:
                    best_score = score
                    best_gap = (start_b, end_b, width, center)
            _, _, _, gap_center = best_gap

            # Desired steering direction = centre of chosen gap
            desired_angle = gap_center
            w = KP_ANGLE * desired_angle
            if w > MAX_W:
                w = MAX_W
            elif w < -MAX_W:
                w = -MAX_W

            # Forward speed proportional to clearance in the direction of the gap
            closest_dist = 6.0
            min_delta = None
            for bearing, dist in scan:
                delta = abs(bearing - gap_center)
                if (min_delta is None) or (delta < min_delta):
                    min_delta = delta
                    closest_dist = dist
            if closest_dist <= SAFETY_DIST:
                v = 0.0
            else:
                # Linear scaling between SAFETY_DIST and SAFE_FRONT_DIST
                scale = (closest_dist - SAFETY_DIST) / (SAFE_FRONT_DIST - SAFETY_DIST)
                if scale < 0.0:
                    scale = 0.0
                if scale > 1.0:
                    scale = 1.0
                v = MAX_SPEED * scale
        else:
            # No free gap found – spin in place toward the goal direction
            desired_angle = angle_to_goal
            w = KP_ANGLE * desired_angle
            if w > MAX_W:
                w = MAX_W
            elif w < -MAX_W:
                w = -MAX_W
            v = 0.0
    else:
        # ------- clear path – drive straight toward the goal -----------------
        desired_angle = angle_to_goal
        w = KP_ANGLE * desired_angle
        if w > MAX_W:
            w = MAX_W
        elif w < -MAX_W:
            w = -MAX_W

        # Speed scaling based on how close the nearest forward obstacle is
        if min_front > SAFE_FRONT_DIST:
            v = MAX_SPEED
        else:
            scale = (min_front - SAFETY_DIST) / (SAFE_FRONT_DIST - SAFETY_DIST)
            if scale < 0.0:
                scale = 0.0
            if scale > 1.0:
                scale = 1.0
            v = MAX_SPEED * scale

        # Slow down when we are very close to the goal to ease stopping
        if dist_to_goal < 0.5:
            v = v * (dist_to_goal / 0.5)

    # ----------------------------------------------------------------------
    # Final clamping (safety)
    # ----------------------------------------------------------------------
    if v < 0.0:
        v = 0.0
    if v > MAX_SPEED:
        v = MAX_SPEED

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
