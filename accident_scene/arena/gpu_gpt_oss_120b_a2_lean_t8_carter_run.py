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
    pose  (x, y, yaw) ground‑truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default – no control code, the robot never
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
MISSION = dict(
    start_xy_yaw=(-9.0, -10.8, 1.0138),
    goal_xy_yaw=(8.0, 16.5, 1.0138),
    timeout_s=120.0,
    position_tolerance_m=0.75,
    yaw_tolerance_rad=0.26,
    max_time_to_goal_s=None,
    check_collision=True,
    boxes=[
        {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},
        {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.25},
        {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.2},
        {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.18},
        {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},
        {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},
        {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},
        {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},
        {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.2},
    ],
)
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def _wrap_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _max_distance_bearing(scan):
    """
    Return the bearing (relative to current heading) that has the largest
    measured distance.  If several beams share the max distance, the first
    encountered is used – deterministic.
    """
    best_bearing = 0.0
    best_dist = -1.0
    for bearing, dist in scan:
        if dist > best_dist:
            best_dist = dist
            best_bearing = bearing
    return best_bearing, best_dist


def controller(t, pose, env):
    """
    Very conservative potential‑field / gap‑following controller.

    1.  Compute the bearing to the goal (pure attractive term).
    2.  From the 180° ray‑scan pick the bearing with the longest clear
        distance – this approximates the centre of the widest free gap.
    3.  Blend the two bearings (goal‑bias weight decreases when we are far
        from the goal, giving the gap more influence).
    4.  Generate an angular command proportional to the blended bearing.
    5.  Linear speed is modest (≤ 0.5 m/s) and is reduced sharply when a
        large turn is required or when something is very close ahead.
    6.  Stop and declare success when position and yaw are within the
        tolerances defined in MISSION.
    """
    # --------------------------------------------------------------
    # Tunable constants (ASCII only)
    # --------------------------------------------------------------
    MAX_SPEED = 0.5            # m/s – safe ceiling speed
    MAX_ANG = 1.0              # rad/s
    Kp_ang = 1.0               # proportional gain for angular error
    GOAL_WEIGHT_FAR = 0.9      # weight of goal bearing when far away
    GOAL_WEIGHT_NEAR = 0.4     # weight of goal bearing when near the goal
    NEAR_DISTANCE = 5.0        # distance at which we start trusting the gap more
    SAFE_FRONT = 0.5           # stop forward motion if something this close ahead
    WALL_MARGIN = 0.8          # keep at least this far from arena edges
    # Approximate arena limits (derived from map dimensions)
    X_MIN, X_MAX = -10.0, 10.0
    Y_MIN, Y_MAX = -12.0, 18.0

    # --------------------------------------------------------------
    # Current pose and goal vector
    # --------------------------------------------------------------
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    bearing_to_goal = math.atan2(dy, dx)          # absolute bearing in world frame
    rel_bearing_to_goal = _wrap_angle(bearing_to_goal - yaw)  # relative to robot heading

    # --------------------------------------------------------------
    # Success test
    # --------------------------------------------------------------
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(gyaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # --------------------------------------------------------------
    # Scan processing – find the direction with the longest clear range
    # --------------------------------------------------------------
    scan = env.raycast_scan()
    gap_bearing, gap_dist = _max_distance_bearing(scan)

    # --------------------------------------------------------------
    # Blend goal bearing with gap bearing
    # --------------------------------------------------------------
    if dist_to_goal > NEAR_DISTANCE:
        goal_weight = GOAL_WEIGHT_FAR
    else:
        goal_weight = GOAL_WEIGHT_NEAR

    blended_bearing = (1.0 - goal_weight) * gap_bearing + goal_weight * rel_bearing_to_goal
    blended_bearing = _wrap_angle(blended_bearing)

    # --------------------------------------------------------------
    # Angular command
    # --------------------------------------------------------------
    ang = Kp_ang * blended_bearing

    # --------------------------------------------------------------
    # Wall‑avoidance bias (small constant nudges)
    # --------------------------------------------------------------
    wall_bias = 0.0
    if x - X_MIN < WALL_MARGIN:
        wall_bias += 0.2          # push right
    if X_MAX - x < WALL_MARGIN:
        wall_bias -= 0.2          # push left
    if y - Y_MIN < WALL_MARGIN:
        wall_bias += 0.15         # push upward (positive yaw)
    if Y_MAX - y < WALL_MARGIN:
        wall_bias -= 0.15         # push downward (negative yaw)
    ang += wall_bias

    # --------------------------------------------------------------
    # Linear speed – proportional to remaining distance, reduced for sharp turns
    # --------------------------------------------------------------
    speed = min(MAX_SPEED, 0.6 * dist_to_goal)   # slower when far, but capped
    if abs(blended_bearing) > 0.6:               # need a noticeable turn → slow down
        speed *= 0.5

    # --------------------------------------------------------------
    # Immediate front safety – stop if something very close is straight ahead
    # --------------------------------------------------------------
    front_blocked = any(
        abs(bearing) < 0.12 and dist < SAFE_FRONT
        for bearing, dist in scan if dist < 6.0
    )
    if front_blocked:
        speed = 0.0

    # --------------------------------------------------------------
    # Clamp angular velocity to the allowed maximum
    # --------------------------------------------------------------
    if ang > MAX_ANG:
        ang = MAX_ANG
    elif ang < -MAX_ANG:
        ang = -MAX_ANG

    return speed, ang, False


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
