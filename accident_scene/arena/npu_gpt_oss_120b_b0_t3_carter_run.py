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
    """
    Conservative potential‑field controller (Track B).

    * Only the 180° ray‑cast sensor is used.
    * An attractive vector points to the goal.
    * Repulsive vectors are generated for every obstacle inside a safety radius.
      The repulsion is stronger the closer the obstacle and the more it lies
      directly in front of the robot (weight by cos(bearing)).
    * Forward speed is kept low (≤ 0.5 m/s) and is reduced when anything is
      closer than the safety radius; it is forced to zero if an obstacle is
      nearer than STOP_DISTANCE.
    * The robot stops (done=True) when it is inside the position tolerance and
      its yaw error is within the yaw tolerance.
    """
    # ----------------------------------------------------------------------
    # unpack current pose and goal pose
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # helper: wrap angle to [-π, π]
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    # ----------------------------------------------------------------------
    # termination test – position + yaw within tolerances
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        yaw_err = wrap_angle(goal_yaw - yaw)
        if abs(yaw_err) <= yaw_tol:
            return 0.0, 0.0, True          # success

    # ----------------------------------------------------------------------
    # sensor scan
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()               # [(bearing, distance), ...]

    # ----------------------------------------------------------------------
    # parameters (tuned conservatively)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.5          # m/s – modest speed to give plenty of reaction time
    MAX_ANG   = 1.0          # rad/s
    SAFE_DIST = 1.0          # meters – start repelling inside this radius
    STOP_DIST = 0.25         # meters – emergency stop if anything this close
    K_ATT = 0.9              # attraction gain
    K_REP = 1.2              # repulsion gain
    KP_ANG = 1.5             # heading‑error → angular speed gain

    # ----------------------------------------------------------------------
    # attractive vector (world frame)
    # ----------------------------------------------------------------------
    att_x = K_ATT * dx
    att_y = K_ATT * dy

    # ----------------------------------------------------------------------
    # repulsive contribution (world frame)
    # ----------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    min_dist = 6.0           # will hold the distance to the closest obstacle

    for bearing, dist in scans:
        # update closest‑obstacle distance (used for speed scaling)
        if dist < min_dist:
            min_dist = dist

        # only consider obstacles inside the safety radius
        if dist < SAFE_DIST:
            # weight by how directly the obstacle lies ahead (cosine of bearing)
            # obstacles to the side have less influence
            front_weight = max(0.0, math.cos(bearing))
            # magnitude grows quickly as distance shrinks
            mag = K_REP * (SAFE_DIST - dist) / (dist * dist) * front_weight

            # bearing is relative to robot heading; convert to world angle
            world_angle = yaw + bearing
            # push *away* from the obstacle → negative direction
            rep_x += -mag * math.cos(world_angle)
            rep_y += -mag * math.sin(world_angle)

    # ----------------------------------------------------------------------
    # combine attractive and repulsive vectors
    # ----------------------------------------------------------------------
    total_x = att_x + rep_x
    total_y = att_y + rep_y

    # desired heading (world frame) – atan2 is safe even if total vector is tiny
    desired_heading = math.atan2(total_y, total_x)
    heading_error = wrap_angle(desired_heading - yaw)

    # ----------------------------------------------------------------------
    # angular command (simple proportional controller)
    # ----------------------------------------------------------------------
    w = max(-MAX_ANG, min(MAX_ANG, KP_ANG * heading_error))

    # ----------------------------------------------------------------------
    # speed command – scale down near obstacles and near the goal
    # ----------------------------------------------------------------------
    if min_dist < STOP_DIST:
        # immediate stop – turn in place until the path clears
        v = 0.0
    else:
        # linear scaling: full speed when min_dist >= SAFE_DIST,
        # zero speed at STOP_DIST
        if min_dist >= SAFE_DIST:
            speed_factor = 1.0
        else:
            speed_factor = (min_dist - STOP_DIST) / (SAFE_DIST - STOP_DIST)

        # additional slowdown when we are close to the goal for a smoother stop
        if dist_to_goal < 2.0:
            speed_factor *= dist_to_goal / 2.0

        v = MAX_SPEED * speed_factor

    # ----------------------------------------------------------------------
    # clamp commands to allowed limits
    # ----------------------------------------------------------------------
    v = max(0.0, min(MAX_SPEED, v))
    w = max(-MAX_ANG, min(MAX_ANG, w))

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
