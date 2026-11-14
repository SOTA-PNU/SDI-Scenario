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
    Reactive potential‑field controller for Track B.

    * Uses only the 180° ray‑cast sensor (env.raycast_scan).
    * Generates an attractive vector toward the goal and repulsive vectors
      away from obstacles that are closer than REPEL_RANGE.
    * The resulting heading is used in a simple proportional controller.
    * Forward speed is reduced when obstacles are nearby and is set to zero
      if anything is closer than STOP_DISTANCE.
    * Stops (done=True) when both position and yaw are within the tolerances
      defined in MISSION.
    """
    # ----------------------------------------------------------------------
    # unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # helper to wrap angles to [-π, π]
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    # ----------------------------------------------------------------------
    # check for termination (goal reached & correctly oriented)
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    if dist_to_goal <= pos_tol:
        yaw_error = wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True      # success

    # ----------------------------------------------------------------------
    # sensor scan
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()          # [(bearing, distance), ...]
    # parameters
    MAX_SPEED = 0.8                     # m/s – keep a modest speed
    MAX_ANG = 1.0                       # rad/s
    REPEL_RANGE = 2.0                   # meters – distance where obstacles push back
    STOP_DISTANCE = 0.25                # meters – immediate stop if anything this close
    K_ATT = 0.8                         # attraction gain
    K_REP = 0.6                         # repulsion gain
    KP_ANG = 1.5                        # heading error → angular speed gain

    # ----------------------------------------------------------------------
    # compute attractive vector (world frame)
    # ----------------------------------------------------------------------
    # direction to goal in world frame
    att_x = K_ATT * dx
    att_y = K_ATT * dy

    # ----------------------------------------------------------------------
    # compute repulsive vectors (robot frame → rotate to world)
    # ----------------------------------------------------------------------
    rep_x_world = 0.0
    rep_y_world = 0.0
    min_dist = 6.0
    for bearing, dist in scans:
        # keep track of the closest obstacle for speed scaling / stop check
        if dist < min_dist:
            min_dist = dist

        if dist < REPEL_RANGE:
            # magnitude of repulsion (stronger the closer)
            mag = K_REP * (1.0 / dist - 1.0 / REPEL_RANGE) / (dist * dist)
            # bearing is relative to robot heading; convert to world frame
            # robot heading vector is (cos(yaw), sin(yaw))
            # a point at (bearing, dist) in robot frame has world angle = yaw + bearing
            world_angle = yaw + bearing
            rep_x_world += -mag * math.cos(world_angle)   # negative → push away
            rep_y_world += -mag * math.sin(world_angle)

    # ----------------------------------------------------------------------
    # combine vectors
    # ----------------------------------------------------------------------
    total_x = att_x + rep_x_world
    total_y = att_y + rep_y_world

    # desired heading in world frame
    desired_heading = math.atan2(total_y, total_x)
    heading_error = wrap_angle(desired_heading - yaw)

    # ----------------------------------------------------------------------
    # angular command (proportional)
    # ----------------------------------------------------------------------
    w = max(-MAX_ANG, min(MAX_ANG, KP_ANG * heading_error))

    # ----------------------------------------------------------------------
    # speed command – scale down when obstacles are near
    # ----------------------------------------------------------------------
    if min_dist < STOP_DISTANCE:
        v = 0.0                         # emergency stop
    else:
        # linear speed proportional to how clear the front is
        # we use a simple linear scaling: full speed when min_dist >= REPEL_RANGE,
        # zero when min_dist == STOP_DISTANCE
        if min_dist >= REPEL_RANGE:
            speed_factor = 1.0
        else:
            # linear interpolation between STOP_DISTANCE and REPEL_RANGE
            speed_factor = (min_dist - STOP_DISTANCE) / (REPEL_RANGE - STOP_DISTANCE)
            speed_factor = max(0.0, min(1.0, speed_factor))

        # also reduce speed when we are close to the goal for smoother stopping
        if dist_to_goal < 2.0:
            speed_factor *= (dist_to_goal / 2.0)

        v = MAX_SPEED * speed_factor

    # ----------------------------------------------------------------------
    # clamp commands
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
