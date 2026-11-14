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

def _wrap_angle(a):
    """Wrap angle to the interval [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Proportional‑plus‑repulsive controller.

    * Drive toward the goal.
    * For every ray that hits something closer than REPEL_RANGE, generate a
      repulsive angular offset that pushes the robot away from the obstacle.
    * Combine the attractive heading (to the goal) with the summed repulsive
      offset, then apply a simple proportional law on the resulting heading
      error.
    * Stop when the robot is inside the position‑ and yaw‑tolerances.
    """
    # ----------------------------------------------------------------------
    # Tunable parameters (kept modest to stay inside the 120 s budget)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.2               # m s⁻¹
    MAX_TURN = 1.0                # rad s⁻¹
    REPEL_RANGE = 2.0             # m – start repelling when something is this close
    SAFE_FRONT = 0.8              # m – if something is this close straight ahead, slow down
    KP_HEADING = 2.5              # proportional gain on heading error
    KP_REPEL = 1.2                # gain for repulsive angular contribution
    # ----------------------------------------------------------------------

    # unpack current pose and goal
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # distance & bearing to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_bearing = math.atan2(dy, dx)          # absolute bearing
    goal_error = _wrap_angle(goal_bearing - yaw)  # error relative to robot heading

    # success test – position AND yaw within tolerances
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(_wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # 1)  Scan for obstacles and build a repulsive angular offset
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()
    repulsive_offset = 0.0
    front_too_close = False

    for bearing, distance in scans:
        # distance == max_range (6.0) means no hit – ignore
        if distance >= REPEL_RANGE:
            continue

        # weight grows the closer the obstacle
        influence = (REPEL_RANGE - distance) / REPEL_RANGE  # in (0,1]

        # repulsive direction is opposite to the bearing
        # (+) bearing => obstacle on left => turn right => negative offset
        # (-) bearing => obstacle on right => turn left => positive offset
        repulsive_offset += -math.copysign(1.0, bearing) * influence

        # check very close obstacle directly ahead (±15°)
        if abs(bearing) < math.radians(15) and distance < SAFE_FRONT:
            front_too_close = True

    # combine attractive and repulsive components
    desired_heading_error = goal_error + KP_REPEL * repulsive_offset
    desired_heading_error = _wrap_angle(desired_heading_error)

    # ----------------------------------------------------------------------
    # 2)  Compute command velocities
    # ----------------------------------------------------------------------
    # angular velocity – proportional on the combined error
    w = max(-MAX_TURN, min(MAX_TURN, KP_HEADING * desired_heading_error))

    # forward speed – reduce when we need to turn sharply or when something is
    # very close ahead
    speed_factor = 1.0 - min(1.0, abs(desired_heading_error) / (math.pi / 2))
    v = MAX_SPEED * speed_factor

    if front_too_close:
        # very conservative forward motion when something is right in front
        v = 0.1

    # ensure we keep moving forward (no reverse) and respect a minimum forward
    # speed when we are roughly aligned with the goal
    if abs(desired_heading_error) < math.radians(10):
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
