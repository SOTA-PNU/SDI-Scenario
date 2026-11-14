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

def _wrap_angle(angle):
    """Wrap angle to [-π, π)."""
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    It combines an attractive vector toward the goal with repulsive vectors
    generated from the ray‑cast scan.  The resulting heading is used to set a
    proportional yaw rate, while forward speed is reduced when heading error
    is large or obstacles are close.

    Parameters
    ----------
    t : float
        Simulation time (seconds). Unused.
    pose : tuple(float, float, float)
        Current robot pose (x, y, yaw) in world coordinates.
    env : ArenaHarness
        Harness providing `raycast_scan`.

    Returns
    -------
    v : float
        Forward speed command [m/s].
    w : float
        Yaw rate command [rad/s].
    done : bool
        True when the robot has reached the goal within tolerances.
    """
    # ----------------------------------------------------------------------
    # 1) Goal handling
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal in world frame
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # If we are inside the position tolerance, just align yaw and stop
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True          # success
        # Rotate in place to fix yaw
        w = max(-0.8, min(0.8, 2.0 * yaw_error))
        return 0.0, w, False

    # ----------------------------------------------------------------------
    # 2) Sensor scan
    # ----------------------------------------------------------------------
    scan = env.raycast_scan()   # list of (bearing_rel_to_heading, distance)

    # ----------------------------------------------------------------------
    # 3) Potential‑field computation (vectors expressed in robot frame)
    # ----------------------------------------------------------------------
    # Attractive component (pointing to goal, expressed in robot frame)
    # Transform world goal vector into robot frame by rotating -yaw
    goal_vec_world_x = dx / (dist_to_goal + 1e-6)
    goal_vec_world_y = dy / (dist_to_goal + 1e-6)
    # rotation: [cos(-yaw) -sin(-yaw); sin(-yaw) cos(-yaw)]
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    att_x = cos_y * goal_vec_world_x - sin_y * goal_vec_world_y
    att_y = sin_y * goal_vec_world_x + cos_y * goal_vec_world_y
    k_att = 0.9
    att_x *= k_att
    att_y *= k_att

    # Repulsive component
    rep_x = 0.0
    rep_y = 0.0
    rep_range = 2.0          # meters – influence distance
    k_rep = 0.6
    for bearing, distance in scan:
        if distance >= rep_range or distance <= 0.0:
            continue
        # magnitude grows when distance gets smaller
        mag = k_rep * (1.0 / distance - 1.0 / rep_range) / (distance ** 2)
        # direction opposite to the bearing (away from obstacle)
        rep_x -= math.cos(bearing) * mag
        rep_y -= math.sin(bearing) * mag

    # Combine
    tot_x = att_x + rep_x
    tot_y = att_y + rep_y

    # Desired heading in robot frame (0 = straight ahead)
    desired_heading = math.atan2(tot_y, tot_x)   # signed angle
    heading_error = desired_heading               # because robot frame

    # ----------------------------------------------------------------------
    # 4) Command generation
    # ----------------------------------------------------------------------
    max_turn = 1.0          # rad/s
    w = max(-max_turn, min(max_turn, 2.0 * heading_error))

    # Speed scaling: slower when heading error large or obstacle extremely close
    # Detect very close obstacle straight ahead (within ±15°)
    close_front = False
    for bearing, distance in scan:
        if -0.262 <= bearing <= 0.262 and distance < 0.6:
            close_front = True
            break

    base_speed = 1.2  # m/s
    # Reduce speed proportionally to heading error
    speed_factor = max(0.2, 1.0 - min(1.0, abs(heading_error) / (math.pi / 2)))
    if close_front:
        speed_factor *= 0.2   # almost stop to turn away

    v = base_speed * speed_factor

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
