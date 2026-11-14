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
    """Wrap an angle into the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def controller(t, pose, env):
    """
    Deterministic reactive controller based on a simple artificial potential
    field (attractive to the goal, repulsive from nearby obstacles).

    * The attractive component points toward the goal.
    * Each laser beam that reports a distance < REPEL_RANGE contributes a
      repulsive vector whose magnitude grows sharply as the distance shrinks.
    * The sum of these vectors yields a desired steering bearing.
    * Forward speed is reduced when the front clearance is low, when a sharp
      turn is required, and when the robot is close to the goal.
    * When the robot reaches the goal disc it aligns its yaw to the goal yaw
      and stops.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – feel free to adjust if needed.
    # ------------------------------------------------------------------
    MAX_SPEED = 0.55            # nominal forward speed [m/s]
    MAX_ANG_SPEED = 1.2         # maximum yaw rate [rad/s]
    Kp_angle = 2.0              # proportional gain for steering
    Kp_yaw = 1.0                # gain for final yaw alignment
    SAFETY_DIST = 0.8           # distance below which we consider the front unsafe
    REPEL_RANGE = 2.0           # range within which obstacles exert repulsion
    K_ATT = 0.6                 # attractive force gain
    K_REP = 0.9                 # repulsive force gain
    CLOSE_GOAL_DIST = 2.0       # start to slow down when this close to the goal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pose & goal information
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Goal bearing expressed in the robot's local frame
    goal_angle = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ------------------------------------------------------------------
    # Goal‑reached handling (position first, then orientation)
    # ------------------------------------------------------------------
    if distance_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = _wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # success – stop the mission
        # Rotate in place to align yaw.
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_yaw * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------------
    scan = env.raycast_scan()          # 61 beams over 180°, max_range = 6.0 m

    # ------------------------------------------------------------------
    # 1) Attractive vector (pointing toward the goal)
    # ------------------------------------------------------------------
    att_x = K_ATT * math.cos(goal_angle)
    att_y = K_ATT * math.sin(goal_angle)

    # ------------------------------------------------------------------
    # 2) Repulsive vectors from obstacles
    # ------------------------------------------------------------------
    rep_x = 0.0
    rep_y = 0.0
    eps = 1e-6  # avoid division by zero

    for bearing, dist in scan:
        # Only consider obstacles that are closer than REPEL_RANGE
        if dist < REPEL_RANGE and dist > eps:
            # Classic inverse‑square repulsion with a cutoff at REPEL_RANGE
            # magnitude = K_REP * (1/dist - 1/REPEL_RANGE) / (dist**2)
            magnitude = K_REP * (1.0 / dist - 1.0 / REPEL_RANGE) / (dist * dist)
            # Repulsion points opposite to the obstacle direction
            rep_x += magnitude * math.cos(bearing + math.pi)
            rep_y += magnitude * math.sin(bearing + math.pi)

    # ------------------------------------------------------------------
    # 3) Combine attractive and repulsive components
    # ------------------------------------------------------------------
    total_x = att_x + rep_x
    total_y = att_y + rep_y

    # Desired steering bearing relative to current heading
    desired_angle = math.atan2(total_y, total_x)

    # Clamp desired_angle to the robot's steering limits (±90°)
    max_steer = math.pi / 2
    if desired_angle > max_steer:
        desired_angle = max_steer
    elif desired_angle < -max_steer:
        desired_angle = -max_steer

    # ------------------------------------------------------------------
    # 4) Angular velocity command
    # ------------------------------------------------------------------
    w = max(-MAX_ANG_SPEED,
            min(MAX_ANG_SPEED, Kp_angle * desired_angle))

    # ------------------------------------------------------------------
    # 5) Forward speed modulation
    # ------------------------------------------------------------------
    # Front clearance – beam whose bearing is closest to zero
    front_beam = min(scan, key=lambda bd: abs(bd[0]))
    front_dist = front_beam[1]

    # Reduce speed when something is close straight ahead
    if front_dist < SAFETY_DIST:
        front_factor = front_dist / SAFETY_DIST
        front_factor = max(0.0, min(1.0, front_factor))
    else:
        front_factor = 1.0

    # Reduce speed when a large turn is required
    turn_factor = max(0.0, 1.0 - abs(desired_angle) / max_steer)

    # Reduce speed when we are very close to the goal (to avoid overshoot)
    if distance_to_goal < CLOSE_GOAL_DIST:
        goal_factor = distance_to_goal / CLOSE_GOAL_DIST
        goal_factor = max(0.0, min(1.0, goal_factor))
    else:
        goal_factor = 1.0

    speed_factor = front_factor * turn_factor * goal_factor
    v = MAX_SPEED * speed_factor

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
