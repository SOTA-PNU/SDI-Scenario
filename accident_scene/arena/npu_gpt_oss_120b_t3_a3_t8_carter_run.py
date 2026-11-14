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
    Deterministic reactive controller with a simple Bug‑style state machine.

    * **go_to_goal** – steer directly toward the goal as long as the front
      is clear.  When the front becomes unsafe or the goal direction is
      blocked, switch to **wall_follow**.
    * **wall_follow** – keep a modest forward speed while maintaining a
      target distance to the left side (i.e. follow the left wall).  When the
      front clears again *and* the goal direction is free for at least the
      sensor range, switch back to **go_to_goal**.
    * Upon reaching the goal disc the robot aligns its yaw to the goal yaw
      and stops.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (feel free to adjust if needed)
    # ------------------------------------------------------------------
    MAX_SPEED = 0.70          # nominal forward speed [m/s]
    MIN_SPEED = 0.12          # forward speed while wall‑following
    MAX_ANG_SPEED = 1.20      # maximum yaw rate [rad/s]
    Kp_angle = 2.0            # gain for steering toward the goal
    Kp_wall = 0.8             # gain for left‑wall distance error
    Kp_yaw = 1.0              # gain for final yaw alignment
    FRONT_SAFE = 0.70         # distance below which the front is considered unsafe
    DESIRED_WALL_DIST = 0.80  # target distance to the left wall (metres)
    CLOSE_GOAL_DIST = 2.0     # start to slow down when this close to the goal
    MAX_RANGE = 6.0           # raycast max range (metres)
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
    scan = env.raycast_scan()               # 61 beams over 180°, max_range = 6.0 m

    # Front clearance – beam whose bearing is closest to zero
    front_beam = min(scan, key=lambda bd: abs(bd[0]))
    front_dist = front_beam[1]

    # ------------------------------------------------------------------
    # Determine whether the goal direction is clear for at least the sensor
    # range.  If the goal bearing lies inside the scanned FOV we look at the
    # nearest beam; otherwise we treat it as “not clear”.
    # ------------------------------------------------------------------
    los_clear = False
    if abs(goal_angle) <= math.pi / 2:
        nearest_beam = min(scan, key=lambda bd: abs(bd[0] - goal_angle))
        # “Clear” means no obstacle within the sensor range (6 m) along that
        # bearing – we keep a tiny margin of 0.2 m.
        los_clear = nearest_beam[1] >= (MAX_RANGE - 0.2)

    # ------------------------------------------------------------------
    # Simple persistent state (function attribute)
    # ------------------------------------------------------------------
    if not hasattr(controller, "mode"):
        controller.mode = "go_to_goal"
    mode = controller.mode

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    if mode == "go_to_goal":
        # Switch to wall‑following if the front is unsafe or the goal direction
        # is blocked (within sensor range).
        if front_dist < FRONT_SAFE or (abs(goal_angle) <= math.pi / 2 and not los_clear):
            controller.mode = "wall_follow"
            mode = "wall_follow"
    else:  # mode == "wall_follow"
        # Return to direct navigation when the front is clear **and** the goal
        # direction is free for at least the sensor range.
        if front_dist >= FRONT_SAFE and los_clear:
            controller.mode = "go_to_goal"
            mode = "go_to_goal"

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------
    if mode == "go_to_goal":
        # -------------------------------------------------
        # Steering toward the goal
        # -------------------------------------------------
        # Small deterministic bias (sinusoid) helps break symmetry in tight
        # passages while remaining fully deterministic.
        bias = 0.05 * math.sin(0.3 * t)
        desired_angle = _wrap_angle(goal_angle + bias)

        # Clamp to the robot's feasible steering range (±90°)
        max_steer = math.pi / 2
        if desired_angle > max_steer:
            desired_angle = max_steer
        elif desired_angle < -max_steer:
            desired_angle = -max_steer

        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_angle * desired_angle))

        # -------------------------------------------------
        # Forward speed modulation
        # -------------------------------------------------
        # 1) Front‑clearance factor
        if front_dist < FRONT_SAFE:
            front_factor = front_dist / FRONT_SAFE
            front_factor = max(0.0, min(1.0, front_factor))
        else:
            front_factor = 1.0

        # 2) Turning penalty – slow down for sharp turns
        turn_factor = max(0.0, 1.0 - abs(desired_angle) / max_steer)

        # 3) Goal proximity – reduce speed when we are very close
        if distance_to_goal < CLOSE_GOAL_DIST:
            goal_factor = distance_to_goal / CLOSE_GOAL_DIST
            goal_factor = max(0.0, min(1.0, goal_factor))
        else:
            goal_factor = 1.0

        speed_factor = front_factor * turn_factor * goal_factor
        v = MAX_SPEED * speed_factor

        # Very small speeds are useless – treat them as zero
        if v < 0.02:
            v = 0.0

        return v, w, False

    else:  # wall_follow mode
        # -------------------------------------------------
        # Left‑wall distance (bearing ≈ +π/2)
        # -------------------------------------------------
        left_beam = max(scan, key=lambda bd: bd[0])   # most positive bearing
        left_dist = left_beam[1]

        # -------------------------------------------------
        # Wall‑following steering
        # -------------------------------------------------
        wall_error = left_dist - DESIRED_WALL_DIST
        w = max(-MAX_ANG_SPEED,
                min(MAX_ANG_SPEED, Kp_wall * wall_error))

        # -------------------------------------------------
        # Forward speed – stop if the front is still unsafe
        # -------------------------------------------------
        if front_dist < FRONT_SAFE:
            v = 0.0
        else:
            v = MIN_SPEED

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
