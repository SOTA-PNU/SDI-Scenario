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
    """Reactive controller that drives toward the goal while avoiding obstacles.

    The controller runs at 60 Hz.  It uses only the forward‑facing 180° raycast
    (61 beams) to steer away from nearby obstacles and to approach the goal.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (feel free to tweak if needed)
    # ------------------------------------------------------------------
    MAX_SPEED = 1.0            # m/s, nominal forward speed
    SAFE_DIST = 4.0            # m, start reacting to obstacles within this range
    MIN_STOP_DIST = 0.3        # m, stop forward motion if an obstacle is this close
    FRONT_ANGLE = math.radians(30)   # rad, sector considered “directly ahead”
    K_YAW = 2.0                # proportional gain for heading to goal
    K_AVOID = 5.0              # gain for obstacle avoidance (larger = stronger turn)
    MAX_ANGULAR = 1.5          # rad/s, hard limit on yaw rate

    # ------------------------------------------------------------------
    # Goal information (global)
    # ------------------------------------------------------------------
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # Current robot state
    # ------------------------------------------------------------------
    x, y, yaw = pose

    # ------------------------------------------------------------------
    # Vector to goal
    # ------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)

    # Helper: wrap angle to [-π, π]
    def wrap_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    yaw_error = wrap_angle(desired_yaw - yaw)

    # ------------------------------------------------------------------
    # Goal‑region handling (stop and align)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if distance_to_goal <= pos_tol:
        # Inside the goal disc: only rotate to match the goal yaw.
        v_cmd = 0.0
        w_cmd = K_YAW * yaw_error
        # Clamp angular speed
        if w_cmd > MAX_ANGULAR:
            w_cmd = MAX_ANGULAR
        elif w_cmd < -MAX_ANGULAR:
            w_cmd = -MAX_ANGULAR
        # Finish when orientation is good enough.
        if abs(yaw_error) <= yaw_tol:
            return 0.0, 0.0, True
        else:
            return v_cmd, w_cmd, False

    # ------------------------------------------------------------------
    # Sensor processing
    # ------------------------------------------------------------------
    scan = env.raycast_scan()   # defaults: 61 beams, 180°, z=0.35, max_range=6.0

    # Track the closest obstacle in a narrow forward sector.
    min_front = float('inf')
    min_front_bearing = 0.0

    # Accumulate a simple repulsive term based on how close obstacles are.
    # For each beam with distance < SAFE_DIST we add a contribution whose
    # magnitude grows the closer the obstacle is, and whose sign is opposite
    # to the bearing (so obstacles on the left cause a right turn, and vice‑versa).
    avoid_sum = 0.0

    for bearing, dist in scan:
        # Front sector (used for speed scaling)
        if abs(bearing) <= FRONT_ANGLE and dist < min_front:
            min_front = dist
            min_front_bearing = bearing

        # Repulsive contribution
        if dist < SAFE_DIST:
            # The closer the obstacle, the larger the term.
            # (SAFE_DIST - dist) is positive for obstacles inside the safe radius.
            # Sign(bearing) tells us which side the obstacle is on.
            avoid_sum += (SAFE_DIST - dist) * (1.0 if bearing > 0.0 else -1.0)

    # If nothing was detected in the front sector, treat it as safe.
    if min_front == float('inf'):
        min_front = SAFE_DIST

    # ------------------------------------------------------------------
    # Compute avoidance angular velocity
    # ------------------------------------------------------------------
    # avoid_sum > 0  -> obstacles predominantly on the left -> turn right (negative w)
    # avoid_sum < 0  -> obstacles predominantly on the right -> turn left (positive w)
    w_avoid = -K_AVOID * (avoid_sum / SAFE_DIST)  # divide to keep magnitude reasonable

    # ------------------------------------------------------------------
    # Combine goal heading and avoidance
    # ------------------------------------------------------------------
    w_cmd = K_YAW * yaw_error + w_avoid

    # Clamp angular velocity.
    if w_cmd > MAX_ANGULAR:
        w_cmd = MAX_ANGULAR
    elif w_cmd < -MAX_ANGULAR:
        w_cmd = -MAX_ANGULAR

    # ------------------------------------------------------------------
    # Forward speed command (slow down when something is close ahead)
    # ------------------------------------------------------------------
    if min_front < MIN_STOP_DIST:
        # Too close – stop and let the turning take effect.
        v_cmd = 0.0
    elif min_front < SAFE_DIST:
        # Linear scaling between MIN_STOP_DIST and SAFE_DIST.
        v_cmd = MAX_SPEED * (min_front - MIN_STOP_DIST) / (SAFE_DIST - MIN_STOP_DIST)
        v_cmd = max(0.0, v_cmd)
    else:
        v_cmd = MAX_SPEED

    # Reduce speed when turning sharply (helps safety and maneuverability).
    turn_factor = max(0.1, 1.0 - abs(w_cmd) / MAX_ANGULAR)
    v_cmd *= turn_factor

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
