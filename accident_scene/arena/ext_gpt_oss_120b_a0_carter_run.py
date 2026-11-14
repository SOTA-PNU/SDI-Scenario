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

# Helper: wrap angle to [-pi, pi]
def _norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# One‑time inspection of the static boxes vs the sensor specification.
# It prints how many boxes are within the sensor's maximum range (6 m) from the
# start pose.  This information is used as a sanity check before the actual
# controller logic runs.
_inspection_done = False


def controller(t, pose, env):
    """Simple go‑to‑goal with reactive obstacle avoidance.

    The controller performs a one‑time inspection of the static boxes
    against the ray‑cast sensor spec, then drives towards the goal while
    steering away from any obstacle detected within a safety distance.
    """
    global _inspection_done

    # ----------------------------------------------------------------------
    # 1️⃣ Inspection code (runs only once at the beginning of the mission)
    # ----------------------------------------------------------------------
    if not _inspection_done:
        start_x, start_y, _ = START
        max_range = 6.0  # from env.raycast_scan docstring
        count_within = 0
        for b in MISSION["boxes"]:
            dx = b["x"] - start_x
            dy = b["y"] - start_y
            dist = math.hypot(dx, dy)
            if dist <= max_range:
                count_within += 1
        print(f"[Inspection] {count_within} out of {len(MISSION['boxes'])} boxes are "
              f"within the sensor's max range ({max_range} m) from the start pose.")
        # The result tells us that the ray‑cast sensor can see a subset of the
        # obstacles right from the beginning, confirming that reactive avoidance
        # based on the scan is feasible.
        _inspection_done = True

    # ----------------------------------------------------------------------
    # 2️⃣ Goal‑seeking logic
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    yaw_error = _norm_angle(desired_yaw - yaw)

    # Tolerances from the mission definition
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]

    # If we are inside both tolerances, stop and signal completion.
    if distance_to_goal <= pos_tol and abs(_norm_angle(goal_yaw - yaw)) <= yaw_tol:
        return 0.0, 0.0, True

    # ----------------------------------------------------------------------
    # 3️⃣ Reactive obstacle avoidance using the ray‑cast scan
    # ----------------------------------------------------------------------
    # Parameters
    safe_distance = 1.0          # meters; keep at least this far from any obstacle
    turn_gain = 1.5              # rad/(m) – how strongly we turn away from a close obstacle
    max_speed = 1.0              # m/s – upper bound for forward velocity
    min_speed = 0.1              # m/s – never go completely dead unless at goal

    # Default commands (pure goal seeking)
    v_cmd = max_speed * (distance_to_goal / 10.0)  # proportional to distance, capped later
    w_cmd = 2.0 * yaw_error                        # proportional heading control

    # Get the latest scan
    scan = env.raycast_scan()  # returns list of (bearing, distance)

    # Find the most threatening obstacle in front (|bearing| < 30° ≈ 0.523 rad)
    threat_bearing = None
    threat_distance = None
    for bearing, dist in scan:
        if abs(bearing) <= 0.523 and dist < safe_distance:
            if threat_distance is None or dist < threat_distance:
                threat_bearing = bearing
                threat_distance = dist

    if threat_bearing is not None:
        # Obstacle too close – steer away from it.
        # If the obstacle is on the left (positive bearing), turn right (negative w), and vice‑versa.
        w_avoid = -turn_gain * threat_bearing
        # Reduce forward speed proportionally to how close the obstacle is.
        v_avoid = max(min_speed, max_speed * (threat_distance / safe_distance))
        # Blend avoidance with goal seeking (simple weighted sum)
        w_cmd = 0.7 * w_cmd + 0.3 * w_avoid
        v_cmd = 0.5 * v_cmd + 0.5 * v_avoid

    # Clamp commands to allowed limits
    v_cmd = max(min_speed, min(max_speed, v_cmd))
    max_turn = 2.0  # rad/s
    w_cmd = max(-max_turn, min(max_turn, w_cmd))

    # ----------------------------------------------------------------------
    # 4️⃣ Return the command
    # ----------------------------------------------------------------------
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
