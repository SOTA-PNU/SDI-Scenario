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

# Global variables for simple stuck‑detection (optional)
_last_pose = None
_last_t = None
_stuck_counter = 0


def controller(t, pose, env):
    """
    Reactive potential‑field controller.

    • Attractive component: steer toward the goal.
    • Repulsive component: steer away from any obstacle detected within a
      safety radius (using the 180° raycast scan).
    • Speed is reduced when turning sharply or when something is directly ahead.
    • When the robot is inside the goal tolerance it aligns its yaw and stops.
    """
    # ----------------------------------------------------------------------
    # Helper utilities
    # ----------------------------------------------------------------------
    def normalize_angle(a):
        """Wrap angle to [-π, π]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ----------------------------------------------------------------------
    # Unpack current pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Vector to goal (world frame)
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Angle to goal expressed in robot‑relative frame
    goal_dir_world = math.atan2(dy, dx)
    angle_to_goal = normalize_angle(goal_dir_world - yaw)  # [-π,π]

    # ----------------------------------------------------------------------
    # Termination / goal‑alignment logic
    # ----------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if dist_to_goal <= pos_tol:
        yaw_error = normalize_angle(yaw - goal_yaw)
        if abs(yaw_error) <= yaw_tol:
            # Goal reached and correctly oriented – stop.
            return 0.0, 0.0, True
        # Within position tolerance but yaw off – rotate in place.
        KP_YAW = 1.5
        w_align = KP_YAW * yaw_error
        w_align = clamp(w_align, -1.0, 1.0)
        return 0.0, w_align, False

    # ----------------------------------------------------------------------
    # Parameters (tuned for the arena)
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.6          # m/s – safe cruising speed
    MAX_TURN = 1.0           # rad/s – hard limit on yaw rate
    KP_ANGLE = 2.0           # proportional gain for heading control
    SAFE_DIST = 1.5          # m – distance at which obstacles start repelling
    REPULSION_GAIN = 2.5     # strength of repulsive influence
    FRONT_BLOCK_DIST = 0.6   # m – stop forward motion if something very close ahead

    # ----------------------------------------------------------------------
    # Scan processing – build repulsive vector in robot frame
    # ----------------------------------------------------------------------
    # Attractive unit vector (robot forward = +x)
    att_x = math.cos(angle_to_goal)
    att_y = math.sin(angle_to_goal)

    rep_x = 0.0
    rep_y = 0.0
    front_dist = None  # nearest obstacle roughly in front

    try:
        scans = env.raycast_scan()  # default: 61 beams, 180°, max_range=6.0
    except Exception:
        scans = []  # safety fallback – treat as no obstacles

    for bearing, distance in scans:
        # distance == max_range (6.0) means no hit → ignore
        if distance >= 6.0:
            continue

        # Track the nearest obstacle directly ahead (|bearing| < ~0.2 rad)
        if abs(bearing) < 0.2:
            if front_dist is None or distance < front_dist:
                front_dist = distance

        # Only consider obstacles inside the safety radius for repulsion
        if distance < SAFE_DIST:
            # Repulsion magnitude grows linearly as we get closer
            rep_mag = (SAFE_DIST - distance) / SAFE_DIST
            # Unit vector pointing *away* from the obstacle (opposite of its bearing)
            rep_dir_x = -math.cos(bearing)
            rep_dir_y = -math.sin(bearing)
            rep_x += rep_dir_x * rep_mag * REPULSION_GAIN
            rep_y += rep_dir_y * rep_mag * REPULSION_GAIN

    # ----------------------------------------------------------------------
    # Combine attractive and repulsive vectors
    # ----------------------------------------------------------------------
    total_x = att_x + rep_x
    total_y = att_y + rep_y

    # Desired heading in robot frame (angle of the combined vector)
    desired_heading = math.atan2(total_y, total_x)  # already robot‑relative
    w = KP_ANGLE * normalize_angle(desired_heading)
    w = clamp(w, -MAX_TURN, MAX_TURN)

    # ----------------------------------------------------------------------
    # Speed selection
    # ----------------------------------------------------------------------
    # If something is too close directly ahead, halt forward motion.
    if front_dist is not None and front_dist < FRONT_BLOCK_DIST:
        v = 0.0
    else:
        # Base speed scaled by how much we need to turn (slower when turning hard)
        turn_factor = max(0.0, 1.0 - abs(w) / MAX_TURN)   # 0..1
        # Also slow down when we are close to the goal (to aid precise stopping)
        if dist_to_goal < 2.0:
            goal_factor = dist_to_goal / 2.0
        else:
            goal_factor = 1.0
        v = MAX_SPEED * turn_factor * goal_factor

    # Ensure forward speed is non‑negative
    v = max(0.0, v)

    # ----------------------------------------------------------------------
    # (Optional) Very simple stuck detection – if robot hasn't moved for a while,
    # inject a small spin to get out of local minima.
    # ----------------------------------------------------------------------
    global _last_pose, _last_t, _stuck_counter
    if _last_pose is not None and _last_t is not None:
        dx_last = x - _last_pose[0]
        dy_last = y - _last_pose[1]
        moved = math.hypot(dx_last, dy_last)
        # If robot moved less than 2 cm over the last 2 s, consider it stuck.
        if t - _last_t >= 2.0:
            if moved < 0.02:
                _stuck_counter += 1
            else:
                _stuck_counter = 0
    _last_pose = (x, y)
    _last_t = t

    if _stuck_counter >= 3:  # after ~6 s of being stuck, spin in place
        v = 0.0
        w = MAX_TURN  # turn right to try to escape
        _stuck_counter = 0  # reset after injecting spin

    # ----------------------------------------------------------------------
    # Return command
    # ----------------------------------------------------------------------
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
