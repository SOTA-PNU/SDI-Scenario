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
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def controller(t, pose, env):
    """
    Two‑stage navigation with a tiny avoidance sub‑state.

    Stage 0 – climb northwards to a safe “mid‑field” line (y = 5 m).  This
    bypasses the dense cluster of boxes around (2 – 4 m, 3 – 5 m).

    Stage 1 – head straight for the final goal while still applying a simple
    potential‑field repulsion from any obstacle that gets too close.

    If an obstacle becomes critically close (CRITICAL_DIST) the controller
    enters an avoidance mode for a short, fixed interval, rotating away from
    the obstacle before resuming the current stage.
    """
    # ------------------------------------------------------------------
    # Tunable parameters (chosen for safety + simplicity)
    MAX_SPEED = 0.8            # m/s – modest speed reduces collision risk
    MAX_TURN = 1.0             # rad/s
    KP_ANGLE = 2.0             # proportional gain for heading error
    ALIGN_TOL = 0.05           # rad – when we consider ourselves aligned
    REPULSE_RANGE = 2.5        # meters – distance that creates repulsion
    K_REPULSE = 1.2            # strength of repulsive component
    SAFE_STOP_DIST = 0.6       # m – stop forward motion if nearer
    CRITICAL_DIST = 0.35       # m – distance that triggers avoidance mode
    AVOID_TIME = 1.0           # seconds to stay in avoidance mode
    MID_Y = 5.0                # y‑coordinate of the intermediate waypoint
    # ------------------------------------------------------------------

    # --------------------------------------------------------------
    # Persistent state (stored on the function object)
    if not hasattr(controller, "stage"):
        controller.stage = 0          # 0 = climb north, 1 = go to final goal
        controller.mode = "go"        # "go" or "avoid"
        controller.avoid_timer = 0.0
        controller.turn_dir = 0.0
    # --------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----- termination ----------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    yaw_err_to_goal = _wrap_angle(goal_yaw - yaw)

    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(yaw_err_to_goal) <= MISSION["yaw_tolerance_rad"]):
        return 0.0, 0.0, True

    # ----- laser scan -------------------------------------------------------
    scans = env.raycast_scan()   # [(bearing, distance), ...]

    # Find the closest obstacle (and its bearing)
    min_dist = 6.0
    min_bearing = 0.0
    for bearing, distance in scans:
        if distance < min_dist:
            min_dist = distance
            min_bearing = bearing

    # ----- critical obstacle → switch to avoidance mode ----------------------
    if min_dist < CRITICAL_DIST and controller.mode != "avoid":
        controller.mode = "avoid"
        # Turn *away* from the obstacle: opposite sign of bearing
        controller.turn_dir = -1.0 if min_bearing > 0 else 1.0
        controller.avoid_timer = 0.0

    # ----- avoidance mode ----------------------------------------------------
    if controller.mode == "avoid":
        controller.avoid_timer += 1.0 / 60.0
        w = controller.turn_dir * MAX_TURN
        v = 0.0
        # Leave avoidance once enough time has passed and the front looks safer
        if (controller.avoid_timer >= AVOID_TIME) and (min_dist > SAFE_STOP_DIST):
            controller.mode = "go"
        return v, w, False

    # ----------------------------------------------------------------------
    # Determine current navigation target (stage handling)
    if controller.stage == 0:
        # Intermediate waypoint directly north of the start position
        target_x, target_y = x, MID_Y
        if abs(y - MID_Y) < 0.5:          # reached intermediate line
            controller.stage = 1
    else:
        target_x, target_y = goal_x, goal_y

    # ----- attractive unit vector toward the chosen target -------------------
    tx = target_x - x
    ty = target_y - y
    dist_to_target = math.hypot(tx, ty)
    if dist_to_target > 0:
        att_x = tx / dist_to_target
        att_y = ty / dist_to_target
    else:
        att_x = att_y = 0.0

    # ----- repulsive vectors from obstacles inside REPULSE_RANGE ------------
    rep_x, rep_y = 0.0, 0.0
    for bearing, distance in scans:
        if distance >= 6.0:               # no hit
            continue
        if distance > REPULSE_RANGE:
            continue

        beam_dir = yaw + bearing
        obs_dir_x = math.cos(beam_dir)
        obs_dir_y = math.sin(beam_dir)

        # Repulsion magnitude grows sharply when distance shrinks
        mag = K_REPULSE * (1.0 / distance - 1.0 / REPULSE_RANGE)
        if mag < 0:
            mag = 0.0

        rep_x += -obs_dir_x * mag
        rep_y += -obs_dir_y * mag

    # ----- combine attractive and repulsive vectors -------------------------
    comb_x = att_x + rep_x
    comb_y = att_y + rep_y
    comb_norm = math.hypot(comb_x, comb_y)
    if comb_norm > 0:
        comb_x /= comb_norm
        comb_y /= comb_norm
    else:
        comb_x, comb_y = att_x, att_y   # fallback to pure attraction

    # Desired heading from combined vector
    desired_heading = math.atan2(comb_y, comb_x)
    angle_error = _wrap_angle(desired_heading - yaw)

    # ----- yaw rate command -------------------------------------------------
    w = max(-MAX_TURN, min(MAX_TURN, KP_ANGLE * angle_error))

    # ----- forward speed command --------------------------------------------
    # Reduce speed when a sharp turn is required
    speed_factor = max(0.0, math.cos(angle_error))
    v = MAX_SPEED * speed_factor

    # Additional safety: stop if something is too close in a narrow front cone
    FRONT_ANGLE = math.radians(15)
    for bearing, distance in scans:
        if abs(bearing) <= FRONT_ANGLE and distance < SAFE_STOP_DIST:
            v = 0.0
            break

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
