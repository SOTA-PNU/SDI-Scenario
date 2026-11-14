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
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple waypoint‑follow controller with a tiny reactive layer.

    Waypoints (relative to the start):
        0) (start_x,   -2.0)   # go north a little, clear the first box
        1) (   8.0,   -2.0)   # go east across the arena
        2) (   8.0,  16.5)   # go north to the goal (final alignment handled separately)

    The controller keeps an internal state (a dict stored on the function
    object) so that it knows which waypoint it is currently heading for.
    A very small reactive check stops the robot and turns it away if an
    obstacle is detected within `obs_stop_dist` directly ahead.
    """
    # ------------------------------------------------------------------ #
    # initialise persistent state on first call
    if not hasattr(controller, "state"):
        controller.state = {
            "waypoints": [
                (START[0], -2.0),   # northward leg
                (8.0, -2.0),        # eastward leg
                (8.0, 16.5),        # final northward leg
                "DONE",             # sentinel
            ],
            "idx": 0,               # current waypoint index
        }

    state = controller.state
    waypoints = state["waypoints"]
    idx = state["idx"]

    # ------------------------------------------------------------------ #
    # unpack current pose
    x, y, yaw = pose

    # ------------------------------------------------------------------ #
    # termination: we have reached the final waypoint and are correctly oriented
    goal_x, goal_y, goal_yaw = GOAL
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    if idx >= len(waypoints) - 1:   # last real waypoint reached
        # check final position & yaw
        if math.hypot(x - goal_x, y - goal_y) <= pos_tol and \
           abs(_wrap_angle(yaw - goal_yaw)) <= yaw_tol:
            return 0.0, 0.0, True
        # otherwise treat the goal itself as the current target
        target_x, target_y = goal_x, goal_y
    else:
        target = waypoints[idx]
        # safety: if the sentinel "DONE" somehow appears, just stop
        if target == "DONE":
            return 0.0, 0.0, True
        target_x, target_y = target

    # ------------------------------------------------------------------ #
    # check if we have arrived at the current waypoint (position tolerance only)
    if math.hypot(x - target_x, y - target_y) <= pos_tol:
        # advance to next waypoint
        state["idx"] = idx + 1
        # recompute target after advancing
        return controller(t, pose, env)   # recursive call with updated state

    # ------------------------------------------------------------------ #
    # compute desired heading toward the current target
    dx = target_x - x
    dy = target_y - y
    desired_yaw = math.atan2(dy, dx)

    # ------------------------------------------------------------------ #
    # reactive obstacle avoidance (very short range)
    scan = env.raycast_scan()
    obs_stop_dist = 0.55          # meters – stop and turn if something this close
    turn_away = 0.0
    for bearing, dist in scan:
        if dist < obs_stop_dist:
            # obstacle on the left (bearing > 0) → turn right, and vice‑versa
            turn_dir = -1.0 if bearing > 0 else 1.0
            # influence grows as distance shrinks
            influence = (obs_stop_dist - dist) / obs_stop_dist
            turn_away += turn_dir * influence * 1.5   # up to ±1.5 rad/s
    # ------------------------------------------------------------------ #
    # angular control (P controller) + avoidance term
    kp_ang = 2.0
    max_ang = 1.5
    yaw_error = _wrap_angle(desired_yaw - yaw)
    angular = kp_ang * yaw_error
    angular = max(-max_ang, min(max_ang, angular))
    angular += turn_away
    angular = max(-max_ang, min(max_ang, angular))

    # ------------------------------------------------------------------ #
    # speed control
    base_speed = 1.6                     # m/s – reasonably quick
    # slower when we need to turn sharply
    speed = base_speed * (1.0 - min(1.0, abs(yaw_error) / (math.pi / 2)))
    # if we are actively avoiding an obstacle, cut speed proportionally
    if turn_away != 0.0:
        speed *= 0.3                     # keep a low crawl while turning away

    # slow down when we are close to the target (for smoother switching)
    dist_to_target = math.hypot(dx, dy)
    if dist_to_target < 1.0:
        speed *= dist_to_target / 1.0

    speed = max(0.0, speed)              # safety clamp

    # ------------------------------------------------------------------ #
    return speed, angular, False


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
