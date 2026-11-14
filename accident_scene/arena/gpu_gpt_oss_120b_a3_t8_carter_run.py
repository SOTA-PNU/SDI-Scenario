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
    Simple deterministic bug‑type controller.

    * **Mode “go**”   – drive toward the goal while the forward sector is clear.
    * **Mode “turn**” – rotate in place (no forward motion) until the forward
      sector becomes safe again, then resume “go”.
    * The controller never moves forward when an obstacle is closer than
      ``SAFE_DIST`` directly ahead, which prevents collisions.
    * Success is reported when both position and yaw are inside the tolerances
      defined in ``MISSION``.
    """
    # ----------------------------------------------------------------------
    # 1) persistent state (survives across calls)
    # ----------------------------------------------------------------------
    if not hasattr(controller, "mode"):
        controller.mode = "go"          # "go" or "turn"
        controller.turn_dir = 0         # -1 = turn right, +1 = turn left
        controller.turn_start = 0.0     # time we entered turn mode

    # ----------------------------------------------------------------------
    # 2) unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # 3) helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    # ----------------------------------------------------------------------
    # 4) geometry to goal (world frame)
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    angle_error = wrap_angle(angle_to_goal - yaw)   # 0 = already facing goal

    # ----------------------------------------------------------------------
    # 5) parameters (tuned for safety & speed)
    # ----------------------------------------------------------------------
    MAX_SPEED = 1.0          # m/s – fast enough to finish well under 120 s
    MAX_ANGULAR = 1.0        # rad/s
    KP_ANGLE = 2.0           # proportional gain for heading error
    SAFE_DIST = 0.7          # meters – must be clear ahead to stay in “go”
    FRONT_CONE = math.radians(30)   # forward sector used for safety test
    SIDE_ANGLE = math.radians(90)   # side sector used for picking turn direction
    TURN_TIMEOUT = 8.0       # seconds – after this we flip turn direction

    # ----------------------------------------------------------------------
    # 6) ray‑scan processing
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()                     # [(bearing, distance), ...]
    # keep only hits that are not the max_range sentinel (6.0)
    hits = [(b, d) for (b, d) in scans if d < 6.0]

    # distance of the closest obstacle inside the forward cone
    front_hits = [d for b, d in hits if abs(b) <= FRONT_CONE]
    min_front = min(front_hits) if front_hits else 6.0

    # ----------------------------------------------------------------------
    # 7) mode handling
    # ----------------------------------------------------------------------
    if controller.mode == "go":
        # --------------------------------------------------------------
        # 7a) safety test – if something too close ahead, switch to turn mode
        # --------------------------------------------------------------
        if min_front < SAFE_DIST:
            # decide which side has more clearance (average distance of side beams)
            left = [d for b, d in hits if 0 < b <= SIDE_ANGLE]
            right = [d for b, d in hits if -SIDE_ANGLE <= b < 0]
            avg_left = sum(left) / len(left) if left else 6.0
            avg_right = sum(right) / len(right) if right else 6.0
            controller.turn_dir = 1 if avg_left > avg_right else -1
            controller.mode = "turn"
            controller.turn_start = t
        else:
            # ----------------------------------------------------------
            # 7b) normal forward motion toward the goal
            # ----------------------------------------------------------
            # angular velocity – proportional to heading error, clipped
            w = max(-MAX_ANGULAR,
                    min(MAX_ANGULAR, KP_ANGLE * angle_error))

            # forward speed – reduced when we need to turn sharply
            # use cosine of the error so speed goes to zero at ±90°
            speed_factor = max(0.0, math.cos(angle_error))
            v = MAX_SPEED * speed_factor

            # additional slowdown when we are close to the goal
            if dist_to_goal < 2.0:
                v *= dist_to_goal / 2.0

            # enforce a tiny forward motion when we are far from goal
            if v < 0.15 and dist_to_goal > 0.5:
                v = 0.15

            # ----------------------------------------------------------
            # goal check (also performed in turn mode)
            # ----------------------------------------------------------
            done = False
            if (dist_to_goal <= MISSION["position_tolerance_m"] and
                    abs(wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
                v = 0.0
                w = 0.0
                done = True

            return v, w, done

    # ----------------------------------------------------------------------
    # 8) turn mode – rotate in place until front clears or timeout expires
    # ----------------------------------------------------------------------
    w = controller.turn_dir * MAX_ANGULAR * 0.8   # rotate, a bit slower than max
    v = 0.0                                      # never move forward while turning

    # if the front becomes sufficiently clear, go back to forward mode
    if min_front > SAFE_DIST * 1.2:
        controller.mode = "go"
    # if we have been turning for a long time, flip direction to avoid dead‑lock
    elif t - controller.turn_start > TURN_TIMEOUT:
        controller.turn_dir *= -1
        controller.turn_start = t

    # --------------------------------------------------------------
    # 9) goal check while turning (still required)
    # --------------------------------------------------------------
    done = False
    if (dist_to_goal <= MISSION["position_tolerance_m"] and
            abs(wrap_angle(goal_yaw - yaw)) <= MISSION["yaw_tolerance_rad"]):
        v = 0.0
        w = 0.0
        done = True

    return v, w, done


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
