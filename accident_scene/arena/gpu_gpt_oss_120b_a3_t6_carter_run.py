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
    Deterministic potential‑field controller.

    * Attractive component points toward the goal.
    * Repulsive component is summed from every ray‑scan hit that lies
      inside ``REPEL_RANGE`` (obstacles farther away are ignored).
    * The resulting vector (attractive + repulsive) defines the desired
      heading.  The robot turns proportionally to the heading error and
      moves forward with a speed that is reduced when the repulsive influence
      is strong.
    * When the robot is inside the position‑ and yaw‑tolerances it stops and
      signals success.
    """
    # ----------------------------------------------------------------------
    # 1) unpack pose and goal
    # ----------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ----------------------------------------------------------------------
    # 2) helpers
    # ----------------------------------------------------------------------
    def wrap_angle(a):
        """Wrap angle to [-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    # ----------------------------------------------------------------------
    # 3) geometric relationship to the goal (robot‑centric)
    # ----------------------------------------------------------------------
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # rotate goal vector into robot frame (heading = 0)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    goal_local_x =  cos_y * dx + sin_y * dy
    goal_local_y = -sin_y * dx + cos_y * dy

    # unit attractive direction (if we are exactly on the goal the vector is zero)
    if goal_local_x == 0 and goal_local_y == 0:
        attr_x, attr_y = 0.0, 0.0
    else:
        norm = math.hypot(goal_local_x, goal_local_y)
        attr_x = goal_local_x / norm
        attr_y = goal_local_y / norm

    # ----------------------------------------------------------------------
    # 4) parameters
    # ----------------------------------------------------------------------
    MAX_SPEED = 0.8          # m/s – safe moderate speed
    MAX_ANGULAR = 1.0        # rad/s
    KP_ANGLE = 2.0           # gain for heading error
    REPEL_RANGE = 2.0        # meters – only obstacles closer than this matter
    K_REPEL = 0.6            # repulsion gain
    SAFETY_STOP = 0.25       # stop forward motion if something this close ahead

    # ----------------------------------------------------------------------
    # 5) ray‑scan → repulsive field (robot frame)
    # ----------------------------------------------------------------------
    scans = env.raycast_scan()                     # [(bearing, distance), ...]
    rep_x, rep_y = 0.0, 0.0                         # accumulated repulsive vector

    # consider only valid hits (distance < max_range == 6.0)
    for bearing, distance in scans:
        if distance >= REPEL_RANGE or distance == 6.0:
            continue                               # ignore far or missed beams
        # classic inverse‑square repulsion (clipped to avoid division by zero)
        influence = K_REPEL * (1.0 / distance - 1.0 / REPEL_RANGE) / (distance ** 2)
        # unit vector pointing **away** from the obstacle (opposite bearing)
        ux = -math.cos(bearing)
        uy = -math.sin(bearing)
        rep_x += influence * ux
        rep_y += influence * uy

    # ----------------------------------------------------------------------
    # 6) combine attractive and repulsive vectors
    # ----------------------------------------------------------------------
    # weight attractive component so that it dominates when no close obstacle
    ATTR_WEIGHT = 1.0
    comb_x = ATTR_WEIGHT * attr_x + rep_x
    comb_y = ATTR_WEIGHT * attr_y + rep_y

    # if the combined vector is (near) zero – e.g. trapped – fall back to
    # turning toward the goal direction.
    if math.hypot(comb_x, comb_y) < 1e-3:
        comb_x, comb_y = attr_x, attr_y

    # desired heading in robot frame (0 = straight ahead)
    desired_heading = math.atan2(comb_y, comb_x)
    heading_error = wrap_angle(desired_heading)      # already robot‑centric

    # ----------------------------------------------------------------------
    # 7) command generation
    # ----------------------------------------------------------------------
    # angular velocity – proportional to heading error, clipped
    w = max(-MAX_ANGULAR, min(MAX_ANGULAR, KP_ANGLE * heading_error))

    # forward speed – reduced when we need to turn sharply
    turn_factor = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
    v = MAX_SPEED * turn_factor

    # additional slowdown when we are close to the goal (smooth stop)
    if dist_to_goal < 1.5:
        v *= dist_to_goal / 1.5

    # emergency stop if something is extremely close directly ahead
    front_beams = [d for b, d in scans if abs(b) <= math.radians(30)]
    min_front = min(front_beams) if front_beams else 6.0
    if min_front < SAFETY_STOP:
        v = 0.0

    # ----------------------------------------------------------------------
    # 8) goal check
    # ----------------------------------------------------------------------
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
