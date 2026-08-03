"""LEVEL 3 mission runner - time-bounded avoidance with a near-goal obstacle.

MISSION (scenarios/nova_carter_warehouse_level3.yaml):
  Same start and goal as level 2: spawn at world (-6.0, -1.0, yaw=pi), goal
  (-6.0, 5.0) facing +y.  The 1.0 m tall box now sits at (-6.2, 3.5) - only
  1.5 m short of the goal, so the detour ends right where the arrival pose
  must be produced.  AND the mission is time-bounded: reaching the goal disc
  later than MAX_TTG seconds of sim time fails, even though the mission
  timeout (120 s) would still accept it.  Be quick AND clean.

PASS CRITERIA (evaluated by the fixed harness, not by this file):
  reached_goal      final distance <= 0.75 m AND |yaw error| <= 0.26 rad
  no_collision      zero chassis contact events (box NOT excluded)
  max_time_to_goal  first entry into the goal disc at t <= MAX_TTG (45 s)
  timeout_s         120 s of sim time

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

The ONLY part meant to be modified to solve the level is the block marked
[EDIT REGION].  Everything outside it is fixed plumbing, identical between
base_carter_run.py and solution_carter_run.py.
"""

import math  # noqa: F401  (available to the controller)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from carter_env import Harness  # noqa: E402

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | "solution"
OUT = str(Path(__file__).resolve().parent / "results")

# scenario givens (from the level YAML - not part of the answer)
GOAL = (-6.0, 5.0, 1.5708)
TIMEOUT_S = 120.0
POS_TOL = 0.75
YAW_TOL = 0.26
MAX_TTG = 45.0
OBSTACLE = {"x": -6.2, "y": 3.5, "height": 1.0, "width": 0.8, "depth": 0.6}

# ===========================================================================
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

CRUISE_V = 0.5     # m/s cruise speed toward the goal
STOP_DIST = 0.30   # m, well inside the 0.75 m position tolerance
K_HEADING = 1.8    # P gain: bearing error -> yaw rate
W_MAX = 1.2        # rad/s yaw rate limit
BEARING_GATE = 0.5  # rad: rotate in place while badly misaligned
K_ALIGN = 2.0      # P gain for the final in-place alignment
ALIGN_W_MAX = 0.8  # rad/s limit during final alignment
YAW_DONE = 0.08    # rad, well inside the 0.26 rad yaw tolerance
# --- L2 additions: reactive avoidance on the planar ray scan ---
THREAT_DIST = 1.4   # m: something this close in the front cone -> avoid
FRONT_CONE = 0.6    # rad: half-angle of the "in my way" cone
AVOID_V = 0.25      # m/s while steering around an obstacle
CLEAR_DIST = 2.2    # m: a candidate direction must see at least this far
SIDE_MARGIN = 0.45  # m: lateral clearance (robot half-width + margin)


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """L1 point controller + reactive gap-follow avoidance (the L2 skill).

    Every step, cast a planar ray fan.  If nothing sits within THREAT_DIST
    of the front cone (or the goal is nearer than the threat), behave exactly
    like the L1 controller.  Otherwise steer at reduced speed toward the
    clear bearing closest to the goal bearing, where "clear" means the ray
    and its neighbours spanning the robot's width all see >= CLEAR_DIST.
    """
    x, y, yaw = pose
    gx, gy, gyaw = GOAL
    d = math.hypot(gx - x, gy - y)

    if d <= STOP_DIST:
        # inside the goal disc: align to the commanded arrival yaw, finish
        yerr = _wrap(gyaw - yaw)
        if abs(yerr) > YAW_DONE:
            return 0.0, max(-ALIGN_W_MAX, min(ALIGN_W_MAX, K_ALIGN * yerr)), False
        return 0.0, 0.0, True

    bearing = math.atan2(gy - y, gx - x)
    err = _wrap(bearing - yaw)
    scan = env.raycast_scan()
    front = [dist for b, dist in scan if abs(b) <= FRONT_CONE]
    threat = min(front) if front else 99.0

    if threat >= THREAT_DIST or threat > d:
        # lane clear (or goal nearer than the obstacle): L1 behaviour
        w = max(-W_MAX, min(W_MAX, K_HEADING * err))
        v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
        return v, w, False

    # avoidance: among sufficiently clear bearings, pick the one whose world
    # direction is closest to the goal bearing; require the neighbour rays
    # covering the robot's width at CLEAR_DIST to be clear too
    half = math.atan2(SIDE_MARGIN, CLEAR_DIST)
    best_b, best_score = None, None
    for b, dist in scan:
        if dist < CLEAR_DIST:
            continue
        nb = [dd for bb, dd in scan if abs(bb - b) <= half]
        if not nb or min(nb) < CLEAR_DIST * 0.8:
            continue
        score = abs(_wrap((yaw + b) - bearing))
        if best_score is None or score < best_score:
            best_score, best_b = score, b
    if best_b is None:
        return 0.0, W_MAX * 0.6, False  # boxed in: rotate in place and rescan
    w = max(-W_MAX, min(W_MAX, K_HEADING * best_b))
    return AVOID_V, w, False


# ========================= [END EDIT REGION] ===============================

h = Harness(
    level="level3",
    variant=VARIANT,
    goal_xy_yaw=GOAL,
    timeout_s=TIMEOUT_S,
    position_tolerance_m=POS_TOL,
    yaw_tolerance_rad=YAW_TOL,
    max_time_to_goal_s=MAX_TTG,
    check_collision=True,
    obstacle=OBSTACLE,
    out_dir=OUT,
).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
