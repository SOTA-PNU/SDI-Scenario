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
  max_time_to_goal  first entry into the goal disc at t <= MAX_TTG (12 s)
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
MAX_TTG = 12.0
OBSTACLE = {"x": -6.2, "y": 3.5, "height": 1.0, "width": 0.8, "depth": 0.6}

# ===========================================================================
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

CRUISE_V = 1.0     # m/s cruise: raised so the mission fits the time budget
STOP_DIST = 0.30   # m, well inside the 0.75 m position tolerance
K_HEADING = 1.8    # P gain: bearing error -> yaw rate
W_MAX = 1.5        # rad/s yaw rate limit: snappier turns save seconds
BEARING_GATE = 0.5  # rad: rotate in place while badly misaligned
K_ALIGN = 2.0      # P gain for the final in-place alignment
ALIGN_W_MAX = 0.8  # rad/s limit during final alignment
YAW_DONE = 0.08    # rad, well inside the 0.26 rad yaw tolerance
# --- L2 additions: obstacle avoidance on the planar ray scan ---
BLOCK_RANGE = 2.2   # m: a hit nearer than this blocks bearings around it
SIDE_MARGIN = 0.55  # m: lateral clearance to keep from any hit point
THREAT_DIST = 1.4   # m: front-cone hit closer than this -> creep speed
FRONT_CONE = 0.6    # rad: half-angle of the "in my way" cone
AVOID_V = 0.5       # m/s while maneuvering: budget-conscious but careful


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def controller(t, pose, env):
    """L1 point controller + bearing-inflation avoidance (the L2 skill).

    Every step, cast a planar ray fan.  Each hit nearer than BLOCK_RANGE
    blocks the bearings around it, inflated by the angle that SIDE_MARGIN of
    lateral clearance subtends at the hit's range: near obstacles block wide
    wedges, far ones narrow slivers.  While the goal direction is clear the
    L1 controller runs unchanged; when it is blocked, steer toward the clear
    bearing nearest the goal direction.  Passing an obstacle keeps its flank
    inside a blocked wedge, which postpones the turn back toward the goal
    until the robot is genuinely clear of it - no explicit state needed.
    """
    x, y, yaw = pose
    gx, gy, gyaw = GOAL
    d = math.hypot(gx - x, gy - y)

    if d <= STOP_DIST:
        # inside the goal disc: align to the commanded arrival yaw, finish
        yerr = _wrap(gyaw - yaw)
        if abs(yerr) > YAW_DONE:
            return 0.0, _clamp(K_ALIGN * yerr, ALIGN_W_MAX), False
        return 0.0, 0.0, True

    err = _wrap(math.atan2(gy - y, gx - x) - yaw)
    scan = env.raycast_scan()
    # hits at or beyond the goal distance cannot be in the way
    blocks = [(b - math.atan2(SIDE_MARGIN, r), b + math.atan2(SIDE_MARGIN, r))
              for b, r in scan if r < min(BLOCK_RANGE, d)]

    def blocked(rb):
        return any(lo <= rb <= hi for lo, hi in blocks)

    front = [r for b, r in scan if abs(b) <= FRONT_CONE]
    threat = min(front) if front else 99.0

    if not blocked(err) and threat >= THREAT_DIST:
        # goal direction clear: L1 behaviour unchanged
        w = _clamp(K_HEADING * err, W_MAX)
        v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
        return v, w, False

    # goal direction blocked: steer to the clear bearing nearest it
    cands = [b for b, r in scan if r >= BLOCK_RANGE and not blocked(b)]
    if not cands:
        return 0.0, W_MAX * 0.6, False  # boxed in: rotate in place and rescan
    best = min(cands, key=lambda b: abs(_wrap(b - err)))
    w = _clamp(K_HEADING * best, W_MAX)
    v = AVOID_V if threat < THREAT_DIST or abs(best) > BEARING_GATE else 0.8 * CRUISE_V
    return v, w, False

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
