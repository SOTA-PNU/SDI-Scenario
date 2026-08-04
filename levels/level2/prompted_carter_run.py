"""LEVEL 2 mission runner - obstacle avoidance in the lane.

MISSION (scenarios/nova_carter_warehouse_level2.yaml):
  Same start and goal as level 1: spawn at world (-6.0, -1.0, yaw=pi), goal
  (-6.0, 5.0) facing +y.  BUT a 1.0 m tall box (0.8 x 0.6 m footprint) now
  blocks the lane at (-6.2, 2.0): it covers x in [-6.6, -5.8], most of the
  driving corridor.  It is tall enough for every onboard sensor to see.
  Perceive it, steer around it without touching it, and still arrive aligned.

PASS CRITERIA (evaluated by the fixed harness, not by this file):
  reached_goal   final distance <= 0.75 m AND |yaw error| <= 0.26 rad
  no_collision   zero chassis contact events (self + floor excluded;
                 the injected box is NOT excluded - touching it fails)
  timeout_s      120 s of sim time

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
OBSTACLE = {"x": -6.2, "y": 2.0, "height": 1.0, "width": 0.8, "depth": 0.6}

# ===========================================================================
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

V_CRUISE = 0.5     # m/s cruise toward the goal
D_STOP = 0.30      # m: end the drive phase well inside the 0.75 m disc
K_TURN = 2.0       # P gain: bearing error -> yaw rate
W_LIMIT = 1.2      # rad/s yaw-rate cap
TURN_ONLY = 0.45   # rad: bearing error above this -> rotate in place
K_FINAL = 1.5      # P gain for arrival-yaw alignment
W_FINAL = 0.7      # rad/s cap during the final alignment
YAW_OK = 0.1       # rad: finish threshold, inside the 0.26 rad tolerance
NEAR_HIT = 2.2     # m: hits nearer than this block bearings around them
CLEARANCE = 0.55   # m: lateral clearance to keep from every hit point
SLOW_ZONE = 1.5    # m: frontal hit nearer than this -> avoidance speed
FRONT_HALF = 0.6   # rad: half-angle of the frontal threat cone
V_AVOID = 0.3      # m/s while maneuvering around the obstacle


def _wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """L1-style point controller plus scan-based detour steering.

    Each scan hit nearer than NEAR_HIT blocks an angular wedge around its
    bearing, inflated by atan2(CLEARANCE, r): close hits block wide wedges,
    distant hits narrow slivers, so passing an obstacle keeps its flank
    "blocked" until the robot is genuinely clear - which also prevents a
    premature turn back toward the goal.  While the goal bearing is clear,
    drive exactly as in L1; when it is blocked, steer to the clear bearing
    closest to the goal direction and slow down.
    """
    x, y, yaw = pose
    gx, gy, gyaw = GOAL
    dist = math.hypot(gx - x, gy - y)

    if dist <= D_STOP:
        yerr = _wrap_pi(gyaw - yaw)
        if abs(yerr) > YAW_OK:
            return 0.0, max(-W_FINAL, min(W_FINAL, K_FINAL * yerr)), False
        return 0.0, 0.0, True

    err = _wrap_pi(math.atan2(gy - y, gx - x) - yaw)
    scan = env.raycast_scan()
    wedges = []
    for b, r in scan:
        if r < min(NEAR_HIT, dist):  # hits beyond the goal cannot block it
            half = math.atan2(CLEARANCE, r)
            wedges.append((b - half, b + half))

    def is_blocked(bearing):
        return any(lo <= bearing <= hi for lo, hi in wedges)

    ahead = [r for b, r in scan if abs(b) <= FRONT_HALF]
    nearest_ahead = min(ahead) if ahead else 99.0

    if not is_blocked(err) and nearest_ahead >= SLOW_ZONE:
        w = max(-W_LIMIT, min(W_LIMIT, K_TURN * err))
        if abs(err) > TURN_ONLY:
            return 0.0, w, False
        return min(V_CRUISE, 0.7 * dist), w, False

    open_dirs = [b for b, r in scan if r >= NEAR_HIT and not is_blocked(b)]
    if not open_dirs:
        return 0.0, 0.6 * W_LIMIT, False  # boxed in: spin slowly, rescan
    pick = min(open_dirs, key=lambda b: abs(_wrap_pi(b - err)))
    w = max(-W_LIMIT, min(W_LIMIT, K_TURN * pick))
    v = V_AVOID if nearest_ahead < SLOW_ZONE or abs(pick) > TURN_ONLY else 0.7 * V_CRUISE
    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = Harness(
    level="level2",
    variant=VARIANT,
    goal_xy_yaw=GOAL,
    timeout_s=TIMEOUT_S,
    position_tolerance_m=POS_TOL,
    yaw_tolerance_rad=YAW_TOL,
    check_collision=True,
    obstacle=OBSTACLE,
    out_dir=OUT,
).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
