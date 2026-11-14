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

CRUISE_V = 0.5     # m/s cruise speed toward the goal
STOP_DIST = 0.30   # m, well inside the 0.75 m position tolerance
K_HEADING = 1.8    # P gain: bearing error -> yaw rate
W_MAX = 1.2        # rad/s yaw rate limit
BEARING_GATE = 0.5  # rad: rotate in place while badly misaligned
K_ALIGN = 2.0      # P gain for the final in-place alignment
ALIGN_W_MAX = 0.8  # rad/s limit during final alignment
YAW_DONE = 0.08    # rad, well inside the 0.26 rad yaw tolerance


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def controller(t, pose, env):
    """Three-phase point controller: rotate -> drive -> final alignment.

    L1 adds two capabilities over the L0 forward-only controller:
    (1) steering - the goal is NOT ahead of the spawn heading, so rotate
        toward the goal bearing and keep correcting it while driving;
    (2) arrival pose - once inside the goal disc, rotate in place until the
        commanded goal yaw is met, then finish.
    """
    x, y, yaw = pose
    gx, gy, gyaw = GOAL
    d = math.hypot(gx - x, gy - y)

    if d > STOP_DIST:
        bearing = math.atan2(gy - y, gx - x)
        err = _wrap(bearing - yaw)
        w = max(-W_MAX, min(W_MAX, K_HEADING * err))
        v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
        return v, w, False

    # inside the goal disc: align to the commanded arrival yaw, then finish
    yerr = _wrap(gyaw - yaw)
    if abs(yerr) > YAW_DONE:
        return 0.0, max(-ALIGN_W_MAX, min(ALIGN_W_MAX, K_ALIGN * yerr)), False
    return 0.0, 0.0, True


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
