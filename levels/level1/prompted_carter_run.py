"""LEVEL 1 mission runner - full-lane drive + arrival pose alignment.

MISSION (scenarios/nova_carter_warehouse_level1.yaml):
  The Nova Carter spawns at world (-6.0, -1.0) facing yaw=pi (-x direction).
  The goal (-6.0, 5.0) lies 6.0 m away along the +y warehouse lane (the lane
  is free the whole way - probe-verified).  Drive there, arrive facing
  yaw=+1.5708 (+y), and never touch anything on the way.

PASS CRITERIA (evaluated by the fixed harness, not by this file):
  reached_goal   final distance <= 0.75 m AND |yaw error| <= 0.26 rad
  no_collision   zero chassis contact events (self + floor excluded)
  timeout_s      120 s of sim time

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams, fov_deg, z, max_range)
          -> [(bearing_rel_to_heading, distance_m), ...]  (not needed at L1)
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

# ===========================================================================
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

CRUISE_V = 0.5        # m/s cruise speed toward the goal
K_ANG = 2.0           # P gain on heading error [1/s]
K_LIN = 0.8           # slowdown gain: v <= K_LIN * distance near the goal
W_MAX = 1.0           # rad/s cap on commanded yaw rate
TURN_GATE = 0.30      # rad: rotate in place while |heading error| exceeds this
GOAL_RADIUS = 0.35    # m: switch to final alignment (well inside 0.75 m tol)
FINAL_YAW_TOL = 0.08  # rad: finish threshold (well inside 0.26 rad tol)


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def controller(t, pose, env):
    """Three-phase point controller: turn to goal, drive, align arrival yaw.

    Phase 1: while the heading error to the goal bearing is large, rotate in
             place (v = 0) under capped P control.
    Phase 2: once roughly aligned, cruise toward the goal, keep correcting the
             bearing, and slow down proportionally to the remaining distance.
    Phase 3: inside the goal disc, rotate in place to the arrival yaw
             GOAL[2] and return done=True when within tolerance.
    """
    x, y, yaw = pose
    dx = GOAL[0] - x
    dy = GOAL[1] - y
    d = math.hypot(dx, dy)

    if d > GOAL_RADIUS:
        # Phases 1-2: steer toward the goal point.
        err = _wrap(math.atan2(dy, dx) - yaw)
        w = max(-W_MAX, min(W_MAX, K_ANG * err))
        if abs(err) > TURN_GATE:
            return 0.0, w, False       # rotate in place until roughly aligned
        v = min(CRUISE_V, K_LIN * d)   # cruise, decelerating near the goal
        return v, w, False

    # Phase 3: align to the arrival yaw, then stop and finish.
    err = _wrap(GOAL[2] - yaw)
    if abs(err) <= FINAL_YAW_TOL:
        return 0.0, 0.0, True          # arrived and aligned - done
    w = max(-W_MAX, min(W_MAX, K_ANG * err))
    return 0.0, w, False


# ========================= [END EDIT REGION] ===============================

h = Harness(
    level="level1",
    variant=VARIANT,
    goal_xy_yaw=GOAL,
    timeout_s=TIMEOUT_S,
    position_tolerance_m=POS_TOL,
    yaw_tolerance_rad=YAW_TOL,
    check_collision=True,
    out_dir=OUT,
).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
