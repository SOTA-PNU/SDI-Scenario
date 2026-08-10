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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "levels" / "common"))
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


def _wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def controller(t, pose, env):
    """Turn toward the goal, drive the free +y lane, align to final yaw, stop.

    Deterministic pose-feedback controller (no randomness, no wall clock):
      Phase 1: while far from the goal, rotate in place until roughly facing
               it, then drive with proportional heading correction and a
               distance-tapered speed.
      Phase 2: inside the capture radius, rotate in place to the goal yaw.
      Phase 3: when position and yaw are both well inside tolerance, stop
               and report done.
    """
    x, y, yaw = pose
    gx, gy, gyaw = GOAL

    dx = gx - x
    dy = gy - y
    dist = math.hypot(dx, dy)

    # -- Phase 1: go to the goal point ------------------------------------
    if dist > 0.25:
        heading_err = _wrap(math.atan2(dy, dx) - yaw)
        # Proportional turn rate, clamped.
        w = max(-1.2, min(1.2, 2.0 * heading_err))
        if abs(heading_err) > 0.25:
            # Face the lane first; turning in place cannot hit anything.
            v = 0.0
        else:
            # Taper speed near the goal; cap cruise speed on the open lane.
            v = max(0.1, min(0.8, 0.7 * dist))
        return v, w, False

    # -- Phase 2: align to the required final yaw -------------------------
    yaw_err = _wrap(gyaw - yaw)
    if abs(yaw_err) > 0.06:
        w = max(-1.0, min(1.0, 2.0 * yaw_err))
        return 0.0, w, False

    # -- Phase 3: settled inside both tolerances - finish -----------------
    return 0.0, 0.0, True


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
