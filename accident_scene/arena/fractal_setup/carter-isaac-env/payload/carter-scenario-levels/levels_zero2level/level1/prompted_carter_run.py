"""Mission runner - identical file for every level (zero2level track).

The mission is NOT hard-coded here: the level comes from the parent folder
name (level0..level3) and its givens from the MISSIONS table below.  What
the controller must accomplish is specified by that level's PROMPT.md.

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

The ONLY part meant to be modified to solve a level is the block marked
[EDIT REGION].  Everything outside it is fixed plumbing, byte-identical
across all four levels and all variants.
"""

import math  # noqa: F401  (available to the controller)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "levels" / "common"))
from carter_env import Harness  # noqa: E402

LEVEL = Path(__file__).resolve().parent.name              # "level0" .. "level3"
VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(Path(__file__).resolve().parent / "results")

# scenario givens per level (from scenarios/*.yaml - not part of the answer)
MISSIONS = {
    "level0": dict(goal_xy_yaw=(-8.0, -1.0, 3.14159265), timeout_s=60.0,
                   position_tolerance_m=1.0, yaw_tolerance_rad=None,
                   max_time_to_goal_s=None, check_collision=False,
                   obstacle=None),
    "level1": dict(goal_xy_yaw=(-6.0, 5.0, 1.5708), timeout_s=120.0,
                   position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
                   max_time_to_goal_s=None, check_collision=True,
                   obstacle=None),
    "level2": dict(goal_xy_yaw=(-6.0, 5.0, 1.5708), timeout_s=120.0,
                   position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
                   max_time_to_goal_s=None, check_collision=True,
                   obstacle={"x": -6.2, "y": 2.0,
                             "height": 1.0, "width": 0.8, "depth": 0.6}),
    "level3": dict(goal_xy_yaw=(-6.0, 5.0, 1.5708), timeout_s=120.0,
                   position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
                   max_time_to_goal_s=12.0, check_collision=True,
                   obstacle={"x": -6.2, "y": 3.5,
                             "height": 1.0, "width": 0.8, "depth": 0.6}),
}
M = MISSIONS[LEVEL]
GOAL = M["goal_xy_yaw"]  # (x, y, yaw) - available to the controller

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

h = Harness(level=LEVEL, variant=VARIANT, out_dir=OUT, **M).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
