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


def controller(t, pose, env, _s={"i": 0, "phase": "drive"}):
    """Waypoint detour around the near-goal box, then align to +y and stop.

    Box footprint spans x in [-6.6, -5.8], y in [3.2, 3.8]; we detour on the
    east side through an x = -5.0 lane (0.8 m clearance from the box edge),
    then cut back to the goal.  A narrow front-cone raycast guard slows or
    stops the robot if anything is unexpectedly close.  Deterministic: no
    randomness, no wall-clock use.
    """
    x, y, yaw = pose

    def wrap(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def clamp(val, lo, hi):
        return lo if val < lo else (hi if val > hi else val)

    GOAL_YAW = 1.5708

    # -- final phase: rotate in place to the required arrival yaw ----------
    if _s["phase"] == "align":
        err = wrap(GOAL_YAW - yaw)
        if abs(err) < 0.05:
            return 0.0, 0.0, True
        return 0.0, clamp(3.0 * err, -1.5, 1.5), False

    # -- drive phase: follow the fixed detour waypoints --------------------
    wps = [
        (-6.0, 2.20),   # straight up the corridor, well south of the box
        (-5.00, 2.70),  # angle out east before the box (box south edge y=3.2)
        (-5.00, 4.35),  # pass alongside the box (east edge x=-5.8, +0.8 m)
        (-6.0, 5.00),   # goal center
    ]
    last = len(wps) - 1

    # advance waypoint index (loop in case a switch lands inside next tol)
    while True:
        tx, ty = wps[_s["i"]]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        tol = 0.20 if _s["i"] == last else 0.35
        if dist >= tol:
            break
        if _s["i"] == last:
            _s["phase"] = "align"
            return 0.0, 0.0, False
        _s["i"] += 1

    # heading control toward current waypoint
    err = wrap(math.atan2(dy, dx) - yaw)
    w = clamp(2.5 * err, -2.0, 2.0)

    if abs(err) > 1.0:
        v = 0.0                      # big heading error: turn in place
    else:
        v = 1.3 * (1.0 - abs(err) / 1.0)
        if _s["i"] == last:
            v = min(v, 0.35 + 0.8 * dist)   # ease into the goal disc
        else:
            v = min(v, 0.45 + 1.2 * dist)   # ease into waypoint turns

    # raycast safety guard: narrow front cone, deterministic thresholds
    if v > 0.0:
        d_front = 6.0
        for bearing, d in env.raycast_scan():
            if abs(bearing) <= 0.30 and d < d_front:
                d_front = d
        if d_front < 0.30:
            v = 0.0                  # something dead ahead: stop, keep turning
            if abs(w) < 0.4:
                w = 0.8 if err >= 0.0 else -0.8
        elif d_front < 0.50:
            v = min(v, 0.25)

    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = Harness(level=LEVEL, variant=VARIANT, out_dir=OUT, **M).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
