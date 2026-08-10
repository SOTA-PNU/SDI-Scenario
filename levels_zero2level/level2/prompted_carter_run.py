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


STATE = {"phase": "drive", "wps": [(-6.0, 5.0)], "i": 0, "detoured": False}


def _wrap(a):
    """Wrap an angle to [-pi, pi] deterministically."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(x, lim):
    return max(-lim, min(lim, x))


def controller(t, pose, env):
    """Scan-driven lane follower: drive north, perceive the box with the ray
    fan, side-step through the freer gap, rejoin the lane, then align."""
    px, py, yaw = pose
    s = STATE

    # ---- final in-place alignment ------------------------------------
    if s["phase"] == "align":
        err = _wrap(1.5708 - yaw)
        if abs(err) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(2.0 * err, 1.2), False

    scan = env.raycast_scan()

    # ---- one-shot obstacle perception (only while heading up the lane)
    if not s["detoured"] and abs(_wrap(yaw - 1.5708)) < 0.6:
        ahead = [d for b, d in scan if abs(b) < 0.6 and d < 2.4]
        if ahead:
            # world-frame hit points of the blocking face
            pts = [(px + d * math.cos(yaw + b), py + d * math.sin(yaw + b))
                   for b, d in scan if abs(b) < 0.9 and d < 3.5]
            if pts:
                east = max(p[0] for p in pts)      # right edge of the box
                west = min(p[0] for p in pts)      # left edge of the box
                front_y = min(p[1] for p in pts)   # near face of the box
                # lateral free space on each side (walls, if any)
                right_hits = [px + d * math.cos(yaw + b)
                              for b, d in scan if b < -0.9 and d < 5.9]
                left_hits = [px + d * math.cos(yaw + b)
                             for b, d in scan if b > 0.9 and d < 5.9]
                r_wall = min([x for x in right_hits if x > px + 0.3],
                             default=None)
                l_wall = max([x for x in left_hits if x < px - 0.3],
                             default=None)
                r_room = (r_wall if r_wall is not None else px + 6.0) - east
                l_room = west - (l_wall if l_wall is not None else px - 6.0)
                if r_room >= l_room:               # pass on the east side
                    dx = east + 0.8
                    if r_wall is not None:
                        dx = min(dx, r_wall - 0.5)
                    dx = max(dx, east + 0.4)
                else:                              # pass on the west side
                    dx = west - 0.8
                    if l_wall is not None:
                        dx = max(dx, l_wall + 0.5)
                    dx = min(dx, west - 0.4)
                y_clear = front_y + 1.7            # past the far face
                s["wps"] = [(dx, front_y - 0.4), (dx, y_clear),
                            (-6.0, min(y_clear + 0.8, 4.3)), (-6.0, 5.0)]
                s["i"] = 0
                s["detoured"] = True

    # ---- waypoint pursuit --------------------------------------------
    wps = s["wps"]
    tx, ty = wps[s["i"]]
    dist = math.hypot(tx - px, ty - py)
    while s["i"] < len(wps) - 1 and dist < 0.35:
        s["i"] += 1
        tx, ty = wps[s["i"]]
        dist = math.hypot(tx - px, ty - py)
    last = s["i"] == len(wps) - 1
    if last and dist < 0.25:
        s["phase"] = "align"
        return 0.0, 0.0, False

    err = _wrap(math.atan2(ty - py, tx - px) - yaw)
    w = _clamp(2.2 * err, 1.8)
    if abs(err) > 1.0:
        v = 0.0                      # large heading error: pivot in place
    else:
        v = min(0.9, 0.4 + 0.8 * dist)
        if abs(err) > 0.3:
            v *= 0.35
        if last:
            v = min(v, 0.1 + 1.0 * dist)

    # ---- last-resort proximity guard ---------------------------------
    front = min([d for b, d in scan if abs(b) < 0.35], default=6.0)
    if front < 0.5:
        v = 0.0
    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = Harness(level=LEVEL, variant=VARIANT, out_dir=OUT, **M).boot()
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
