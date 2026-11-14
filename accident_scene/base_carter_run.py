"""Mission runner - ACCIDENT SCENE (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/nova_carter_warehouse_accident.yaml): the robot starts at the
harness spawn (-6.0, -1.0, yaw=pi) and must reach GOAL; one static box stands
in the lane.

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "levels" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "arena"))
from carter_env import Harness  # noqa: E402
from arena_harness import apply_record_quality, recording  # noqa: E402

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(Path(__file__).resolve().parent / "results")

# scenario givens (from accident_scene/nova_carter_warehouse_accident.yaml - not part of the answer)
MISSION = dict(goal_xy_yaw=(-6.0, 5.0, 1.5708), timeout_s=120.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               obstacle={"x": -6.0, "y": 3.8,
                         "height": 0.25, "width": 1.2, "depth": 0.6})
GOAL = MISSION["goal_xy_yaw"]  # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def controller(t, pose, env):
    """No control code yet: the robot never moves."""
    return 0.0, 0.0, False


# ========================= [END EDIT REGION] ===============================

h = Harness(level="accident_scene", variant=VARIANT, out_dir=OUT, **MISSION).boot()
if recording():
    apply_record_quality()  # visualization only: path tracer (see arena/REPORT.md)
result = h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
