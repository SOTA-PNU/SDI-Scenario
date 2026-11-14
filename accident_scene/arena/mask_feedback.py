"""Redact a result.json down to what a model may see between turns.

Track B forbids learning the layout, but the raw result leaks it three ways:
  * the "obstacle" block carries all 14 box geometries verbatim;
  * collision_partners names the prim ("/World/arena_box_04"), whose index is a
    row of that table;
  * sim_time_s minus the harness' 5 s abort and 1 s settle recovers the exact
    first-contact time, which with START and the average speed localises the
    contact to under a metre.
Everything kept below is outcome, not geometry: which oracle failed, how far
from the goal the robot ended, how many contacts, and whether the run was cut
short. That is enough signal to fix a controller and not enough to map a room.

    python3 mask_feedback.py results/<variant>_result.json
"""

import json
import sys
from pathlib import Path

KEEP_METRICS = ("time_to_goal_s", "final_dist_m", "final_yaw_err_rad",
                "collision_count", "min_goal_dist_m")


def mask(result):
    m = result.get("metrics", {})
    out = {
        "verdict": result["verdict"],
        "criteria": result.get("criteria", {}),
        "metrics": {k: m.get(k) for k in KEEP_METRICS},
        "oracles": [{"name": o["name"], "passed": o["passed"], "reason": o.get("reason")}
                    for o in result.get("oracles", [])],
        "collided": bool(m.get("collision_count", 0)),
        "aborted_early": bool(result.get("aborted_after_collision")),
    }
    # the contact's timing and partner are withheld on purpose; say so, so the
    # model does not read the absence as "no contact"
    if out["collided"]:
        out["note"] = ("A collision occurred. The time and the identity of what was "
                       "struck are withheld.")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    print(json.dumps(mask(json.loads(src.read_text())), indent=2, ensure_ascii=False))
