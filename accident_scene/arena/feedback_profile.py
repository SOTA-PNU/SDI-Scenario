"""Two feedback profiles for the richness ablation, and nothing else.

EXPERIMENTS.md asks whether richer feedback converges a turn loop. The earlier
comparison used track A against track B, which differ in feedback richness AND
in whether the layout may be read - a confounded pair. These profiles vary
richness alone, inside track A.

    lean  verdict, final distance, yaw error, collision count, elapsed
    rich  the same, plus what was struck, when, and where the robot was

mask_feedback.mask() is untouched: that is track B's layout redaction, a
different job. turn_loop.py keeps its default behaviour when no profile is
given, so every run recorded before this file existed stays reproducible.
"""

LEAN_METRICS = ("final_dist_m", "final_yaw_err_rad", "collision_count", "sim_time_s")
RICH_EXTRA = ("collision_partners", "first_collision_t", "final_pose")


def profile(result, name):
    """Return the dict a model is shown after a failing run."""
    if name not in ("lean", "rich"):
        raise ValueError(f"unknown feedback profile: {name}")
    m = result.get("metrics", {})
    out = {
        "verdict": result["verdict"],
        "metrics": {k: m.get(k) for k in LEAN_METRICS},
    }
    if name == "rich":
        out["metrics"].update({k: m.get(k) for k in RICH_EXTRA})
        out["oracles"] = [{"name": o["name"], "passed": o["passed"],
                           "reason": o.get("reason"), "detail": o.get("detail")}
                          for o in result.get("oracles", [])]
    else:
        out["oracles"] = [{"name": o["name"], "passed": o["passed"]}
                          for o in result.get("oracles", [])]
    return out
