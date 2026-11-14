"""Verify scenarios/*.yaml bodies match the givens hardcoded in levels/*/base runners.

The runner files deliberately embed the scenario givens as constants (single
self-contained benchmark file for the NPU LLM); this checker keeps that
duplication honest.  Run from the repo root:

    python3 levels/common/check_yaml_sync.py            # all levels
    python3 levels/common/check_yaml_sync.py level2     # one level

Exits non-zero on any mismatch.
"""

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ["level0", "level1", "level2", "level3"]


def runner_givens(level):
    """Parse module-level constant assignments from the base runner via ast."""
    tree = ast.parse((ROOT / "levels" / level / "base_carter_run.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def yaml_givens(level):
    n = level.replace("level", "")
    doc = yaml.safe_load(
        (ROOT / "scenarios" / f"nova_carter_warehouse_level{n}.yaml").read_text())
    sc = doc["scenario"]
    crit = {}
    for entry in doc["acceptance_criteria"]:
        crit.update(entry.get("params") or {})
    return sc, crit


def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


def check(level):
    r = runner_givens(level)
    sc, crit = yaml_givens(level)
    errs = []

    gx, gy, gyaw = r["GOAL"]
    for name, rv, yv in (("goal.x", gx, sc["goal"]["x"]),
                         ("goal.y", gy, sc["goal"]["y"]),
                         ("goal.yaw", gyaw, sc["goal"]["yaw"]),
                         ("timeout_s", r["TIMEOUT_S"], sc["timeout_s"]),
                         ("position_tolerance_m", r["POS_TOL"],
                          crit["position_tolerance_m"])):
        if not close(rv, yv):
            errs.append(f"{name}: runner {rv} != yaml {yv}")

    for rkey, ckey in (("YAW_TOL", "yaw_tolerance_rad"),
                       ("MAX_TTG", "max_time_to_goal_s")):
        if (rkey in r) != (ckey in crit):
            errs.append(f"{ckey}: present in only one side "
                        f"(runner={rkey in r}, yaml={ckey in crit})")
        elif rkey in r and not close(r[rkey], crit[ckey]):
            errs.append(f"{ckey}: runner {r[rkey]} != yaml {crit[ckey]}")

    robs, yobs = r.get("OBSTACLE"), sc.get("debug_obstacle")
    if (robs is None) != (yobs is None):
        errs.append(f"debug_obstacle: present in only one side "
                    f"(runner={robs is not None}, yaml={yobs is not None})")
    elif robs:
        for k in ("x", "y", "height", "width", "depth"):
            if not close(robs[k], yobs[k]):
                errs.append(f"debug_obstacle.{k}: runner {robs[k]} != yaml {yobs[k]}")

    tag = "OK" if not errs else "MISMATCH"
    print(f"{level}: yaml<->runner {tag}")
    for e in errs:
        print(f"  - {e}")
    return not errs


def main():
    targets = [a.rstrip("/").split("/")[-1] for a in sys.argv[1:]] or LEVELS
    ok = all(check(t) for t in targets)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
