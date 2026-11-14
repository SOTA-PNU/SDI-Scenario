"""Turn a directory of graded runs into the paper's result table.

Two numbers here are not in the harness' own result.json and carry the paper:

  trap_exposure_m   how close the robot ever got to a sub-scan-plane box
                    (0.18-0.25 m tall, invisible to the default z=0.35 m fan).
                    Measured to the box footprint, not its centre. This is the
                    evidence for the central claim - a controller that never
                    comes near those boxes was never tested by them, whatever
                    its verdict says.

  failure_class     stall | visible_collision | trap_collision | pass
                    Distinguishing which KIND of obstacle was struck is the
                    difference between "could not avoid what it could see" and
                    "could not perceive what it could not see".

    python3 score_runs.py results/            -> table + results.csv
"""

import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = json.load(open(os.path.join(HERE, "layout.json")))
SCAN_Z = 0.35
TRAPS = [i for i, b in enumerate(LAYOUT["boxes"]) if b["height"] < SCAN_Z]


def footprint(b):
    return (b["x"] - b["width"] / 2, b["x"] + b["width"] / 2,
            b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2)


TRAP_BOXES = [(i, footprint(LAYOUT["boxes"][i])) for i in TRAPS]


def dist_to_box(x, y, bb):
    x0, x1, y0, y1 = bb
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def trap_exposure(traj_path):
    """Closest approach to any sub-scan-plane box over the whole run."""
    best, which = float("inf"), None
    with open(traj_path) as f:
        for row in csv.DictReader(f):
            x, y = float(row["x"]), float(row["y"])
            for i, bb in TRAP_BOXES:
                d = dist_to_box(x, y, bb)
                if d < best:
                    best, which = d, i
    return best, which


def classify(result):
    m = result["metrics"]
    if result["verdict"] == "pass":
        return "pass"
    if not m["collision_count"]:
        return "stall"
    for p in m.get("collision_partners") or []:
        name = p.rsplit("/", 1)[-1]
        if name.startswith("arena_box_") and int(name[-2:]) in TRAPS:
            return "trap_collision"
    return "visible_collision"


EXC_RE = re.compile(r"^([A-Za-z_.]*(?:Error|Exception))\b:?(.*)$", re.M)


def runtime_error_rows(d):
    """Submissions whose controller raised: Isaac never wrote a result.json.

    These are not missing data - the run happened and the code failed in it.
    Dropping them would quietly flatter the model that crashed, so they get
    their own class and a row like any other failure. There is no trajectory,
    so trap exposure is undefined rather than zero.
    """
    out = []
    for lp in sorted(glob.glob(os.path.join(d, "*.isaac.log"))):
        variant = os.path.basename(lp)[: -len(".isaac.log")]
        if os.path.exists(os.path.join(d, f"{variant}_result.json")):
            continue
        blob = re.sub(r"^.*?\[py stderr\]:\s?", "", open(lp, errors="replace").read(),
                      flags=re.M)
        hits = [f"{a}:{b}".strip() for a, b in EXC_RE.findall(blob)
                if "omni" not in b and "Extension python module" not in b]
        out.append(dict(variant=variant, verdict="fail", failure_class="runtime_error",
                        time_to_goal_s=None, final_dist_m=None, collisions=None,
                        trap_exposure_m=None, nearest_trap=None,
                        detail=(hits[-1][:120] if hits else "controller raised")))
    return out


def main(d):
    rows = []
    for rp in sorted(glob.glob(os.path.join(d, "*_result.json"))):
        variant = os.path.basename(rp)[: -len("_result.json")]
        tp = os.path.join(d, f"{variant}_trajectory.csv")
        if not os.path.exists(tp):
            continue
        r = json.load(open(rp))
        m = r["metrics"]
        exp, which = trap_exposure(tp)
        rows.append(dict(variant=variant, verdict=r["verdict"], failure_class=classify(r),
                         time_to_goal_s=m["time_to_goal_s"], final_dist_m=m["final_dist_m"],
                         collisions=m["collision_count"],
                         trap_exposure_m=round(exp, 2), nearest_trap=which))
    for r in rows:
        r.setdefault("detail", "")
    rows += runtime_error_rows(d)
    rows.sort(key=lambda z: (z["verdict"] != "pass",
                             z["trap_exposure_m"] is None,
                             z["trap_exposure_m"] or 0.0))
    w = max((len(r["variant"]) for r in rows), default=8)
    print(f"{'variant':{w}}  {'verdict':7} {'class':17} {'ttg':>7} {'dist':>7} {'col':>4} {'trap':>7}")
    print("-" * (w + 54))
    for r in rows:
        ttg = f"{r['time_to_goal_s']:.1f}" if r["time_to_goal_s"] else "-"
        if r["trap_exposure_m"] is None:
            print(f"{r['variant']:{w}}  {r['verdict']:7} {r['failure_class']:17} "
                  f"{'-':>7} {'-':>7} {'-':>4} {'-':>7}  {r['detail']}")
            continue
        print(f"{r['variant']:{w}}  {r['verdict']:7} {r['failure_class']:17} {ttg:>7} "
              f"{r['final_dist_m']:7.2f} {r['collisions']:4d} {r['trap_exposure_m']:6.2f}m")
    out = os.path.join(d, "runs.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n함정(스캔 평면 아래) 박스: {TRAPS}  ·  {len(rows)}행 -> {out}")


main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
