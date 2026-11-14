"""Join the turn logs to the graded runs, one row per turn, and name the shape.

turn_summary.py reports collisions and final distance per turn; runs.csv holds
the per-run geometry, including trap_exposure_m - the closest the robot ever
came to a box that sits under the scan plane. Every turn was graded in Isaac
and its trajectory kept, so the join needs no new simulation.

The phenotype rules are fixed in REPORT.md before this ran. They are applied
here in that order, first match wins, so nothing is assigned by eye.

    python3 analysis/turn_trace.py [--turns-dir results/turns/cap8] [--csv out.csv]
"""

import argparse
import csv
import json
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA = os.path.dirname(HERE)


def oracle_violations(t):
    """How many of the two scored oracles this turn failed, or None if unscorable.

    A static disqualification or a runtime error never reached the oracles, so
    it has no count - counting it as zero would read as success and counting it
    as two would read as a worse controller than was measured.
    """
    if t.get("verdict") == "static_dq" or t.get("runtime_error"):
        return None
    if t.get("verdict") == "pass":
        return 0
    n = 0
    if (t.get("collisions") or 0) > 0:
        n += 1
    if t.get("time_to_goal_s") is None:
        n += 1
    return n


def phenotype(turns):
    """Name the shape of a run, rules fixed in REPORT.md before this ran.

    rev.2. rev.1 asked only that the oracle-violation count be non-increasing,
    which cannot tell "fixed the collisions and gave up on the goal" from
    "closed both" - both end on one violation. It labelled the archetypal
    stall (collisions 360 -> 0 while the final distance grew from 14.5 m to
    31.9 m) as sequential closure. rev.2 makes distance improvement a
    necessary condition for closure, and rev.1's output is kept beside this
    one in turn_trace_rev1.csv.
    """
    scored = [t for t in turns if oracle_violations(t) is not None]
    if len(turns) < 3 or not scored:
        return "other"
    viol = [oracle_violations(t) for t in scored]
    dists = [t.get("final_dist_m") for t in scored if t.get("final_dist_m") is not None]
    passed = any(t.get("verdict") == "pass" for t in scored)
    improved = bool(dists) and len(dists) > 1 and dists[-1] < dists[0]
    monotone = all(b <= a for a, b in zip(viol, viol[1:]))

    if monotone and (passed or improved):
        return "sequential_closure"
    last3 = scored[-3:]
    stalled = len(last3) == 3 and all((t.get("collisions") or 0) == 0 for t in last3)
    if stalled or (monotone and not improved and not passed):
        return "stall_converge"
    kinds = [("dq" if t.get("verdict") == "static_dq" else
              "err" if t.get("runtime_error") else
              "pass" if t.get("verdict") == "pass" else
              "stall" if (t.get("collisions") or 0) == 0 else "collide") for t in turns]
    flips = sum(1 for a, b in zip(kinds, kinds[1:]) if a != b)
    if flips >= len(turns) / 2:
        return "oscillate"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-dir", default=os.path.join(ARENA, "results", "turns", "cap8"))
    ap.add_argument("--runs-csv", default=os.path.join(ARENA, "results", "runs.csv"))
    ap.add_argument("--csv", default=os.path.join(ARENA, "results", "analysis", "turn_trace.csv"))
    a = ap.parse_args()

    geo = {r["variant"]: r for r in csv.DictReader(open(a.runs_csv))}
    rows, missing = [], 0
    for p in sorted(glob.glob(os.path.join(a.turns_dir, "*_turns.json"))):
        log = json.loads(open(p).read())
        ph = phenotype(log["turns"])
        accel = "npu" if log["tag"].startswith("npu_") else "gpu"
        cond = log.get("condition", "default")
        # the ablation runs two conditions off the same seed, so the condition
        # is part of a run's identity - without it lean and rich merge into one
        # 16-turn run and every per-run statistic is wrong
        stem = f"{log['tag']}_{log['track']}{log['sample']}"
        rid = stem if cond == "default" else f"{stem}_{cond}"
        for t in log["turns"]:
            # turn 1 is forked from the un-suffixed run and graded under that name
            v = t["variant"]
            g = geo.get(v) or (geo.get(stem, {}) if t["turn"] == 1 else {})
            if not g:
                missing += 1
            rows.append(dict(
                run_id=rid, condition=cond,
                track=log["track"], accel=accel, model=log["model"],
                turn_idx=t["turn"], variant=t["variant"],
                verdict=t.get("verdict", ""),
                failure_type=g.get("failure_class", ""),
                collision_count=t.get("collisions"),
                trap_exposure_m=g.get("trap_exposure_m", ""),
                final_dist_m=t.get("final_dist_m"),
                yaw_err_rad="", elapsed_s=t.get("gen_wall_s"),
                oracle_violations=oracle_violations(t),
                phenotype=ph))
    if not rows:
        print("대상 없음", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    runs = {}
    for r in rows:
        runs[r["run_id"]] = r["phenotype"]
    print(f"{len(rows)}턴 / {len(runs)}실행  (runs.csv 미매칭 {missing}턴)")
    print("\nphenotype 분포:", dict(Counter(runs.values())))
    print(f"\n{'실행':30}{'phenotype':20}{'턴별 함정노출 m'}")
    for rid in sorted(runs):
        ex = [r["trap_exposure_m"] for r in rows if r["run_id"] == rid]
        print(f"{rid:30}{runs[rid]:20}" +
              " ".join(f"{float(x):5.2f}" if x else "    -" for x in ex))
    print(f"\n-> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
