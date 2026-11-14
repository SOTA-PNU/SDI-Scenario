"""Regression for the A-2-1 scene tools (REPORT.md 실험 15), no Isaac, no LLM.

(a) check_route on the 15 model routes of A-1..A-4/furio (results/analysis/route_clearance_check.json):
    scored overlaps on every failing route, none on the passing ones.
(b) every archived Isaac trajectory with a result: the grid gap from the robot outline to the
    harness's first collision partner at first_collision_t, and, for passing runs, the smallest
    gap to any scored object.
Writes results/analysis/a21_regress.json.

    python3 analysis/a21_regress.py
"""
import csv
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene as S  # noqa: E402

START, GOAL = (-9.0, -10.8, 1.0138), (8.0, 16.5, 1.0138)


def main():
    sc = S.Scene()
    out = {"scene_sha16": sc.sha16, "routes": [], "trajectories": []}
    rc = json.loads((ARENA / "results/analysis/route_clearance_check.json").read_text())
    passed_variants = set()
    for f in ARENA.glob("results/turns/**/*_result.json"):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("verdict") == "pass":
            passed_variants.add(r.get("variant"))
    for e in rc:
        wp = e.get("waypoints")
        if not wp:
            continue
        r = S.check_route(sc, wp, START, GOAL)
        scored_ov = sorted({o[0] for ov in r["overlaps"] for o in ov["objects"] if o[1]}
                           | {t[4] for t in r["turns"] if t[3] <= 0.0 and t[5]})
        unscored_ov = sorted({o[0] for ov in r["overlaps"] for o in ov["objects"] if not o[1]}
                             | {t[4] for t in r["turns"] if t[3] <= 0.0 and not t[5]})
        passed = e["variant"] in passed_variants
        out["routes"].append({"prompt": e["prompt"], "variant": e["variant"], "passed": passed,
                              "min_gap": r["min_gap"], "scored_overlaps": scored_ov,
                              "unscored_overlaps": unscored_ov})
    seen = set()
    for res in sorted(ARENA.glob("results/turns/**/isaac_logs/*_result.json")):
        v = res.name[:-len("_result.json")]
        tr = res.with_name(f"{v}_trajectory.csv")
        if not tr.exists() or v in seen:
            continue
        seen.add(v)
        r = json.loads(res.read_text())
        if r.get("level") != "accident_arena" or r.get("goal", {}).get("x") != 8.0:
            continue
        m = r["metrics"]
        rows = [(float(q["t"]), float(q["x"]), float(q["y"]), float(q["yaw"])) for q in csv.DictReader(open(tr))]
        rec = {"variant": v, "dir": str(res.parent.relative_to(ARENA)), "verdict": r["verdict"],
               "partners": m.get("collision_partners"), "first_collision_t": m.get("first_collision_t")}
        if m.get("first_collision_t") is not None and m.get("collision_partners"):
            t0 = m["first_collision_t"]
            near = [q for q in rows if abs(q[0] - t0) <= 0.1] or [min(rows, key=lambda q: abs(q[0] - t0))]
            want = S.short_name(m["collision_partners"][0])
            best = None
            for q in near:
                px, py = sc.outline_points(q[1], q[2], q[3])
                j, i = sc.cell_of(px, py)
                ok = (j >= 0) & (j < sc.ny) & (i >= 0) & (i < sc.nx)
                j, i = j[ok], i[ok]
                # gap to the named partner only: nearest occupied cell of that name
                import numpy as np
                own = sc.own
                mask = np.zeros(len(sc.owners) + 1, dtype=bool)
                for k, n in enumerate(sc.names, 1):
                    mask[k] = (n == want)
                cells = np.argwhere(mask[own])
                if not len(cells):
                    continue
                cx = sc.x0 + (cells[:, 1] + 0.5) * sc.c
                cy = sc.y0 + (cells[:, 0] + 0.5) * sc.c
                d = np.min(np.hypot(px[:, None] - cx[None, :], py[:, None] - cy[None, :])) - 0.5 * sc.c
                d = max(0.0, float(d))
                if best is None or d < best:
                    best = d
            rec["gap_to_partner_at_first_collision_m"] = None if best is None else round(best, 3)
            rec["partner_name"] = want
        if r["verdict"] == "pass":
            step = max(1, len(rows) // 600)
            res_c = sc.clearance([(q[1], q[2], q[3]) for q in rows[::step]])
            sg = [g for g, own, _ in res_c if own >= 0 and sc.is_scored[own]]
            rec["min_gap_to_scored_m"] = min(sg) if sg else None
        out["trajectories"].append(rec)
    p = ARENA / "results/analysis/a21_regress.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    fails = [r for r in out["routes"] if not r["passed"]]
    passes = [r for r in out["routes"] if r["passed"]]
    print(f"routes: {len(out['routes'])}; failing with scored overlap {sum(1 for r in fails if r['scored_overlaps'])}/{len(fails)}; "
          f"passing with scored overlap {sum(1 for r in passes if r['scored_overlaps'])}/{len(passes)}")
    for r in out["routes"]:
        print(f"  {r['prompt']:10s} {r['variant'][-26:]:26s} pass={r['passed']!s:5s} min_gap={r['min_gap']} "
              f"scored={r['scored_overlaps']} unscored={r['unscored_overlaps'][:3]}")
    col = [t for t in out["trajectories"] if t.get("gap_to_partner_at_first_collision_m") is not None]
    print(f"trajectories: {len(out['trajectories'])}; with a scored collision {len(col)}; "
          f"gap to partner at first collision: max {max(t['gap_to_partner_at_first_collision_m'] for t in col) if col else None}")
    for t in col:
        print(f"  {t['variant'][-34:]:34s} {t['partner_name']:24s} t={t['first_collision_t']} gap={t['gap_to_partner_at_first_collision_m']}")
    ps = [t for t in out["trajectories"] if t["verdict"] == "pass"]
    print(f"passing runs: {len(ps)}; min gap to scored objects: " + ", ".join(f"{t['variant'][-22:]}={t['min_gap_to_scored_m']}" for t in ps))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
