"""How close does each submitted waypoint route come to an obstacle? (REPORT.md 실험 12)

For every graded submission that holds a literal waypoint list, START plus that list is
walked as a polyline and the smallest distance from the line to any obstacle is taken.
The map is the one analysis/a3_route.py plans on: arena_map.json static structures
(walls, west shelf, both forklifts, NW fixture) plus the 14 MISSION boxes. 0.00 means
the planned line itself runs through an obstacle - before the robot's own width. The boxes
are exact rectangles; the static structures are 0.25 m sweep cells, so near them a value
can be up to one cell (0.25 m) smaller than the true gap (A-4 a1's corner at (-9.8, -4.0)
reads 0.00 and the robot passed it without contact).

    python3 analysis/route_clearance_check.py   (writes results/analysis/route_clearance_check.json)
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(HERE))
import a3_route as R   # noqa: E402

SUBS = [("A-1", "a1", 1, "npu_gpt_oss_120b_pA1_a1"), ("A-1", "a0", 2, "npu_gpt_oss_120b_pA1_a0_t2"),
        ("A-1", "a3", 2, "npu_gpt_oss_120b_pA1_a3_t2"), ("A-1", "a4", 2, "npu_gpt_oss_120b_pA1_a4_t2"),
        ("A-2", "a0", 1, "npu_gpt_oss_120b_pA2_a0"), ("A-2", "a0", 2, "npu_gpt_oss_120b_pA2_a0_t2"),
        ("A-2", "a1", 1, "npu_gpt_oss_120b_pA2_a1"), ("A-2", "a1", 2, "npu_gpt_oss_120b_pA2_a1_t2"),
        ("A-3", "a0", 1, "npu_gpt_oss_120b_pA3_a0"), ("A-3", "a1", 1, "npu_gpt_oss_120b_pA3_a1"),
        ("A-4", "a0", 1, "npu_gpt_oss_120b_pA4_a0"), ("A-4", "a1", 1, "npu_gpt_oss_120b_pA4_a1"),
        ("A-4 furio", "a0", 1, "npu_furio_pA4_a0"), ("A-4 furio", "a0", 2, "npu_furio_pA4_a0_t2"),
        ("A-4 furio", "a1", 1, "npu_furio_pA4_a1")]


def region(p):
    m = re.search(r"# \[EDIT REGION\][^\n]*\n(.*?)\n# =+ \[END EDIT REGION\]", p.read_text(), re.S)
    return m.group(1) if m else ""


def first_pair_list(code):
    code = re.sub(r"#[^\n]*", "", code)
    m = re.search(r"\[\s*\(\s*-?\d", code)
    if not m:
        return None
    i, depth = m.start(), 0
    for j in range(i, len(code)):
        depth += {"[": 1, "]": -1}.get(code[j], 0)
        if depth == 0:
            break
    body = re.sub(r"GOAL\s*\[\s*0\s*\]", "8.0", code[i:j + 1])
    body = re.sub(r"GOAL\s*\[\s*1\s*\]", "16.5", body)
    return [(float(x), float(y)) for x, y in
            re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", body)]


def main():
    occ = R.build()
    clear = R.clearance_fn(occ)
    rows = []
    for exp, run, turn, v in SUBS:
        pts = first_pair_list(region(ARENA / f"{v}_carter_run.py"))
        if not pts:
            rows.append({"prompt": exp, "run": run, "turn": turn, "variant": v, "waypoints": None})
            continue
        pts = [(-9.0, -10.8)] + pts
        worst = (9e9, None)
        for p, q in zip(pts, pts[1:]):
            n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05))
            for k in range(n + 1):
                x, y = p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n
                c = clear(x, y)
                if c < worst[0]:
                    worst = (c, [round(x, 2), round(y, 2)])
        rows.append({"prompt": exp, "run": run, "turn": turn, "variant": v, "waypoints": pts[1:],
                     "min_clearance_m": round(worst[0], 2), "at": worst[1]})
        print(f"{exp} {run} t{turn}: {len(pts) - 1} waypoints, min clearance {worst[0]:.2f} m at {worst[1]}")
    (ARENA / "results" / "analysis" / "route_clearance_check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
