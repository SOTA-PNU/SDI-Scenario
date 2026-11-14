"""실험 31 (B-5-6) measures: does each test / graded code compute its route or carry hand-picked coordinates,
and how close did the actual drive come to mapped objects. Reads only; writes results/analysis/b56_route_<tag>.json.

    python3 analysis/c_b56_route_check.py results/turns/prompt_b5_6 gpu_gpt_oss_120b_pB56

hand pairs: 2-number tuples/lists of literals inside the arena walls with max(|x|, |y|) >= 2 (robot-scale constants such
as speeds or margins stay below 2), not counting array shapes given to np.zeros/ones/empty/full/reshape; a hand route is
a list or tuple holding two or more of them.
traj_min_clear: min map_clearance over the trajectory rows (pose point to the nearest mapped object or wall)."""
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

MAP = (ARENA / "base_carter_run_ros_map.py").read_text()
ns = {"np": np, "math": math}
exec(MAP[MAP.index("MISSION = dict("):MAP.index("\nSTART = ")], ns)
exec(MAP[MAP.index("# ---- 지도 기본 코드 ----"):MAP.index("# [EDIT REGION] mission controller")], ns)
WX0, WX1, WY0, WY1 = ns["WALLS"]
RECTS = ns["map_rects"]()
NAMES = [f"box_{i:02d}" for i in range(len(ns["MISSION"]["boxes"]))] + [o[0] for o in ns["FIXED_OBSTACLES"]]


def num(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = num(n.operand)
        return None if v is None else -v
    return None


def is_pair(n):
    if isinstance(n, (ast.Tuple, ast.List)) and len(n.elts) == 2:
        x, y = num(n.elts[0]), num(n.elts[1])
        return (x is not None and y is not None and WX0 <= x <= WX1 and WY0 <= y <= WY1
                and max(abs(x), abs(y)) >= 2.0)
    return False


def code_facts(reg):
    try:
        tree = ast.parse(reg)
    except SyntaxError:
        return {"parse": False}
    shape_args = {id(c.args[0]) for c in ast.walk(tree) if isinstance(c, ast.Call) and c.args
                  and getattr(c.func, "attr", getattr(c.func, "id", "")) in ("zeros", "ones", "empty", "full", "reshape")}
    pairs = [n for n in ast.walk(tree) if is_pair(n) and id(n) not in shape_args]  # np.zeros((0, 3)) is a shape
    routes = [n for n in ast.walk(tree) if isinstance(n, (ast.List, ast.Tuple)) and sum(is_pair(e) for e in n.elts) >= 2]
    calls = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
              {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    return {"parse": True, "hand_pairs": [[num(p.elts[0]), num(p.elts[1])] for p in pairs],
            "hand_route_lists": len(routes),
            "map_calls": sorted(c for c in calls if c in ("map_clearance", "segment_clearance", "map_rects", "unmapped_points")),
            "imports": sorted(i for i in imports if i in ("heapq", "collections"))}


def nearest(x, y):
    """The mapped object or wall closest to (x, y) and its distance."""
    d = [math.hypot(max(r[0] - x, 0, x - r[1]), max(r[2] - y, 0, y - r[3])) for r in RECTS]
    d += [x - WX0, WX1 - x, y - WY0, WY1 - y]
    names = NAMES + ["wall_west", "wall_east", "wall_south", "wall_north"]
    i = int(np.argmin(d))
    return names[i], round(d[i], 2)


def traj_facts(variant):
    cands = list((ARENA / "results").glob(f"turns/*/isaac_logs/{variant}_trajectory.csv")) + \
        [ARENA / "results" / f"{variant}_trajectory.csv"]
    f = next((c for c in cands if c.exists()), None)
    if f is None:
        return {}
    rows = [(float(r["t"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(f.open())]
    cl = [(ns["map_clearance"](x, y), t, x, y) for t, x, y in rows]
    c, t, x, y = min(cl)
    return {"traj_min_clear": round(c, 2), "at": [round(t, 1), round(x, 2), round(y, 2)], "near": nearest(x, y)}


def main(run_dir, tag):
    run_dir = ARENA / run_dir
    logs = run_dir / "isaac_logs"
    out = {}
    for tf in sorted(run_dir.glob(f"{tag}_a*_turns.json")):
        T = json.loads(tf.read_text())
        rows = []
        for turn in T["turns"]:
            items = [("test", t["test"], t["variant"], t) for t in turn.get("tests") or []] + \
                    [("graded", None, turn["variant"], turn)]
            for kind, n, v, rec in items:
                f = logs / f"{v}_carter_run.py"
                reg = A.region_of(f.read_text()) if f.exists() else ""
                row = {"turn": turn["turn"], "kind": kind, "n": n, "verdict": rec.get("verdict"),
                       "partners": [p.rsplit("/", 1)[-1] for p in rec.get("partners") or []],
                       "dist": rec.get("final_dist_m"), "runtime_error": rec.get("runtime_error"),
                       **code_facts(reg), **traj_facts(v)}
                rows.append(row)
                print(f"a{T['sample']} t{row['turn']} {kind:6} {n or '':>2} {row['verdict'] or '-':5} "
                      f"{','.join(row['partners']) or '-':24} dist={row['dist']} hand_pairs={len(row.get('hand_pairs', []))} "
                      f"hand_routes={row.get('hand_route_lists')} map={row.get('map_calls')} imp={row.get('imports')} "
                      f"traj_min={row.get('traj_min_clear')} at={row.get('at')} near={row.get('near')}"
                      + (f" ERR={row['runtime_error'][:60]}" if row.get("runtime_error") else ""))
        out[f"a{T['sample']}"] = rows
    (ARENA / "results" / "analysis" / f"b56_route_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
