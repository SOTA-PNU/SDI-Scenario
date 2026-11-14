"""Gate G1 analysis (REPORT.md 실험 15): grid gap bins vs live PhysX overlap bins.

Reads results/analysis/a21_g1_probe.json (analysis/a21_g1_probe.py, one Isaac run) and compares,
per pose and per robot-outline box, the live bin (overlap with the box grown by 0 / 0.05 /
0.10 m) with the 0.05 m grid's gap from a21_scene. Accept when no pose is more than one bin
apart, i.e. the grid is right to one 0.05 m cell (the prompt states a 0.05 m error). The
grown box is square-cornered, so near corners it reaches up to 0.07 m. Also prints the wall faces the rays measured and the
contact offsets. Writes results/analysis/a21_g1_check.json.
"""
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene as S  # noqa: E402


def live_bin(b):
    if b["0.0"]:
        return 0
    if b["0.05"]:
        return 1
    if b["0.1"]:
        return 2
    return 3


def grid_bin(g):
    if g <= 0.0:
        return 0
    if g <= 0.05:
        return 1
    if g <= 0.10:
        return 2
    return 3


def main():
    sc = S.Scene()
    d = json.loads((ARENA / "results/analysis/a21_g1_probe.json").read_text())
    rows, bad = [], []
    for rec in d["poses"]:
        pose = tuple(rec["pose"])
        for k, box in enumerate(S.FOOTPRINT):
            gap, own, _ = sc.clearance([pose], boxes=(box,))[0]
            lb, gb = live_bin(rec["boxes"][k]), grid_bin(gap)
            rows.append((lb, gb, gap))
            if abs(lb - gb) > 1:
                bad.append({"pose": pose, "box": k, "grid_gap": gap, "live_bin": lb, "grid_bin": gb,
                            "live": rec["boxes"][k]})
    n = len(rows)
    agree = sum(1 for lb, gb, _ in rows if lb == gb)
    off1 = sum(1 for lb, gb, _ in rows if abs(lb - gb) == 1)
    print(f"G1: {n} box queries; same bin {agree} ({agree / n:.1%}), one bin apart {off1}, "
          f"more than one bin apart: {len(bad)}")
    conf = {}
    for lb, gb, _ in rows:
        conf[f"live{lb}_grid{gb}"] = conf.get(f"live{lb}_grid{gb}", 0) + 1
    print("  confusion (bins 0: overlap, 1: <=0.05, 2: <=0.10, 3: >0.10):", dict(sorted(conf.items())))
    for b in bad[:10]:
        print("  BAD", b)
    walls = sc.walls_inner_face()
    print("  wall faces, grid:", walls)
    for name, r in d["rays"].items():
        print(f"  ray {name}: face {r['face']} hit {r['hit']}")
    offs = d["offsets"]
    vals = {}
    for p, o in offs.items():
        key = (o["contactOffset"], o["restOffset"])
        vals.setdefault(key, []).append(p)
    for (co, ro), ps in vals.items():
        print(f"  contactOffset={co} restOffset={ro}: {len(ps)} colliders, e.g. {ps[:3]}")
    out = {"n": n, "same_bin": agree, "one_bin_apart": off1, "bad": bad, "confusion": conf,
           "walls_grid": walls, "rays": d["rays"], "offsets": {f"{k[0]}|{k[1]}": v for k, v in vals.items()},
           "accepted": not bad}
    (ARENA / "results/analysis/a21_g1_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("ACCEPTED" if not bad else "NOT ACCEPTED")


if __name__ == "__main__":
    main()
