"""Warn, before Isaac runs, when a controller ignores the sub-plane obstacles.

The scene's decisive fact is a product of two numbers that live in different
files: the box heights in base's MISSION table, and the default z of
env.raycast_scan() in the harness. A box whose top is below that z is never
reported by the scanner, so a controller that neither treats it as a map
obstacle nor lowers its own scan plane is going to drive into it.

Both numbers are read from source, never hard-coded here: the heights by
parsing base_carter_run.py's MISSION literal, the plane by reading the default
argument of carter_env.raycast_scan. If the harness changes, this check
changes with it instead of quietly going stale.

HONEST LIMIT, to be repeated in the paper: this checker was written knowing the
answer. What it demonstrates is that this failure is of a mechanically
checkable KIND - not that we built a general detector for unknown scenes.

    python3 analysis/spec_consistency_check.py 'npu_*_carter_run.py' [--csv out.csv]
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ARENA))
BASE = os.path.join(ARENA, "base_carter_run.py")
ENV = os.path.join(REPO, "levels", "common", "carter_env.py")


def scan_plane_z(path=ENV):
    """The default z of raycast_scan, read from the harness' own signature."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "raycast_scan":
            args = node.args.args[len(node.args.args) - len(node.args.defaults):]
            for a, d in zip(args, node.args.defaults):
                if a.arg == "z":
                    return float(ast.literal_eval(d))
    raise RuntimeError(f"raycast_scan(z=...) not found in {path}")


def mission_boxes(path=BASE):
    """The box table as base states it.

    base writes MISSION as dict(...), which literal_eval refuses, so the
    keyword is picked out of the call node instead of evaluating the whole
    expression - no import of base, no execution of it.
    """
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "MISSION" for t in node.targets)):
            continue
        v = node.value
        if isinstance(v, ast.Dict):
            return ast.literal_eval(v)["boxes"]
        if isinstance(v, ast.Call):
            for kw in v.keywords:
                if kw.arg == "boxes":
                    return ast.literal_eval(kw.value)
    raise RuntimeError("MISSION boxes not found in base")


def sub_plane_boxes():
    z = scan_plane_z()
    boxes = mission_boxes()
    return z, [(i, b) for i, b in enumerate(boxes) if b["height"] < z]


def edit_region(src):
    b = src.find("# [EDIT REGION]")
    e = src.find("# ========================= [END EDIT REGION]")
    return src[b:e] if (b >= 0 and e > b) else src


def lowered_plane(region, min_height):
    """Does the controller ever scan below the lowest sub-plane box top?

    Three shapes appear in real submissions and all three count:
      * a literal z= below min_height;
      * z= bound to a name, with a literal below min_height elsewhere in the
        region (the multi-plane fans are written as a table of tuples and
        indexed in a loop, so the value never reaches the call site as a
        literal);
      * z= bound to a name with no such literal - reported as dynamic and NOT
        counted, because nothing static can say where that plane ends up.
    """
    lowered, dynamic = False, False
    for m in re.finditer(r"raycast_scan\s*\(([^)]*)\)", region):
        zm = re.search(r"\bz\s*=\s*([^,)\s]+)", m.group(1))
        if not zm:
            continue                       # default plane, i.e. not lowered
        try:
            if float(zm.group(1)) < min_height:
                lowered = True
        except ValueError:
            dynamic = True
    if dynamic and not lowered:
        # a plane value below the lowest box top, written anywhere in the region
        for lit in re.findall(r"(?<![\d.])0\.\d+", region):
            if float(lit) < min_height:
                lowered = True
                break
    return lowered, dynamic


def reads_box_table(region):
    """Does the controller consume MISSION's box table at run time?

    A controller that indexes MISSION["boxes"] has every height in hand; it
    cannot be said to have ignored the table. This is the route the passing
    track-A entries took, and a coordinate-literal test misses it entirely -
    they plan waypoints around the boxes without ever writing a box centre.
    """
    return bool(re.search(r"MISSION\s*\[\s*[\"']boxes[\"']\s*\]", region)
                or re.search(r"\bboxes\b", region) and re.search(r"MISSION", region))


def treats_as_map(region, subs, tol=0.05):
    """Are the sub-plane boxes present as coordinates the controller plans around?

    A box counts as handled when both its centre coordinates appear as numeric
    literals within tol. Track A may do this; track B may not, which is why the
    warning is reported alongside the track rather than folded into it. Note
    this test alone is too narrow - a controller can plan around a box without
    ever writing its centre - which is what reads_box_table() covers.
    """
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", region)]
    found = []
    for i, b in subs:
        hx = any(abs(n - b["x"]) <= tol for n in nums)
        hy = any(abs(n - b["y"]) <= tol for n in nums)
        if hx and hy:
            found.append(i)
    return found


def check(path, z, subs, min_height):
    src = open(path).read()
    region = edit_region(src)
    lowered, dynamic = lowered_plane(region, min_height)
    mapped = treats_as_map(region, subs)
    reads_table = reads_box_table(region)
    handled = lowered or reads_table or len(mapped) == len(subs)
    return dict(
        run_id=os.path.basename(path)[: -len("_carter_run.py")],
        warned=not handled,
        route=("lowered_plane" if lowered else "reads_box_table" if reads_table
               else "mapped_coords" if len(mapped) == len(subs) else "none"),
        lowered_scan_plane=lowered,
        dynamic_scan_z=dynamic,
        reads_box_table=reads_table,
        mapped_boxes="|".join(str(i) for i in mapped),
        mapped_count=len(mapped),
        sub_plane_count=len(subs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--runs-csv", default=os.path.join(ARENA, "results", "runs.csv"),
                    help="verdicts to measure agreement against")
    a = ap.parse_args()

    z, subs = sub_plane_boxes()
    min_h = min(b["height"] for _, b in subs)
    print(f"스캔 평면 z={z} m (carter_env.raycast_scan 기본값)")
    print(f"평면 아래 박스 {len(subs)}개: " +
          ", ".join(f"{i}(h={b['height']})" for i, b in subs))

    actual = {}
    if os.path.exists(a.runs_csv):
        for r in csv.DictReader(open(a.runs_csv)):
            actual[r["variant"]] = r["failure_class"]

    rows = []
    for pat in a.patterns:
        for f in sorted(glob.glob(os.path.join(ARENA, pat) if not os.path.isabs(pat) else pat)):
            r = check(f, z, subs, min_h)
            r["actual_failure_type"] = actual.get(r["run_id"], "")
            r["agreement"] = ("경고∩함정충돌" if r["warned"] and
                              r["actual_failure_type"] == "trap_collision"
                              else "무경고∩함정충돌" if r["actual_failure_type"] == "trap_collision"
                              else "경고∩기타" if r["warned"] else "무경고∩기타")
            rows.append(r)
    if not rows:
        print("대상 없음", file=sys.stderr)
        return 1

    warned = sum(r["warned"] for r in rows)
    traps = [r for r in rows if r["actual_failure_type"] == "trap_collision"]
    caught = sum(r["warned"] for r in traps)
    print(f"\n{len(rows)}건 중 사전 경고 {warned}건 ({warned / len(rows) * 100:.0f}%)")
    if traps:
        print(f"실제 함정 충돌 {len(traps)}건 중 경고된 것 {caught}건")
    graded = [r for r in rows if r["actual_failure_type"]]
    if graded:
        print(f"채점된 {len(graded)}건 기준: 경고 {sum(r['warned'] for r in graded)}건, "
              f"통과 {sum(1 for r in graded if r['actual_failure_type'] == 'pass')}건 "
              f"(그중 경고 {sum(1 for r in graded if r['actual_failure_type'] == 'pass' and r['warned'])}건)")
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
