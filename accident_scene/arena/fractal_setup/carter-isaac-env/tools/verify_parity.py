#!/usr/bin/env python3
"""Compare this host's benchmark results against the source-host reference.

    python3 tools/verify_parity.py                  # compare everything found
    python3 tools/verify_parity.py --level 0 3      # only these levels
    python3 tools/verify_parity.py --strict         # require exact float match

What is compared, and why:

  verdict, oracles, collision_count   exact — these are the benchmark's answer
  time_to_goal_s, final_dist_m,       within tolerance — PhysX is deterministic,
  final_yaw_err_rad, sim_time_s       so on the same GPU these reproduce to 3-4
                                      decimals; a different GPU may drift by a
                                      physics step or two

  wall_time_s, frames                 IGNORED — pure wall-clock/render measures.
                                      The source host is an A100 with no RT
                                      cores at ~17 fps; any other GPU differs.
                                      Comparing them flags every correct
                                      reproduction as a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# One physics step is 1/60 s, so the default tolerance is ~3 steps.
DEFAULT_TOL = {
    "time_to_goal_s": 0.05,
    "sim_time_s": 0.05,
    "final_dist_m": 0.02,
    "final_yaw_err_rad": 0.02,
}
EXACT_FIELDS = ("collision_count",)
IGNORED = ("wall_time_s", "frames")


def load_reference(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"reference not found: {path}")
    return json.loads(path.read_text())


def find_results(repo: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for tree in ("levels", "levels_zero2level"):
        for f in sorted((repo / tree).glob("level*/results/*_result.json")):
            stem = f.name[: -len("_result.json")]
            out[f"{tree}/{f.parent.parent.name}/{stem}"] = f
    return out


def compare(ref: dict, got: dict, tol: dict, strict: bool) -> list[str]:
    """Return a list of human-readable mismatches (empty == parity)."""
    diffs: list[str] = []

    if ref["verdict"] != got["verdict"]:
        diffs.append(f"verdict {ref['verdict']} -> {got['verdict']}")

    ref_or = {o["name"]: o["passed"] for o in ref.get("oracles", [])}
    got_or = {o["name"]: o["passed"] for o in got.get("oracles", [])}
    for name in sorted(set(ref_or) | set(got_or)):
        if name not in ref_or:
            diffs.append(f"oracle {name} appeared")
        elif name not in got_or:
            diffs.append(f"oracle {name} vanished")
        elif ref_or[name] != got_or[name]:
            diffs.append(f"oracle {name} {ref_or[name]} -> {got_or[name]}")

    m = got.get("metrics", {})
    for field in EXACT_FIELDS:
        a, b = ref.get(field), m.get(field)
        if a != b:
            diffs.append(f"{field} {a} -> {b}")

    for field, limit in tol.items():
        a, b = ref.get(field), m.get(field)
        if a is None and b is None:
            continue
        if a is None or b is None:
            diffs.append(f"{field} {a} -> {b}")
            continue
        delta = abs(a - b)
        if strict:
            if round(a, 3) != round(b, 3):
                diffs.append(f"{field} {a} -> {b} (delta {delta:.4f}, strict)")
        elif delta > limit:
            diffs.append(f"{field} {a} -> {b} (delta {delta:.4f} > {limit})")
    return diffs


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("CARTER_REPO",
                    str(Path.home() / "carter-scenario-levels")),
                    help="deployed benchmark repo (default: $CARTER_REPO)")
    ap.add_argument("--reference", default=str(here.parent / "reference" / "expected_results.json"),
                    help="source-host reference JSON shipped with this package")
    ap.add_argument("--level", nargs="*", help="only these level numbers, e.g. --level 0 3")
    ap.add_argument("--strict", action="store_true",
                    help="require the physics metrics to match to 3 decimals")
    ap.add_argument("--tol", type=float, default=None,
                    help="override the tolerance for every float field (seconds/metres/radians)")
    args = ap.parse_args()

    repo = Path(args.repo)
    if not (repo / "levels").is_dir():
        sys.exit(f"no benchmark repo at {repo} (set --repo or $CARTER_REPO)")

    reference = load_reference(Path(args.reference))
    results = find_results(repo)
    tol = dict(DEFAULT_TOL)
    if args.tol is not None:
        tol = {k: args.tol for k in tol}

    keys = sorted(set(reference) & set(results))
    if args.level:
        wanted = {f"level{n}" for n in args.level}
        keys = [k for k in keys if k.split("/")[1] in wanted]

    missing = sorted(set(reference) - set(results))
    extra = sorted(set(results) - set(reference))

    if not keys:
        print("no results to compare yet — run the benchmark first "
              "(setup/40_run_benchmark.sh)")
        for k in missing:
            print(f"    not run: {k}")
        return 2

    print(f"reference : {args.reference}")
    print(f"repo      : {repo}")
    print(f"mode      : {'strict (3dp)' if args.strict else 'tolerance ' + repr(tol)}")
    print()
    print(f"{'run':<40} {'verdict':<16} {'time_to_goal_s':<26} result")
    print("-" * 104)

    n_ok = n_bad = 0
    details: list[tuple[str, list[str]]] = []
    for k in keys:
        ref = reference[k]
        got = json.loads(results[k].read_text())
        diffs = compare(ref, got, tol, args.strict)
        rv, gv = ref["verdict"], got["verdict"]
        rt = ref.get("time_to_goal_s")
        gt = got.get("metrics", {}).get("time_to_goal_s")
        verdict_col = rv if rv == gv else f"{rv} -> {gv}"
        ttg_col = f"{rt} -> {gt}" if rt != gt else f"{rt}"
        if diffs:
            n_bad += 1
            print(f"{k:<40} {verdict_col:<16} {ttg_col:<26} MISMATCH")
            details.append((k, diffs))
        else:
            n_ok += 1
            print(f"{k:<40} {verdict_col:<16} {ttg_col:<26} match")

    if details:
        print("\nmismatch detail")
        print("-" * 104)
        for k, diffs in details:
            print(f"  {k}")
            for d in diffs:
                print(f"      {d}")

    if missing:
        print(f"\nnot run on this host ({len(missing)}):")
        for k in missing:
            print(f"    {k}")
    if extra:
        print(f"\nno reference for ({len(extra)}):")
        for k in extra:
            print(f"    {k}")

    print()
    print(f"{n_ok} matched, {n_bad} mismatched, {len(missing)} not run")
    print(f"(ignored on purpose: {', '.join(IGNORED)} — GPU-dependent)")

    if n_bad:
        print("\nA mismatch in `verdict` means the reproduction is wrong: investigate.")
        print("Small drift in the float metrics on a different GPU model can be legitimate;")
        print("re-run once to see whether it is stable, then judge against the oracle detail.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
