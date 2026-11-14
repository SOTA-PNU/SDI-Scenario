"""Closed-loop phase sweep of a B-5 model controller in the 2D stand-in (analysis/claude_ros_sim2d.py): the base's map
block (WALLS, FIXED_OBSTACLES, map_clearance, segment_clearance, unmapped_points, ...) is executed into the controller's
namespace so model codes that call the base map functions run unchanged. Contact is judged twice: with the outline that
includes the wheels (x -0.59..0.14, y +-0.25; any touch) and with the chassis only (x -0.59..0.13, y +-0.18; what the
Isaac harness scores). The stand-in has no contact physics: a mover that touches a wheel is not pushed off, so the
chassis-only count is pessimistic where Isaac pushes the robot away. Mover positions only draw scans and judge contact.

    python3 analysis/c_llm_sim_sweep.py RUN_FILE BASE_FILE [--phases v1|random:N[:seed]] [--out results/analysis/...json]
"""
import argparse
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA / "analysis"))
import claude_ros_sim2d as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_file"); ap.add_argument("base_file")
ap.add_argument("--phases", default="v1"); ap.add_argument("--out", default="")
a = ap.parse_args()
base = Path(a.base_file).read_text()
MAP = base[base.index("# ---- 지도 기본 코드 ----"):base.index("# ---- DBG 기록 ----")]
_orig = S.load_region


def load_region(run_file):
    region, ns = _orig(run_file)
    exec(MAP, ns)
    return region, ns


S.load_region = load_region
OUTLINES = {"outline_with_wheels": (-0.59, 0.14, -0.25, 0.25), "chassis_only": (-0.59, 0.13, -0.18, 0.18)}
sets = S.phase_sets(a.phases)
out = {"run_file": a.run_file, "base_file": a.base_file, "phases": a.phases, "results": {}}
for name, ol in OUTLINES.items():
    S.OUTLINE = ol
    tally, rows = {}, []
    for k, ms in enumerate(sets):
        r = S.run(a.run_file, ms, quiet=True)
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        rows.append({"k": k, "verdict": r["verdict"], "t": r["t"], "hit": r["hit"], "phases": [round(m.get("phase", 0), 2) for m in ms]})
        print(f"[{name} {k}] {r['verdict']} t={r['t']} hit={r['hit']}", flush=True)
    out["results"][name] = {"tally": tally, "runs": rows}
    print(f"== {name}: {tally} of {len(sets)}", flush=True)
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
