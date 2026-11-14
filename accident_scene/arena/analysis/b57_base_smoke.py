"""Isaac smoke test of base_carter_run_ros_map_dbg.py (실험 32 base) before any model sees it.

Splices the B-5-6 a0 turn-2 graded EDIT REGION (실험 31) into the new base, runs it once through the normal harness
(turn_loop.grade, the serial carter-run lock) with the grading movers (ARENA_MOVERS=v1), and compares it with the
original graded run: first contact time and partner, trajectory difference over the common time range, DBG lines.
Run files and logs go to results/analysis/b57_base_smoke/. One Isaac run (about 40 s)."""
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/b57_base_smoke"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT / "isaac_logs"
TL.ISAAC_LOG_DIR.mkdir(exist_ok=True)
ref_v = "gpu_gpt_oss_120b_pB56_a0_t2"
ref_run = (ARENA / f"results/turns/prompt_b5_6/isaac_logs/{ref_v}_carter_run.py").read_text()
base = (ARENA / "base_carter_run_ros_map_dbg.py").read_text()
b, e = TL.BEGIN_MARK, TL.END_MARK
region = ref_run[ref_run.index(b):ref_run.index(e)]
run_text = base[:base.index(b)] + region + base[base.index(e):]
variant = "selftest_b57_base_smoke"
path = ARENA / f"{variant}_carter_run.py"
assert not path.exists()
path.write_text(run_text)
os.environ["ARENA_MOVERS"] = "v1"
try:
    result = TL.grade(variant)
finally:
    shutil.move(str(path), OUT / path.name)
m = result.get("metrics", {})
ref = json.loads((ARENA / f"results/{ref_v}_result.json").read_text())["metrics"]
print("new :", result.get("verdict"), m.get("first_collision_t"), m.get("collision_partners"), m.get("final_dist_m"),
      "runtime_error:", result.get("_runtime_error"))
print("ref :", "fail", ref.get("first_collision_t"), ref.get("collision_partners"), ref.get("final_dist_m"))


def traj(f):
    return {round(float(r["t"]), 2): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(f))}


tn = traj(ARENA / f"results/{variant}_trajectory.csv")
tr = traj(ARENA / f"results/{ref_v}_trajectory.csv")
common = sorted(set(tn) & set(tr))
dev = max(math.hypot(tn[t][0] - tr[t][0], tn[t][1] - tr[t][1]) for t in common)
print(f"trajectory: {len(common)} common rows up to {common[-1]} s, max position difference {dev:.3f} m")
so = next(TL.ISAAC_LOG_DIR.glob(f"{variant}*stdout.log")).read_text()
ref_so = (ARENA / f"results/turns/prompt_b5_6/isaac_logs/{ref_v}.carter-run.stdout.log").read_text()
dn = [ln for ln in so.splitlines() if ln.strip().startswith("DBG")]
dr = [ln for ln in ref_so.splitlines() if ln.strip().startswith("DBG")]
print(f"DBG lines: new {len(dn)} (ref {len(dr)}); first new: {dn[0][:100] if dn else None}")
print("new DBG times:", [ln.split()[1][:6] for ln in dn[:14]], "...", [ln.split()[1][:6] for ln in dn[-3:]])
json.dump({"result": result, "dev_m": dev, "dbg_new": len(dn), "dbg_ref": len(dr)},
          open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=str)
