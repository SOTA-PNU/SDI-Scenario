"""Did the 실험 33 a0 pass (B-5-8, turn 2) come from the lidar reaction or from timing? (cf. 실험 24 B-4 a1 t2)

Runs the graded answer once more on the fixed map (ARENA_MOVERS unset: no moving obstacles) through the normal
harness, then compares that drive with the graded one: the first time the two trajectories part (more than 0.03 m;
plain reruns spread up to 0.025 m), and the minimum gap to each mover along the graded drive
(analysis/c_mover_clearance.py, chassis outline). One Isaac run. Writes results/gpu_gpt_oss_120b_pB58_a0_t2_fixedmap_*."""
import csv
import math
import os
import shutil
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401  (sets TL.CARTER_RUN to carter_run_locked.sh)
import turn_loop as TL  # noqa: E402

src = ARENA / "results/turns/prompt_b5_8/isaac_logs/gpu_gpt_oss_120b_pB58_a0_t2_carter_run.py"
variant = "gpu_gpt_oss_120b_pB58_a0_t2_fixedmap"
path = ARENA / f"{variant}_carter_run.py"
assert not path.exists() and not (ARENA / f"results/{variant}_result.json").exists()
path.write_text(src.read_text())
os.environ.pop("ARENA_MOVERS", None)
TL.ISAAC_LOG_DIR = ARENA / "results/analysis/b58_fixedmap_isaac_logs"
TL.ISAAC_LOG_DIR.mkdir(parents=True, exist_ok=True)
try:
    res = TL.grade(variant)
finally:
    shutil.move(str(path), TL.ISAAC_LOG_DIR / path.name)
m = res.get("metrics", {})
print("fixed map:", res.get("verdict"), "time_to_goal", m.get("time_to_goal_s"), "final_dist", m.get("final_dist_m"),
      "collisions", m.get("collision_count"), "runtime_error", res.get("_runtime_error"))


def traj(f):
    return [(float(r["t"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(open(f))]


g = traj(ARENA / "results/gpu_gpt_oss_120b_pB58_a0_t2_trajectory.csv")
f = traj(ARENA / f"results/{variant}_trajectory.csv")
fd = {round(t, 3): (x, y) for t, x, y in f}
first = next(((t, x, y, fd[round(t, 3)]) for t, x, y in g if round(t, 3) in fd
              and math.hypot(x - fd[round(t, 3)][0], y - fd[round(t, 3)][1]) > 0.03), None)
print("graded time_to_goal 71.817; first divergence (> 0.03 m):",
      None if first is None else f"t={first[0]:.2f} graded=({first[1]:.2f},{first[2]:.2f}) fixed=({first[3][0]:.2f},{first[3][1]:.2f})")
