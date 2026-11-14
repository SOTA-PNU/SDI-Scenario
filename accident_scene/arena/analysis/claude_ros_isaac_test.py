"""Run claude_ros_carter_run.py once in Isaac through the normal harness (turn_loop.grade, carter_run_locked.sh) with the
grading movers (ARENA_MOVERS=v1, or another name), under a fresh variant name; keep logs and the run file in
results/analysis/claude_ros_isaac/. Prints verdict, contact, timing and the DBG lines.

    python3 analysis/claude_ros_isaac_test.py <variant> [movers]
"""
import os
import shutil
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401  (TL.CARTER_RUN = carter_run_locked.sh)
import turn_loop as TL  # noqa: E402

variant = sys.argv[1]
movers = sys.argv[2] if len(sys.argv) > 2 else "v1"
OUT = ARENA / "results/analysis/claude_ros_isaac"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT
path = ARENA / f"{variant}_carter_run.py"
assert not path.exists() and not (ARENA / f"results/{variant}_result.json").exists(), variant
shutil.copy(ARENA / "claude_ros_carter_run.py", path)
if movers:
    os.environ["ARENA_MOVERS"] = movers
else:
    os.environ.pop("ARENA_MOVERS", None)
try:
    res = TL.grade(variant)
finally:
    shutil.move(str(path), OUT / path.name)
m = res.get("metrics", {})
print("verdict", res.get("verdict"), "| time_to_goal", m.get("time_to_goal_s"), "| final", m.get("final_pose"),
      "dist", m.get("final_dist_m"), "yaw_err", m.get("final_yaw_err_rad"), "| collisions", m.get("collision_count"),
      m.get("collision_partners"), "first", m.get("first_collision_t"), "| sim", m.get("sim_time_s"), "wall", res.get("_wall_s"),
      "| runtime_error", res.get("_runtime_error"))
so = next(OUT.glob(f"{variant}*stdout.log")).read_text()
for ln in so.splitlines():
    if ln.strip().startswith("DBG") or "[mission]" in ln:
        print(ln.strip()[:260])
