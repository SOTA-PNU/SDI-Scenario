"""Isaac checks before 실험 33 (B-5-8). Two runs through the normal harness (turn_loop.grade, carter_run_locked.sh):
1. wall limit: a controller that sleeps 0.3 s per call, CARTER_WALL_LIMIT_S=70. Expect the wrapper to stop the whole
   simulator group, a 'TimeoutError:' runtime error from both extractors, no simulator left, the lock free again.
2. base_carter_run_ros_map_dbg_heapq.py: the 실험 32 a1 turn-1 answer (NameError: heapq) with the movers, 60 s sim.
   Expect no NameError. Files go to results/analysis/b58_smoke/."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402  (sets TL.CARTER_RUN to carter_run_locked.sh)
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/b58_smoke"
(OUT / "isaac_logs").mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT / "isaac_logs"
BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text()
B, E = TL.BEGIN_MARK, TL.END_MARK
assert str(TL.CARTER_RUN).endswith("carter_run_locked.sh"), TL.CARTER_RUN


def run(variant, region, sim_limit=None, env=None):
    text = BASE[:BASE.index(B)] + region + BASE[BASE.index(E):]
    if sim_limit:
        text = text.replace(A.TIMEOUT_LITERAL, f"timeout_s={sim_limit:.1f},", 1)
    path = ARENA / f"{variant}_carter_run.py"
    assert not path.exists()
    path.write_text(text)
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    t0 = time.time()
    try:
        res = TL.grade(variant)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.move(str(path), OUT / path.name)
    so = next((OUT / "isaac_logs").glob(f"{variant}*stdout.log")).read_text()
    se = next((OUT / "isaac_logs").glob(f"{variant}*stderr.log")).read_text()
    return res, so, se, round(time.time() - t0, 1)


slow = (B + "\n# ===========================================================================\nimport time\n\n\n"
        "def controller(t, pose, env):\n    time.sleep(0.3)\n    print('DBG', round(t, 2))\n    return 0.0, 0.0, False\n\n\n")
res, so, se, wall = run("selftest_b58_walllimit", slow, env={"CARTER_WALL_LIMIT_S": "70"})
left = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
left = [ln for ln in left.splitlines() if "selftest_b58_walllimit_carter_run" in ln and "ps -eo" not in ln]
lock_free = subprocess.run(["flock", "-n", os.environ.get("CARTER_LOCK", "/tmp/carter_run_serial.lock"), "true"]).returncode == 0
rt = A.runtime_error_text(so, se)
print(f"1. wall {wall} s, verdict {res.get('verdict')}, _runtime_error: {res.get('_runtime_error')}")
print(f"   runtime_error_text: {rt}")
print(f"   simulator processes left: {len(left)}, lock free: {lock_free}")
s1 = {"wall_s": wall, "result": res, "runtime_error_text": rt, "left": left, "lock_free": lock_free}

reg = (ARENA / "results/turns/prompt_b5_7/isaac_logs/gpu_gpt_oss_120b_pB57_a1_carter_run.py").read_text()
reg = reg[reg.index(B):reg.index(E)]
res2, so2, se2, wall2 = run("selftest_b58_heapq_base", reg, sim_limit=60.0, env={"ARENA_MOVERS": "v1"})
m = res2.get("metrics", {})
print(f"2. wall {wall2} s, verdict {res2.get('verdict')}, runtime_error {res2.get('_runtime_error')}, "
      f"first_collision {m.get('first_collision_t')} {m.get('collision_partners')}, sim {m.get('sim_time_s')}, "
      f"DBG lines {sum(1 for ln in so2.splitlines() if ln.strip().startswith('DBG'))}")
json.dump({"walllimit": s1, "heapq_base": {"wall_s": wall2, "result": res2}}, open(OUT / "summary.json", "w"),
          ensure_ascii=False, indent=1, default=str)
