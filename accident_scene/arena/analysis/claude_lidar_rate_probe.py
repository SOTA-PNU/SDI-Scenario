"""How often does each 2D lidar deliver a new scan to controller()? Three short Isaac runs on the fixed map (no movers):
standstill reading front first, standstill reading back first, and turning in place (w = 0.6) reading front first.
Each counts new array objects from lidar2d_points() over 6 s of sim time. Logs go to results/analysis/claude_lidar_rate/.

    python3 analysis/claude_lidar_rate_probe.py
"""
import os
import re
import shutil
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/claude_lidar_rate"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT
base = (ARENA / "base_carter_run_ros.py").read_text()
B, E = "# [EDIT REGION] mission controller - modify ONLY this block\n", "# ========================= [END EDIT REGION]"
REGION = '''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
_C = {"front": [0, None, None], "back": [0, None, None]}


def controller(t, pose, env):
    for which in ORDER:
        p = lidar2d_points(which)
        if p is not _C[which][1]:
            _C[which][1] = p
            _C[which][0] += 1
            if _C[which][2] is None and len(p):
                _C[which][2] = t
    if abs(t - round(t)) < 1e-6 or t > 5.99:
        print("DBG t=%.3f front=%d back=%d first_front=%s first_back=%s" % (t, _C["front"][0], _C["back"][0], _C["front"][2], _C["back"][2]))
    return 0.0, W, t > 6.0


'''
os.environ.pop("ARENA_MOVERS", None)
for name, order, w in (("still_front_first", '("front", "back")', 0.0), ("still_back_first", '("back", "front")', 0.0),
                       ("turning_front_first", '("front", "back")', 0.6)):
    variant = f"claude_lidar_rate_{name}"
    region = REGION.replace("_C = {", f"ORDER = {order}\nW = {w}\n_C = {{", 1)
    path = ARENA / f"{variant}_carter_run.py"
    assert not path.exists()
    path.write_text(base[:base.index(B)] + region + base[base.index(E):])
    try:
        TL.grade(variant)
    finally:
        shutil.move(str(path), OUT / path.name)
    so = next(OUT.glob(f"{variant}*stdout.log")).read_text()
    lines = [ln.strip() for ln in so.splitlines() if ln.strip().startswith("DBG")]
    last = lines[-1] if lines else "no DBG"
    m = re.search(r"t=([\d.]+) front=(\d+) back=(\d+)", last)
    if m:
        t, f, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
        print(f"{name}: over {t:.2f} s front {f} new scans ({f / t:.1f} Hz), back {b} ({b / t:.1f} Hz) | {last}")
    else:
        print(name, last)
