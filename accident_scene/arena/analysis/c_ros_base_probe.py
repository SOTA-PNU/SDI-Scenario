"""Write the B-2-1 end-to-end probe (validation only, never shown to a model).

Same probe controller as c_allsensor_base_probe.py, but its EDIT REGION holds only the probe
controller: the sensor functions it calls come from base_carter_run_ros.py, above the EDIT
REGION, exactly as a B-2-1 answer would use them. The runner is built with splice() on the ros
base - the path agent_loop --base-file takes for test drives and grading.

    python3 analysis/c_ros_base_probe.py   -> c_ros_base_probe_carter_run.py
    check: python3 analysis/c_allsensor_base_probe_check.py --dir results/analysis/c_ros_base_probe
"""
import ast
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import splice as SP  # noqa: E402

src = (ARENA / "analysis" / "c_allsensor_base_probe.py").read_text()
PROBE = next(n.value.value for n in ast.parse(src).body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROBE")
OUTDIR = ARENA / "results" / "analysis" / "c_ros_base_probe"
SP.BASE = (ARENA / "base_carter_run_ros.py").read_text()
runner = SP.splice(PROBE.replace("__OUTDIR__", str(OUTDIR)))
assert "def lidar2d_points" in runner.split("# [EDIT REGION]")[0], "sensor block must sit above the EDIT REGION"
assert "def lidar2d_points" not in runner.split("# [EDIT REGION]")[1], "the probe must not redefine the base code"
out = ARENA / "c_ros_base_probe_carter_run.py"
out.write_text(runner)
print("wrote", out)
