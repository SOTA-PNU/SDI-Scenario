"""Write the B-4 end-to-end probe (validation only, never shown to a model).

The c_allsensor_base_probe.py probe controller, spliced into base_carter_run_ros_noenv.py the way
agent_loop --base-file does it. Besides what the sensor functions return, it records what env is
inside controller() and what calling env.raycast_scan() does there (B-4 gives env = None).

    python3 analysis/c_noenv_base_probe.py   -> c_noenv_base_probe_carter_run.py
    check: python3 analysis/c_allsensor_base_probe_check.py --dir results/analysis/c_noenv_base_probe
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
OUTDIR = ARENA / "results" / "analysis" / "c_noenv_base_probe"
probe = PROBE.replace("__OUTDIR__", str(OUTDIR))
head = "def controller(t, pose, env):\n"
assert probe.count(head) == 1
probe = probe.replace(head, head + '''    if "env" not in _ENV:
        _ENV["env"] = repr(env)
        try:
            env.raycast_scan()
            _ENV["raycast"] = "callable"
        except Exception as exc:  # noqa: BLE001
            _ENV["raycast"] = f"{type(exc).__name__}: {exc}"
''')
meta_key = '"box_uv": uv.tolist(), "box_ok": ok.tolist()}'
assert probe.count(meta_key) == 1
probe = probe.replace(meta_key, '"box_uv": uv.tolist(), "box_ok": ok.tolist(), "env_check": dict(_ENV)}')
probe = "_ENV = {}\n" + probe
SP.BASE = (ARENA / "base_carter_run_ros_noenv.py").read_text()
runner = SP.splice(probe)
assert "def lidar2d_points" in runner.split("# [EDIT REGION]")[0]
out = ARENA / "c_noenv_base_probe_carter_run.py"
out.write_text(runner)
print("wrote", out)
