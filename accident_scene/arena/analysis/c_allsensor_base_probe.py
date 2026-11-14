"""Write the end-to-end probe for the B-2 sensor base code (validation only, never shown to a model).

The EDIT REGION holds analysis/c_sensor_base_all_snippet.py verbatim (the exact text the B-2
prompt will carry) plus a probe controller that calls every base-code function on every step,
times each call, and at two rest poses (start, then +90 deg in place) saves what the functions
return. The robot never steers by sensor data. Check offline with c_allsensor_base_probe_check.py.

    python3 analysis/c_allsensor_base_probe.py   -> c_allsensor_base_probe_carter_run.py
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUTDIR = ARENA / "results" / "analysis" / "c_allsensor_base_probe"
SNIPPET = (ARENA / "analysis" / "c_sensor_base_all_snippet.py").read_text()

PROBE = r'''

# ---- probe controller (validation only) ----
import json as _json
import time as _time

_OUT = "__OUTDIR__"
_T = {"lidar2d_front": [], "lidar2d_back": [], "lidar3d": [], "camera": []}
_PS = {"phase": 0, "yaw0": None, "t_phase": 0.0}


def _timed(key, fn, *a):
    t0 = _time.perf_counter()
    out = fn(*a)
    _T[key].append((_time.perf_counter() - t0) * 1000.0)
    return out


def _psnap(name, t, pose, f2, b2, p3, img):
    import os
    os.makedirs(_OUT, exist_ok=True)
    np.save(f"{_OUT}/snap{name}_front2d.npy", f2)
    np.save(f"{_OUT}/snap{name}_back2d.npy", b2)
    np.save(f"{_OUT}/snap{name}_lidar3d.npy", p3)
    np.save(f"{_OUT}/snap{name}_front2d_world.npy", to_world(f2, pose))
    np.save(f"{_OUT}/snap{name}_back2d_world.npy", to_world(b2, pose))
    np.save(f"{_OUT}/snap{name}_lidar3d_world.npy", to_world(p3, pose))
    np.save(f"{_OUT}/snap{name}_cam.npy", img)
    boxes = MISSION["boxes"]
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rel = []
    for b in boxes:   # box centre at half height, world -> robot
        dx, dy = b["x"] - x, b["y"] - y
        rel.append((c * dx + s * dy, -s * dx + c * dy, b["height"] / 2.0))
    uv, ok = camera_project(np.array(rel))
    stats = {k: {"n": len(v), "mean_ms": sum(v) / len(v), "max_ms": max(v),
                 "p95_ms": sorted(v)[int(0.95 * (len(v) - 1))]} for k, v in _T.items() if v}
    meta = {"name": name, "t": t, "pose": list(pose), "timing": stats,
            "sizes": {"front2d": int(len(f2)), "back2d": int(len(b2)), "lidar3d": int(len(p3)),
                      "cam": None if img is None else list(img.shape)},
            "box_uv": uv.tolist(), "box_ok": ok.tolist()}
    with open(f"{_OUT}/snap{name}.json", "w") as f:
        f.write(_json.dumps(meta, indent=1))
    print("BASEPROBE snapshot", name, "t=%.2f" % t, meta["sizes"],
          {k: round(v["mean_ms"], 2) for k, v in stats.items()}, flush=True)


def controller(t, pose, env):
    f2 = _timed("lidar2d_front", lidar2d_points, "front")
    b2 = _timed("lidar2d_back", lidar2d_points, "back")
    p3 = _timed("lidar3d", lidar3d_points)
    img = _timed("camera", camera_image)
    ready = len(f2) > 0 and len(b2) > 0 and len(p3) > 0 and img is not None and len(_S["cloud"]) == 6
    if _PS["phase"] == 0:
        if t >= 3.0 and ready:
            _psnap("A", t, pose, f2, b2, p3, img)
            _PS["phase"], _PS["yaw0"] = 1, pose[2]
        elif t >= 15.0:
            _psnap("A_partial", t, pose, f2, b2, p3, img)
            return 0.0, 0.0, True
        return 0.0, 0.0, False
    if _PS["phase"] == 1:
        dyaw = math.atan2(math.sin(pose[2] - _PS["yaw0"]), math.cos(pose[2] - _PS["yaw0"]))
        if dyaw < 1.5708:
            return 0.0, 0.5, False
        _PS["phase"], _PS["t_phase"] = 2, t
        return 0.0, 0.0, False
    if t - _PS["t_phase"] >= 1.5:
        _psnap("B", t, pose, f2, b2, p3, img)
        return 0.0, 0.0, True
    return 0.0, 0.0, False


'''
HEAD = ("# [EDIT REGION] mission controller - modify ONLY this block\n"
        "# ===========================================================================\n")
base = (ARENA / "base_carter_run_a2.py").read_text()
b = base.index("# [EDIT REGION] mission controller")
end = "# ========================= [END EDIT REGION] ===============================\n"
e = base.index(end) + len(end)
out = ARENA / "c_allsensor_base_probe_carter_run.py"
out.write_text(base[:b] + HEAD + SNIPPET + PROBE.replace("__OUTDIR__", str(OUTDIR)) + end + base[e:])
print("wrote", out)
