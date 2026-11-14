"""Isaac check of the B-5-13 base tool observed_objects() against the real moving boxes (scenario v1). The probe stands at
START (1.7-3.3 m from M1's lane), still for 20 s, turning in place at +0.4 rad/s (20-35 s) and -0.4 rad/s (35-50 s), still
again to 65 s, and writes every call's observed_objects() to a JSON-lines file. Offline, each reported object is compared with
the true movers (arena_movers.position at the object's seen_t): tracking of M1, position and velocity error, id switches,
false objects. Waits for $WAIT_PID first (no parallel Isaac runs). Output results/analysis/b513_objects_isaac.json.

    WAIT_PID=<pid> python3 analysis/b513_objects_isaac_check.py
"""
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401
import arena_movers as AM  # noqa: E402
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/b513_objects_isaac"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT
LOG = OUT / "observed_objects.jsonl"
base = (ARENA / "base_carter_run_ros_map_lidartime_objects.py").read_text()
B, E = "# [EDIT REGION] mission controller - modify ONLY this block\n", "# ========================= [END EDIT REGION]"
REGION = '''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# experimenter probe for observed_objects() (not a model answer)
import json as _json
_F = open(os.environ["B513_PROBE_OUT"], "w")


def controller(t, pose, env):
    objs = observed_objects()
    _F.write(_json.dumps({"t": round(t, 4), "pose": [round(v, 4) for v in pose],
                          "objs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()} for o in objs]}) + "\\n")
    if t >= 65.0:
        _F.flush()
        return 0.0, 0.0, True
    if 20.0 <= t < 35.0:
        return 0.0, 0.4, False
    if 35.0 <= t < 50.0:
        return 0.0, -0.4, False
    return 0.0, 0.0, False


'''
EVAL_ONLY = "--eval-only" in sys.argv          # re-score the saved observed_objects.jsonl without Isaac
wait = os.environ.get("WAIT_PID")
if wait and not EVAL_ONLY:
    while Path(f"/proc/{wait}").exists():
        time.sleep(20)
os.environ["ARENA_MOVERS"] = "v1"
os.environ["CARTER_CONTACT_ALL_LINKS"] = "1"
os.environ["B513_PROBE_OUT"] = str(LOG)
variant = "b513_objects_isaac_probe"
if not EVAL_ONLY:
    path = ARENA / f"{variant}_carter_run.py"
    assert not path.exists()
    path.write_text(base[:base.index(B)] + REGION + base[base.index(E):])
    try:
        TL.grade(variant)
    finally:
        shutil.move(str(path), OUT / path.name)

movers = AM.scenario("v1")
rows = [json.loads(l) for l in open(LOG)]


def truth(t):
    out = []
    for m in movers:
        x, y = AM.position(m, t)
        x2, y2 = AM.position(m, t + 0.05)
        x1, y1 = AM.position(m, t - 0.05)
        out.append((m["name"], x, y, (x2 - x1) / 0.1, (y2 - y1) / 0.1))
    return out


stats = {"still": [], "turning": []}
false_calls, calls_m1_in_range, calls_m1_tracked, ids_m1, dup_calls = 0, 0, 0, [], 0
for r in rows:
    t = r["t"]
    phase = "turning" if 20.0 <= t < 50.0 else "still"
    x0, y0 = r["pose"][0], r["pose"][1]
    tm = truth(t - 0.1)
    m1 = [q for q in tm if q[0] == "start_zone"][0]
    in_range = math.hypot(m1[1] - x0, m1[2] - y0) < 7.5
    calls_m1_in_range += in_range
    matched = False
    has_false = False
    near_m1 = []
    for o in r["objs"]:
        tt = truth(o["seen_t"])
        best = min(tt, key=lambda q: math.hypot(q[1] - o["x"], q[2] - o["y"]))
        d = math.hypot(best[1] - o["x"], best[2] - o["y"])
        if d > 1.0:
            has_false = True
            continue
        if best[0] == "start_zone":
            matched = True
            near_m1.append(o)
            stats[phase].append((d, math.hypot(best[3] - o["vx"], best[4] - o["vy"])))
    if near_m1:
        ids_m1.append(min(near_m1, key=lambda o: o["dist"])["id"])     # the id a controller would take for M1
        dup_calls += len(near_m1) > 1
    false_calls += has_false
    calls_m1_tracked += matched and in_range
res = {"calls": len(rows), "calls_M1_within_7.5m": calls_m1_in_range, "calls_M1_tracked": calls_m1_tracked,
       "calls_with_false_object": false_calls, "M1_nearest_id_changes": int(sum(1 for a, b in zip(ids_m1, ids_m1[1:]) if a != b)),
       "calls_with_two_objects_on_M1": dup_calls, "M1_distinct_ids": len(set(ids_m1))}
for ph, v in stats.items():
    if v:
        a = np.array(v)
        res[ph] = {"n": len(v), "pos_err_median": round(float(np.median(a[:, 0])), 3), "pos_err_p95": round(float(np.percentile(a[:, 0], 95)), 3),
                   "vel_err_median": round(float(np.median(a[:, 1])), 3), "vel_err_p95": round(float(np.percentile(a[:, 1], 95)), 3)}
ok = (calls_m1_in_range > 0 and calls_m1_tracked >= 0.9 * calls_m1_in_range and false_calls <= 0.02 * len(rows)
      and all(res[p]["pos_err_median"] <= 0.35 for p in ("still", "turning") if p in res))
res["verdict"] = "ok" if ok else "not ok"
import hashlib  # noqa: E402
res["base_sha16"] = hashlib.sha256(base.encode()).hexdigest()[:16]      # the base the probe ran on
(ARENA / "results/analysis/b513_objects_isaac.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, ensure_ascii=False))
