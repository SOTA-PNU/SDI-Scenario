"""B-5-14 60 s Isaac smoke; run outside the preparation sandbox.

One simulator launch, always through carter_run_locked.sh. The controller uses
only the static map and observed_objects(). M1 ground truth is loaded ONLY by
the post-run evaluator, never inserted into the runner/base/model prompt.
An ordinary mission FAIL (e.g. the 60 s cutoff before reaching GOAL) is not an
execution error. Smoke PASS requires a complete run, no execution/tracker
errors, median check_motion <= 2 ms and an observed approaching-M1 interval
with decreasing path-follow candidate t_hit_obj. Missing evidence is FAIL.

python3 analysis/b514_check_isaac_smoke.py
python3 analysis/b514_check_isaac_smoke.py --eval-only
python3 analysis/b514_check_isaac_smoke.py --selftest  # no Isaac or network
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results/analysis/b514_check_isaac"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check.py"
VARIANT = "b514_check_smoke"
BEGIN = "# [EDIT REGION] mission controller - modify ONLY this block\n"
END = "# ========================= [END EDIT REGION]"

# Probe instrumentation may use files and a wall clock; candidate selection
# depends only on sensor/map values and simulation time.
REGION = r'''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
import json as _probe_json
import time as _probe_time

_PROBE_F = open(os.environ["B514_PROBE_OUT"], "x", buffering=1)
_PROBE = {"path": None, "i": 1, "tick": -1, "cmd": (0.0, 0.0)}


def _probe_plan():
    res, need = 0.5, 1.0
    x0, x1, y0, y1 = WALLS
    nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
    pt = lambda c: (x0 + (c[0] + 0.5) * res, y0 + (c[1] + 0.5) * res)
    cell = lambda p: (int((p[0] - x0) / res), int((p[1] - y0) / res))
    start, goal = cell(START), cell(GOAL)
    costs, came, opened, free_cache = {start: 0.0}, {start: None}, [(0.0, start)], {}
    def free(c):
        if c not in free_cache:
            free_cache[c] = (0 <= c[0] < nx and 0 <= c[1] < ny
                             and map_clearance(*pt(c)) >= need)
        return free_cache[c]
    while opened:
        _, c = heapq.heappop(opened)
        if c == goal:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = c[0] + di, c[1] + dj
                if n == c or not free(n) or segment_clearance(pt(c), pt(n)) < need:
                    continue
                nc = costs[c] + math.hypot(di, dj) * res
                if nc < costs.get(n, float("inf")):
                    costs[n], came[n] = nc, c
                    heapq.heappush(opened, (nc + math.hypot(n[0] - goal[0], n[1] - goal[1]) * res, n))
    if goal not in came:
        raise RuntimeError("static-map A* found no path")
    path, c = [tuple(GOAL[:2])], goal
    while c is not None:
        path.append(pt(c))
        c = came[c]
    path = [tuple(START[:2])] + path[::-1]
    # Greedy line-of-sight simplification, always verified against the map.
    simple, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and segment_clearance(path[i], path[j]) < need:
            j -= 1
        simple.append(path[j])
        i = j
    if any(segment_clearance(p, q) < need for p, q in zip(simple, simple[1:])):
        raise RuntimeError("static-map path has insufficient clearance")
    return simple


def _probe_follow(pose):
    x, y, yaw = pose
    p = _PROBE["path"]
    while _PROBE["i"] < len(p) - 1 and math.hypot(p[_PROBE["i"]][0] - x, p[_PROBE["i"]][1] - y) < 0.35:
        _PROBE["i"] += 1
    tx, ty = p[_PROBE["i"]]
    dist = math.hypot(tx - x, ty - y)
    e = math.atan2(math.sin(math.atan2(ty - y, tx - x) - yaw), math.cos(math.atan2(ty - y, tx - x) - yaw))
    v, w = (0.0 if abs(e) > 0.35 else min(0.3, 0.5 * dist + 0.1)), max(-0.4, min(0.4, 1.5 * e))
    return v, w, (tx, ty)


def controller(t, pose, env):
    tick = int((t + 1e-6) / 0.1)
    if tick == _PROBE["tick"]:
        return *_PROBE["cmd"], False
    _PROBE["tick"] = tick
    try:
        if _PROBE["path"] is None:
            _PROBE["path"] = _probe_plan()
        v, w, target = _probe_follow(pose)
        candidates = [("follow", v, w), ("slow", v * 0.5, w), ("stop", 0.0, 0.0),
                      ("left", 0.0, 0.4), ("right", 0.0, -0.4), ("reverse", -0.15, 0.0)]
        checks = []
        for name, cv, cw in candidates:
            before = _probe_time.perf_counter_ns()
            result = check_motion(cv, cw, horizon_s=3.0)
            elapsed_ms = (_probe_time.perf_counter_ns() - before) / 1e6
            checks.append({"name": name, "v": cv, "w": cw, "ms": elapsed_ms, "result": result})
        safe = [c for c in checks if c["result"]["t_hit"] is None]
        if safe:
            choice = min(safe, key=lambda c: math.hypot(c["result"]["end_pose"][0] - target[0],
                                                       c["result"]["end_pose"][1] - target[1]))
        else:
            choice = max(checks, key=lambda c: c["result"]["t_hit"])
        _PROBE["cmd"] = choice["v"], choice["w"]
        _PROBE_F.write(_probe_json.dumps({"t": t, "pose": list(pose), "objects": observed_objects(),
                                          "checks": checks, "chosen": choice["name"],
                                          "tracking_error": _BASE_OBJ_ST["error"], "error": None}) + "\n")
        return *_PROBE["cmd"], False
    except Exception as exc:
        _PROBE_F.write(_probe_json.dumps({"t": t, "error": repr(exc)}) + "\n")
        raise


'''


def archive(path):
    """Copy before replacing any prior output, preserving its relative path."""
    if path.exists():
        root = ARENA / "results" / ("archive_b514_smoke_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}")
        dest = root / path.relative_to(ARENA)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)


def decreasing_interval(rows, position):
    """Find >=4 consecutive observations of approaching M1 with falling TTC.

    A valid 0.3+ s window has the same matched sensor ID, at most 0.25 s
    between samples, >=0.02 m approach, >=0.1 s TTC decrease and slope
    <=-0.2 s/s. Each step permits one 0.1 s prediction-grid rise for noise.
    M1 must itself move toward the robot (radial speed < -0.02 m/s).
    """
    streak, accepted = [], []
    for row in rows:
        t, pose = row.get("t"), row.get("pose")
        checks = row.get("checks", [])
        follow = next((c["result"] for c in checks if c["name"] == "follow"), None)
        if pose is None or follow is None or follow.get("t_hit_obj") is None:
            streak = []
            continue
        ob = next((o for o in row["objects"] if o["id"] == follow.get("hit_obj_id")), None)
        if not ob or not ob["vel_ok"]:
            streak = []
            continue
        seen_xy = position(ob["seen_t"])
        mx, my = position(t)
        px, py = position(t - 0.05)
        nx, ny = position(t + 0.05)
        dist = math.hypot(mx - pose[0], my - pose[1])
        radial = ((mx - pose[0]) * (nx - px) + (my - pose[1]) * (ny - py)) / (0.1 * max(dist, 1e-9))
        if math.hypot(ob["x"] - seen_xy[0], ob["y"] - seen_xy[1]) > 0.8 or radial >= -0.02:
            streak = []
            continue
        item = {"t": t, "ttc": follow["t_hit_obj"], "distance": dist, "id": ob["id"]}
        if streak and (item["id"] != streak[-1]["id"] or not 0 < t - streak[-1]["t"] <= 0.25
                       or dist > streak[-1]["distance"] + 0.005
                       or item["ttc"] > streak[-1]["ttc"] + 0.100001):
            streak = []
        streak.append(item)
        if len(streak) >= 4 and t - streak[0]["t"] >= 0.299:
            slope = float(np.polyfit([a["t"] for a in streak], [a["ttc"] for a in streak], 1)[0])
            if (streak[0]["ttc"] - item["ttc"] >= 0.099
                    and streak[0]["distance"] - dist >= 0.02 and slope <= -0.2):
                accepted.append({"first": streak[0], "last": item, "samples": len(streak), "slope": slope})
                streak = []
    return accepted


def evaluate(rows, result, returncode, position):
    reasons = []
    errors = [r for r in rows if r.get("error") or r.get("tracking_error")]
    if returncode not in (0, 1):
        reasons.append(f"launcher_exit={returncode}")
    if not result:
        reasons.append("missing_mission_result")
    elif result.get("metrics", {}).get("sim_time_s", 0) < 59.8:
        reasons.append("incomplete_60s_run")
    if errors:
        reasons.append(f"execution_errors={len(errors)}")
    timings = [c["ms"] for r in rows for c in r.get("checks", [])]
    median = float(np.median(timings)) if timings else None
    if median is None or not math.isfinite(median) or median > 2.0:
        reasons.append(f"median_ms={median}")
    valid_rows = [r for r in rows if not r.get("error")]
    intervals = decreasing_interval(valid_rows, position)
    if not intervals:
        reasons.append("no_decreasing_M1_follow_t_hit_obj_interval")
    if len(valid_rows) < 590 or not valid_rows or valid_rows[-1]["t"] < 59.8:
        reasons.append("incomplete_0.1s_probe_log")
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons,
            "execution_errors": len(errors), "calls": len(timings), "rows": len(rows),
            "median_ms": median, "approach_intervals": intervals,
            "mission_verdict": result.get("verdict") if result else None, "launcher_exit": returncode}


def selftest():
    """Synthetic gate tests: no simulator imports, processes, or network."""
    def position(t):
        return 10.0 - 0.1 * t, 0.0
    rows = []
    for i in range(600):
        t = i * 0.1
        x, y = position(t)
        rows.append({"t": t, "pose": [0, 0, 0], "error": None, "tracking_error": None,
                     "objects": [{"id": 1, "x": x, "y": y, "seen_t": t, "vel_ok": True}],
                     "checks": [{"name": "follow", "ms": 0.5,
                                 "result": {"hit_obj_id": 1, "t_hit_obj": max(0, 3.0 - t)}}]})
    result = {"verdict": "fail", "metrics": {"sim_time_s": 60.0}}
    assert evaluate(rows, result, 1, position)["verdict"] == "PASS"
    assert evaluate(rows, result, 75, position)["verdict"] == "FAIL"
    assert evaluate(rows, {}, 1, position)["verdict"] == "FAIL"
    assert evaluate(rows[:30], result, 1, position)["verdict"] == "FAIL"
    bad = json.loads(json.dumps(rows))
    for row in bad:
        row["checks"][0]["result"]["t_hit_obj"] = 1.0
    assert evaluate(bad, result, 1, position)["verdict"] == "FAIL"
    bad = json.loads(json.dumps(rows))
    for row in bad:
        row["checks"][0]["ms"] = 2.01
    assert evaluate(bad, result, 1, position)["verdict"] == "FAIL"
    bad = json.loads(json.dumps(rows))
    bad[10]["tracking_error"] = "test tracker failure"
    assert evaluate(bad, result, 1, position)["verdict"] == "FAIL"
    compile(REGION, "probe_region", "exec")
    print("smoke evaluator selftest: 8/8 PASS (no Isaac)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.eval_only:
        archive(OUT)
        if OUT.exists():
            shutil.rmtree(OUT)  # copied in full above
    OUT.mkdir(parents=True, exist_ok=True)
    path = ARENA / f"{VARIANT}_carter_run.py"
    originals = list((ARENA / "results").glob(f"{VARIANT}*"))
    try:
        if not args.eval_only:
            if path.exists():
                raise RuntimeError(f"probe runner already exists: {path}; inspect its PID before removing")
            for old in originals:
                archive(old)
                if old.is_file():
                    old.unlink()
            base = BASE.read_text()
            text = base[:base.index(BEGIN)] + REGION + base[base.index(END):]
            assert text.count("timeout_s=600.0,") == 1
            text = text.replace("timeout_s=600.0,", "timeout_s=60.0,", 1)
            compile(text, str(path), "exec")
            with path.open("x") as f:
                f.write(text)
            env = {**os.environ, "ARENA_MOVERS": "v1", "CARTER_CONTACT_ALL_LINKS": "1",
                   "CARTER_REPO": str(ARENA.parents[1]), "B514_PROBE_OUT": str(OUT / "observations.jsonl")}
            env.pop("CARTER_WHEEL_TOE_DEG", None)
            try:
                with (OUT / "stdout.log").open("x") as stdout, (OUT / "stderr.log").open("x") as stderr:
                    # The locked wrapper owns its wall timeout and simulator PGID.
                    run = subprocess.run([str(ARENA / "carter_run_locked.sh"),
                                          f"accident_scene/arena/{path.name}"], cwd=ARENA,
                                         env=env, stdout=stdout, stderr=stderr, check=False)
                (OUT / "launch.json").write_text(json.dumps({"returncode": run.returncode,
                    "base_sha16": hashlib.sha256(base.encode()).hexdigest()[:16], "sim_seconds": 60}, indent=2))
            finally:
                shutil.move(str(path), OUT / path.name)
                for artifact in (ARENA / "results").glob(f"{VARIANT}*"):
                    if artifact.is_file():
                        shutil.move(str(artifact), OUT / artifact.name)
        sys.path.insert(0, str(ARENA))
        import arena_movers
        m1 = next(m for m in arena_movers.scenario("v1") if m["name"] == "start_zone")
        launch = json.loads((OUT / "launch.json").read_text())
        log = OUT / "observations.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        result_file = OUT / f"{VARIANT}_result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        summary = evaluate(rows, result, launch["returncode"], lambda t: arena_movers.position(m1, t))
        summary["base_sha16"] = launch["base_sha16"]
    except Exception as exc:
        summary = {"verdict": "FAIL", "reasons": [f"{type(exc).__name__}: {exc}".replace("\n", " ")]}
    for name in ("summary.json", "verdict.txt"):
        archive(OUT / name)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    verdict = summary["verdict"] + ((" " + "; ".join(summary["reasons"])) if summary["reasons"] else "")
    (OUT / "verdict.txt").write_text(verdict + "\n")
    print(verdict, flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
