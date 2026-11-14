"""B-5-14 sensor accuracy: A follows a map path, B stands at START for 60 s.

Default execution is for the GPU host only. Each simulation uses the exclusive
carter_run_locked.sh wrapper. --selftest is entirely offline; --eval-only reads
saved outputs. Truth is added only after each run and never used for control.
"""
import argparse
import fcntl
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

import b514_check_isaac_smoke as smoke

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results/analysis/b514_accuracy_isaac"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v2.py"
BEGIN, END = smoke.BEGIN, smoke.END
NAMES = {"follow", "slow", "stop", "left", "right", "reverse"}

# Reuse only the static-map path planner and follow helper, never old candidate
# selection. The probe moves on the nominal path in A and sends zero in B.
REGION = smoke.REGION.split("def controller(t, pose, env):", 1)[0] + r'''def controller(t, pose, env):
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
            hits = [result[k] for k in ("t_hit_obj", "t_hit_map") if result[k] is not None]
            assert all(math.isfinite(q) and 0 <= q <= 3.0 for q in hits)
            assert result["t_hit"] == (min(hits) if hits else None)
            assert all(math.isfinite(q) for q in result["end_pose"])
            assert not math.isnan(result["min_gap_obj"])
            checks.append({"name": name, "v": cv, "w": cw, "ms": elapsed_ms, "result": result})
        before = _probe_time.perf_counter_ns()
        current = check_motion(0.0, 0.0, horizon_s=0.0)
        current_ms = (_probe_time.perf_counter_ns() - before) / 1e6
        mode = os.environ["B514_ACCURACY_MODE"]
        _PROBE["cmd"] = (v, w) if mode == "A" else (0.0, 0.0)
        _PROBE_F.write(_probe_json.dumps({"t": t, "pose": list(pose), "objects": observed_objects(),
            "scan_points_t": dict(_BASE_OBJ_ST.get("pts_t", {})),
            "checks": checks, "current": current, "current_ms": current_ms,
            "mode": mode, "command": list(_PROBE["cmd"]),
            "tracking_error": _BASE_OBJ_ST["error"], "error": None}) + "\n")
        return *_PROBE["cmd"], False
    except Exception as exc:
        _PROBE_F.write(_probe_json.dumps({"t": t, "error": repr(exc)}) + "\n")
        raise


'''


def archive(path):
    if path.exists():
        root = ARENA / "results" / ("archive_b514_accuracy_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}")
        dest = root / path.relative_to(ARENA)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)


def write_json(path, data):
    archive(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def distribution(values):
    a = np.asarray(values, dtype=float)
    return ({"n": len(values), "min": float(a.min()), "p05": float(np.percentile(a, 5)),
             "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
            if values else {"n": 0})


def enrich(rows, truth_fn=None, gaps_fn=None):
    if truth_fn is None or gaps_fn is None:
        from b514_existing_accuracy import truth_objects, true_body_gaps
        truth_fn, gaps_fn = truth_fn or truth_objects, gaps_fn or true_body_gaps
    out = []
    for row in rows:
        if row.get("error") or "pose" not in row:
            out.append(dict(row))
            continue
        t, pose = row["t"], row["pose"]
        truth = truth_fn(t)
        gaps = gaps_fn(pose, truth)
        pairs = []
        for obj in row.get("objects", []):
            at_seen = truth_fn(obj["seen_t"])
            nearest = min(at_seen, key=lambda o: math.hypot(o["x"] - obj["x"], o["y"] - obj["y"]))
            distance = math.hypot(nearest["x"] - obj["x"], nearest["y"] - obj["y"])
            pairs.append({"id": obj["id"], "nearest_truth": nearest["name"],
                          "matched_truth": nearest["name"] if distance <= 0.8 else None,
                          "center_error_m": distance, "at_seen_t": obj["seen_t"],
                          "velocity_error_m_s": math.hypot(obj["vx"] - nearest["vx"], obj["vy"] - nearest["vy"]),
                          "distance_to_robot_pose_m": math.hypot(obj["x"] - pose[0], obj["y"] - pose[1])})
        current_gap = row.get("current", {}).get("min_gap_obj")
        truth_gap = min(gaps.values())
        error = current_gap - truth_gap if current_gap is not None and math.isfinite(current_gap) else None
        out.append({**row, "truth_objects": truth, "true_body_gaps": gaps, "true_min_gap": truth_gap,
                    "object_matches": pairs, "current_gap_error_m": error,
                    "false_overlap": bool(truth_gap > 0.3 and current_gap is not None and current_gap < 0)})
    return out


def evaluate(mode, rows, result, returncode, log_errors=()):
    """Gate each run separately; a long stopped run cannot dilute A's errors."""
    reasons = []
    mt = result.get("metrics", {}) if result else {}
    first_contact = mt.get("first_collision_t")
    first_contact = float(first_contact) if first_contact is not None else None
    errors = [r for r in rows if r.get("error") or r.get("tracking_error")]
    runtime_errors = list(log_errors)
    if result.get("_runtime_error") or result.get("runtime_error"):
        runtime_errors.append(str(result.get("_runtime_error") or result.get("runtime_error")))
    if returncode not in (0, 1):
        reasons.append(f"launcher_exit={returncode}")
    if not result:
        reasons.append("missing_mission_result")
    sim_time = mt.get("sim_time_s", 0)
    full = isinstance(sim_time, (int, float)) and math.isfinite(sim_time) and sim_time >= 59.8
    collision_end = (mode == "A" and result.get("aborted_after_collision") is True
                     and first_contact is not None and math.isfinite(first_contact) and first_contact > 0)
    if not full and not collision_end:
        reasons.append("incomplete_run_without_accepted_contact_end")
    valid = [r for r in rows if not r.get("error") and "pose" in r]
    if valid:
        times = [r["t"] for r in valid]
        if times[0] > 0.15 or any(not 0 < b - a <= 0.25 for a, b in zip(times, times[1:])):
            reasons.append("invalid_0.1s_sampling")
        if full and (len(valid) < 590 or times[-1] < 59.8):
            reasons.append("incomplete_60s_probe_log")
        if collision_end and times[-1] < first_contact - 0.15:
            reasons.append("probe_ends_before_contact")
        if any({c["name"] for c in r.get("checks", [])} != NAMES or "current" not in r for r in valid):
            reasons.append("missing_candidates_or_current_measurement")
        if mode == "B" and any(r.get("command") != [0.0, 0.0] for r in valid):
            reasons.append("B_not_commanded_stop")
    else:
        reasons.append("missing_observations")
    # After contact, physics can push the body. All accuracy statistics use
    # strictly earlier rows, including B should a contact unexpectedly occur.
    pre = [r for r in valid if first_contact is None or r["t"] < first_contact]
    measured = [r for r in pre if r.get("current_gap_error_m") is not None
                and math.isfinite(r["current_gap_error_m"])]
    gap_errors = [r["current_gap_error_m"] for r in measured]
    abs_errors = [abs(e) for e in gap_errors]
    # Missing detections must not disappear from the acceptance median: that
    # would let ten accurate frames hide hundreds of missed nearby objects.
    # Keep the finite error distribution for diagnostics, but count every
    # unmeasured pre-contact frame as unbounded error in the threshold test.
    gate_errors = abs_errors + [math.inf] * (len(pre) - len(measured))
    median_abs = float(np.median(gate_errors)) if gate_errors else None
    if len(measured) < 10:
        reasons.append("insufficient_current_gap_measurements")
    if median_abs is None or median_abs > 0.15:
        reasons.append(f"median_absolute_current_error_m={median_abs}")
    far = [r for r in measured if r["true_min_gap"] > 0.3]
    false_n = sum(r["false_overlap"] for r in far)
    false_rate = false_n / len(far) if far else None
    if false_rate is None or false_rate > 0.05:
        reasons.append(f"false_overlap_fraction_given_true_gap_gt_0.3={false_rate}")
    timings = [c["ms"] for r in valid for c in r.get("checks", [])] + [r["current_ms"] for r in valid if "current_ms" in r]
    median_ms = float(np.median(timings)) if timings else None
    if median_ms is None or not all(math.isfinite(t) and t >= 0 for t in timings) or median_ms > 2.0:
        reasons.append(f"median_call_ms={median_ms}")
    if errors or runtime_errors:
        reasons.append(f"execution_errors={len(errors) + len(runtime_errors)}")
    warnings = []
    for r in pre:
        follow = next((c["result"] for c in r.get("checks", []) if c["name"] == "follow"), {})
        if follow.get("t_hit_obj") is not None:
            matched = next((m["matched_truth"] for m in r.get("object_matches", [])
                            if m["id"] == follow.get("hit_obj_id")), None)
            warnings.append({"t": r["t"], "t_hit_obj": follow["t_hit_obj"], "matched_truth": matched,
                             "seconds_before_actual_contact": first_contact - r["t"] if first_contact is not None else None})
    window = [{"t": r["t"], "matches": r.get("object_matches", [])} for r in pre if 6.3 <= r["t"] <= 9.0]
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons, "mode": mode,
            "gate": {"median_absolute_current_error_max_m": 0.15, "conditional_false_overlap_max": 0.05,
                     "execution_errors_max": 0, "median_call_max_ms": 2.0},
            "rows": len(rows), "precontact_rows": len(pre), "current_gap_measurements": len(measured),
            "unmeasured_precontact_rows": len(pre) - len(measured),
            "current_gap_error_estimated_minus_true_m": distribution(gap_errors),
            "absolute_current_gap_error_m": distribution(abs_errors),
            "gate_median_absolute_current_error_m": median_abs if median_abs is not None and math.isfinite(median_abs) else None,
            "gate_median_unbounded": bool(median_abs is not None and not math.isfinite(median_abs)),
            "gate_unmeasured_as_infinite_error": True,
            "false_overlap_n": false_n, "true_gap_gt_0.3_n": len(far),
            "false_overlap_fraction_given_true_gap_gt_0.3": false_rate,
            "false_overlap_fraction_all_measured": false_n / len(measured) if measured else None,
            "median_call_ms": median_ms, "execution_errors": len(errors) + len(runtime_errors),
            "runtime_log_errors": runtime_errors, "first_collision_t": first_contact,
            "first_follow_warning": warnings[0] if warnings else None,
            "first_positive_follow_warning": next((w for w in warnings if w["t_hit_obj"] > 0), None),
            "follow_warnings": warnings, "matches_6.3_to_9.0": window,
            "mission_verdict": result.get("verdict"), "sim_time_s": sim_time, "launcher_exit": returncode,
            "contact_end_accepted": bool(collision_end)}


def evaluate_directory(mode, directory):
    variant = f"b514_accuracy_{mode.lower()}"
    launch = json.loads((directory / "launch.json").read_text())
    observations = directory / "observations.jsonl"
    rows = [json.loads(line) for line in observations.read_text().splitlines()] if observations.exists() else []
    result_path = directory / f"{variant}_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    enriched = enrich(rows)
    truth_log = directory / "truth_observations.jsonl"
    archive(truth_log)
    truth_log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in enriched))
    blob = "\n".join((directory / n).read_text(errors="replace") for n in ("stdout.log", "stderr.log") if (directory / n).exists())
    # Normalize only the filename used by the inherited targeted traceback detector.
    log_errors = smoke.runtime_log_errors(blob.replace(variant + "_carter_run.py", smoke.VARIANT + "_carter_run.py"))
    summary = evaluate(mode, enriched, result, launch["returncode"], log_errors)
    summary["launch"] = launch
    summary["truth_source"] = "arena_movers.scenario(v1)/position at simulation time; post-run only"
    write_json(directory / "summary.json", summary)
    return summary


def launch_one(mode, base):
    directory = OUT / mode
    directory.mkdir(parents=True, exist_ok=True)
    variant = f"b514_accuracy_{mode.lower()}"
    runner = ARENA / f"{variant}_carter_run.py"
    if runner.exists():
        raise RuntimeError(f"probe runner already exists: {runner}; inspect its PID")
    for old in (ARENA / "results").glob(f"{variant}*"):
        archive(old)
        if old.is_file():
            old.unlink()
    text = base[:base.index(BEGIN)] + REGION + base[base.index(END):]
    assert text.count("timeout_s=600.0,") == 1
    text = text.replace("timeout_s=600.0,", "timeout_s=60.0,", 1)
    # A may end after physical contact. B must run its whole 60-second window.
    if mode == "B":
        assert text.count("abort_after_collision_s=5.0") == 1
        text = text.replace("abort_after_collision_s=5.0", "abort_after_collision_s=None", 1)
    compile(text, str(runner), "exec")
    with runner.open("x") as f:
        f.write(text)
    env = {**os.environ, "ARENA_MOVERS": "v1", "CARTER_CONTACT_ALL_LINKS": "1",
           "CARTER_REPO": str(ARENA.parents[1]), "B514_PROBE_OUT": str(directory / "observations.jsonl"),
           "B514_ACCURACY_MODE": mode}
    env.pop("CARTER_WHEEL_TOE_DEG", None)
    metadata = {"mode": mode, "base_file": BASE.name,
                "base_sha16": hashlib.sha256(base.encode()).hexdigest()[:16],
                "runner_sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "sim_seconds": 60,
                "accuracy_script_sha16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "prompt_sha16": hashlib.sha256((ARENA / "PROMPT_B5_14.md").read_bytes()).hexdigest()[:16]}
    try:
        with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
            run = subprocess.run([str(ARENA / "carter_run_locked.sh"), f"accident_scene/arena/{runner.name}"],
                                 cwd=ARENA, env=env, stdout=stdout, stderr=stderr, check=False)
        metadata["returncode"] = run.returncode
        write_json(directory / "launch.json", metadata)
    finally:
        shutil.move(str(runner), directory / runner.name)
        for artifact in (ARENA / "results").glob(f"{variant}*"):
            if artifact.is_file():
                shutil.move(str(artifact), directory / artifact.name)
    return evaluate_directory(mode, directory)


def selftest():
    import copy
    import tempfile
    from types import SimpleNamespace
    rows = []
    for i in range(600):
        t = i * 0.1
        checks = [{"name": name, "ms": 0.5, "result": {"t_hit_obj": None, "hit_obj_id": None}} for name in NAMES]
        rows.append({"t": t, "pose": [0, 0, 0], "checks": checks, "current": {"min_gap_obj": 1.1},
                     "current_ms": 0.2, "command": [0.0, 0.0], "current_gap_error_m": 0.1,
                     "true_min_gap": 1.0, "false_overlap": False, "objects": [], "object_matches": []})
    result = {"verdict": "fail", "metrics": {"sim_time_s": 60.0, "first_collision_t": None}}
    checks = []
    def test(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
    test("B_complete_zero_command_pass", evaluate("B", rows, result, 1)["verdict"] == "PASS")
    contact = {"verdict": "fail", "aborted_after_collision": True,
               "metrics": {"sim_time_s": 15.0, "first_collision_t": 9.667}}
    test("A_contact_end_pass", evaluate("A", rows[:147], contact, 1)["verdict"] == "PASS")
    test("B_contact_end_not_full_fail", evaluate("B", rows[:147], contact, 1)["verdict"] == "FAIL")
    bad = copy.deepcopy(rows)
    for row in bad:
        row["current_gap_error_m"] = -0.151
    test("negative_bias_does_not_evade_absolute_gate", evaluate("B", bad, result, 1)["verdict"] == "FAIL")
    bad = copy.deepcopy(rows)
    for row in bad[:31]:
        row["false_overlap"] = True
    test("false_overlap_over_five_percent_fail", evaluate("B", bad, result, 1)["verdict"] == "FAIL")
    test("runtime_error_fail", evaluate("B", rows, {**result, "_runtime_error": "test"}, 1)["verdict"] == "FAIL")
    test("busy_wrapper_fail", evaluate("B", rows, result, 75)["verdict"] == "FAIL")
    bad = copy.deepcopy(rows)
    for row in bad:
        for c in row["checks"]:
            c["ms"] = 2.01
    test("slow_calls_fail", evaluate("B", bad, result, 1)["verdict"] == "FAIL")
    test("missing_samples_fail", evaluate("B", [], result, 1)["verdict"] == "FAIL")
    bad = copy.deepcopy(rows)
    for row in bad[:590]:
        row["current_gap_error_m"] = None
        row["current"]["min_gap_obj"] = math.inf
    test("mostly_missing_detections_cannot_pass_on_ten_frames", evaluate("B", bad, result, 1)["verdict"] == "FAIL")
    # Load only map/tracker/helper declarations from the real base, no harness.
    base = BASE.read_text()
    ns = {"np": np, "math": math}
    exec(base[base.index("MISSION = dict("):base.index("\nSTART = ")], ns)
    exec(base[base.index("# ---- 지도 기본 코드 ----"):base.index("# ---- DBG 기록 ----")], ns)
    ns["START"], ns["GOAL"] = ns["MISSION"]["start_xy_yaw"], ns["MISSION"]["goal_xy_yaw"]
    for mode in ("A", "B"):
        with tempfile.TemporaryDirectory(prefix="b514_accuracy_probe_") as tmp:
            probe_log = Path(tmp) / "observations.jsonl"
            ns["os"] = SimpleNamespace(environ={"B514_PROBE_OUT": str(probe_log), "B514_ACCURACY_MODE": mode})
            exec(REGION, ns)
            try:
                for t in (0.0, 0.05, 0.1):
                    ns["_BASE_POSES"][:] = [(t, *ns["START"])]
                    cmd = ns["controller"](t, ns["START"], None)
                    assert all(math.isfinite(v) for v in cmd[:2])
            finally:
                ns["_PROBE_F"].close()
            recorded = [json.loads(line) for line in probe_log.read_text().splitlines()]
            test(mode + "_actual_probe_six_candidates_and_current", len(recorded) == 2 and all(
                 len(r["checks"]) == 6 and "current" in r and not r["error"] for r in recorded)
                 and (mode != "B" or all(r["command"] == [0.0, 0.0] for r in recorded)))
    print(json.dumps({"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks}))
    return 0 if all(c["pass"] for c in checks) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    lock_path = ARENA / "results/analysis/.b514_accuracy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    run_lock = lock_path.open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FAIL accuracy check already active", file=sys.stderr)
        return 75
    if not args.eval_only:
        for mode in ("a", "b"):
            runner = ARENA / f"b514_accuracy_{mode}_carter_run.py"
            if runner.exists():
                print(f"FAIL probe runner already exists: {runner}", file=sys.stderr)
                return 73
        archive(OUT)
        if OUT.exists():
            shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for mode in ("A", "B"):
        try:
            summaries[mode] = (evaluate_directory(mode, OUT / mode) if args.eval_only
                               else launch_one(mode, BASE.read_text()))
        except Exception as exc:
            summaries[mode] = {"verdict": "FAIL", "reasons": [f"{type(exc).__name__}: {exc}".replace("\n", " ")]}
            (OUT / mode).mkdir(exist_ok=True)
            write_json(OUT / mode / "summary.json", summaries[mode])
        print(mode, summaries[mode]["verdict"], flush=True)
        # A numerical accuracy failure does not suppress B's needed diagnosis.
        # Busy/runtime failure is an unsafe basis for another simulator launch.
        if not args.eval_only and (summaries[mode].get("launcher_exit") not in (0, 1)
                                  or summaries[mode].get("execution_errors", 1) != 0):
            break
    reasons = [f"{m}: " + "; ".join(summaries[m]["reasons"]) if m in summaries else f"{m}: not executed"
               for m in ("A", "B") if m not in summaries or summaries[m]["verdict"] != "PASS"]
    summary = {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons, "runs": summaries}
    write_json(OUT / "summary.json", summary)
    verdict = summary["verdict"] + ((" " + " | ".join(reasons)) if reasons else "")
    archive(OUT / "verdict.txt")
    (OUT / "verdict.txt").write_text(verdict + "\n")
    print(verdict, flush=True)
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
