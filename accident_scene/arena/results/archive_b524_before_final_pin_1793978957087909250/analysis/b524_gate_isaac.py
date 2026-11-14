"""B-5-24 200-second gate. Default launches one locked Isaac run on the GPU host.

--selftest uses CPU fixtures only. --eval-only reads saved output. The reference supplies exactly three model hooks; the actual v12 _robot_only
dispatches the base-owned loop. Instrumentation preserves its returned tuple. Ground truth is attached after
the run solely for inspection. Goal completion is recorded, never required.
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
import uuid

import numpy as np

import b514_check_isaac_smoke as smoke

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results/analysis/b524_gate_isaac"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v12.py"
VARIANT = "b524_gate_probe"
BEGIN, END = smoke.BEGIN, smoke.END

REFERENCE = ARENA / "analysis/b524_reference.py"
BASE_SHA256 = "B524_V12_HASH_PENDING"
REFERENCE_SHA256 = "29418d703ffe7ce16b8e88c42f7ccbca15c643ac114d9efa4e4aa3f98c2c2f3b"

INSTRUMENTATION = r'''
import json as _gate_json
import time as _gate_time
_GATE_OBS = open(os.environ["B524_OBSERVATIONS_OUT"], "x", buffering=1)
_GATE_TIMES = open(os.environ["B524_TIMINGS_OUT"], "x", buffering=1)
_GATE_AUDIT = {"tick": None, "t": None, "calls": []}
_GATE_ACTUAL_CHOOSE = choose_command


def _b524_record_choose(v_follow, w_follow, target_xy, *args, **kwargs):
    before = _gate_time.perf_counter_ns()
    result = _GATE_ACTUAL_CHOOSE(v_follow, w_follow, target_xy, *args, **kwargs)
    ms = (_gate_time.perf_counter_ns() - before) / 1e6
    _GATE_TIMES.write(_gate_json.dumps({"t": _GATE_AUDIT["t"], "ms": ms}) + "\n")
    _GATE_AUDIT["calls"].append({"nominal": [v_follow, w_follow],
        "target_xy": list(target_xy), "command": list(result[:2]), "info": result[2], "ms": ms})
    return result


choose_command = _b524_record_choose


def _b524_audit_step(actual_robot_only):
    def step(t, pose, env):
        _GATE_AUDIT["t"], _GATE_AUDIT["calls"] = t, []
        try:
            result = actual_robot_only(t, pose, env)
            cv, cw, done = result
            calls = _GATE_AUDIT["calls"]
            assert len(calls) == 1, "base loop must call chooser exactly once"
            call = calls[0]
            detail = _BASE_LOOP_ST["last"]
            assert tuple(call["command"]) == (cv, cw), "base return differs from chooser"
            assert tuple(detail["command"]) == (cv, cw) and detail["done"] == done
            assert tuple(call["nominal"]) == tuple(detail["nominal"])
            assert tuple(call["target_xy"]) == tuple(detail["target_xy"])
            assert math.isfinite(cv) and math.isfinite(cw), "nonfinite command"
            tick = int((t + 1e-6) / 0.1)
            if tick != _GATE_AUDIT["tick"] or done or detail.get("error"):
                _GATE_AUDIT["tick"] = tick
                _GATE_OBS.write(_gate_json.dumps({"t": t, "pose": list(pose),
                    "objects": observed_objects(), "scan_points_t": dict(_BASE_OBJ_ST.get("pts_t", {})),
                    "nominal": call["nominal"], "target_xy": call["target_xy"],
                    "command": [cv, cw], "info": call["info"], "choose_ms": call["ms"],
                    "done": done, "base_loop": detail, "chooser_calls": len(calls),
                    "tracking_error": _BASE_OBJ_ST["error"], "error": detail.get("error")}) + "\n")
            return result
        except Exception as exc:
            _GATE_OBS.write(_gate_json.dumps({"t": t, "error": repr(exc)}) + "\n")
            raise
    return step
'''


def load_reference():
    text = REFERENCE.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("reference source hash does not match the validated hooks")
    return text


def make_region(reference):
    if hashlib.sha256(reference.encode()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("reference source hash mismatch")
    assert reference.count(BEGIN) == 1 and reference.count(END) == 1
    body = reference[reference.index(BEGIN) + len(BEGIN):reference.index(END)]
    return BEGIN + body + "\n" + INSTRUMENTATION + "\n"


def make_runner(base, reference):
    if hashlib.sha256(base.encode()).hexdigest() != BASE_SHA256:
        raise RuntimeError("v12 base source hash mismatch")
    text = base[:base.index(BEGIN)] + make_region(reference) + base[base.index(END):]
    assert text.count("timeout_s=600.0,") == 1
    text = text.replace("timeout_s=600.0,", "timeout_s=200.0,", 1)
    assert text.count("ctrl = _robot_only\n") == 1
    text = text.replace("ctrl = _robot_only\n", "ctrl = _b524_audit_step(_robot_only)\n", 1)
    compile(text, "b524_gate_probe_carter_run.py", "exec")
    return text


def archive(path):
    if path.exists():
        root = ARENA / "results" / ("archive_b524_gate_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}")
        dest = root / path.relative_to(ARENA)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)


def save_json(path, data):
    archive(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def evaluate(rows, timings, result, returncode, log_errors=()):
    reasons = []
    mt = result.get("metrics", {})
    runtime_errors = list(log_errors)
    if result.get("_runtime_error") or result.get("runtime_error"):
        runtime_errors.append(str(result.get("_runtime_error") or result.get("runtime_error")))
    errors = [r for r in rows if r.get("error") or r.get("tracking_error")]
    if returncode not in (0, 1):
        reasons.append(f"launcher_exit={returncode}")
    if not result:
        reasons.append("missing_mission_result")
    collisions = mt.get("collision_count")
    if not finite_number(collisions) or collisions != 0:
        reasons.append(f"collision_count={collisions}")
    sim_time, goal_time = mt.get("sim_time_s"), mt.get("time_to_goal_s")
    goal_reached = finite_number(goal_time) and 0 <= goal_time <= 200.0
    full = finite_number(sim_time) and 199.8 <= sim_time <= 201.5
    # A controller may finish early only after legitimate goal attainment.
    early_goal = (result.get("verdict") == "pass" and goal_reached
                  and finite_number(sim_time) and goal_time <= sim_time <= 201.5)
    if not full and not early_goal:
        reasons.append("premature_or_invalid_run_duration")
    valid = [r for r in rows if not r.get("error") and "pose" in r]
    if not valid:
        reasons.append("missing_observations")
    else:
        ts = [r["t"] for r in valid]
        if (not all(finite_number(t) for t in ts) or ts[0] > 0.15
                or any(not 0 < b - a <= 0.25 for a, b in zip(ts, ts[1:]))):
            reasons.append("invalid_0.1s_observation_sampling")
        if full and (len(valid) < 1990 or ts[-1] < 199.8):
            reasons.append("incomplete_200s_observations")
        if not full and early_goal and (not valid[-1].get("done") or ts[-1] < goal_time
                                       or sim_time - ts[-1] > 1.2):
            reasons.append("missing_legitimate_goal_finish_observation")
        if any(not isinstance(r.get("info"), dict) or not r["info"].get("chosen") for r in valid):
            reasons.append("missing_choose_command_info")
    elapsed = [r["ms"] for r in timings if "ms" in r]
    median = float(np.median(elapsed)) if elapsed else None
    if (median is None or not all(finite_number(t) and t >= 0 for t in elapsed)
            or median > 3.0):
        reasons.append(f"median_choose_ms={median}")
    if valid:
        tts = [r.get("t") for r in timings]
        if (len(timings) < len(valid) or not tts or not all(finite_number(t) for t in tts)
                or tts[0] > 0.05 or tts[-1] < valid[-1]["t"]
                or any(not 0 < b - a <= 0.05 for a, b in zip(tts, tts[1:]))):
            reasons.append("incomplete_control_tick_timings")
    if errors or runtime_errors:
        reasons.append(f"execution_errors={len(errors) + len(runtime_errors)}")
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons,
            "execution_errors": len(errors) + len(runtime_errors), "runtime_log_errors": runtime_errors,
            "collision_count": collisions, "median_choose_ms": median, "choose_calls": len(elapsed),
            "observation_rows": len(rows), "sim_time_s": sim_time, "goal_reached": goal_reached,
            "time_to_goal_s": goal_time, "mission_verdict": result.get("verdict"),
            "early_goal_finish_accepted": bool(early_goal and not full), "launcher_exit": returncode,
            "gate_limits": {"collision_count": 0, "execution_errors": 0, "median_choose_ms_max": 3.0,
                            "sim_seconds": 200, "goal_required": False}}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evaluate_saved():
    from b514_existing_accuracy import truth_objects, true_body_gaps
    launch = json.loads((OUT / "launch.json").read_text())
    rows = read_lines(OUT / "observations.jsonl")
    timings = read_lines(OUT / "choose_timings.jsonl")
    result_file = OUT / f"{VARIANT}_result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    truth_path = OUT / "truth_observations.jsonl"
    archive(truth_path)
    with truth_path.open("w") as f:
        for row in rows:
            if "pose" in row:
                truth = truth_objects(row["t"])
                row = {**row, "truth_objects": truth, "true_body_gaps": true_body_gaps(row["pose"], truth)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    blob = "\n".join((OUT / name).read_text(errors="replace") for name in ("stdout.log", "stderr.log") if (OUT / name).exists())
    log_errors = smoke.runtime_log_errors(blob.replace(VARIANT + "_carter_run.py", smoke.VARIANT + "_carter_run.py"))
    summary = evaluate(rows, timings, result, launch["returncode"], log_errors)
    summary["launch"] = launch
    summary["truth_source"] = "arena_movers.position at simulation time, post-run analysis only"
    return summary


def run_gate(base, run_process=subprocess.run):
    runner = ARENA / f"{VARIANT}_carter_run.py"
    if runner.exists():
        raise RuntimeError(f"probe runner exists: {runner}; inspect its PID")
    for old in (ARENA / "results").glob(f"{VARIANT}*"):
        archive(old)
        if old.is_file():
            old.unlink()
    reference = load_reference()
    text = make_runner(base, reference)
    compile(text, str(runner), "exec")
    with runner.open("x") as f:
        f.write(text)
    env = {**os.environ, "ARENA_MOVERS": "v1", "CARTER_CONTACT_ALL_LINKS": "1",
           "CARTER_REPO": str(ARENA.parents[1]), "B524_OBSERVATIONS_OUT": str(OUT / "observations.jsonl"),
           "B524_TIMINGS_OUT": str(OUT / "choose_timings.jsonl")}
    env.pop("CARTER_WHEEL_TOE_DEG", None)
    metadata = {"run_id": os.environ.get("B524_GATE_RUN_ID") or uuid.uuid4().hex,
                "reference_file": "analysis/b524_reference.py",
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                "reference_sha16": hashlib.sha256(reference.encode()).hexdigest()[:16],
                "control_entry": "_robot_only -> _base_control_loop -> model hooks -> choose_command",
                "base_file": BASE.name, "base_sha16": hashlib.sha256(base.encode()).hexdigest()[:16],
                "runner_sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "sim_seconds": 200,
                "script_sha16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "prompt_sha16": hashlib.sha256((ARENA / "PROMPT_B5_24.md").read_bytes()).hexdigest()[:16]}
    try:
        with (OUT / "stdout.log").open("x") as stdout, (OUT / "stderr.log").open("x") as stderr:
            run = run_process([str(ARENA / "carter_run_locked.sh"), f"accident_scene/arena/{runner.name}"],
                              cwd=ARENA, env=env, stdout=stdout, stderr=stderr, check=False)
        metadata["returncode"] = run.returncode
        save_json(OUT / "launch.json", metadata)
    finally:
        shutil.move(str(runner), OUT / runner.name)
        for artifact in (ARENA / "results").glob(f"{VARIANT}*"):
            if artifact.is_file():
                shutil.move(str(artifact), OUT / artifact.name)
    return evaluate_saved()


def selftest():
    global ARENA, OUT
    import copy
    import tempfile
    from types import SimpleNamespace
    checks = []
    def test(name, value):
        checks.append({"name": name, "pass": bool(value)})
    rows = [{"t": i * 0.1, "pose": [-9.0, -10.8, 1.0138], "info": {"chosen": "stop"}, "done": False}
            for i in range(2000)]
    timings = [{"t": i / 60, "ms": 1.2} for i in range(12000)]
    result = {"verdict": "fail", "metrics": {"sim_time_s": 201.0, "collision_count": 0, "time_to_goal_s": None}}
    test("full_200s_no_goal_pass", evaluate(rows, timings, result, 1)["verdict"] == "PASS")
    test("collision_fail", evaluate(rows, timings, {**result, "metrics": {**result["metrics"], "collision_count": 1}}, 1)["verdict"] == "FAIL")
    test("runtime_error_fail", evaluate(rows, timings, {**result, "_runtime_error": "test"}, 1)["verdict"] == "FAIL")
    test("no_result_fail", evaluate(rows, timings, {}, 0)["verdict"] == "FAIL")
    test("no_observation_fail", evaluate([], timings, result, 1)["verdict"] == "FAIL")
    test("busy_lock_fail", evaluate(rows, timings, result, 75)["verdict"] == "FAIL")
    test("truncated_timing_fail", evaluate(rows, timings[:100], result, 1)["verdict"] == "FAIL")
    test("median_over_3ms_fail", evaluate(rows, [{**t, "ms": 3.01} for t in timings], result, 1)["verdict"] == "FAIL")
    early_rows = copy.deepcopy(rows[:101]); early_rows[-1]["done"] = True
    early_result = {"verdict": "pass", "metrics": {"sim_time_s": 11.0, "collision_count": 0, "time_to_goal_s": 9.0}}
    test("legitimate_early_goal_pass", evaluate(early_rows, timings[:601], early_result, 0)["verdict"] == "PASS")
    test("premature_non_goal_fail", evaluate(early_rows, timings[:601], {**result, "metrics": {**result["metrics"], "sim_time_s": 11.0}}, 1)["verdict"] == "FAIL")
    import ast
    import b515_chooser_sim as SIM
    base, reference = BASE.read_text(), load_reference()
    runner_source = make_runner(base, reference)
    def function_ast(source, names):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name in names}
    test("generated_runner_preserves_actual_v12_dispatch_and_loop",
         function_ast(runner_source, {"_robot_only", "_base_control_loop"})
         == function_ast(base, {"_robot_only", "_base_control_loop"}))
    test("generated_runner_preserves_three_reference_hooks_and_stub",
         function_ast(runner_source, {"plan_path", "lookahead_target", "follow_cmd", "controller"})
         == function_ast(reference, {"plan_path", "lookahead_target", "follow_cmd", "controller"}))
    test("instrumentation_wraps_actual_robot_only",
         "ctrl = _b524_audit_step(_robot_only)\n" in runner_source)
    for name, wrong_base, wrong_reference in (("base", base + "\n", reference),
                                               ("reference", base, reference + "\n")):
        rejected = False
        try:
            make_runner(wrong_base, wrong_reference)
        except RuntimeError:
            rejected = True
        test("changed_" + name + "_hash_rejected", rejected)
    ns = SIM.load_base(BASE)
    ns["_BASE_DBG_STATE"] = {"t": 0.0}
    ns["lidar2d_points"] = lambda which="front": np.zeros((0, 2))
    robot_fn = next(n for n in ast.parse(base).body if isinstance(n, ast.FunctionDef) and n.name == "_robot_only")
    exec(compile(ast.Module(body=[robot_fn], type_ignores=[]), "actual_v12_robot_only", "exec"), ns)
    with tempfile.TemporaryDirectory(prefix="b524_gate_cpu_") as tmp:
        obs, cost = Path(tmp) / "observations.jsonl", Path(tmp) / "timings.jsonl"
        ns["os"] = SimpleNamespace(environ={"B524_OBSERVATIONS_OUT": str(obs), "B524_TIMINGS_OUT": str(cost)})
        exec(make_region(reference), ns)
        step = ns["_b524_audit_step"](ns["_robot_only"])
        try:
            returned = []
            for t in (0.0, 1 / 60, 2 / 60, 0.1):
                result = step(t, ns["START"], None)
                returned.append(result)
                assert result == (*ns["_BASE_LOOP_ST"]["last"]["command"], ns["_BASE_LOOP_ST"]["last"]["done"])
            recorded, costs = read_lines(obs), read_lines(cost)
            test("actual_v12_dispatch_chooses_once_each_tick", len(costs) == 4
                 and all(row["chooser_calls"] == 1 for row in recorded))
            test("actual_reference_plan_is_accepted_by_base", recorded[0]["base_loop"]["plan_attempts"][0]["accepted"]
                 and recorded[0]["base_loop"]["phase"] == "path_follow")
            test("observations_sample_at_point_one_seconds", len(recorded) == 2
                 and [row["t"] for row in recorded] == [0.0, 0.1])
            test("choose_arguments_and_base_diagnostics_match", all(
                 row["nominal"] == row["base_loop"]["nominal"]
                 and row["target_xy"] == row["base_loop"]["target_xy"]
                 and row["command"] == row["base_loop"]["command"] for row in recorded))
            def poisoned_controller(*args):
                raise AssertionError("model controller must not be called")
            ns["controller"] = poisoned_controller
            result = step(0.2, ns["START"], None)
            test("model_controller_override_is_ignored_by_actual_robot_only",
                 len(read_lines(cost)) == 5 and ns["_BASE_LOOP_ST"]["last"]["error"] is None)
            goal = ns["GOAL"]
            original_target = ns["lookahead_target"]
            ns["lookahead_target"] = lambda path, pose: tuple(goal[:2])
            result = step(0.3, ns["START"], None)
            corrected = read_lines(obs)[-1]
            test("bad_GOAL_target_is_corrected_and_actual_choice_preserved",
                 corrected["base_loop"]["target_corrected"]
                 and corrected["base_loop"]["target_reject_reason"] == "too_far"
                 and corrected["target_xy"] != list(goal[:2])
                 and 0.3 - 1e-12 <= corrected["base_loop"]["target_corrected_ahead_m"] <= 2.0 + 1e-12
                 and tuple(corrected["command"]) == tuple(result[:2])
                 and corrected["chooser_calls"] == 1)
            ns["lookahead_target"] = original_target
            pose = (goal[0] - 0.1, goal[1], goal[2] - 0.14)
            result = step(0.4, pose, None)
            test("base_goal_stop_done_recorded_without_override", result == (0.0, 0.0, True)
                 and read_lines(obs)[-1]["done"] is True)
            original_choose = ns["_GATE_ACTUAL_CHOOSE"]
            ns["_GATE_ACTUAL_CHOOSE"] = lambda *args, **kw: (.15, 0.0, {"chosen": "escape", "candidates": {"follow": {"t_hit_map": None}}})
            result = step(0.5, pose, None)
            test("base_goal_escape_return_preserved", result == (.15, 0.0, False)
                 and read_lines(obs)[-1]["command"] == [.15, 0.0])
            ns["_GATE_ACTUAL_CHOOSE"] = original_choose
            def broken_target(*args):
                raise ValueError("reference hook fixture")
            ns["lookahead_target"] = broken_target
            result = step(0.6, ns["START"], None)
            error_row = read_lines(obs)[-1]
            test("caught_model_hook_error_is_recorded", error_row["error"]["hook"] == "lookahead_target"
                 and error_row["chooser_calls"] == 1 and error_row["nominal"] == [0.0, 0.0])
            bad_rows = copy.deepcopy(rows); bad_rows[10]["error"] = error_row["error"]
            test("caught_hook_error_fails_zero_error_gate", evaluate(bad_rows, timings, result={"verdict": "fail",
                 "metrics": {"sim_time_s": 201.0, "collision_count": 0}}, returncode=1)["verdict"] == "FAIL")
        finally:
            ns["_GATE_OBS"].close(); ns["_GATE_TIMES"].close()
    saved_arena, saved_out = ARENA, OUT
    with tempfile.TemporaryDirectory(prefix="b524_locked_launch_mock_") as tmp:
        ARENA = Path(tmp) / "repo/accident_scene/arena"
        OUT = ARENA / "results/analysis/b524_gate_isaac"
        OUT.mkdir(parents=True)
        (ARENA / "PROMPT_B5_24.md").write_text("mock prompt")
        launched = []
        def fake_run(command, **kwargs):
            launched.append(command)
            assert Path(command[0]).name == "carter_run_locked.sh"
            assert kwargs["env"]["CARTER_CONTACT_ALL_LINKS"] == "1"
            probe = ARENA / f"{VARIANT}_carter_run.py"
            assert probe.read_text() == runner_source
            (OUT / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in early_rows))
            (OUT / "choose_timings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timings[:601]))
            (ARENA / "results" / f"{VARIANT}_result.json").write_text(json.dumps(early_result))
            return SimpleNamespace(returncode=0)
        try:
            checked = run_gate(base, run_process=fake_run)
            test("mock_one_locked_launch_preserves_artifacts", len(launched) == 1 and checked["verdict"] == "PASS"
                 and (OUT / f"{VARIANT}_carter_run.py").exists() and (OUT / "truth_observations.jsonl").exists()
                 and not (ARENA / f"{VARIANT}_carter_run.py").exists())
            test("launch_records_actual_reference_and_base_hashes", checked["launch"]["reference_sha256"] == REFERENCE_SHA256
                 and checked["launch"]["base_sha16"] == BASE_SHA256[:16])
        finally:
            ARENA, OUT = saved_arena, saved_out
    report = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
              "base_sha16": BASE_SHA256[:16], "base_sha256": BASE_SHA256,
              "reference_sha256": REFERENCE_SHA256,
              "execution": "CPU actual v12 dispatcher and locked-launch substitute only; no GPU, HTTP, or real simulator"}
    save_json(ARENA / "results/analysis/b524_gate_selftest.json", report)
    print(json.dumps(report))
    return 0 if all(c["pass"] for c in checks) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    lock = ARENA / "results/analysis/.b524_gate.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = lock.open("a")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FAIL gate already active", file=sys.stderr)
        return 75
    if not args.eval_only:
        runner = ARENA / f"{VARIANT}_carter_run.py"
        if runner.exists():
            print(f"FAIL probe runner exists: {runner}; inspect its PID", file=sys.stderr)
            return 73
        archive(OUT)
        if OUT.exists():
            shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        summary = evaluate_saved() if args.eval_only else run_gate(BASE.read_text())
    except Exception as exc:
        summary = {"verdict": "FAIL", "reasons": [f"{type(exc).__name__}: {exc}".replace("\n", " ")]}
    save_json(OUT / "summary.json", summary)
    verdict = summary["verdict"] + (" " + "; ".join(summary["reasons"]) if summary["reasons"] else "")
    archive(OUT / "verdict.txt")
    run_id = summary.get("launch", {}).get("run_id", "")
    (OUT / "verdict.txt").write_text(verdict + "\nRUN_ID=" + run_id + "\n")
    print(verdict, flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
