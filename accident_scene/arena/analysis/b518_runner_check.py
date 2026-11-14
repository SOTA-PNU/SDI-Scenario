#!/usr/bin/env python3
"""Exercise B-5-18 launch control with local substitutes; no HTTP, GPU, or Isaac."""
import ast
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime

A = Path(__file__).resolve().parents[1]
TAG = "gpu_gpt_oss_120b_pB518"
D = Path("results/turns/prompt_b5_18")
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def script_args(path):
    text = path.read_text().replace("\\\n", " ")
    cmd = text.split("python3 agent_loop.py ", 1)[1].split(" > ", 1)[0]
    return shlex.split(cmd)


def run_sequential(name, outcomes, expected_samples, expected_lines):
    with tempfile.TemporaryDirectory(prefix="b518_sequential_mock_") as tmp:
        root = Path(tmp)
        shutil.copy2(A / "run_b518_sequential.sh", root)
        (root / "agent_loop.py").write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
s = int(args[args.index("--sample") + 1])
spec = json.loads(os.environ["B518_MOCK_CASE"])[s]
assert os.environ.get("CARTER_CONTACT_ALL_LINKS") == "1"
with open("calls.jsonl", "a") as f: f.write(json.dumps(args) + "\\n")
d = pathlib.Path(args[args.index("--out-dir") + 1])
p = d / ("gpu_gpt_oss_120b_pB518_a%d_turns.json" % s)
data = {"tag": "gpu_gpt_oss_120b_pB518", "sample": s, "turns": spec.get("turns", [])}
if spec.get("artifact") == "malformed": p.write_text("{")
elif spec.get("artifact") != "missing": p.write_text(json.dumps(data))
sys.exit(spec["rc"])
''')
        env = dict(os.environ, B518_MOCK_CASE=json.dumps(outcomes))
        proc = subprocess.run(["bash", "run_b518_sequential.sh"], cwd=root, env=env,
                              capture_output=True, text=True, timeout=15)
        calls = [json.loads(s) for s in (root / "calls.jsonl").read_text().splitlines()]
        samples = [int(a[a.index("--sample") + 1]) for a in calls]
        log = (root / D / "sequential.log").read_text()
        check(name, proc.returncode == 0 and samples == expected_samples
              and all(line in log for line in expected_lines)
              and log.splitlines()[-1].endswith(" done"),
              {"samples": samples, "log": log, "returncode": proc.returncode})
        before = {p.relative_to(root).as_posix(): p.read_bytes()
                  for p in (root / D).rglob("*") if p.is_file()}
        retry = subprocess.run(["bash", "run_b518_sequential.sh"], cwd=root, env=env,
                               capture_output=True, text=True, timeout=15)
        after = {p.relative_to(root).as_posix(): p.read_bytes()
                 for p in (root / D).rglob("*") if p.is_file()}
        check(name + "_preserves_existing", retry.returncode == 73 and before == after)


def run_gpu(name, verdict, smoke_rc, expect_run, stale=False, timeout=False, busy=False,
            ps_failure=False, fresh_artifacts=True, artifact_fault=None,
            previous_busy_rounds=0, previous_scan_failure_at=None, sleep_failure=False):
    with tempfile.TemporaryDirectory(prefix="b518_gpu_mock_") as tmp:
        root = Path(tmp)
        script = (A / "run_b518_gpu_steps.sh").read_text()
        if timeout:
            # Accelerate only the copied fixture's deadline; production remains 1800 s.
            assert "MAX_WAIT_S=1800" in script
            script = script.replace("MAX_WAIT_S=1800", "MAX_WAIT_S=0")
        (root / "run_b518_gpu_steps.sh").write_text(script)
        (root / "bin").mkdir()
        curl = root / "bin/curl"
        curl.write_text('#!/bin/sh\necho curl >> mock_order.log\nprintf \'%s\\n\' \'{"data":[{"id":"gpt-oss-120b"}]}\'\n')
        curl.chmod(0o755)
        ps = root / "bin/ps"
        process_config = {"previous_busy_rounds": previous_busy_rounds,
                          "previous_scan_failure_at": previous_scan_failure_at,
                          "agent_busy": busy, "agent_scan_failure": ps_failure}
        ps.write_text("#!/usr/bin/env python3\nconfig = " + repr(process_config) + "\n" + '''
from pathlib import Path
import sys
assert sys.argv[1:] == ['-eo', 'args']
counter = Path('mock_ps_count')
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
with Path('mock_order.log').open('a') as f: f.write('ps:' + str(n) + '\\n')
if n == config['previous_scan_failure_at']:
    raise SystemExit(1)
if n <= config['previous_busy_rounds']:
    print('bash /example/run_b517_gpu_steps.sh')
    print('python3 agent_loop.py')
elif n == config['previous_busy_rounds'] + 1:
    print('COMMAND')
else:
    assert n == config['previous_busy_rounds'] + 2
    if config['agent_scan_failure']:
        raise SystemExit(1)
    print('python3 agent_loop.py' if config['agent_busy'] else 'COMMAND')
''')
        ps.chmod(0o755)
        sleeper = root / "bin/sleep"
        sleeper.write_text("#!/bin/sh\necho sleep:$* >> mock_order.log\nexit "
                           + ("1" if sleep_failure else "0") + "\n")
        sleeper.chmod(0o755)
        (root / "analysis").mkdir()
        verdict_path = Path("results/analysis/b518_gate_isaac/verdict.txt")
        config = dict(verdict=verdict, rc=smoke_rc, fresh=fresh_artifacts, fault=artifact_fault)
        (root / "analysis/b518_gate_isaac.py").write_text("config = " + repr(config) + "\n" + '''
from pathlib import Path
import json, os, shutil
Path('smoke_called').write_text('yes')
d = Path('results/analysis/b518_gate_isaac')
if config['fresh']:
    if d.exists():
        shutil.copytree(d, Path('results/archive_mock_previous_gate'))
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if config['verdict'] is not None:
        verdict_id = 'previous-run' if config['fault'] == 'old_verdict' else os.environ['B518_GATE_RUN_ID']
        (d / 'verdict.txt').write_text(config['verdict'] + '\\nRUN_ID=' + verdict_id + '\\n')
    launch = {'run_id': os.environ['B518_GATE_RUN_ID'],
              'base_file': 'base_carter_run_ros_map_lidartime_objects_check_v7.py'}
    summary = {'verdict': 'PASS' if config['verdict'] == 'PASS' else 'FAIL', 'launch': dict(launch)}
    if config['fault'] == 'old_launch': launch['run_id'] = 'previous-run'
    if config['fault'] == 'old_summary': summary['launch']['run_id'] = 'previous-run'
    if config['fault'] == 'fail_summary': summary['verdict'] = 'FAIL'
    if config['fault'] == 'old_base': launch['base_file'] = 'base_carter_run_ros_map_lidartime_objects_check_v6.py'
    if config['fault'] != 'missing_launch': (d / 'launch.json').write_text(json.dumps(launch))
    if config['fault'] != 'missing_summary': (d / 'summary.json').write_text(json.dumps(summary))
raise SystemExit(config['rc'])
''')
        (root / "run_b518_sequential.sh").write_text("#!/bin/sh\necho yes > sequential_called\nexit 0\n")
        if stale:
            (root / verdict_path).parent.mkdir(parents=True)
            (root / verdict_path).write_text("PASS\n")
            old = {"run_id": "previous-run", "base_file": "base_carter_run_ros_map_lidartime_objects_check_v6.py"}
            (root / verdict_path.parent / "launch.json").write_text(json.dumps(old))
            (root / verdict_path.parent / "summary.json").write_text(json.dumps({"verdict": "PASS", "launch": old}))
            (root / verdict_path.parent / "old_observations.jsonl").write_text("keep old gate observations\n")
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
        proc = subprocess.run(["bash", "run_b518_gpu_steps.sh"], cwd=root, env=env,
                              capture_output=True, text=True, timeout=15)
        log = (root / D / "gpu_steps.log").read_text()
        blocked = previous_scan_failure_at is not None or sleep_failure
        expected_rc = 1 if blocked or ps_failure else 75 if busy else 124 if timeout else 0 if expect_run else 1
        check(name, (root / "sequential_called").exists() == expect_run
              and proc.returncode == expected_rc
              and (not (blocked or timeout or busy or ps_failure) or not (root / "smoke_called").exists()),
              {"returncode": proc.returncode, "log": log})
        order = (root / "mock_order.log").read_text().splitlines()
        if previous_busy_rounds or previous_scan_failure_at is not None:
            expected_order = []
            for n in range(1, previous_busy_rounds + 3):
                expected_order.append("ps:" + str(n))
                if n == previous_scan_failure_at:
                    break
                if n <= previous_busy_rounds:
                    expected_order.append("sleep:30")
                    if sleep_failure:
                        break
            if not (blocked or timeout or busy or ps_failure):
                expected_order.append("curl")
            check(name + "_wait_order", order == expected_order,
                  {"actual": order, "expected": expected_order})
        if stale and fresh_artifacts and not (blocked or timeout or busy or ps_failure):
            check(name + "_archives_previous_observations",
                  (root / "results/archive_mock_previous_gate/old_observations.jsonl").read_text()
                  == "keep old gate observations\n")


def main():
    old = script_args(A / "run_b517_sequential.sh")
    expected = [s.replace("B5_17", "B5_18").replace("pB517", "pB518")
                .replace("base_carter_run_ros_map_lidartime_objects_check_v6.py",
                         "base_carter_run_ros_map_lidartime_objects_check_v7.py") for s in old]
    check("identical_agent_flags_except_requested_paths_tag", script_args(A / "run_b518_sequential.sh") == expected)
    gate = (A / "analysis/b518_gate_isaac.py").read_text()
    normalized = gate
    for new, old_name in (("b518", "b517"), ("B518", "B517"),
                          ("B5_18", "B5_17"), ("B-5-18", "B-5-17"),
                          ("v7", "v6")):
        normalized = normalized.replace(new, old_name)
    prior = (A / "analysis/b517_gate_isaac.py").read_text()
    def shared_functions(text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name != "selftest"}
    check("gate_evaluator_launcher_and_freshness_unchanged",
          shared_functions(normalized) == shared_functions(prior))
    def planner_prefix(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "REGION" for t in n.targets))
        return ast.dump(node.value.left, include_attributes=False)
    check("same_existing_planner_and_path_follower_prefix",
          planner_prefix(normalized) == planner_prefix(prior))
    check("gate_cpu_compiles", bool(compile(gate, "b518_gate_isaac.py", "exec")))
    for filename in ("run_b518_sequential.sh", "run_b518_gpu_steps.sh"):
        result = subprocess.run(["bash", "-n", str(A / filename)], capture_output=True, text=True)
        check(filename + "_syntax", result.returncode == 0, result.stderr)
    p1 = {"rc": 0, "turns": [{"turn": 1, "verdict": "pass"}]}
    p2 = {"rc": 0, "turns": [{"turn": 1, "verdict": "fail"}, {"turn": 2, "verdict": "pass"}]}
    fail = {"rc": 1, "turns": [{"turn": 1, "verdict": "fail"}, {"turn": 2, "verdict": "fail"}]}
    run_sequential("both_pass_run_all_even_a2_failure", [p1, p2, fail, p1], [0, 1, 2, 3],
                   ["a0 exit=0 grade=PASS turn=1", "a1 exit=0 grade=PASS turn=2",
                    "a2 exit=1 grade=FAIL turn=-", "a3 exit=0 grade=PASS turn=1"])
    run_sequential("a0_fail_still_run_a1_then_stop", [fail, p1], [0, 1],
                   ["a0 exit=1 grade=FAIL turn=-", "a1 exit=0 grade=PASS turn=1"])
    run_sequential("a1_fail_stop", [p1, fail], [0, 1], ["a1 exit=1 grade=FAIL turn=-"])
    test_pass = {"rc": 0, "turns": [{"turn": 1, "verdict": "fail", "tests": [{"verdict": "pass"}]}]}
    run_sequential("test_pass_and_zero_exit_do_not_pass_grade", [test_pass, p1], [0, 1],
                   ["a0 exit=0 grade=FAIL turn=-"])
    run_sequential("missing_malformed_grade_keep_real_exit", [{"rc": 27, "artifact": "missing"},
                   {"rc": 19, "artifact": "malformed"}], [0, 1],
                   ["a0 exit=27 grade=FAIL turn=-", "a1 exit=19 grade=FAIL turn=-"])
    run_gpu("gpu_pass_launches", "PASS", 0, True)
    run_gpu("gpu_fail_does_not_launch", "FAIL bad clearance", 0, False)
    run_gpu("gpu_nonzero_pass_does_not_launch", "PASS", 7, False)
    run_gpu("gpu_missing_verdict_does_not_launch", None, 0, False)
    run_gpu("gpu_prior_pass_new_pass_launches", "PASS", 0, True, stale=True)
    run_gpu("gpu_unchanged_prior_pass_zero_exit_does_not_launch", "PASS", 0, False, stale=True, fresh_artifacts=False)
    run_gpu("gpu_prior_pass_new_fail_does_not_launch", "FAIL collision", 0, False, stale=True)
    for fault in ("old_verdict", "old_launch", "old_summary", "fail_summary", "old_base", "missing_launch", "missing_summary"):
        run_gpu("gpu_fresh_verdict_" + fault + "_does_not_launch", "PASS", 0, False,
                stale=True, artifact_fault=fault)
    run_gpu("gpu_timeout_does_not_smoke_or_launch", "PASS", 0, False, timeout=True)
    run_gpu("gpu_busy_agent_blocks_before_gate", "PASS", 0, False, busy=True)
    run_gpu("gpu_process_scan_error_blocks", "PASS", 0, False, ps_failure=True)
    run_gpu("b517_running_twice_then_finished_launches", "PASS", 0, True,
            previous_busy_rounds=2)
    run_gpu("b517_first_process_scan_error_blocks_before_models", "PASS", 0, False,
            previous_scan_failure_at=1)
    run_gpu("b517_process_scan_error_after_wait_blocks_before_models", "PASS", 0, False,
            previous_busy_rounds=2, previous_scan_failure_at=2)
    run_gpu("b517_wait_sleep_error_blocks_before_models", "PASS", 0, False,
            previous_busy_rounds=2, sleep_failure=True)
    sources = ("analysis/b518_gate_isaac.py", "analysis/b514_check_isaac_smoke.py",
               "run_b518_sequential.sh", "run_b518_gpu_steps.sh", "analysis/b518_runner_check.py")
    report = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
              "source_sha16": {s: hashlib.sha256((A / s).read_bytes()).hexdigest()[:16] for s in sources},
              "execution": "local mocks only; no real HTTP, GPU, Isaac, or agent_loop"}
    out = A / "results/analysis/b518_runner_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        archive = A / "results" / ("archive_b518_runner_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        shutil.copy2(out, archive)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "file": str(out)}))
    for c in checks:
        if not c["pass"]:
            print(json.dumps(c, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
