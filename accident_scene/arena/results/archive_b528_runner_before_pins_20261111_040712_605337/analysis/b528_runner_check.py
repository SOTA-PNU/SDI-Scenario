#!/usr/bin/env python3
"""Exercise B-5-28 launch control with local substitutes; no HTTP, GPU, or Isaac."""
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
TAG = "gpu_gpt_oss_120b_pB528"
D = Path("results/turns/prompt_b5_28")
BASE_FILE = "base_carter_run_ros_map_lidartime_objects_check_v16.py"
BASE_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
REFERENCE_FILE = "analysis/b528_reference.py"
REFERENCE_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def output_contract(sources):
    """Task30 literals, independent of copied fixtures and old runner flags."""
    seq, gpu, gate = (sources[k] for k in ("sequential", "gpu", "gate"))
    return (re.search(r"^D=results/turns/prompt_b5_28$", seq, re.M) is not None
            and re.search(r"^D=results/turns/prompt_b5_28$", gpu, re.M) is not None
            and '"results/turns/prompt_b5_28/isaac_logs/omniverse_logs_offsets_start.json"' in seq
            and "TAG=gpu_gpt_oss_120b_pB528" in seq
            and "--prompt-file PROMPT_B5_28.md" in seq
            and all(old not in text for text in sources.values()
                    for old in ("prompt_b5_27", "pB527", "PROMPT_B5_27.md", "b527_gate_isaac"))
            and "[r]un_b527_gpu_steps.sh" not in gpu
            and "results/analysis/b528_gate_isaac" in gate)


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def previous_output_fixture(root):
    previous = root / "results/turns/prompt_b5_27"
    previous.mkdir(parents=True)
    for name in ("gpu_steps.log", "sequential.log", ".gpu_steps.lock", ".sequential.lock"):
        (previous / name).write_text("untouched B527 fixture: " + name + "\n")
    return previous, snapshot(previous)


def script_args(path):
    text = path.read_text().replace("\\\n", " ")
    cmd = text.split("python3 agent_loop.py ", 1)[1].split(" > ", 1)[0]
    return shlex.split(cmd)


def run_sequential(name, outcomes, expected_samples, expected_lines):
    with tempfile.TemporaryDirectory(prefix="b528_sequential_mock_") as tmp:
        root = Path(tmp)
        previous, previous_before = previous_output_fixture(root)
        script = (A / "run_b528_sequential.sh").read_text()
        # Isolate the copied fixture's Omniverse metadata snapshot from host logs.
        host_log_path = 'd = os.path.expanduser("~/.nvidia-omniverse/logs")'
        assert script.count(host_log_path) == 1
        (root / "run_b528_sequential.sh").write_text(script.replace(host_log_path, 'd = "mock_omniverse_logs"'))
        (root / "mock_omniverse_logs").mkdir()
        (root / "mock_omniverse_logs/mock.log").write_text("fixture\n")
        (root / "agent_loop.py").write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
s = int(args[args.index("--sample") + 1])
spec = json.loads(os.environ["B528_MOCK_CASE"])[s]
assert os.environ.get("CARTER_CONTACT_ALL_LINKS") == "1"
with open("calls.jsonl", "a") as f: f.write(json.dumps(args) + "\\n")
d = pathlib.Path(args[args.index("--out-dir") + 1])
assert d.as_posix() == "results/turns/prompt_b5_28"
assert args[args.index("--tag") + 1] == "gpu_gpt_oss_120b_pB528"
assert args[args.index("--prompt-file") + 1] == "PROMPT_B5_28.md"
assert args[args.index("--base-file") + 1] == "base_carter_run_ros_map_lidartime_objects_check_v16.py"
p = d / ("gpu_gpt_oss_120b_pB528_a%d_turns.json" % s)
data = {"tag": "gpu_gpt_oss_120b_pB528", "sample": s, "turns": spec.get("turns", [])}
if spec.get("artifact") == "malformed": p.write_text("{")
elif spec.get("artifact") != "missing": p.write_text(json.dumps(data))
sys.exit(spec["rc"])
''')
        env = dict(os.environ, B528_MOCK_CASE=json.dumps(outcomes))
        proc = subprocess.run(["bash", "run_b528_sequential.sh"], cwd=root, env=env,
                              capture_output=True, text=True, timeout=15)
        calls = [json.loads(s) for s in (root / "calls.jsonl").read_text().splitlines()]
        samples = [int(a[a.index("--sample") + 1]) for a in calls]
        log = (root / D / "sequential.log").read_text()
        check(name, proc.returncode == 0 and samples == expected_samples
              and all(line in log for line in expected_lines)
              and log.splitlines()[-1].endswith(" done"),
              {"samples": samples, "log": log, "returncode": proc.returncode})
        offsets = json.loads((root / D / "isaac_logs/omniverse_logs_offsets_start.json").read_text())
        check(name + "_snapshot_uses_only_mock_logs", offsets["files"] == {"mock.log": 8}
              and offsets["dir"] == "mock_omniverse_logs")
        before = {p.relative_to(root).as_posix(): p.read_bytes()
                  for p in (root / D).rglob("*") if p.is_file()}
        retry = subprocess.run(["bash", "run_b528_sequential.sh"], cwd=root, env=env,
                               capture_output=True, text=True, timeout=15)
        after = {p.relative_to(root).as_posix(): p.read_bytes()
                 for p in (root / D).rglob("*") if p.is_file()}
        check(name + "_preserves_existing", retry.returncode == 73 and before == after)
        check(name + "_does_not_touch_B527_outputs", snapshot(previous) == previous_before)


def run_gpu(name, verdict, smoke_rc, expect_run, stale=False, timeout=False, busy=False,
            ps_failure=False, fresh_artifacts=True, artifact_fault=None,
            previous_wrapper_present=False,
            corrupt_base_before_gate=False, corrupt_reference_before_gate=False):
    with tempfile.TemporaryDirectory(prefix="b528_gpu_mock_") as tmp:
        root = Path(tmp)
        previous, previous_before = previous_output_fixture(root)
        shutil.copy2(A / BASE_FILE, root / BASE_FILE)
        if corrupt_base_before_gate:
            with (root / BASE_FILE).open("a") as f:
                f.write("\n# corrupted mock fixture\n")
        script = (A / "run_b528_gpu_steps.sh").read_text()
        if timeout:
            # Accelerate only the copied fixture's deadline; production remains 1800 s.
            assert "MAX_WAIT_S=1800" in script
            script = script.replace("MAX_WAIT_S=1800", "MAX_WAIT_S=0")
        (root / "run_b528_gpu_steps.sh").write_text(script)
        (root / "bin").mkdir()
        curl = root / "bin/curl"
        curl.write_text('#!/bin/sh\necho curl >> mock_order.log\nprintf \'%s\\n\' \'{"data":[{"id":"gpt-oss-120b"}]}\'\n')
        curl.chmod(0o755)
        ps = root / "bin/ps"
        process_config = {"previous_wrapper_present": previous_wrapper_present,
                          "agent_busy": busy, "agent_scan_failure": ps_failure}
        ps.write_text("#!/usr/bin/env python3\nconfig = " + repr(process_config) + "\n" + '''
from pathlib import Path
import sys
assert sys.argv[1:] == ['-eo', 'args']
counter = Path('mock_ps_count')
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
with Path('mock_order.log').open('a') as f: f.write('ps:' + str(n) + '\\n')
assert n == 1
if config['agent_scan_failure']:
    raise SystemExit(1)
if config['previous_wrapper_present']:
    print('bash /example/run_b527_gpu_steps.sh')
print('python3 agent_loop.py' if config['agent_busy'] else 'COMMAND')
''')
        ps.chmod(0o755)
        sleeper = root / "bin/sleep"
        sleeper.write_text("#!/bin/sh\necho sleep:$* >> mock_order.log\nexit 0\n")
        sleeper.chmod(0o755)
        (root / "analysis").mkdir()
        shutil.copy2(A / REFERENCE_FILE, root / REFERENCE_FILE)
        if corrupt_reference_before_gate:
            with (root / REFERENCE_FILE).open("a") as f:
                f.write("\n# corrupted mock reference\n")
        verdict_path = Path("results/analysis/b528_gate_isaac/verdict.txt")
        config = dict(verdict=verdict, rc=smoke_rc, fresh=fresh_artifacts, fault=artifact_fault)
        (root / "analysis/b528_gate_isaac.py").write_text("config = " + repr(config) + "\n" + '''
from pathlib import Path
import hashlib, json, os, shutil
Path('smoke_called').write_text('yes')
d = Path('results/analysis/b528_gate_isaac')
if config['fresh']:
    if d.exists():
        shutil.copytree(d, Path('results/archive_mock_previous_gate'))
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if config['verdict'] is not None:
        verdict_id = 'previous-run' if config['fault'] == 'old_verdict' else os.environ['B528_GATE_RUN_ID']
        (d / 'verdict.txt').write_text(config['verdict'] + '\\nRUN_ID=' + verdict_id + '\\n')
    launch = {'run_id': os.environ['B528_GATE_RUN_ID'],
              'base_file': 'base_carter_run_ros_map_lidartime_objects_check_v16.py',
              'base_sha16': hashlib.sha256(Path('base_carter_run_ros_map_lidartime_objects_check_v16.py').read_bytes()).hexdigest()[:16],
              'reference_file': 'analysis/b528_reference.py',
              'reference_sha256': hashlib.sha256(Path('analysis/b528_reference.py').read_bytes()).hexdigest(),
              'reference_sha16': hashlib.sha256(Path('analysis/b528_reference.py').read_bytes()).hexdigest()[:16]}
    summary = {'verdict': 'PASS' if config['verdict'] == 'PASS' else 'FAIL', 'launch': dict(launch)}
    if config['fault'] == 'old_launch': launch['run_id'] = 'previous-run'
    if config['fault'] == 'old_summary': summary['launch']['run_id'] = 'previous-run'
    if config['fault'] == 'fail_summary': summary['verdict'] = 'FAIL'
    if config['fault'] == 'old_base': launch['base_file'] = 'base_carter_run_ros_map_lidartime_objects_check_v15.py'
    if config['fault'] == 'old_summary_base': summary['launch']['base_file'] = 'base_carter_run_ros_map_lidartime_objects_check_v15.py'
    if config['fault'] == 'wrong_hash': launch['base_sha16'] = '0' * 16
    if config['fault'] == 'missing_hash': launch.pop('base_sha16')
    if config['fault'] == 'old_summary_hash': summary['launch']['base_sha16'] = '0' * 16
    if config['fault'] == 'wrong_reference': launch['reference_sha256'] = '0' * 64
    if config['fault'] == 'wrong_reference_file': launch['reference_file'] = 'analysis/b527_reference.py'
    if config['fault'] == 'missing_reference': launch.pop('reference_sha256')
    if config['fault'] == 'wrong_summary_reference': summary['launch']['reference_sha256'] = '0' * 64
    if config['fault'] != 'missing_launch': (d / 'launch.json').write_text(json.dumps(launch))
    if config['fault'] != 'missing_summary': (d / 'summary.json').write_text(json.dumps(summary))
    if config['fault'] == 'base_mutated_during_gate':
        with Path('base_carter_run_ros_map_lidartime_objects_check_v16.py').open('a') as f:
            f.write('\\n# mutated during mock gate\\n')
    if config['fault'] == 'reference_mutated_during_gate':
        with Path('analysis/b528_reference.py').open('a') as f:
            f.write('\\n# mutated during mock gate\\n')
raise SystemExit(config['rc'])
''')
        (root / "run_b528_sequential.sh").write_text("#!/bin/sh\necho yes > sequential_called\nexit 0\n")
        if stale:
            (root / verdict_path).parent.mkdir(parents=True)
            (root / verdict_path).write_text("PASS\n")
            old = {"run_id": "previous-run", "base_file": "base_carter_run_ros_map_lidartime_objects_check_v15.py",
                   "base_sha16": "bbe902ad613caa5e"}
            (root / verdict_path.parent / "launch.json").write_text(json.dumps(old))
            (root / verdict_path.parent / "summary.json").write_text(json.dumps({"verdict": "PASS", "launch": old}))
            (root / verdict_path.parent / "old_observations.jsonl").write_text("keep old gate observations\n")
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
        proc = subprocess.run(["bash", "run_b528_gpu_steps.sh"], cwd=root, env=env,
                              capture_output=True, text=True, timeout=15)
        log = (root / D / "gpu_steps.log").read_text()
        blocked = corrupt_base_before_gate or corrupt_reference_before_gate
        expected_rc = 1 if blocked or ps_failure else 75 if busy else 124 if timeout else 0 if expect_run else 1
        check(name, (root / "sequential_called").exists() == expect_run
              and proc.returncode == expected_rc
              and (not (blocked or timeout or busy or ps_failure) or not (root / "smoke_called").exists()),
              {"returncode": proc.returncode, "log": log})
        order = (root / "mock_order.log").read_text().splitlines()
        expected_order = ["ps:1"]
        if not (timeout or busy or ps_failure):
            expected_order.append("curl")
        check(name + "_independent_agent_scan_order", order == expected_order,
              {"actual": order, "expected": expected_order})
        if stale and fresh_artifacts and not (blocked or timeout or busy or ps_failure):
            check(name + "_archives_previous_observations",
                  (root / "results/archive_mock_previous_gate/old_observations.jsonl").read_text()
                  == "keep old gate observations\n")
        check(name + "_does_not_touch_B527_outputs", snapshot(previous) == previous_before)


def main():
    production = {"sequential": (A / "run_b528_sequential.sh").read_text(),
                  "gpu": (A / "run_b528_gpu_steps.sh").read_text(),
                  "gate": (A / "analysis/b528_gate_isaac.py").read_text()}
    check("task27_independent_output_path_tag_prompt_contract", output_contract(production))
    bad_gpu = {**production, "gpu": production["gpu"].replace(
        "D=results/turns/prompt_b5_28", "D=results/turns/prompt_b5_27")}
    check("contract_rejects_previous_gpu_output_directory", not output_contract(bad_gpu))
    bad_seq = {**production, "sequential": production["sequential"].replace(
        "prompt_b5_28/isaac_logs/omniverse_logs_offsets_start.json",
        "prompt_b5_27/isaac_logs/omniverse_logs_offsets_start.json")}
    check("contract_rejects_previous_offset_snapshot_directory", not output_contract(bad_seq))
    check("v16_source_selection_and_frozen_hash",
          hashlib.sha256((A / BASE_FILE).read_bytes()).hexdigest() == BASE_SHA256
          and BASE_FILE in production["sequential"] and BASE_FILE in production["gate"]
          and "BASE_FILE=" + BASE_FILE in production["gpu"]
          and "BASE_SHA256=" + BASE_SHA256 in production["gpu"])
    check("reference_source_selection_and_frozen_hash",
          hashlib.sha256((A / REFERENCE_FILE).read_bytes()).hexdigest() == REFERENCE_SHA256
          and "REFERENCE_FILE=" + REFERENCE_FILE in production["gpu"]
          and "REFERENCE_SHA256=" + REFERENCE_SHA256 in production["gpu"])
    old = script_args(A / "run_b527_sequential.sh")
    expected = [s.replace("B5_27", "B5_28").replace("pB527", "pB528")
                .replace("check_v15.py", "check_v16.py") for s in old]
    check("identical_agent_flags_except_requested_paths_tag_base", script_args(A / "run_b528_sequential.sh") == expected)
    gate = (A / "analysis/b528_gate_isaac.py").read_text()
    normalized = gate
    for new, old_name in (("b528", "b527"), ("B528", "B527"),
                          ("B5_28", "B5_27"), ("B-5-28", "B-5-27")):
        normalized = normalized.replace(new, old_name)
    prior = (A / "analysis/b527_gate_isaac.py").read_text()
    def selected_functions(text, names):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name in names}
    check("gate_criteria_evaluator_unchanged",
          selected_functions(normalized, {"evaluate", "finite_number", "read_lines"})
          == selected_functions(prior, {"evaluate", "finite_number", "read_lines"}))
    import b528_gate_isaac as current_gate
    reference = current_gate.load_reference()
    base = (A / BASE_FILE).read_text()
    probe = current_gate.make_runner(base, reference)
    check("gate_pins_final_base_and_reference",
          current_gate.BASE_SHA256 == BASE_SHA256 and current_gate.REFERENCE_SHA256 == REFERENCE_SHA256)
    check("actual_robot_dispatch_and_control_loop_AST_preserved",
          selected_functions(probe, {"_robot_only", "_base_control_loop"})
          == selected_functions(base, {"_robot_only", "_base_control_loop"}))
    check("exact_reference_three_hooks_and_stub_AST_preserved",
          selected_functions(probe, {"plan_path", "lookahead_target", "follow_cmd", "controller"})
          == selected_functions(reference, {"plan_path", "lookahead_target", "follow_cmd", "controller"}))
    check("auditor_wraps_actual_dispatch_without_controller_override",
          "ctrl = _b528_audit_step(_robot_only)" in probe
          and "ctrl = _robot_only" not in probe
          and "timeout_s=200.0," in probe)
    check("gate_cpu_compiles", bool(compile(gate, "b528_gate_isaac.py", "exec")))
    for filename in ("run_b528_sequential.sh", "run_b528_gpu_steps.sh"):
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
    for fault in ("old_verdict", "old_launch", "old_summary", "fail_summary", "old_base", "missing_launch", "missing_summary",
                  "old_summary_base", "wrong_hash", "missing_hash", "old_summary_hash", "base_mutated_during_gate",
                  "wrong_reference", "wrong_reference_file", "missing_reference", "wrong_summary_reference",
                  "reference_mutated_during_gate"):
        run_gpu("gpu_fresh_verdict_" + fault + "_does_not_launch", "PASS", 0, False,
                stale=True, artifact_fault=fault)
    run_gpu("gpu_timeout_does_not_smoke_or_launch", "PASS", 0, False, timeout=True)
    run_gpu("gpu_busy_agent_blocks_before_gate", "PASS", 0, False, busy=True)
    run_gpu("gpu_process_scan_error_blocks", "PASS", 0, False, ps_failure=True)
    run_gpu("removed_previous_wrapper_wait_does_not_block", "PASS", 0, True,
            previous_wrapper_present=True)
    run_gpu("changed_v16_source_blocks_before_gate", "PASS", 0, False,
            corrupt_base_before_gate=True)
    run_gpu("changed_reference_source_blocks_before_gate", "PASS", 0, False,
            corrupt_reference_before_gate=True)
    sources = ("analysis/b528_gate_isaac.py", "analysis/b514_check_isaac_smoke.py",
               "run_b528_sequential.sh", "run_b528_gpu_steps.sh", "analysis/b528_runner_check.py", BASE_FILE,
               REFERENCE_FILE)
    report = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
              "source_sha16": {s: hashlib.sha256((A / s).read_bytes()).hexdigest()[:16] for s in sources},
              "execution": "local mocks only; no real HTTP, GPU, Isaac, or agent_loop"}
    out = A / "results/analysis/b528_runner_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        archive = A / "results" / ("archive_b528_runner_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        shutil.copy2(out, archive)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "file": str(out)}))
    for c in checks:
        if not c["pass"]:
            print(json.dumps(c, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
