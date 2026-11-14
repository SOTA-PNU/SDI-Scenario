"""Execute the real agent_loop.main preflight, stopping before Run construction.

No model, network, process or simulator execution is possible through the tested
path: Run is replaced by a sentinel and URL/process entry points are tripwires.
The scene guard is extracted only for provenance, never duplicated as a checker.
"""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest import mock

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A

SENTENCE = "시험 주행(`run_isaac_sim`)과 최종 채점은 같은 장면에서 돕니다."
OUT = ARENA / "results/analysis/b525_prompt_guard.json"
GLOBALS = ("A21", "A22", "A23", "A4T", "A4T_SEED", "B_MODE", "B_MOVERS", "BASE_FILE", "BASE", "UNLIMITED", "TEST_MOVERS", "TURN_NOTE", "DBG_FULL_CHARS")


class PreflightReached(BaseException):
    pass


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def preflight(prompt_text, base_file):
    saved = {name: getattr(A, name) for name in GLOBALS}
    saved_splice = A.SP.BASE
    saved_env = os.environ.get("ARENA_MOVERS")
    external_calls, sentinel_calls = [], []
    outcome = {"accepted": False, "system_exit": None}
    def stop_before_constructor(args):
        sentinel_calls.append({"prompt_file": args.prompt_file, "base_file": args.base_file,
                               "b5": args.b5, "b_test_movers": args.b_test_movers})
        raise PreflightReached
    def no_external_call(*args, **kwargs):
        external_calls.append("unexpected network/process entry point")
        raise AssertionError("preflight must not make external calls")
    try:
        with tempfile.TemporaryDirectory(prefix="b525_actual_preflight_") as temp:
            prompt_path, run_output = Path(temp) / "prompt.md", Path(temp) / "must_not_be_created"
            prompt_path.write_text(prompt_text)
            argv = ["agent_loop.py", "--b5", "--b-test-movers", "--prompt-file", str(prompt_path),
                    "--base-file", str(base_file), "--turn-note-file", "PROMPT_B5_10_turn_note.md",
                    "--model", "gpt-oss-120b", "--tag", "b525_preflight_only", "--sample", "0",
                    "--base", "http://127.0.0.1:8420/v1", "--out-dir", str(run_output),
                    "--max-turns", "2", "--test-budget", "0", "--max-tokens", "16384",
                    "--temperature", "0.7", "--top-p", "0.95", "--seed-base", "1000"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(A, "Run", stop_before_constructor), \
                    mock.patch.object(A.urllib.request, "urlopen", no_external_call), \
                    mock.patch.object(A.TL.subprocess, "run", no_external_call), \
                    mock.patch.object(A.TL.subprocess, "Popen", no_external_call):
                try:
                    A.main()
                except PreflightReached:
                    outcome["accepted"] = True
                except SystemExit as error:
                    outcome["system_exit"] = str(error.code).replace(str(prompt_path), "<fixture prompt>")
            outcome["run_output_created"] = run_output.exists()
    finally:
        for name, value in saved.items():
            setattr(A, name, value)
        A.SP.BASE = saved_splice
        if saved_env is None:
            os.environ.pop("ARENA_MOVERS", None)
        else:
            os.environ["ARENA_MOVERS"] = saved_env
    outcome.update(sentinel_calls=len(sentinel_calls), external_calls=external_calls,
                   run_constructor_called=False, model_calls=0,
                   selected_base=str(base_file), prompt_sha256=sha(prompt_text))
    return outcome



def run(prompt_file, base_file, save=True):
    source_path = ARENA / "agent_loop.py"
    source = source_path.read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If)
              and {"장면에 없는 물체", "최종 채점은 같은 장면"}.issubset(
                  {v.value for v in ast.walk(n.test) if isinstance(v, ast.Constant) and isinstance(v.value, str)})]
    if len(guards) != 1:
        raise AssertionError("the real main scene guard is missing or ambiguous")
    guard = guards[0]
    fixed = Path(prompt_file).read_text()
    controls = {"b524_control": (ARENA / "PROMPT_B5_24.md").read_text(),
                "b523_control": (ARENA / "PROMPT_B5_23.md").read_text(),
                "b519_control": (ARENA / "PROMPT_B5_19.md").read_text()}
    texts = {"b525_current": fixed, **controls,
             "missing_required_phrase": fixed.replace("최종 채점은 같은 장면", "최종 채점은 동일한 장면"),
             "forbidden_phrase_added": fixed + "\n장면에 없는 물체\n"}
    cases = {name: preflight(text, base_file) for name, text in texts.items()}
    expected = {"b525_current": True, "b524_control": True, "b523_control": True, "b519_control": True,
                "missing_required_phrase": False, "forbidden_phrase_added": False}
    checks = []
    def check(name, passed):
        checks.append({"name": name, "pass": bool(passed)})
    check("exact scene sentence preserved in B525 and historical controls", fixed.count(SENTENCE) == 1 and all(SENTENCE in text for text in controls.values()))
    for name, accepted in expected.items():
        result = cases[name]
        check("actual main preflight: " + name, result["accepted"] is accepted
              and result["sentinel_calls"] == int(accepted)
              and (accepted or "시험 장면을 채점 장면과 다르게 설명합니다" in (result["system_exit"] or "")))
    check("all actual-main cases stop before run creation with zero external calls", all(not r["external_calls"] and not r["run_output_created"] and not r["run_constructor_called"] and r["model_calls"] == 0 for r in cases.values()))
    check("actual agent_loop and historical prompt files unchanged", source_path.read_text() == source and controls["b523_control"] == (ARENA / "PROMPT_B5_23.md").read_text() and controls["b519_control"] == (ARENA / "PROMPT_B5_19.md").read_text())
    report = {"verdict": "pass" if all(c["pass"] for c in checks) else "fail", "passed": sum(c["pass"] for c in checks), "total": len(checks),
              "method": "Actual imported agent_loop.main with B5 runtime flags; sentinel replaces Run before its constructor. URL/process entry points are tripwires; no copied guard condition is evaluated.",
              "agent_loop_file": str(source_path), "agent_loop_sha256": sha(source),
              "actual_guard_line": guard.lineno, "actual_guard_end_line": guard.end_lineno,
              "actual_guard_source": ast.get_source_segment(source, guard), "actual_main_sha256": sha(ast.get_source_segment(source, main)),
              "prompt_sha256": sha(fixed), "control_sha256": {name: sha(text) for name, text in controls.items()},
              "scene_sentence": SENTENCE, "cases": cases, "checks": checks,
              "limitations": ["Stops before Run construction: proves preflight acceptance, not model invocation, Isaac behavior or mission success.", "Scene wording is checked by exact substrings in the unchanged harness."]}
    if save:
        if OUT.exists():
            shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run(ARENA / "PROMPT_B5_25.md", ARENA / "base_carter_run_ros_map_lidartime_objects_check_v13.py")
    print(json.dumps({k: report[k] for k in ("verdict", "passed", "total", "actual_guard_line")}, ensure_ascii=False))
    raise SystemExit(0 if report["verdict"] == "pass" else 1)
