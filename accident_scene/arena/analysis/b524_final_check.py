"""Read-only task26 evidence integration; no simulator, model, or process query."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ARENA = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ARENA), str(ARENA / "analysis")]
import agent_loop as A
import b524_build_base as BB
import b524_build_prompt as BP


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = ARENA / "results/analysis"
    out = evidence / "b524_final_validation.json"
    if out.exists():
        raise SystemExit("Archive prior final validation before rerunning")
    documents = {}
    def read(name):
        path = evidence / name
        documents[str(path.relative_to(ARENA))] = digest(path)
        return json.loads(path.read_text())
    checks = []
    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})

    before = read("b524_protected_before.json")
    report = read("b524_report_update.json")
    preserved = {p: (ARENA / p).is_file() and digest(ARENA / p) == h
                 for p, h in before.items() if p != "REPORT.md"}
    check("All 173 protected source and completed-result files preserved", len(preserved) == 173 and all(preserved.values()))
    check("REPORT backed up exactly before change", digest(ARENA / report["archive"]) == before["REPORT.md"] == report["before_sha256"])
    check("REPORT contains only the reviewed addition", (ARENA / "REPORT.md").read_bytes() == (ARENA / report["archive"]).read_bytes() + (ARENA / report["addition_file"]).read_bytes())
    addition = (ARENA / report["addition_file"]).read_text()
    check("New REPORT sections contain no middle dot", "·" not in addition)
    check("REPORT records experiment49 outcomes and experiment50 preparation", all(s in addition for s in ("### 실험 49 결과", "### 실험 50 준비", "78.317", "63.283", "4/4", "a3", "결과 미확인")))

    base = BP.BASE.read_text()
    prompt = BP.OUT.read_text()
    base_hash = digest(BP.BASE)
    ref = ARENA / "analysis/b524_reference.py"
    ref_hash = digest(ref)
    check("v12 rebuild matches saved source", BB.build(BB.SRC.read_text()) == base)
    check("B524 prompt rebuild matches saved source", BP.build(BP.SRC.read_text(), base) == prompt)
    check("Reference remains byte-identical to B523", ref.read_bytes() == (ARENA / "analysis/b523_reference.py").read_bytes())
    selftest = read("b524_selftest.json")
    A.BASE_FILE, A.BASE, A.SP.BASE = BP.BASE, base, base
    A.B_MODE, A.A21, A.A22, A.A23, A.A4T = "b5", False, False, False, False
    model_input = A.model_input(BP.OUT.name)
    check("Final input hash and length match actual harness composition", hashlib.sha256(model_input.encode()).hexdigest()[:16] == selftest["prompt_sha16"] and len(model_input) == selftest["prompt_chars"])
    check("Prompt file hash and length match selftest", digest(BP.OUT)[:16] == selftest["prompt_file_sha16"] and len(prompt) == selftest["prompt_file_chars"])
    check("Selftest applies to final base and reference", selftest["base_sha256"] == base_hash and selftest["reference_sha256"] == ref_hash)
    for name, count in (("b524_selftest.json", 79), ("b524_prompt_guard.json", 8),
                        ("b524_check_offline.json", 62), ("b524_gate_selftest.json", 29),
                        ("b524_runner_check.json", 152)):
        data = selftest if name == "b524_selftest.json" else read(name)
        check(name + " all checks passed", data["passed"] == data["total"] == count)

    sim = read("b524_reference_sim.json")
    bad = read("b524_bad_functions.json")
    for name, data, count in (("normal", sim, 61), ("bad6", bad, 6)):
        check(name + " CPU evidence uses final base and reference", data["base_sha256"] == base_hash and data["reference_sha256"] == ref_hash)
        check(name + " CPU batch complete with exact-loop audit", data["complete"] and data["audit_pass"] and len(data["results"]) == count)
        check(name + " CPU engine unchanged", digest(ARENA / "analysis/b515_chooser_sim.py") == data["calibrated_engine_sha256"])
    check("Normal CPU reference60+v1 pass without errors", sim["random_passed"] == sim["random_total"] == 60 and sim["v1_pass"] and sim["runtime_errors"] == sim["hook_error_ticks"] == sim["plan_error_attempts"] == 0)
    check("All six injected fixtures contact-free, exactly two mission successes", all(r["hit"] is None for r in bad["results"]) and sum(r["verdict"] == "pass" for r in bad["results"]) == 2)
    outcomes = {r["label"]: r["verdict"] for r in bad["results"]}
    check("Grid endpoint fixture passes alongside GOAL target fixture", {k for k, v in outcomes.items() if v == "pass"} == {"grid_endpoints_v1", "goal_target_v1"})
    comparison = read("b524_comparison.json")
    check("Comparison evidence checks all pass", comparison["passed"] == comparison["total"] and comparison["total"] > 0)

    prep = read("b524_launch_preparation.json")
    check("Launch manifest pins all current operational sources", all(digest(ARENA / p) == h for p, h in prep["source_sha256"].items()))
    check("Launch manifest pins gate and runner evidence", all(digest(ARENA / v["artifact"]) == v["sha256"] for v in prep["validation"].values()))
    check("New actual runtime output remains absent", all(not (ARENA / p).exists() for p in prep["new_actual_runtime_artifacts_absent"]))
    check("B523 a3 was not inspected by launch preparation", prep["B523_a3_artifacts_accessed"] is False)
    historical = read("b523_a0_a1_results_evidence.json")
    check("Experiment49 completed evidence source hashes match", all(digest(ARENA / p) == h for p, h in historical["source_sha256"].items()))
    for source, expected in selftest["harness_before"].items():
        check("Harness preserved: " + source, digest(Path(source)) == expected == selftest["harness_after"][source])
    for name in ("run_b524_gpu_steps.sh", "run_b524_sequential.sh"):
        result = subprocess.run(["bash", "-n", str(ARENA / name)], capture_output=True, text=True)
        check("Shell syntax: " + name, result.returncode == 0)

    current_sources = sorted(set(prep["source_sha256"]) | {"analysis/b524_build_base.py", "analysis/b524_build_prompt.py",
        "analysis/b524_selftest.py", "analysis/b524_prompt_guard.py", "analysis/b524_check_offline.py",
        "analysis/b524_reference_sim.py", "analysis/b524_compare.py", "analysis/b524_final_check.py", "REPORT.md"})
    result = {
        "task": 26, "verdict": "pass" if all(c["pass"] for c in checks) else "fail",
        "passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
        "base_file": BP.BASE.name, "base_sha256": base_hash, "base_sha16": base_hash[:16],
        "prompt_sha16": selftest["prompt_sha16"], "prompt_chars": selftest["prompt_chars"],
        "prompt_file_sha16": selftest["prompt_file_sha16"], "prompt_file_chars": selftest["prompt_file_chars"],
        "protected_files_unchanged": preserved, "report_update": report,
        "source_sha256": {p: digest(ARENA / p) for p in current_sources}, "evidence_sha256": documents,
        "launch_cmd": "bash " + str(ARENA / "run_b524_gpu_steps.sh"),
        "execution": "Preparation and CPU checks only; no actual GPU, Isaac, model, HTTP, host process inspection or B523 a3 access.",
        "limitations": "CPU proxy does not establish Isaac success or model a0-a3 4/4 stability; a3 result not inspected."
    }
    with out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"FINAL={result['verdict']} {result['passed']}/{result['total']}")
    for c in checks:
        if not c["pass"]:
            print("FAIL", c["name"])
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
