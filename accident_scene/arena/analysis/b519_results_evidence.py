"""Cross-check completed B519 turns, result artifacts, sequential log and gate."""
import collections
import hashlib
import json
from pathlib import Path

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
RUNS = ARENA / "results/turns/prompt_b5_19"
LOGS = RUNS / "isaac_logs"
GATE = ARENA / "results/analysis/b519_gate_isaac"
OUT = ARENA / "results/analysis/b519_results_evidence.json"
PREFIX = "gpu_gpt_oss_120b_pB519"


def provenance(path):
    return {"path": str(path.relative_to(ARENA)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def line_refs(path, tokens):
    return [{"line": i, "text": line.strip()} for i, line in enumerate(path.read_text().splitlines(), 1)
            if any(token in line for token in tokens)]


def main():
    sources, samples, checks = {}, [], {}
    def read(path):
        sources[str(path.relative_to(ARENA))] = provenance(path)
        return json.loads(path.read_text())
    metric_mapping = {"final_dist_m": "final_dist_m", "collisions": "collision_count",
                      "partners": "collision_partners", "first_collision_t": "first_collision_t",
                      "time_to_goal_s": "time_to_goal_s", "sim_time_s": "sim_time_s"}
    all_test_rows = []
    for index in range(4):
        path = RUNS / f"{PREFIX}_a{index}_turns.json"
        turns = read(path)
        rows = []
        for turn in turns["turns"]:
            result_path = LOGS / (turn["variant"] + "_result.json")
            artifact = read(result_path) if result_path.exists() else None
            item = {k: turn.get(k) for k in ("turn", "variant", "tests_used", "spliced", "verdict", "detail", *metric_mapping)}
            item["model_calls"] = len(turn["calls"])
            item["result_file"] = provenance(result_path) if artifact is not None else None
            item["grade_metrics"] = artifact["metrics"] if artifact is not None else None
            if artifact is not None:
                equal = artifact["verdict"] == turn["verdict"] and all(turn.get(k) == artifact["metrics"].get(v) for k, v in metric_mapping.items())
                checks[turn["variant"]+"_turn_vs_result"] = equal
                item["result_metric_line_refs"] = line_refs(result_path, ('"time_to_goal_s":', '"final_dist_m":', '"collision_count":', '"first_collision_t":', '"sim_time_s":'))
            else:
                checks[turn["variant"]+"_no_result_explained"] = turn["verdict"] == "static_dq" and not turn["spliced"]
            tests = []
            for test in turn["tests"]:
                tr = dict(test)
                tp = LOGS / (test["variant"] + "_result.json")
                test_result = read(tp) if tp.exists() else None
                tr["result_file"] = provenance(tp) if test_result is not None else None
                if test_result is not None:
                    checks[test["variant"]+"_turn_vs_result"] = test_result["verdict"] == test["verdict"] and all(
                        test.get(k) == test_result["metrics"].get(v) for k, v in metric_mapping.items())
                else:
                    checks[test["variant"]+"_no_result_explained"] = bool(test.get("runtime_error"))
                tests.append(tr)
                all_test_rows.append({"sample": index, "turn": turn["turn"], **tr})
            item["tests"] = tests
            checks[turn["variant"]+"_test_count"] = len(tests) == turn["tests_used"]
            rows.append(item)
        final = rows[-1]
        samples.append({"sample": index, "passed_at_turn": turns["passed_at_turn"], "turns_used": turns["turns_used"],
                        "started": turns["started"], "ended": turns["ended"], "base_file": turns["base_file"],
                        "input_sha16": turns["input_sha16"], "model_calls": turns["model_calls"],
                        "total_tests_all_turns": sum(r["tests_used"] for r in rows),
                        "final_verdict": final["verdict"], "final_turn_tests": final["tests_used"],
                        "final_time_to_goal_s": final["time_to_goal_s"], "turns": rows})
    seq_path = RUNS / "sequential.log"
    sources[str(seq_path.relative_to(ARENA))] = provenance(seq_path)
    sequential = [{"line": i, "text": line} for i, line in enumerate(seq_path.read_text().splitlines(), 1)]
    seq_text = seq_path.read_text()
    for sample in samples:
        grade = "PASS" if sample["passed_at_turn"] is not None else "FAIL"
        checks[f"a{sample['sample']}_sequential_agrees"] = f"a{sample['sample']} exit={'0' if grade == 'PASS' else '1'} grade={grade}" in seq_text
    checks["sequential_completed"] = seq_text.rstrip().endswith("done")

    gate_summary = read(GATE / "summary.json")
    gate_launch = read(GATE / "launch.json")
    gate_result = read(GATE / "b519_gate_probe_result.json")
    base = ARENA / gate_launch["base_file"]
    base_sha16 = hashlib.sha256(base.read_bytes()).hexdigest()[:16]
    checks["gate_summary_result_and_launch_agree"] = (gate_summary["verdict"] == "PASS" and gate_result["verdict"] == "pass"
        and gate_summary["launch"] == gate_launch and gate_summary["time_to_goal_s"] == gate_result["metrics"]["time_to_goal_s"]
        and gate_summary["collision_count"] == gate_result["metrics"]["collision_count"] == 0
        and gate_launch["base_sha16"] == base_sha16 and gate_launch["returncode"] == 0)

    shield_path = ARENA / "results/analysis/b519_shield_sim.json"
    shield = read(shield_path)
    shield_counts = {}
    improved, changed_graded = [], []
    for cohort in ("failed_grading", "completed_test"):
        comparisons = [c for c in shield["comparisons"] if c["cohort"] == cohort]
        shield_counts[cohort] = {}
        for mode in ("none", "ahead", "goal"):
            rs = [c["modes"][mode] for c in comparisons]
            shield_counts[cohort][mode] = {"pass": sum(r["mission_score"]["verdict"] == "pass" for r in rs),
                                           "collision": sum(bool(r.get("hit")) for r in rs), "cases": len(rs)}
            stored = shield["aggregate_results"][cohort]["modes"][mode]
            checks[f"shield_{cohort}_{mode}_aggregate"] = all(shield_counts[cohort][mode][k] == stored[k] for k in ("pass", "collision"))
        for c in comparisons:
            if cohort == "failed_grading" and c["modes"]["none"]["shield_stats"]["unverified_ticks"]:
                changed_graded.append(c["id"])
            for mode in ("ahead", "goal"):
                if c["modes"]["none"]["mission_score"]["verdict"] != "pass" and c["modes"][mode]["mission_score"]["verdict"] == "pass":
                    improved.append({"id": c["id"], "mode": mode})
    analysis_path = ARENA / "results/analysis/b519_a2_analysis.md"
    sources[str(analysis_path.relative_to(ARENA))] = provenance(analysis_path)
    result = {"scope": "Completed B519 a0-a3 and saved gate/results only; a3 completion is authorized by task17. No execution, GPU, HTTP or process inspection.",
              "verdict": "pass" if all(checks.values()) else "fail", "passed_checks": sum(checks.values()), "total_checks": len(checks),
              "checks": checks, "sources": sources, "samples": samples,
              "aggregate": {"passed_samples": sum(s["passed_at_turn"] is not None for s in samples), "samples": len(samples),
                            "turns_total": sum(s["turns_used"] for s in samples), "tests_total": len(all_test_rows),
                            "tests_by_verdict": dict(collections.Counter(t["verdict"] for t in all_test_rows)),
                            "test_runtime_errors": sum(bool(t.get("runtime_error")) for t in all_test_rows),
                            "tests_with_collision": sum(bool(t.get("collisions")) for t in all_test_rows)},
              "sequential_log": {"source": provenance(seq_path), "lines": sequential},
              "gate": {"summary": gate_summary, "launch": gate_launch, "metrics": gate_result["metrics"],
                       "criteria": gate_result["criteria"], "base_current_sha16": base_sha16},
              "shield": {"counts_recomputed": shield_counts, "fail_to_pass_cases": improved,
                         "unverified_failed_grading_ids": changed_graded,
                         "separately_simulated_jobs": shield["separately_simulated_jobs"],
                         "verified_identity_jobs": shield["verified_identity_jobs"],
                         "physical_execution_attempts": shield["physical_execution_attempts"],
                         "report_wording": "No FAIL-to-PASS improvement from the analysis-only outer filter; already checked outputs stayed unchanged. Some raw B514 commands avoided contact but still failed the mission."},
              "report_notes": ["a1 tested13 times in round1, but its successful round2 submission usedzero tests; do not call the whole sample untested.",
                               "a2 round2 is static_dq with no graded Isaac run. Its6 executed tests first hit M1 at8.700-8.867 simulated seconds;36.1-38.1 are wall-clock durations.",
                               "Gate requested200s and accepted legitimate early goal completion at65.517s total; time-to-goal62.133s is a separate metric.",
                               "Shield analysis has31 separately simulated conditions plus32 identity-derived conditions; no new Isaac shield runs were made."]}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(f"{result['verdict']} {result['passed_checks']}/{result['total_checks']} {OUT}")
    print(result["aggregate"])
    for name, ok in checks.items():
        if not ok: print("FAIL", name)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["verdict"] == "pass" else 1)
