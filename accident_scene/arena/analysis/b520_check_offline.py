"""B520 rechecks unchanged v8 with the B519 suite, preserving every old result."""
import hashlib
import json
from pathlib import Path

import b519_check_offline as V8_CHECKS
from b514_build_base import archive_before_write
from b515_check_offline import safe_json

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results/analysis/b520_check_offline.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    old_result = V8_CHECKS.OUT
    before = digest(old_result)
    prior = json.loads(old_result.read_text())
    base_before = digest(V8_CHECKS.BASE)
    # Only the Python module's output destination changes; source/base stay intact.
    V8_CHECKS.OUT = OUT
    try:
        report = V8_CHECKS.run()
    finally:
        V8_CHECKS.OUT = old_result
    report["checks"]["b520_uses_identical_v8"] = {
        "pass": digest(V8_CHECKS.BASE) == base_before and base_before[:16] == prior["base_sha16"],
        "b519_base_sha16": prior["base_sha16"], "current_sha16": base_before[:16]}
    report["checks"]["b519_offline_result_preserved"] = {"pass": digest(old_result) == before, "sha256": before}
    report["passed"] = sum(c["pass"] for c in report["checks"].values())
    report["total"] = len(report["checks"])
    report["verdict"] = "pass" if report["passed"] == report["total"] else "fail"
    report["b520_provenance"] = {
        "base_changed": False, "suite": str(Path(V8_CHECKS.__file__).relative_to(ARENA)),
        "suite_sha256": digest(V8_CHECKS.__file__), "prior_result": str(old_result.relative_to(ARENA)),
        "prior_result_sha256": before,
        "random61_rerun": False,
        "reason": "B520 changes the prompt/response contract; the v8 geometry, chooser and path helpers are identical."}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"B520 {report['verdict']} {report['passed']}/{report['total']} base={report['base_sha16']}")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run()["verdict"] == "pass" else 1)
