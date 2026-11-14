"""Per-test and per-turn summary of a B-5 family run directory, for the 실험 29 / 30 comparison.

    python3 analysis/c_b5x_method_summary.py results/turns/prompt_b5_5 gpu_gpt_oss_120b_pB55

For every sample a<N>: each test drive (verdict, partners, first contact, DBG lines and how many the model
did not see, the '# [시험 N]' records in its code and whether the newest record's result line matches the
test it names), each graded answer (tested or not: same code as a test of that turn up to comments,
docstrings and print calls), whether turn 1 opened with a test, and whether a later turn opened by
re-testing the previous graded answer. Reads only; writes results/analysis/b5x_method_<tag>.json."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

A.A23 = True  # code_key drops docstrings too
SEEN = A.DBG_HEAD + A.DBG_TAIL
REC = re.compile(r"#\s*\[시험\s*(\d+)\]\s*(결과|추린 DBG|원인|이번 변경)\s*:\s*(.*)")


def short(p):
    return p.rsplit("/", 1)[-1].replace("arena_", "")


def region(variant, logs):
    f = logs / f"{variant}_carter_run.py"
    return A.region_of(f.read_text()) if f.exists() else ""


def key_no_print(reg):
    """AST key without docstrings and without bare print(...) statements; None if unparsable."""
    try:
        tree = A.drop_docstrings(ast.parse(reg))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            b = getattr(node, field, None)
            if isinstance(b, list):
                kept = [s for s in b if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                             and getattr(s.value.func, "id", "") == "print")]
                if len(kept) != len(b):
                    setattr(node, field, kept or [ast.Pass()])
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:12]


def records(reg):
    out = {}
    for m in REC.finditer(reg):
        out.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3).strip()
    return out


def result_line_ok(line, t):
    """Does a '결과' record line agree with the real test t: verdict word, every partner, first contact +-0.5 s."""
    if not line or not t:
        return None
    low = line.lower()
    ok = (t["verdict"] in low)
    for p in t.get("partners") or []:
        name = short(p).lower()
        ok &= name in low or name.replace("_", "") in low.replace("_", "")
    if t.get("first_collision_t") is not None:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", line)]
        ok &= any(abs(x - t["first_collision_t"]) <= 0.5 for x in nums)
    return bool(ok)


def summarize(run_dir, tag):
    run_dir = ARENA / run_dir
    logs = run_dir / "isaac_logs"
    out = {}
    for tf in sorted(run_dir.glob(f"{tag}_a*_turns.json")):
        T = json.loads(tf.read_text())
        s = f"a{T['sample']}"
        rows, prev_sub_key = [], None
        for turn in T["turns"]:
            tests = turn.get("tests") or []
            keys = {}
            first_call_test = bool(turn["calls"]) and turn["calls"][0].get("tool_calls", 0) > 0
            for t in tests:
                reg = region(t["variant"], logs)
                recs = records(reg)
                newest = max(recs) if recs else None
                named = next((u for u in tests if u["test"] == newest), None) if newest else None
                k = key_no_print(reg)
                keys[t["test"]] = (A.code_key(reg) if reg else None, k)
                stdout = logs / f"{t['variant']}.carter-run.stdout.log"
                dbg = [ln.strip() for ln in stdout.read_text(errors="replace").splitlines()
                       if ln.strip().startswith("DBG")] if stdout.exists() else []
                rows.append({
                    "turn": turn["turn"], "kind": "test", "n": t["test"], "verdict": t["verdict"] if t["static_ok"] else "static_dq",
                    "partners": [short(p) for p in t.get("partners") or []], "first_contact_s": t.get("first_collision_t"),
                    "contacts": t.get("collisions"), "final_dist_m": t.get("final_dist_m"), "sim_s": t.get("sim_time_s"),
                    "dbg_lines": t.get("dbg_lines"), "dbg_unseen": max(0, (t.get("dbg_lines") or 0) - SEEN),
                    "dbg_seg_clear_first40": sum(1 for ln in dbg[:40] if re.search(r"seg|clear", ln, re.I)),
                    "records": sorted(recs), "newest_record_result_ok": result_line_ok(recs.get(newest, {}).get("결과"), named),
                    "newest_record_has_cause": bool(newest and recs[newest].get("원인")),
                    "same_as_prev_graded_up_to_prints": prev_sub_key is not None and k == prev_sub_key,
                    "turn_opened_with_test": first_call_test if t is tests[0] else None,
                })
            sub = region(turn["variant"], logs)
            sk, sk_np = (A.code_key(sub), key_no_print(sub)) if sub else (None, None)
            tested = [n for n, (a, b) in keys.items() if a == sk or (sk_np and b == sk_np)]
            passed = [n for n in tested if next(u for u in tests if u["test"] == n)["verdict"] == "pass"]
            rows.append({
                "turn": turn["turn"], "kind": "graded", "verdict": turn.get("verdict"),
                "partners": [short(p) for p in turn.get("partners") or []], "first_contact_s": turn.get("first_collision_t"),
                "contacts": turn.get("collisions"), "final_dist_m": turn.get("final_dist_m"), "sim_s": turn.get("sim_time_s"),
                "tests_in_turn": len(tests), "same_code_as_tests": tested, "same_code_as_passed_test": passed,
                "records": sorted(records(sub)), "turn_opened_with_test": first_call_test,
            })
            prev_sub_key = sk_np
        out[s] = {"input_sha16": T["input_sha16"], "passed_at_turn": T.get("passed_at_turn"), "rows": rows}
    return out


def show(out):
    for s, d in out.items():
        print(f"== {s} input {d['input_sha16']} passed_at_turn {d['passed_at_turn']}")
        for r in d["rows"]:
            if r["kind"] == "test":
                print(f"  t{r['turn']} test{r['n']:02d} {r['verdict']:9} {','.join(r['partners']) or '-':28} "
                      f"t0={r['first_contact_s']} dist={r['final_dist_m']} sim={r['sim_s']} dbg={r['dbg_lines']} "
                      f"unseen={r['dbg_unseen']} seg40={r['dbg_seg_clear_first40']} rec={r['records'][-3:] if r['records'] else '-'} "
                      f"rec_ok={r['newest_record_result_ok']} cause={r['newest_record_has_cause']} "
                      f"retest_prev={r['same_as_prev_graded_up_to_prints']}"
                      + (f" opened_with_test={r['turn_opened_with_test']}" if r["turn_opened_with_test"] is not None else ""))
            else:
                print(f"  t{r['turn']} GRADED {r['verdict']:6} {','.join(r['partners']) or '-':28} t0={r['first_contact_s']} "
                      f"dist={r['final_dist_m']} sim={r['sim_s']} tests={r['tests_in_turn']} same_as_tests={r['same_code_as_tests']} "
                      f"same_as_passed={r['same_code_as_passed_test']} rec={len(r['records'])} opened_with_test={r['turn_opened_with_test']}")


if __name__ == "__main__":
    run_dir, tag = sys.argv[1], sys.argv[2]
    res = summarize(run_dir, tag)
    show(res)
    (ARENA / "results" / "analysis" / f"b5x_method_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
