"""B523 prompt, real unchanged harness, and actual dispatcher CPU contract checks.

No Isaac, network, process inspection or B522 a1 artifacts. The dispatcher tests
execute the saved source AST; only hardware tracking and log sinks are stubbed.
"""
import ast
import copy
from datetime import datetime
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ARENA), str(ARENA / "analysis")]
import agent_loop as A
import b523_build_base_v11 as BB
import b523_build_prompt as BP
import b523_prompt_guard as PG
from b515_check_offline import load_ns

REFERENCE = ARENA / "analysis/b523_reference.py"
OUT = ARENA / "results/analysis/b523_selftest.json"
HOOKS = ("plan_path", "lookahead_target", "follow_cmd")
SIGNATURES = (["start_xy", "goal_xy", "clearance_m"], ["path", "pose"], ["pose", "target_xy"], ["t", "pose", "env"])


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def masked_hooks(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HOOKS:
            node.body = [ast.Pass()]
    return ast.dump(tree)


def controller_variant(reference, mode):
    if mode == "wrapper":
        return reference
    tree = ast.parse(reference)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "controller")
    fn.body = ast.parse("return 0., 0., False" if mode == "zero" else "raise RuntimeError('EDIT controller must not run')").body
    return BP.SCAFFOLD.split("def plan_path", 1)[0] + ast.unparse(tree) + "\n" + A.SP.END + "\n"


def run_dispatcher(spliced_source, reference):
    """Load actual selected base helpers and actual _robot_only from built code."""
    ns = load_ns(BP.BASE)
    ns.update(heapq=heapq, START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"], _BASE_DBG_STATE={"t": 0.})
    tree = ast.parse(spliced_source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (*HOOKS, "controller", "_robot_only")]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<actual spliced B523 functions>", "exec"), ns)
    calls, output, details, emitted = [], [], [], []
    ns["builtins"].print = lambda *args, **kwargs: emitted.append(" ".join(str(a) for a in args))
    for name in (*HOOKS, "choose_command", "controller"):
        original = ns[name]
        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        ns[name] = record
    # There is no ROS/LiDAR stream in this CPU dispatch test. Keep the original
    # empty observed-object store and the full actual map/chooser implementation.
    ns["_base_track_step"] = lambda t: None
    for t in (0., 1./60.):
        output.append(ns["_robot_only"](t, ns["START"], None))
        details.append(copy.deepcopy(ns["_BASE_LOOP_ST"]["last"]))
    return {"calls": calls, "outputs": output, "phases": [d["phase"] for d in details],
            "nominal": [d["nominal"] for d in details], "choice": [d["info"]["chosen"] for d in details],
            "plan_attempts": details[0]["plan_attempts"], "target_corrected": [d["target_corrected"] for d in details],
            "target_requested": [d["target_requested"] for d in details], "target_xy": [d["target_xy"] for d in details],
            "target_reject_reason": [d["target_reject_reason"] for d in details],
            "target_ahead_m": [d["target_ahead_m"] for d in details],
            "target_corrected_ahead_m": [d.get("target_corrected_ahead_m") for d in details],
            "fault_logs": [line for line in emitted if line.startswith("DBG base_loop ")],
            "unchanged_return": all(tuple(d["command"]) == tuple(o[:2]) for d, o in zip(details, output))}


def target_contract_checks():
    import math
    ns = load_ns(BP.BASE)
    guard = ns["_base_loop_guard_target"]
    straight = [(0., 0.), (5., 0.)]
    checks = []
    for ahead in (.3, 2.):
        target = (1. + ahead, 0.)
        got, detail = guard(straight, (1., 0., 0.), target, 0.)
        checks.append((f"target arc boundary{ahead}m accepted", got == target and detail["target_reject_reason"] is None))
    got, detail = guard(straight, (3.5, 0., 0.), (5., 0.), 0.)
    checks.append(("last-segment GOAL in forward range retained", got == (5., 0.) and detail["target_ahead_m"] == 1.5 and not detail["target_corrected"]))
    got, detail = guard(straight, (1., 0., 0.), (2., 1.), 0.)
    checks.append(("path-distance1m boundary accepted with valid forward arc", got == (2., 1.) and detail["target_reject_reason"] is None))
    got, detail = guard([(0., 0.), (5., 0.), (5., 5.)], (4., 0., 0.), (5., 1.2), 0.)
    checks.append(("target uses path arc rather than shorter Euclidean distance", math.dist((4., 0.), (5., 1.2)) < 2. and detail["target_reject_reason"] == "too_far" and got == (5., 0.)))
    got, detail = guard([(0., 0.), (1., 0.), (1., 2.)], (.9, 0., 0.), (0., 0.), 0.)
    checks.append(("short-segment fallback advances0.3m along following segment", math.dist(got, (1., .2)) < 1e-12 and abs(detail["target_corrected_ahead_m"] - .3) < 1e-12))
    got, detail = guard([(0., 0.), (1., 0.)], (.9, 1., 0.), (1., 0.), 0.)
    checks.append(("insufficient remaining arc requests checked-zero branch", got is None and detail["target_reject_reason"] == "no_forward_range" and detail["target_fallback_unavailable"]))
    logs = []
    ns["builtins"].print = lambda *args, **kwargs: logs.append(" ".join(str(a) for a in args))
    ns["_BASE_LOOP_ST"]["target_note_t"] = {}
    for t in (0., 9.999, 10.):
        guard(straight, (1., 0., 0.), (5., 0.), t)
    guard(straight, (1., 0., 0.), (0., 0.), 10.)
    checks.append(("target correction DBG throttles each reason independently for10s", len(logs) == 3 and sum("reason=too_far" in s for s in logs) == 2 and sum("reason=behind" in s for s in logs) == 1))
    return checks


def main():
    checks = []
    def check(name, passed):
        checks.append({"name": name, "pass": bool(passed)})
        print(("PASS " if passed else "FAIL ") + name)
    before = json.loads((ARENA / "results/analysis/b523_harness_before.json").read_text())
    prompt, prior, base, reference = BP.OUT.read_text(), BP.SRC.read_text(), BP.BASE.read_text(), REFERENCE.read_text()
    ref_sha = sha(reference)
    check("prompt reproduces from declared B519 source and selected v11", BP.build(prior, base) == prompt)
    check("exact requested B523 title", prompt.splitlines()[0] == BP.TITLE == "# ARENA 미션 프롬프트 B-5-23 (제어 루프를 base 가 돌리고, 모델은 경로 계산, 앞쪽 목표점, 추종 명령의 세 함수만 작성)")
    check("selected v11 reproduces from unchanged v10", sha(base)[:16] == BP.BASE_SHA16 and sha(BB.SRC.read_text()) == BB.SOURCE_SHA256 and BB.build(BB.SRC.read_text()) == base)
    preserved = json.loads((ARENA / "results/analysis/b523_v10_preserved_before_v11.json").read_text())
    check("old v10 base, builder, reference and CPU raw summaries preserved", all(hashlib.sha256((ARENA / name).read_bytes()).hexdigest() == digest for name, digest in preserved.items()))
    check("prior B519 prompt preserved and new text shorter", sha(prior)[:16] == "b25dbb6a6db121fa" and len(prompt) < len(prior))
    check("mission prose retained exactly from B519", prior[prior.index("## 미션\n"):prior.index("## 성공 조건")] in prompt)
    check("four-of-four criterion and unchanged harness decisions explicit", "모두 통과(4/4)" in prompt and "하네스·splice·EDIT 밖 코드는 변경하지 않습니다" in prompt)
    fns = [n for n in ast.parse(BP.SCAFFOLD).body if isinstance(n, ast.FunctionDef)]
    check("only required hooks plus compatibility wrapper in scaffold", [n.name for n in fns] == list(HOOKS) + ["controller"] and [[a.arg for a in n.args.args] for n in fns] == list(SIGNATURES))
    check("reference fills only the three hook bodies", masked_hooks(reference) == masked_hooks(BP.SCAFFOLD))
    check("model scaffold supplies no filled hook implementation", all(len(n.body) == 1 and isinstance(n.body[0], ast.Raise) for n in fns[:3]) and "heapq.heappop" not in BP.SCAFFOLD)
    check("model base placeholders also leave all three algorithms unimplemented", all(isinstance(n.body[-1], ast.Raise) for n in ast.parse(base).body if isinstance(n, ast.FunctionDef) and n.name in HOOKS))
    check("deterministic A-star, valid endpoints and anchored simplification retained", all(text in prompt for text in ("격자 A*", "동점 순서를 고정", "실제 좌표와 격자 칸의 연결", "기준점(anchor)", "모든 연결과 양 끝점", "미검사 직선 대체는 금지")))
    check("center clearances and check_path margins are unambiguous", "0.5/0.45/0.4/0.35m" in prompt and "0.25/0.20/0.15/0.10m" in prompt and "margin_m=max(0,clearance_m-0.25)" in prompt and "기본margin0.25를 고집하지 않습니다" in prompt)
    check("path progress, corner and direct-GOAL constraints retained", all(text in prompt for text in ("경로를 따라0.3~2.0m 앞", "현재 선분 끝", "다음 선분으로 전환", "멀리 있는 GOAL을 바로 반환하지 않습니다", "GOAL이 같은 앞거리0.3~2.0m 범위 안")))
    cfg = load_ns(BP.BASE)["_BASE_LOOP_CFG"]
    check("v11 target documentation matches actual fixed limits and log cadence", tuple(cfg[k] for k in ("target_path_limit_m", "target_min_ahead_m", "target_max_ahead_m", "target_note_s")) == (1., .3, 2., 10.) and all(text in prompt for text in ("거리(arc)가0.3~2.0m", "총0.3m 진행한 점", "유효한 앞점이 없어 정지 입력", "이유별 DBG는10초", "시각 역행")))
    for name, passed in target_contract_checks():
        check(name, passed)
    check("B519 follow formula and clipping explicit", all(text in prompt for text in ("abs(e)>0.35", "min(0.3,0.5*dist+0.1)", "clamp(1.5*e,-0.4,0.4)", "v∈[0,0.3]", "w∈[-0.4,0.4]")))
    check("base-only loop, single chooser, goal and checked-failure behavior explicit", all(text in prompt for text in ("choose_command()` 를 한 번", "목표0.5m", "yaw 오차0.15rad", "실제 반환 명령 둘 다0", "choose_command(0,0,...)", "탈출 명령이 선택")))
    check("initial and >10s blocked replanning only; no invented periodic retry", "계획 호출은 최초와 지도 막힘 정지가10초를 넘은 다음 틱에만" in prompt and "1초 간격" not in prompt)
    check("sensor/manual-loop override instructions removed", "base 센서 API 문서의 controller() 호출 예시는 base 내부 루프에 적용합니다" in prompt and "세 함수에서는 센서나 chooser를 직접 호출하지 않습니다" in prompt and "물체 거리로0으로 덮거나 별도 정지·후진·기다림" in prompt and "base 함수·상태를 재정의하거나 덮어쓰지" in prompt)
    check("first response uses tool and every call specifies200 seconds", prompt.split("## 지금 할 일\n", 1)[1].strip().splitlines()[0] == BP.FIRST_ACTION and prompt.count('{"code": "...", "sim_seconds": 200}') == 2 and "생략하지 않습니다" in prompt)
    check("final output is exact tested EDIT in one actual Python fence", all(text in prompt for text in ("한 글자도 바꾸지 않고 python 코드 펜스 하나", "최종 답은 JSON 객체/문자열이 아닙니다", "실제 줄바꿈", "바꾸려면 다시 시험해 pass")))
    banned = (r"\bM[1-5]\b", r"arena_mover", r"start_zone", r"goal_south", r"왕복", r"11\.6")
    check("model input supplies no mover truth or filled reference", not any(re.search(p, prompt + base) for p in banned) and reference not in prompt + base and "b523_reference.py" not in prompt + base and all(ast.get_source_segment(reference, n) not in prompt + base for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef) and n.name in HOOKS))
    A.BASE_FILE, A.BASE, A.SP.BASE = BP.BASE, base, base
    A.B_MODE, A.A21, A.A22, A.A23, A.A4T = "b5", False, False, False, False
    model_input = A.model_input(BP.OUT.name)
    check("real model_input appends selected base bytes exactly", model_input == prompt + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + base + "```\n")
    prompt_guard = PG.run(BP.OUT, BP.BASE)
    for item in prompt_guard["checks"]:
        check("actual prompt preflight: " + item["name"], item["pass"])
    scaffold_full = A.build(A.TL.extract("```python\n" + BP.SCAFFOLD + "\n```"))
    check("real static check and splice accept scaffold syntax", A.static_check(scaffold_full)[0] and A.TL._outside(scaffold_full) == A.TL._outside(base))
    tool_args = json.loads(json.dumps({"code": reference, "sim_seconds": 200}))
    final_full = A.build(A.TL.extract("```python\n" + reference + "\n```"))
    check("real final extraction and tool JSON splice produce identical accepted source", A.build(tool_args["code"]) == final_full and A.static_check(final_full)[0] and A.TL._outside(final_full) == A.TL._outside(base))
    check("real harness rejects JSON-string and explanation-only final answers", not A.static_check(A.build(A.TL.extract(json.dumps({"code": reference}))))[0] and not A.static_check(A.build(A.TL.extract("설명만 제출")))[0])
    dispatcher = next(n for n in ast.parse(final_full).body if isinstance(n, ast.FunctionDef) and n.name == "_robot_only")
    check("actual external dispatcher calls base directly rather than EDIT controller", isinstance(dispatcher.body[-1], ast.Return) and isinstance(dispatcher.body[-1].value, ast.Call) and isinstance(dispatcher.body[-1].value.func, ast.Name) and dispatcher.body[-1].value.func.id == "_base_control_loop" and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "controller" for n in ast.walk(dispatcher)))
    dispatch_results = {}
    for mode in ("wrapper", "zero", "raise"):
        full = A.build(controller_variant(reference, mode))
        check("real harness accepts EDIT-controller variant: " + mode, A.static_check(full)[0] and A.TL._outside(full) == A.TL._outside(base))
        got = run_dispatcher(full, reference)
        dispatch_results[mode] = got
        check("actual dispatcher uses hooks and one unchanged chooser result: " + mode, got["calls"] == ["plan_path", "lookahead_target", "follow_cmd", "choose_command", "lookahead_target", "follow_cmd", "choose_command"] and got["unchanged_return"] and got["phases"] == ["path_follow", "path_follow"])
    check("zero/raising EDIT controllers cannot change actual dispatcher output", dispatch_results["wrapper"] == dispatch_results["zero"] == dispatch_results["raise"])
    goal_tree = ast.parse(reference)
    goal_hook = next(n for n in goal_tree.body if isinstance(n, ast.FunctionDef) and n.name == "lookahead_target")
    goal_hook.body = ast.parse("return tuple(GOAL[:2])").body
    goal_edit = BP.SCAFFOLD.split("def plan_path", 1)[0] + ast.unparse(goal_tree) + "\n" + A.SP.END + "\n"
    goal_full = A.build(goal_edit)
    goal_dispatch = run_dispatcher(goal_full, reference)
    check("real-spliced direct-GOAL hook is corrected before following and one chooser", A.static_check(goal_full)[0] and goal_dispatch["calls"] == dispatch_results["wrapper"]["calls"] and goal_dispatch["target_corrected"] == [True, True] and goal_dispatch["target_reject_reason"] == ["too_far", "too_far"] and goal_dispatch["target_corrected_ahead_m"] == [2., 2.] and goal_dispatch["unchanged_return"])
    faulty_tree = ast.parse(reference)
    faulty_follow = next(n for n in faulty_tree.body if isinstance(n, ast.FunctionDef) and n.name == "follow_cmd")
    faulty_follow.body = ast.parse("raise RuntimeError('synthetic repeated hook failure')").body
    faulty_edit = BP.SCAFFOLD.split("def plan_path", 1)[0] + ast.unparse(faulty_tree) + "\n" + A.SP.END + "\n"
    faulty_full = A.build(faulty_edit)
    check("real harness accepts valid-Python throwing hook for runtime guard test", A.static_check(faulty_full)[0])
    faulty = run_dispatcher(faulty_full, reference)
    check("actual dispatcher catches each hook exception and calls chooser once per tick", faulty["calls"] == dispatch_results["wrapper"]["calls"] and faulty["phases"] == ["hook_error", "hook_error"] and faulty["nominal"] == [(0., 0.), (0., 0.)] and faulty["unchanged_return"])
    check("repeated hook exception emits exactly one base DBG line every failing tick", len(faulty["fault_logs"]) == 2 and all("hook=follow_cmd" in line and "RuntimeError" in line for line in faulty["fault_logs"]))
    offline_path = ARENA / "results/analysis/b523_v11_check_offline.json"
    offline = json.loads(offline_path.read_text()) if offline_path.exists() else {}
    check("new offline evidence passes on selected v11", offline.get("verdict") == "pass" and offline.get("base_sha16") == BP.BASE_SHA16 and offline.get("passed") == offline.get("total") and offline.get("total", 0) > 0)
    sim_path = ARENA / "results/analysis/b523_v11_reference_sim.json"
    sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    check("CPU reference evidence matches exact reference and selected base", sim.get("complete") is True and sim.get("reference_sha256") == ref_sha and sim.get("base_sha16") == BP.BASE_SHA16)
    rows = sim.get("results", [])
    random_rows = [r for r in rows if r.get("label") != "v1"]
    v1_rows = [r for r in rows if r.get("label") == "v1"]
    snapshot = Path(sim.get("reference_snapshot", "/nonexistent"))
    check("CPU61 contains sixty random phases plus v1 and exact reference snapshot", len(rows) == len({r.get("label") for r in rows}) == 61 and len(random_rows) == sim.get("random_total") == 60 and len(v1_rows) == 1 and sim.get("random_passed") == sum(r.get("verdict") == "pass" for r in random_rows) and sim.get("v1_pass") == (v1_rows[0]["verdict"] == "pass") and snapshot.is_file() and snapshot.read_text() == reference)
    check("CPU61 audits each actual base tick and exact source identity", sim.get("audit_pass") is True and sim.get("runtime_errors") == 0 and sim.get("hook_error_ticks") == 0 and bool(rows) and all(r.get("reference_sha256") == ref_sha and r.get("base_sha256") == sha(base) and r.get("audit", {}).get("exact_one_choose_per_tick") is True and all(r["audit"].get(k) == 0 for k in ("call_order_errors", "returned_command_changes", "goal_finish_errors", "controller_stub_calls")) for r in rows))
    bad_path = ARENA / "results/analysis/b523_v11_bad_functions.json"
    bad = json.loads(bad_path.read_text()) if bad_path.exists() else {}
    bad_rows = {r.get("label"): r for r in bad.get("results", [])}
    expected_bad = {"goal_target_v1", "empty_path_v1", "plan_path_exception_v1", "lookahead_target_exception_v1", "follow_cmd_exception_v1"}
    check("five bad-function CPU cases use exact selected base and reference", bad.get("complete") is True and bad.get("base_sha256") == sha(base) and bad.get("reference_sha256") == ref_sha and set(bad_rows) == expected_bad and bad.get("audit_pass") is True)
    goal_bad = bad_rows.get("goal_target_v1", {})
    bad_summary = {label: {k: r.get(k) for k in ("verdict", "t", "hit")} for label, r in bad_rows.items()}
    check("all five bad functions avoid contact; direct-GOAL target is corrected", bool(bad_rows) and all(r.get("verdict") in ("pass", "timeout") and r.get("hit") is None for r in bad_rows.values()) and goal_bad.get("audit", {}).get("target_corrected_ticks", 0) > 0 and "경로 끝점 GOAL 자체는 이 거리 검사에 걸리지" not in prompt)
    after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
    check("all shared harness files including both carter_env copies unchanged", before == after)
    check("selected base and original B519 prompt unchanged after tests", sha(BP.BASE.read_text()) == sha(base) and sha(BP.SRC.read_text()) == sha(prior))
    summary = {"verdict": "pass" if all(c["pass"] for c in checks) else "fail", "passed": sum(c["pass"] for c in checks), "total": len(checks),
               "base_sha16": sha(base)[:16], "base_sha256": sha(base), "base_file": BP.BASE.name,
               "prompt_sha16": sha(model_input)[:16], "prompt_chars": len(model_input), "prompt_file_sha16": sha(prompt)[:16], "prompt_file_chars": len(prompt),
               "prior_prompt_file_chars": len(prior), "prompt_reduction_chars": len(prior)-len(prompt), "reference_sha256": ref_sha,
               "reference_in_model_input": False, "harness_before": before, "harness_after": after, "dispatch": dispatch_results, "throwing_hook_dispatch": faulty, "direct_goal_dispatch": goal_dispatch,
               "target_limits": {k: cfg[k] for k in ("target_path_limit_m", "target_min_ahead_m", "target_max_ahead_m", "target_note_s")}, "preserved_v10_artifacts": preserved,
               "prompt_guard_file": str(PG.OUT.relative_to(ARENA)), "prompt_guard": {k: prompt_guard[k] for k in ("verdict", "passed", "total", "method", "actual_guard_line", "agent_loop_sha256")},
               "shortenings": BP.SHORTENINGS, "offline_file": str(offline_path.relative_to(ARENA)), "reference_sim_file": str(sim_path.relative_to(ARENA)),
               "reference_sim": {k: sim.get(k) for k in ("random_passed", "random_total", "v1_pass", "runtime_errors", "audit_pass")},
               "bad_function_file": str(bad_path.relative_to(ARENA)), "bad_function_outcomes": bad_summary,
               "limitations": ["Dispatcher tests execute the actual saved _robot_only and base loop on CPU; tracking input is empty, no Isaac run is performed.",
                               "Ignored zero/raising controller bodies cannot bypass the three hooks. Arbitrary rebinding of private base helpers/globals is prohibited by prompt, not sandboxed by these tests.",
                               "The scaffold passes syntax/splicing only and must be completed before running.",
                               "The reference remains byte-identical to v10 and can request3m lookahead; the v11 base corrects out-of-range targets. The prompt asks new model hooks for0.3–2m. Empty/failed plans and caught exceptions need not complete the mission.",
                               "CPU outcomes and matching source hashes do not guarantee Isaac or model-generated controller success."], "checks": checks}
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"SELFTEST={summary['passed']}/{summary['total']} PROMPT_SHA16={summary['prompt_sha16']} PROMPT_CHARS={summary['prompt_chars']} FILE_SHA16={summary['prompt_file_sha16']} FILE_CHARS={len(prompt)}")
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
