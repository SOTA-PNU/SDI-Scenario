"""Validate B521 scaffold, filled reference and actual EDIT-only harness path.

No GPU, HTTP, process queries, B520 a1 reads, or operational file changes.
Runtime fixtures use the reference body with controlled map/chooser callbacks;
the reference's separate full CPU suite is linked by hash.
"""
import ast
from datetime import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A
import b521_build_prompt as BP
from b515_check_offline import load_ns
from b515_selftest import AGENT_LOOP_SHA16, sha16

REFERENCE = ARENA / "analysis/b521_reference.py"
EXPECTED_FUNCTIONS = ("plan_path", "lookahead_target", "follow_cmd", "controller")


def ast_without_bodies(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [ast.Pass()]
    return ast.dump(tree)


def reference_namespace(source):
    ns = load_ns(BP.BASE)
    ns.update(heapq=heapq, START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"])
    exec(compile(source, "<B521 reference EDIT>", "exec"), ns)
    return ns


def controller_contract_checks(source):
    results = []

    def fixture(plan_ok=True, target_ok=True, command=(.17, -.23), chosen="follow", map_hit=None):
        ns = reference_namespace(source)
        events = []
        def plan():
            events.append(("plan_path", ns["_PLAN_POSE"], ns["_PLAN_CLEARANCE"]))
            return [tuple(ns["_PLAN_POSE"][:2]), (0.5, 0.2), tuple(ns["GOAL"][:2])] if plan_ok else None
        def target(path, pose):
            events.append(("lookahead_target", path, pose))
            return (0.5, 0.2) if target_ok else None
        def follow(pose, target):
            events.append(("follow_cmd", pose, target))
            return .3, .1
        def choose(v, w, target):
            events.append(("choose_command", v, w, tuple(target)))
            return command[0], command[1], {"chosen": chosen, "candidates": {"follow": {"t_hit_map": map_hit}}}
        # Route geometry is independently covered offline; this fixture isolates
        # the real controller's state/call ordering with an accepted test path.
        ns.update(plan_path=plan, lookahead_target=target, follow_cmd=follow, choose_command=choose,
                  check_path=lambda path: {"first_bad_segment": None, "min_gap_map": .75})
        return ns, events

    # One fresh approach tick traverses the named helpers in the required order.
    ns, events = fixture()
    pose = (-2., -2., 0.)
    out = ns["controller"](0., pose, None)
    results.append(("filled reference approach uses plan-lookahead-follow-choose once and preserves output",
                    [e[0] for e in events] == list(EXPECTED_FUNCTIONS[:3]) + ["choose_command"] and out == (.17, -.23, False) and events[0][1] == pose and events[-1][3] == (.5, .2)))
    ns, events = fixture(plan_ok=False)
    out = ns["controller"](0., pose, None)
    results.append(("planning failure requests zero once but preserves a nonzero escape and doneFalse",
                    [e[0] for e in events] == ["plan_path", "choose_command"] and events[-1][1:3] == (0., 0.) and out == (.17, -.23, False) and ns["_PLAN_FAILED"]))
    events.clear()
    ns["_NEXT_PLAN_T"] = 5.
    out = ns["controller"](1., pose, None)
    results.append(("pending retry tick keeps failed-plan zero request without rerunning planner",
                    [e[0] for e in events] == ["choose_command"] and events[-1][1:3] == (0., 0.) and out == (.17, -.23, False)))
    ns, events = fixture(target_ok=False)
    out = ns["controller"](0., pose, None)
    results.append(("invalid lookahead enters failed-plan stop instead of direct GOAL following",
                    [e[0] for e in events] == ["plan_path", "lookahead_target", "choose_command"] and events[-1][1:3] == (0., 0.) and out == (.17, -.23, False) and ns["_PLAN_FAILED"]))
    goal_checks = []
    for yaw_error, command, expect_done in [(.2, (.17, -.23), False), (.15, (0., 0.), True), (0., (.17, -.23), False)]:
        ns, events = fixture(command=command)
        gx, gy, ga = ns["GOAL"]
        # Use zero goal yaw in this boundary fixture to avoid cancellation in
        # ga-(ga-.15); the operational GOAL and full CPU suite are unchanged.
        ns["GOAL"] = (gx, gy, 0.0)
        ga = 0.0
        out = ns["controller"](0., (gx, gy, ga-yaw_error), None)
        call = events[-1]
        goal_checks.append([e[0] for e in events] == ["choose_command"] and call[3] == (gx, gy) and out == (*command, expect_done)
                           and (call[1:3] == (0., 0.) if yaw_error <= .15 else call[1] == 0. and math.isclose(call[2], 1.5*yaw_error)))
    results.append(("goal mode overrides planning, aligns above threshold, and completes only on returned zero", all(goal_checks)))
    ns, events = fixture(command=(0., 0.), chosen="stop", map_hit=.1)
    ns["controller"](0., pose, None)
    events.clear()
    ns["controller"](10., pose, None)
    boundary_ok = not ns["_REPLAN_PENDING"] and [e[0] for e in events].count("plan_path") == 0
    events.clear()
    ns["controller"](10.1, pose, None)
    deferred_ok = ns["_REPLAN_PENDING"] and [e[0] for e in events].count("plan_path") == 0 and [e[0] for e in events].count("choose_command") == 1
    events.clear()
    current = (-1.8, -1.9, .1)
    ns["controller"](10.2, current, None)
    results.append(("map stop threshold is strictly over10s and replans from next-tick pose with larger clearance",
                    boundary_ok and deferred_ok and events[0][0] == "plan_path" and events[0][1] == current and events[0][2] > 1.0
                    and [e[0] for e in events].count("choose_command") == 1))
    return results


def main():
    checks = []
    def check(name, ok):
        checks.append({"name": name, "pass": bool(ok)})
        print(("PASS " if ok else "FAIL ") + name)

    original, prompt, base = BP.SRC.read_text(), BP.OUT.read_text(), BP.BASE.read_text()
    reference = REFERENCE.read_text()
    scaffold_tree, ref_tree = ast.parse(BP.SCAFFOLD), ast.parse(reference)
    scaffold_fns = [n for n in scaffold_tree.body if isinstance(n, ast.FunctionDef)]
    ref_fns = [n for n in ref_tree.body if isinstance(n, ast.FunctionDef)]
    check("B521 prompt reproduces from declared B520 edits", BP.build(original, base) == prompt)
    check("exact task19 title and unchanged v8 base", prompt.splitlines()[0] == BP.TITLE == "# ARENA 미션 프롬프트 B-5-21 (B-5-20 에 EDIT REGION 뼈대 코드를 더함)" and sha16(base) == BP.BASE_SHA16 == "d92eed4891798942")
    check("original B520 prompt is unchanged and new prompt is no longer", sha16(original) == "a32f40a48deafb74" and len(prompt) <= len(original))
    check("scaffold has exactly four top-level functions in the fixed order", tuple(n.name for n in scaffold_fns) == EXPECTED_FUNCTIONS)
    signatures = [tuple(a.arg for a in fn.args.args) for fn in scaffold_fns]
    check("literal zero-arg planner and specified public signatures", signatures == [(), ("path", "pose"), ("pose", "target"), ("t", "pose", "env")])
    check("reference fills only function bodies and preserves global declarations", ast_without_bodies(reference) == ast_without_bodies(BP.SCAFFOLD))
    check("reference is complete while scaffold clearly leaves algorithm TODOs", not any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(ref_tree)) and not any(isinstance(n, ast.Name) and n.id == "NotImplementedError" for n in ast.walk(ref_tree)) and all(len(fn.body) == 1 and isinstance(fn.body[0], ast.Raise) for fn in scaffold_fns[:3]) and "# TODO" in BP.SCAFFOLD)
    check("structure-only scaffold provides no A-star expansion implementation", not any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.DictComp)) for fn in scaffold_fns[:3] for n in ast.walk(fn)) and "완성 제어기나 경로 알고리즘의 구현이 아닙니다" in prompt)
    check("structure/body-only instructions and nested helpers are explicit", "이 뼈대의 구조, 함수 이름, 호출 순서를 바꾸지 않는다. 몸통만 채운다" in prompt and "보조 함수는 해당 함수 몸통 안에" in prompt)
    def controller_calls(fn):
        return [n.func.id for n in sorted(ast.walk(fn), key=lambda n: getattr(n, "lineno", 0)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in (*EXPECTED_FUNCTIONS[:3], "choose_command")]
    expected_calls = [*EXPECTED_FUNCTIONS[:3], "choose_command"]
    check("scaffold and filled reference preserve ordered helper and single chooser anchors", controller_calls(scaffold_fns[-1]) == expected_calls and controller_calls(ref_fns[-1]) == expected_calls)
    returns = [n for n in ast.walk(ref_fns[-1]) if isinstance(n, ast.Return)]
    check("filled controller has one common unchanged velocity return", len(returns) == 1 and isinstance(returns[0].value, ast.Tuple) and [ast.unparse(n) for n in returns[0].value.elts[:2]] == ["v", "w"])
    forbidden_calls = {"choose_command", "observed_objects", "lidar2d_points", "lidar3d_points", "camera_image"}
    check("planning/target/follow helper bodies contain no chooser or manual sensor policy", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in forbidden_calls for fn in ref_fns[:3] for n in ast.walk(fn)))
    check("zero-argument planner receives current pose and declared clearance state", "controller 가 계획 직전에 `_PLAN_POSE` 를 현재 `pose` 로 설정" in prompt and "_PLAN_POSE = pose" in BP.SCAFFOLD and "_PLAN_CLEARANCE = 1.0" in BP.SCAFFOLD)
    check("initial and replanned paths validate current start, connectors and minimum gap", BP.PATH_DOC in prompt and "실제 현재 위치부터 GOAL 까지" in prompt and "min_gap_map >= 0.25" in prompt and "최소 0.5 m" in prompt and "현재 위치를 빼거나 최소 기준을 낮추지 않습니다" in prompt)
    check("route targets cannot jump directly to GOAL before the final path segment", "target 은 경로 위 앞쪽 점이며, GOAL 을 바로 넣지 않습니다" in prompt and "마지막 경로 구간에서만 경로의 끝점" in prompt and "유효 연결이 없으면 None" in BP.SCAFFOLD)
    check("failed plan keeps one zero request, preserves escape and never completes", BP.PLAN_FAILURE_DOC in prompt and "검증된 새 경로를 얻을 때까지 실패 상태를 유지" in prompt and "강제로 `(0, 0)` 을 내지 않습니다" in prompt and "실패 정지는 완료가 아닙니다" in prompt)
    check("map-stop info schedules only next-tick replanning with proper reset semantics", BP.REPLAN_DOC in prompt and "연속 10 s 를 넘으면 다음 틱 재계획" in prompt and "조건 해제·시각 역행·재계획 시도 뒤" in prompt)
    check("goal mode preserves0.5m/0.15rad and precedence over pending planning", BP.GOAL_DOC in prompt and "예약된 계획보다 목표 마무리를 우선" in prompt and "d_goal > 0.5" in prompt)
    check("old duplicate tick description and stop-only example were removed", BP.PREV.TICK_DOC not in prompt and BP.PREV.GOOD_FINAL not in prompt and BP.PREV.FORMAT_DOC not in prompt)
    check("deterministic A-star, stable ties and checked anchored simplification remain", BP.PREV.PREV.DETERMINISTIC_DOC in prompt and "보존할 현재 기준점(anchor)을 고정" in prompt and "경유점 간격은 통과 반경 0.35 m 보다 충분히 길게" in prompt)
    check("route progress uses current pose without requiring an extra waypoint-index global", "현재 `pose` 로 경로\n  진행 위치를 계산해 통과한 경유점을 처리" in prompt and "통과한 경유점은 상태로 관리하되" not in prompt)
    banned = (r"\bM[1-5]\b", r"arena_mover", r"start_zone", r"goal_south", r"0\.5 m/s", r"11\.6", r"왕복", r"되돌아", r"끝에서", r"방향을 바꾼", r"상자면")
    check("model input has no mover truth, timing, lanes or supplied route coordinates", not any(re.search(p, prompt + "\n" + base) for p in banned) and not any(isinstance(n, ast.Constant) and isinstance(n.value, (float, int)) and n.value < 0 for n in ast.walk(scaffold_tree)))
    check("reference implementation is absent from model prompt", reference not in prompt and all(ast.get_source_segment(reference, fn) not in prompt for fn in ref_fns) and "b521_reference.py" not in prompt)
    check("first response tool and every200s top-level argument remain required", prompt.split("## 지금 할 일\n", 1)[1].strip().splitlines()[0] == BP.FIRST_ACTION and prompt.count(BP.CALL_EXAMPLE) == 3 and "매 호출의 최상위 JSON 에 `code` 와 `sim_seconds` 두 키" in prompt)
    check("final one-fence EDIT and exact PASS-code submission remain explicit", BP.FORMAT_DOC in prompt and "시험의 `code` 와 최종 답에는 수정한 `[EDIT REGION]` 블록만" in prompt and "통과 뒤 바꾸려면 다시 시험해 통과한 코드만 제출" in prompt)

    A.BASE_FILE, A.BASE, A.SP.BASE = BP.BASE, base, base
    A.B_MODE = "b5"
    A.A21 = A.A22 = A.A23 = A.A4T = False
    model_input = A.model_input(BP.OUT.name)
    marker = "\n\n---\n\n`base_carter_run.py`:\n\n```python\n"
    check("actual model_input contains prompt and unchanged base only", model_input == prompt + marker + base + "```\n" and reference not in model_input)
    scaffold_file = A.build(A.TL.extract("```python\n" + BP.SCAFFOLD + "\n```"))
    check("unfinished scaffold is syntactically valid and actual harness accepts its region", A.static_check(scaffold_file)[0] and A.TL._outside(scaffold_file) == A.TL._outside(base))
    payload = json.loads(json.dumps({"code": reference, "sim_seconds": 200}))
    tool_file = A.build(payload["code"])
    final_file = A.build(A.TL.extract("```python\n" + reference + "\n```"))
    check("filled reference tool and final extraction produce identical executable files", tool_file == final_file and A.static_check(final_file)[0] and A.TL._outside(final_file) == A.TL._outside(base))
    actual_controller = next(n for n in ast.parse(final_file).body if isinstance(n, ast.FunctionDef) and n.name == "controller")
    check("actual spliced controller is the exact filled reference AST", ast.dump(actual_controller) == ast.dump(ref_fns[-1]))
    bad_json = json.dumps({"code": reference})
    check("JSON wrapper and explanation-only final answers remain rejected", not A.static_check(A.build(A.TL.extract(bad_json)))[0] and not A.static_check(A.build(A.TL.extract("설명만 제출합니다.")))[0])
    for name, passed in controller_contract_checks(reference):
        check(name, passed)
    check("base and shared harness remain unchanged after CPU contract checks", sha16(BP.BASE.read_text()) == BP.BASE_SHA16 and sha16((ARENA / "agent_loop.py").read_text()) == AGENT_LOOP_SHA16)

    offline_path = ARENA / "results/analysis/b521_check_offline.json"
    offline = json.loads(offline_path.read_text()) if offline_path.exists() else {}
    reference_sha256 = hashlib.sha256(reference.encode()).hexdigest()
    check("new offline evidence passes for this unchanged v8 and exact reference", offline.get("verdict") == "pass" and offline.get("base_sha16") == sha16(base) and offline.get("reference_sha256") == reference_sha256 and offline.get("passed") == offline.get("total") and offline.get("total", 0) > 0)
    sim_path = ARENA / "results/analysis/b521_reference_sim.json"
    sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    rows = sim.get("results", [])
    expected_labels = {"v1"} | {f"seed{seed}_a{i:02d}" for seed in (1, 2) for i in range(30)}
    snapshot = Path(sim.get("reference_snapshot", "/nonexistent"))
    check("completed CPU61 uses this exact reference, snapshot and unchanged base",
          sim.get("complete") is True and sim.get("exact_source_matches") is True
          and sim.get("reference_sha256") == reference_sha256 and sim.get("base_sha16") == sha16(base)
          and snapshot.is_file() and snapshot.read_text() == reference
          and len(rows) == 61 and {r.get("label") for r in rows} == expected_labels
          and all(r.get("reference_sha256") == reference_sha256 and r.get("base_sha16") == sha16(base) for r in rows))
    random_rows = [r for r in rows if r.get("label") != "v1"]
    observed_random_passed = sum(r.get("verdict") == "pass" for r in random_rows)
    v1_rows = [r for r in rows if r.get("label") == "v1"]
    check("CPU61 reports observed outcomes without imposing a fabricated all-pass threshold",
          len(random_rows) == sim.get("random_total") == 60
          and observed_random_passed == sim.get("random_passed")
          and len(v1_rows) == 1 and (v1_rows[0].get("verdict") == "pass") == sim.get("v1_pass")
          and sim.get("runtime_errors") == 0 and all(not r.get("error") for r in rows))
    check("every simulated reference tick calls chooser once and preserves its returned command",
          sim.get("audit_pass") is True and bool(rows)
          and all(r.get("audit", {}).get("exact_one_choose_per_tick") is True
                  and r["audit"].get("choose_calls") == r["audit"].get("ticks")
                  and all(r["audit"].get(k) == 0 for k in ("call_order_errors", "returned_command_changes", "goal_finish_errors")) for r in rows))
    budget200 = sim.get("completion_budgets_s", {}).get("200", {})
    check("200-second reference outcome count is derived from the completed CPU traces",
          budget200.get("random_total") == 60
          and budget200.get("random_passed") == sum(r.get("verdict") == "pass" and r.get("t", math.inf) < 200 for r in random_rows)
          and len(v1_rows) == 1 and budget200.get("v1_pass") == (v1_rows[0].get("verdict") == "pass" and v1_rows[0].get("t", math.inf) < 200))
    summary = {"verdict": "pass" if all(c["pass"] for c in checks) else "fail", "passed": sum(c["pass"] for c in checks), "total": len(checks), "base_sha16": sha16(base), "base_changed": False, "prompt_file_sha16": sha16(prompt), "prompt_file_chars": len(prompt), "prompt_sha16": sha16(model_input), "prompt_chars": len(model_input), "prior_prompt_chars": len(original), "prompt_reduction_chars": len(original)-len(prompt), "reference_file": str(REFERENCE.relative_to(ARENA)), "reference_sha16": sha16(reference), "reference_in_model_input": False, "offline_file": str(offline_path.relative_to(ARENA)), "shortenings": [{"section": title, "removed_chars": len(old), "replacement_chars": len(new)} for title, old, new in BP.SHORTENINGS], "scaffold_chars": len(BP.SCAFFOLD), "limitations": ["Unfilled scaffold only passes syntax/region checks; it is intentionally not runnable before TODO completion.", "CPU callback fixtures verify reference control flow, not future model compliance or Isaac safety.", "Full reference CPU61 results are produced separately, not embedded in model input."], "checks": checks}
    summary.update(reference_sha256=reference_sha256,
                   reference_sim_file=str(sim_path.relative_to(ARENA)),
                   reference_sim={"random_passed": observed_random_passed, "random_total": len(random_rows),
                                  "v1_pass": sim.get("v1_pass"), "runtime_errors": sim.get("runtime_errors"),
                                  "ticks_audited": sum(r.get("audit", {}).get("ticks", 0) for r in rows),
                                  "completion_budgets_s": sim.get("completion_budgets_s"),
                                  "numeric_pass_threshold_required": False})
    output = ARENA / "results/analysis/b521_selftest.json"
    if output.exists():
        shutil.copy2(output, ARENA / "results" / f"archive_b521_selftest.json_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"SELFTEST={summary['passed']}/{summary['total']} BASE_SHA16={summary['base_sha16']} PROMPT_FILE_SHA16={summary['prompt_file_sha16']} PROMPT_SHA16={summary['prompt_sha16']} PROMPT_CHARS={summary['prompt_chars']}")
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
