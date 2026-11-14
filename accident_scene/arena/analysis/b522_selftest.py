"""B522 prompt/harness/immutable-shell and filled-reference CPU evidence checks.

Contract mutation checks are preparation checks, not enforcement added to the
shared operational harness. No GPU, HTTP, process inspection or a1 data access.
"""
import ast
import copy
from datetime import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import sys
from types import FunctionType

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A
import b522_build_prompt as BP
from b521_selftest import controller_contract_checks as prior_controller_checks
from b515_check_offline import load_ns
from b515_selftest import AGENT_LOOP_SHA16, sha16

REFERENCE = ARENA / "analysis/b522_reference.py"
NAMES = ("plan_path", "lookahead_target", "follow_cmd", "controller")
FORBIDDEN = {"observed_objects", "lidar2d_points", "lidar3d_points", "camera_image", "camera_project", "camera_ray"}


def reference_namespace(source):
    """Bind all reference/contract geometry fixtures to the selected v9 base."""
    ns = load_ns(BP.BASE)
    ns.update(heapq=heapq, START=ns["MISSION"]["start_xy_yaw"], GOAL=ns["MISSION"]["goal_xy_yaw"])
    exec(compile(source, "<B522 reference EDIT with selected base>", "exec"), ns)
    return ns


def controller_contract_checks(source):
    # Reuse the previous fixture logic with an explicit current-base loader.
    # Copy globals rather than mutating the imported B521 module or its files.
    fixture_globals = {**prior_controller_checks.__globals__, "reference_namespace": reference_namespace}
    bound_checks = FunctionType(prior_controller_checks.__code__, fixture_globals)
    return bound_checks(source)


def retention_only_ast(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_base_track_step", "observed_objects"):
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == "_BASE_OBJ_CFG" for n in node.targets):
            for keyword in node.value.keywords:
                if keyword.arg == "keep":
                    keyword.value = ast.Constant(value=1.0)
    return ast.dump(tree)


def masked(source):
    tree = ast.parse(source)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            if fn.name in ("lookahead_target", "follow_cmd"):
                fn.body = [ast.Pass()]
            elif fn.name == "plan_path":
                for nested in fn.body:
                    if isinstance(nested, ast.FunctionDef) and nested.name == "_search_path":
                        nested.body = [ast.Pass()]
    return ast.dump(tree)


def contract_ok(source):
    try:
        tree = ast.parse(source)
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        return masked(source) == masked(BP.SCAFFOLD) and not (set(calls) & FORBIDDEN) and calls.count("choose_command") == 1
    except (SyntaxError, ValueError):
        return False


def mutations(source):
    tree = ast.parse(source)
    variants = []
    def altered(label, change):
        candidate = copy.deepcopy(tree)
        change(candidate)
        variants.append((label, ast.unparse(candidate)))
    def fn(t, name):
        return next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
    altered("remove lookahead helper", lambda t: t.body.remove(fn(t, "lookahead_target")))
    altered("insert manual sensor zero in controller", lambda t: fn(t, "controller").body.insert(-2, ast.parse("if any(o['dist'] < 2 for o in observed_objects()):\n    v, w = 0., 0.").body[0]))
    def forced_replan(t):
        c = fn(t, "controller")
        assignment = next(n for n in c.body if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "replan_now" for v in n.targets))
        assignment.value = ast.parse("_REPLAN_PENDING or t >= _NEXT_PLAN_T", mode="eval").body
        for n in ast.walk(c):
            if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "_NEXT_PLAN_T" for v in n.targets):
                n.value = ast.parse("t + 5.0", mode="eval").body
    altered("force five-second replanning", forced_replan)
    def infinite(t):
        f = fn(t, "plan_path")
        loop = next(n for n in f.body if isinstance(n, ast.For))
        body = ast.parse("need = _PLAN_CLEARANCE").body + loop.body + ast.parse("_PLAN_CLEARANCE = max(.5, _PLAN_CLEARANCE-.1)").body
        f.body[f.body.index(loop)] = ast.While(test=ast.parse("_PLAN_CLEARANCE >= .5", mode="eval").body, body=body, orelse=[])
    altered("replace finite margins by saturated while-loop", infinite)
    altered("force zero after chooser", lambda t: setattr(fn(t, "controller").body[-1], "value", ast.parse("(0.,0.,done)", mode="eval").body))
    def drop_plan(t):
        c = fn(t, "controller")
        for n in ast.walk(c):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "plan_path":
                n.func.id = "ignored_planner"
    altered("bypass planner call anchor", drop_plan)
    altered("add global waypoint index", lambda t: t.body.insert(0, ast.parse("_PATH_IDX=0").body[0]))
    altered("read sensor from editable follow body", lambda t: fn(t, "follow_cmd").body.insert(0, ast.parse("observed_objects()").body[0]))
    altered("add a second chooser in an editable helper", lambda t: fn(t, "follow_cmd").body.insert(0, ast.parse("choose_command(0,0,target)").body[0]))
    return variants


def finite_shell_checks():
    results = []
    tree = ast.parse(BP.FIXED_PLAN)
    search = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef))
    search.body = ast.parse("return _search_spy(center_clearance)").body
    code = compile(ast.fix_missing_locations(tree), "<fixed B522 planner shell>", "exec")
    def run(requested, candidate=None, blocked_start=False):
        tried = []
        def search_spy(need):
            tried.append(need)
            return candidate
        ns = {"math": math, "_PLAN_CLEARANCE": requested, "_PLAN_POSE": (0., 0., 0.), "GOAL": (2., 0., 0.),
              "map_clearance": lambda x, y: .4 if blocked_start and x == 0 else 10.,
              "segment_clearance": lambda p, q: 10., "check_path": lambda path: {"first_bad_segment": None, "min_gap_map": 9.75},
              "_search_spy": search_spy}
        exec(code, ns)
        return ns["plan_path"](), tried
    for requested, count in [(1., 6), (.5, 1), (2., 16)]:
        returned, tried = run(requested)
        results.append((f"fixed planner returns after finite failure at requested{requested}", returned is None and len(tried) == count and tried[-1] == .5 and tried.count(.5) == 1))
    returned, tried = run(1., [(0.2, 0.), (2., 0.)])
    results.append(("fixed planner rejects a grid-only start even when gap check passes", returned is None and len(tried) == 6))
    returned, tried = run(1., [(0., 0.), (2., 0.)])
    results.append(("fixed planner accepts actual endpoints and checked complete path", returned == [(0., 0.), (2., 0.)] and tried == [1.]))
    returned, tried = run(1., blocked_start=True)
    results.append(("actual start below0.5m is never omitted or relaxed", returned is None and tried == []))
    return results


def cached_path_check(reference):
    ns = reference_namespace(reference)
    plans, chooses = [], []
    pose = (-2., -2., 0.)
    def plan():
        plans.append(ns["_PLAN_POSE"])
        return [tuple(pose[:2]), tuple(ns["GOAL"][:2])]
    def choose(v, w, target):
        chooses.append((v, w))
        return v, w, {"chosen": "follow", "candidates": {"follow": {"t_hit_map": None}}}
    ns.update(plan_path=plan, lookahead_target=lambda path, pose: (.5, .2), follow_cmd=lambda pose, target: (.3, .1), choose_command=choose,
              check_path=lambda path: {"first_bad_segment": None, "min_gap_map": .75})
    for t in (0., 5., 10., 100.):
        ns["controller"](t, pose, None)
    return len(plans) == 1 and len(chooses) == 4


def main():
    checks = []
    def check(name, passed):
        checks.append({"name": name, "pass": bool(passed)})
        print(("PASS " if passed else "FAIL ") + name)
    prompt, prior, base = BP.OUT.read_text(), BP.SRC.read_text(), BP.BASE.read_text()
    reference = REFERENCE.read_text()
    ref_sha = hashlib.sha256(reference.encode()).hexdigest()
    ref_tree = ast.parse(reference)
    fns = [n for n in ref_tree.body if isinstance(n, ast.FunctionDef)]
    check("prompt reproduces from declared B521 changes", BP.build(prior, base) == prompt)
    check("exact B522 title", prompt.splitlines()[0] == BP.TITLE == "# ARENA 미션 프롬프트 B-5-22 (B-5-21 에 제어 흐름 고정과 유한 재시도를 더함)")
    v8 = (ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py").read_text()
    check("B521 prompt and prior v8 preserved; selected v9 hash exact", sha16(prior) == "d7bad20e17058563" and sha16(v8) == "d92eed4891798942" and sha16(base) == BP.BASE_SHA16 == "45b6b77cd2fa8cf0")
    check("v9 changes only track retention and its two documentation strings", retention_only_ast(base) == retention_only_ast(v8) and load_ns(BP.BASE)["_BASE_OBJ_CFG"]["keep"] == 3.0)
    check("fixed scaffold and filled reference are byte-identical to task20", sha16(BP.SCAFFOLD) == "b6b0e03c692396e3" and ref_sha == "80de7d39821096fc516f7cdd4c8b32a6d6810389de6cc4cc56c06cc1205c9d98")
    check("B522 prompt length no greater than B521", len(prompt) <= len(prior))
    check("reference fills exactly the three allowed bodies", contract_ok(reference))
    check("four public functions retain names order and zero-argument planner", [n.name for n in fns] == list(NAMES) and [[a.arg for a in n.args.args] for n in fns] == [[], ["path", "pose"], ["pose", "target"], ["t", "pose", "env"]])
    old_ref = (ARENA / "analysis/b521_reference.py").read_text()
    old_controller = next(n for n in ast.parse(old_ref).body if isinstance(n, ast.FunctionDef) and n.name == "controller")
    check("fixed controller is the unchanged verified B521 reference controller", ast.dump(old_controller) == ast.dump(fns[-1]) == ast.dump(ast.parse(BP.FIXED_CONTROLLER).body[0]) and "TODO" not in BP.FIXED_CONTROLLER)
    check("scaffold supplies no complete A-star implementation", "heapq.heappop" not in BP.SCAFFOLD and "raise NotImplementedError" in BP.SCAFFOLD and "_search_path(center_clearance)" in BP.SCAFFOLD)
    check("planner shell preserves finite floor and endpoint plus complete-path checks", "dict.fromkeys(needs)" in BP.FIXED_PLAN and "math.dist(path[0], start_xy)" in BP.FIXED_PLAN and "math.dist(path[-1], goal_xy)" in BP.FIXED_PLAN and "checked['min_gap_map'] >= 0.25" in BP.FIXED_PLAN)
    check("fixed-flow and only-three-editable-body instructions explicit", BP.CONTRACT in prompt and "허용된 세 TODO 몸통만 채워" in prompt and "controller()` 본문과 `plan_path()` 의 재시도·검사 틀은 고정" in prompt)
    check("EDIT observation calls forbidden including diagnostic exception", BP.INPUT_DOC in prompt and "진단 용도도 금지" in prompt and "추가 진단에만 사용" not in prompt and "observed_objects()` 가 돌려준 가까운 물체" not in prompt)
    check("sensor/controller-DBG permission conflicts removed", "그 값을 어떻게 쓸지는 직접 작성" not in prompt and "controller()` 안에서 `print" not in prompt and "진단을 위해 고정 controller 를 바꾸거나" in prompt)
    check("three-second retention docs use scan time and base-only observation access", "스캔 기준(t−0.1 s)으로 3.0 s" in prompt and "age_s는 약3.1 s" in prompt and "1.0 s 넘게 보이지" not in prompt and "EDIT 직접 호출은 금지합니다" in prompt)
    check("target progress and blocked-anchor correction do not lower safety margin", BP.PATH_DOC in prompt and "양의 거리 앞" in prompt and "0.5 m 하한을 낮추지 않습니다" in prompt and "다른 유효 칸과 안전한 연결" in prompt)
    check("goal thresholds and chooser escape preservation retained", BP.FOLLOW_DOC in prompt and "선택된 탈출 명령도 그대로 반환" in prompt and "5초마다 재계산" in prompt)
    check("first response is tool and all tool calls explicitly200s", prompt.split("## 지금 할 일\n", 1)[1].strip().splitlines()[0] == BP.FIRST_ACTION and prompt.count(BP.CALL_EXAMPLE) == 3 and "최상위 sim_seconds 를 빠뜨리지 않습니다" in prompt)
    check("final output remains one actual EDIT code fence with exact PASS-code rule", BP.FORMAT_DOC in prompt and "시험의 `code` 와 최종 답에는 수정한 `[EDIT REGION]` 블록만" in prompt and "pass 뒤 코드가 바뀌면 다시 시험" in prompt)
    banned = (r"\bM[1-5]\b", r"arena_mover", r"start_zone", r"goal_south", r"왕복", r"되돌아", r"11\.6")
    check("model input adds no mover truth or reference implementation", not any(re.search(p, prompt + base) for p in banned) and reference not in prompt and "b522_reference.py" not in prompt and ast.get_source_segment(reference, fns[0]) not in prompt)
    mutant_variants = mutations(reference)
    for name, candidate in mutant_variants:
        compile(candidate, "<valid contract mutant>", "exec")
        check("contract rejects mutation: " + name, not contract_ok(candidate))
    check("all rejected mutants remain valid executable Python", len(mutant_variants) == 9)
    for name, passed in finite_shell_checks():
        check(name, passed)
    for name, passed in controller_contract_checks(reference):
        check(name, passed)
    check("healthy cached path is not replanned at5s or repeated intervals", cached_path_check(reference))
    selected_ns = reference_namespace(reference)
    selected_ns["_PLAN_POSE"] = selected_ns["START"]
    planned = selected_ns["plan_path"]()
    path_check = selected_ns["check_path"](planned)
    check("actual reference path geometry executes against selected v9 helpers", selected_ns["_BASE_OBJ_CFG"]["keep"] == 3.0 and path_check["first_bad_segment"] is None and path_check["min_gap_map"] >= .25 and tuple(planned[0]) == tuple(selected_ns["START"][:2]) and tuple(planned[-1]) == tuple(selected_ns["GOAL"][:2]))
    A.BASE_FILE, A.BASE, A.SP.BASE = BP.BASE, base, base
    A.B_MODE = "b5"
    A.A21 = A.A22 = A.A23 = A.A4T = False
    model_input = A.model_input(BP.OUT.name)
    marker = "\n\n---\n\n`base_carter_run.py`:\n\n```python\n"
    check("actual model input is prompt plus exact unchanged base", model_input == prompt + marker + base + "```\n")
    scaffold_file = A.build(A.TL.extract("```python\n" + BP.SCAFFOLD + "\n```"))
    check("actual harness accepts scaffold syntax and EDIT boundary", A.static_check(scaffold_file)[0] and A.TL._outside(scaffold_file) == A.TL._outside(base))
    args = json.loads(json.dumps({"code": reference, "sim_seconds": 200}))
    final_file = A.build(A.TL.extract("```python\n" + reference + "\n```"))
    check("filled reference real tool splice and final extraction are identical", A.build(args["code"]) == final_file and A.static_check(final_file)[0] and A.TL._outside(final_file) == A.TL._outside(base))
    actual = next(n for n in ast.parse(final_file).body if isinstance(n, ast.FunctionDef) and n.name == "controller")
    check("spliced controller is exact fixed AST", ast.dump(actual) == ast.dump(fns[-1]))
    check("JSON-wrapper and explanation-only finals rejected", not A.static_check(A.build(A.TL.extract(json.dumps({"code": reference}))))[0] and not A.static_check(A.build(A.TL.extract("설명만 제출")))[0])
    offline_path = ARENA / "results/analysis/b522_gate_repair_offline.json"
    offline = json.loads(offline_path.read_text()) if offline_path.exists() else {}
    check("new repair offline evidence passes on current reference and selected v9", offline.get("verdict") == "pass" and offline.get("base_sha16") == sha16(base) and offline.get("reference_sha256") == ref_sha and offline.get("passed") == offline.get("total") and offline.get("total", 0) > 0)
    sim_path = ARENA / "results/analysis/b522_v9_reference_sim.json"
    sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    rows = sim.get("results", [])
    snapshot = Path(sim.get("reference_snapshot", "/nonexistent"))
    expected_labels = {"v1"} | {f"seed{seed}_a{i:02d}" for seed in (1, 2) for i in range(30)}
    check("completed CPU61 matches exact reference and snapshot", sim.get("complete") is True and sim.get("exact_source_matches") is True and sim.get("reference_sha256") == ref_sha and sim.get("base_sha16") == sha16(base) and snapshot.is_file() and snapshot.read_text() == reference and len(rows) == 61 and {r.get("label") for r in rows} == expected_labels)
    random_rows = [r for r in rows if r.get("label") != "v1"]
    v1 = [r for r in rows if r.get("label") == "v1"]
    check("CPU61 reports actual outcomes and zero runtime errors", sim.get("random_total") == len(random_rows) == 60 and sim.get("random_passed") == sum(r.get("verdict") == "pass" for r in random_rows) and len(v1) == 1 and sim.get("v1_pass") == (v1[0].get("verdict") == "pass") and sim.get("runtime_errors") == 0)
    check("CPU61 audits exact one chooser and unchanged return every tick", sim.get("audit_pass") is True and bool(rows) and all(r.get("reference_sha256") == ref_sha and r.get("base_sha16") == sha16(base) and r.get("audit", {}).get("exact_one_choose_per_tick") is True and all(r["audit"].get(k) == 0 for k in ("call_order_errors", "returned_command_changes", "goal_finish_errors")) for r in rows))
    check("base and shared harness remain unchanged", sha16(BP.BASE.read_text()) == BP.BASE_SHA16 and sha16((ARENA / "agent_loop.py").read_text()) == AGENT_LOOP_SHA16)
    summary = {"verdict": "pass" if all(c["pass"] for c in checks) else "fail", "passed": sum(c["pass"] for c in checks), "total": len(checks), "base_sha16": sha16(base), "base_changed": True, "base_version": "v9", "prior_v8_preserved": sha16(v8) == "d92eed4891798942", "base_repair": "keep1s_to3s_only",
               "prompt_file_sha16": sha16(prompt), "prompt_file_chars": len(prompt), "prompt_sha16": sha16(model_input), "prompt_chars": len(model_input), "prior_prompt_chars": len(prior), "prompt_reduction_chars": len(prior)-len(prompt),
               "reference_sha256": ref_sha, "reference_in_model_input": False, "fixed_controller_from_prior_reference": True,
               "reference_sim_file": str(sim_path.relative_to(ARENA)), "reference_sim": {k: sim.get(k) for k in ("random_passed", "random_total", "v1_pass", "runtime_errors")},
               "offline_file": str(offline_path.relative_to(ARENA)), "shortenings": BP.SHORTENINGS, "mutation_cases": len(mutations(reference)),
               "limitations": ["Immutable scaffold validation and mutation detection are preparation selftests; the unchanged operational harness does not enforce this AST contract.", "Unfilled scaffold passes only syntax/splicing checks; its TODOs must be implemented before running.", "Synthetic callbacks verify fixed control flow and finite retry mechanics; real geometry and CPU61 evidence are separate.", "CPU mission outcomes do not guarantee Isaac/model-generated behavior, nor reproduce all original sensor/physics effects."], "checks": checks}
    out = ARENA / "results/analysis/b522_selftest.json"
    if out.exists():
        shutil.copy2(out, ARENA / "results" / f"archive_{out.name}_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"SELFTEST={summary['passed']}/{summary['total']} PROMPT_SHA16={summary['prompt_sha16']} PROMPT_CHARS={summary['prompt_chars']} FILE_SHA16={summary['prompt_file_sha16']} FILE_CHARS={len(prompt)}")
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
