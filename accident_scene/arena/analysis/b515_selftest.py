"""B-5-15 preparation validation; no HTTP, GPU, Isaac, or process control.

Uses the actual agent_loop extraction/build/static-check path for full executable
file answers. Selection geometry and timing are covered by b515_check_offline.
"""
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A
import b515_build_base as BB
import b515_build_prompt as BP

AGENT_LOOP_SHA16 = "f6ab9cd0d094ba9b"


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def helper_namespace(source):
    """Execute only the existing CPU helpers; no runtime imports or simulator."""
    ns = {"math": math, "np": np, "builtins": SimpleNamespace(print=lambda *a, **k: None)}
    exec(source[source.index("MISSION = dict("):source.index("\nSTART = ")], ns)
    exec(source[source.index("def to_world("):source.index("# ---- 지도 기본 코드 ----")], ns)
    exec(source[source.index("# ---- 지도 기본 코드 ----"):source.index("# ---- DBG 기록 ----")], ns)
    ns["_BASE_POSES"][:] = [(10.0, 0.0, 0.0, 0.0)]
    ns["WALLS"] = (-100.0, 100.0, -100.0, 100.0)
    ns["_MAP_RECTS"][:] = [np.array([[50.0, 51.0, 50.0, 51.0]])]
    return ns


def finite_rejections(base):
    ns = helper_namespace(base)
    rejected = 0
    for index in range(4):
        for bad in (math.nan, math.inf, -math.inf):
            pose = [10.0, 0.0, 0.0, 0.0]
            pose[index] = bad
            ns["_BASE_POSES"][:] = [tuple(pose)]
            try:
                ns["check_motion"](0.3, 0.1)
            except ValueError:
                rejected += 1
    ns["_BASE_POSES"][:] = [(10.0, 0.0, 0.0, 0.0)]
    for key in ("vx", "vy", "seen_t"):
        for vel_ok in (False, True):
            for bad in (math.nan, math.inf, -math.inf):
                item = dict(id=1, vx=0.2, vy=0.1, seen_t=10.0, vel_ok=vel_ok, points=[[2.0, 0.0]])
                item[key] = bad
                ns["observed_objects"] = lambda: [item]
                try:
                    ns["check_motion"](0.3, 0.1)
                except ValueError:
                    rejected += 1
    return rejected == 30


def raw_robust_semantics(old_base, base):
    old, new = helper_namespace(old_base), helper_namespace(base)
    item = dict(id=1, vx=-0.4, vy=0.0, seen_t=10.0, vel_ok=True,
                points=[[1.5, -0.1], [1.5, 0.1]])
    old["observed_objects"] = new["observed_objects"] = lambda: [item]
    a, b = old["check_motion"](0.0, 0.0), new["check_motion"](0.0, 0.0)
    return (all(a[k] == b[k] for k in ("min_gap_obj", "min_gap_map", "t_hit_map", "end_pose"))
            and b["raw_t_hit"] == a["t_hit"] and b["raw_t_hit_obj"] == a["t_hit_obj"]
            and b["t_hit"] < b["raw_t_hit"] and b["robust_min_gap_obj"] < b["min_gap_obj"])


def main():
    checks = []

    def check(name, value):
        checks.append({"name": name, "pass": bool(value)})
        print(("PASS " if value else "FAIL ") + name)

    old_prompt, prompt = BP.SRC.read_text(), BP.OUT.read_text()
    old_base = BB.SRC.read_text()
    base = BP.BASE.read_text()
    old_tree, tree = ast.parse(old_base), ast.parse(base)
    old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    check("prompt reproduces from B-5-14 and the declared edits", BP.build(old_prompt, base) == prompt)
    check("base reproduces from v3 with the B-5-15 v4 builder", BB.build(old_base) == base)
    check("title is exact and passes only the B-5 gate", prompt.splitlines()[0] == BP.TITLE and all((m == "b5") == (f"프롬프트 B-{m[1]}" in prompt.splitlines()[0]) for m in ("b1", "b2", "b3", "b4", "b5")))
    check("decision paragraph follows the updated task06 selection rules", BP.DECISION in prompt and "이 순서를 앞세워" not in BP.DECISION)
    start = base.index("# [EDIT REGION] mission controller")
    end = base.index(A.END_LINE)
    check("choose_command is outside EDIT REGION after check_motion", base.index("def check_motion(") < base.index("def choose_command(") < start < end)
    choose = functions["choose_command"]
    check("choose_command signature has the exact requested default", [p.arg for p in choose.args.args] == ["v_follow", "w_follow", "target_xy", "horizon_s"] and [ast.literal_eval(d) for d in choose.args.defaults] == [3.0])
    preserved = ("map_clearance", "observed_objects", "_base_track_step", "_base_obj_pose_at", "_base_obj_cluster", "_base_obj_velocity", "_drop_body")
    check("map geometry, sensor filtering and tracking are unchanged from v3", all(ast.dump(functions[name]) == ast.dump(old_functions[name]) for name in preserved))
    def givens(t):
        names = {"MISSION", "START", "GOAL", "WALLS", "FIXED_OBSTACLES", "_BASE_SELF_BOX", "_BASE_BODY_BOX", "_BASE_OBJ_CFG"}
        return {target.id: ast.dump(node.value) for node in t.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name) and target.id in names}
    check("scene, body and tracker constants are unchanged", givens(tree) == givens(old_tree))
    banned = (r"\bM[1-5]\b", r"arena_mover", r"start_zone", r"goal_south", r"0\.5 m/s", r"11\.6", r"왕복", r"되돌아", r"끝에서", r"방향을 바꾼", r"상자면")
    check("no mover identity, position, lane, cycle, timing or path hint added", not any(re.search(p, prompt + "\n" + base) for p in banned) and prompt.count("오가") == old_prompt.count("오가") == 1)
    config = BP.base_config(base)
    chooser_doc = BP.chooser_doc(config)
    check("detector docs derive candidate priorities and info schema from the exact base", config == BB.CONFIG and chooser_doc in prompt and all(f"`{name}" in chooser_doc for name in ("follow", "slow", "stop", "fast", "turn_l", "turn_r", "back", "chosen", "reason", "candidates")))
    check("motion docs distinguish raw geometry and robust safety using the configured formula", BP.motion_doc(config) in prompt and "min_gap=min(robust_min_gap_obj, min_gap_map)" in prompt and all(name in prompt for name in ("raw_t_hit_obj", "robust_min_gap_obj", "future_min_gap_obj", "ValueError")))
    check("nonfinite pose, timestamp and even unconfirmed object velocity are rejected", finite_rejections(base))
    check("raw geometry is preserved while robust checks can report an earlier risk", raw_robust_semantics(old_base, base))
    check("all motion including terminal stop passes through choose_command", "choose_command(0.0, 0.0, GOAL[:2])" in prompt and "반환한 `(v, w)` 가 모두 0 일 때만" in prompt and "(0.0, 0.0, True)` 를 돌려줘" not in prompt)
    check("manual wait and duplicate selected-command DBG instructions reconciled", "별도 대기 분기를 만들지 않습니다" in prompt and "자동으로 기록합니다" in prompt and "명령 선택 기록을 중복해서 찍지 않습니다" in prompt)
    check("chooser DBG is independent of the ordinary model DBG limit", "builtins.print(" in ast.get_source_segment(base, choose) and "모델의 추가 DBG\n  줄 제한과 별도로" in prompt and "자동 DBG 는 그 제한과 별도로" in prompt and "처음 15초쯤을" not in prompt)
    check("full executable file is required in both testing and submission", "답은 언제나 실행할 수 있는 전체 코드 파일" in prompt and "- `code` (필수): 실행 가능한 전체 코드 파일" in prompt and "그 시험에 보낸 실행 가능한 전체 코드를" in prompt and "**답은 `[EDIT REGION]` 블록만" not in prompt and "`code` 에는 `[EDIT REGION]` 블록만" not in prompt)
    check("outside-EDIT byte-preservation rule remains explicit", "블록 밖은 base 와 한 글자도\n다르지 않게 그대로 포함" in prompt and "`[EDIT REGION]` 블록 밖은 한 글자도 바꾸지 마십시오" in prompt)
    check("both required sections provide explicit 200-second tool example", prompt.count(BP.CALL_EXAMPLE) == 2 and "생략하면 도구 기본값 600 s 가 적용되므로 생략하지 마십시오" in prompt)
    check("old block-submission line-number lesson removed", not re.search(r"이번 base 에서는 답의 첫 줄이 \d+행", prompt) and "그 전체 파일에서 확인합니다" in prompt)
    A.BASE_FILE = BP.BASE
    A.BASE = A.SP.BASE = base
    A.B_MODE = "b5"
    A.A21 = A.A22 = A.A23 = A.A4T = False
    model_input = A.model_input(BP.OUT.name)
    marker = "\n\n---\n\n`base_carter_run.py`:\n\n```python\n"
    check("real model input embeds the exact current base", model_input == prompt + marker + base + "```\n")
    original = ast.get_source_segment(base, functions["controller"])
    replacement = 'def controller(t, pose, env):\n    v, w, info = choose_command(0.0, 0.0, GOAL[:2])\n    return v, w, False'
    full_source = base.replace(original, replacement, 1)
    extracted = A.TL.extract("```python\n" + full_source + "```\n")
    rebuilt = A.build(extracted)
    ok, detail = A.static_check(rebuilt)
    check("actual full executable file survives extraction/build/static check", ok and rebuilt == full_source and A.TL._outside(rebuilt) == A.TL._outside(base) and "choose_command(0.0, 0.0, GOAL[:2])" in rebuilt[start:end])
    changed_plumbing = rebuilt.replace("MISSION = dict(", "MISSION = dict(\n", 1)
    check("full-file answer with outside-EDIT change is rejected", not A.static_check(changed_plumbing)[0])
    refusal = A.build(A.TL.extract("[No answer provided due to inability to achieve a passing solution.]"))
    check("explanation-only final answer fails actual static check", not A.static_check(refusal)[0])
    check("shared agent_loop and tool schema were not modified", sha16((ARENA / "agent_loop.py").read_text()) == AGENT_LOOP_SHA16)
    offline_path = ARENA / "results/analysis/b515_check_offline.json"
    offline = json.loads(offline_path.read_text()) if offline_path.exists() else {}
    check("offline choice scenarios and timing passed on this exact base", offline.get("verdict") == "pass" and offline.get("base_sha16") == sha16(base))
    summary = {"verdict": "pass" if all(c["pass"] for c in checks) else "fail", "passed": sum(c["pass"] for c in checks), "total": len(checks), "base_file": BP.BASE.name, "base_sha16": sha16(base), "prompt_file_sha16": sha16(prompt), "prompt_sha16": sha16(model_input), "prompt_chars": len(model_input), "answer_format": "full_executable_file", "shared_agent_loop_sha16": AGENT_LOOP_SHA16, "isaac_gate": "pending_gpu", "checks": checks}
    output = ARENA / "results/analysis/b515_selftest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.copy2(output, ARENA / "results" / f"archive_b515_selftest.json_before_rerun_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"SELFTEST={summary['passed']}/{summary['total']} BASE_SHA16={summary['base_sha16']} PROMPT_SHA16={summary['prompt_sha16']} PROMPT_CHARS={summary['prompt_chars']}")
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
