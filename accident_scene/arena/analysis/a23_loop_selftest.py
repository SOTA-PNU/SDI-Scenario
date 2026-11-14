"""Self-test for prompt A-2-3 (REPORT.md 실험 17): agent_loop --a23 plumbing, no Isaac, no LLM.

Scripted fake model and fake grader, small budgets (2 tests, 0 queries, so 8 calls a turn):
- turn 1: rule 1 unchanged (a full file with an edit outside the EDIT REGION is not run); an
  answer that differs from the passing test only in docstrings and comments is graded at once;
- turn 2: untested answers are held more than three times while tests are left; when the
  tests are used up the next answer is graded;
- turn 3: two over-budget queries switch tool_choice to "none", a held answer opens the tools
  again; untested answers are held until the call cap, and the answer at the cap is graded.
Also: the first message is the A-2-3 prompt with the scene; --a22 gate keys are unchanged.

    python3 analysis/a23_loop_selftest.py <real a21diag.json> [out_dir]
"""
import ast
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

DIAG = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/a23_loop_selftest").resolve()
if OUT.exists():
    shutil.rmtree(OUT)
GOOD = '''def controller(t, pose, env):
    """good"""
    return 0.0, 0.0, True
'''
GOOD_REDOC = '''"""a module docstring the test did not have"""
# other comments
def controller(t, pose, env):
    """a different docstring"""
    # nothing else changed
    return 0.0, 0.0, True
'''
OTHER = '''def controller(t, pose, env):
    """other"""
    return 0.1, 0.0, True
'''
THIRD = '''def controller(t, pose, env):
    """third"""
    return 0.2, 0.0, True
'''
FULL_WITH_OUTSIDE_EDIT = "# -*- coding: utf-8 -*-\n" + A.BASE.replace(
    '    """No control code yet: the robot never moves."""\n    return 0.0, 0.0, False',
    '    """good"""\n    return 0.0, 0.0, True')


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def resp(content=None, calls=None):
    msg = {"role": "assistant", "content": content, "reasoning": "r"}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg, "finish_reason": "stop", "logprobs": {"content": []}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                      "completion_tokens_details": {"reasoning_tokens": 10}}}


def answer(code):
    return resp(content="```python\n" + code + "```\n")


ROUTE = {"points": [[-3.7, 5.1], [8.0, 16.5]]}
SCRIPT = [
    # turn 1 (2 tests, 0 queries: cap 8 calls a turn)
    resp(calls=[call("run_isaac_sim", {"code": FULL_WITH_OUTSIDE_EDIT}, "a1")]),   # test 1: static DQ, not run
    resp(calls=[call("run_isaac_sim", {"code": GOOD}, "a2")]),                     # test 2: pass
    answer(GOOD_REDOC),                                                           # docstrings only -> graded (fail)
    # turn 2
    answer(OTHER), answer(OTHER), answer(OTHER), answer(OTHER),                   # held x4, tests left 2
    resp(calls=[call("run_isaac_sim", {"code": THIRD}, "b1")]),                   # test 1 (pass)
    resp(calls=[call("run_isaac_sim", {"code": THIRD}, "b2")]),                   # test 2 (pass), none left
    answer(OTHER),                                                                # no test left -> graded (fail)
    # turn 3
    resp(calls=[call("check_route", ROUTE, "c1")]),                               # over query budget 1
    resp(calls=[call("check_route", ROUTE, "c2")]),                               # over 2 -> tool_choice none
    answer(OTHER),                                                                # held (sent with none) -> auto
    answer(OTHER), answer(OTHER), answer(OTHER), answer(OTHER),                   # held
    answer(OTHER),                                                                # 8th call = cap -> graded (pass)
]
sent = []


def fake_post(base, payload, timeout):
    sent.append(payload)
    if len(sent) > len(SCRIPT):
        return json.loads(json.dumps(answer(OTHER)))
    return json.loads(json.dumps(SCRIPT[len(sent) - 1]))


graded = []


def fake_grade(variant):
    iso = A.TL.ISAAC_LOG_DIR
    runner = A.HERE / f"{variant}_carter_run.py"
    if runner.exists():
        shutil.copy(runner, iso / runner.name)
    (iso / f"{variant}.carter-run.stdout.log").write_text("")
    (iso / f"{variant}.carter-run.stderr.log").write_text("")
    (iso / f"{variant}.kept.json").write_text(json.dumps({"result_files": []}))
    if "_test" in variant:
        shutil.copy(DIAG, iso / f"{variant}_a21diag.json")
        verdict = "pass"
    else:
        graded.append((variant, runner.read_text()))
        verdict = "pass" if len(graded) == 3 else "fail"
    return {"verdict": verdict, "criteria": {}, "oracles": [],
            "metrics": {"final_dist_m": 0.2, "collision_count": 0, "collision_partners": [],
                        "first_collision_t": None, "time_to_goal_s": None, "sim_time_s": 30.0}, "_wall_s": 1.0}


# --a22 keys must not change: same as the plain AST hash used in 실험 16
A.A21, A.A22, A.A23 = True, True, False
key22 = A.code_key(GOOD_REDOC)
plain = hashlib.sha256(ast.dump(ast.parse(GOOD_REDOC)).encode()).hexdigest()[:12]
key22_good = A.code_key(GOOD)

A.post = fake_post
A.TL.grade = fake_grade
A.A21 = A.A22 = A.A23 = True
A.A21_BLOCK = None
a = types.SimpleNamespace(model="fake", tag="selftest23", sample=0, base="http://fake", out_dir=str(OUT),
                          prompt_file="PROMPT_A2_3.md", max_turns=3, test_budget=2, query_budget=0,
                          max_tokens=100, temperature=0.7, top_p=0.95, seed_base=1000, budget_chars=400000,
                          request_timeout=10, resume=False, redo_turn=None, redo_result=None, redo_first_call=None)
rc = A.Run(a).main()
for f in A.HERE.glob("selftest23_a0*_carter_run.py"):
    shutil.move(str(f), str(OUT / f.name))
left = sorted(p.name for p in A.HERE.glob("selftest23_a0*"))
log = json.loads((OUT / "selftest23_a0_turns.json").read_text())
ev = [json.loads(l) for l in open(OUT / "selftest23_a0_events.jsonl")]
gates = [(e["turn"], e["decision"]) for e in ev if e["type"] == "submission_gate"]
checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


first = sent[0]["messages"][0]["content"]
ok("first message: A-2-3 prompt + scene", first.startswith("# ARENA 미션 프롬프트 A-2-3")
   and '"obstacles":[' in first and "contact_offset_m" in first and "실제로 로봇을" in first)
ok("tools: check_route says scored-0 overlaps block", "scored 가 0 인 물체" in json.dumps(sent[0]["tools"], ensure_ascii=False))
ok("--a22 gate key unchanged (docstrings still count there)", key22 == plain and key22 != key22_good, (key22, plain))
t1 = json.loads((OUT / "selftest23_a0_c01_tool1.json").read_text())["result"]
ok("rule 1 kept: outside edit is a static DQ, not run", "EDIT REGION 밖" in t1.get("static_check", "")
   and "result" not in t1, t1.get("static_check"))
g1 = [d for t, d in gates if t == 1]
ok("turn 1: docstring-only edit of the passing test is graded at once", g1 == ["passed_test"], g1)
g2 = [d for t, d in gates if t == 2]
ok("turn 2: held more than 3 times while tests are left, graded when none left",
   g2 == ["held"] * 4 + ["no_test_left"], g2)
choices = [p.get("tool_choice") for p in sent]
# turn 1 sends 3 requests, turn 2 sends 7: request 12 is the answer after two over-budget queries
ok("held answer re-opens the tools after tool_choice none", choices[10:14] == ["auto", "auto", "none", "auto"],
   choices[10:14])
g3 = [d for t, d in gates if t == 3]
ok("turn 3: held until the call cap, the answer at the cap is graded",
   g3 == ["held"] * 5 + ["call_cap"] and len(log["turns"][2]["calls"]) == A.call_cap(a), (g3, A.call_cap(a)))
notes = [m.get("content") or "" for m in sent[4]["messages"] if m["role"] == "user"]
ok("gate note names the passed tests and the new rule", any("pass 를 받은 시험: 없음" in n and "docstring" in n
                                                            and "남아 있는 동안에는" in n for n in notes))
ok("three graded submissions, file is base + region", len(graded) == 3
   and all(A.TL._outside(g[1]) == A.TL._outside(A.BASE) for g in graded))
ok("passed at turn 3", rc == 0 and log["passed_at_turn"] == 3, log.get("passed_at_turn"))
ok("header says a22 and a23", log.get("a22") is True and log.get("a23") is True and log.get("query_budget") == 0)
ok("nothing left in the arena tree", not left, left)
for n, good, det in checks:
    print(("PASS " if good else "FAIL ") + n + ("" if good else f"  -> {det}"))
print(f"{sum(g for _, g, _ in checks)}/{len(checks)} checks passed; files in {OUT}")
sys.exit(0 if all(g for _, g, _ in checks) else 1)
