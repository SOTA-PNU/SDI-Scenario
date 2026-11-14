"""Self-test for prompt A-2-2 (REPORT.md 실험 16): agent_loop --a22 plumbing, no Isaac, no LLM.

Checks the A-2-2 changes with a scripted fake model and a fake grader: scene in the first
message; rule 1 unchanged (a full file with an edit outside the EDIT REGION is not run);
check_route overlaps carry rects and xy; the submission gate holds an answer that fails the
static check or differs from the code that passed a test, accepts the same code with other
comments, and stops holding after three notes in a turn.

    python3 analysis/a22_loop_selftest.py <real a21diag.json> [out_dir]
"""
import json
import shutil
import sys
import types
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

DIAG = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/a22_loop_selftest").resolve()
if OUT.exists():
    shutil.rmtree(OUT)
GOOD = '''def controller(t, pose, env):
    """good"""
    return 0.0, 0.0, True
'''
GOOD_COMMENTED = '''# same code, other comments
def controller(t, pose, env):
    """good"""
    # nothing else changed
    return 0.0, 0.0, True
'''
OTHER = '''def controller(t, pose, env):
    """other"""
    return 0.1, 0.0, True
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


SCRIPT = [
    # turn 1
    resp(calls=[call("check_route", {"points": [[-3.7, 5.1], [8.0, 16.5]]}, "c1")]),        # overlaps (box 08)
    resp(calls=[call("run_isaac_sim", {"code": FULL_WITH_OUTSIDE_EDIT}, "c2")]),           # rule 1: static DQ, not run
    resp(calls=[call("run_isaac_sim", {"code": GOOD}, "c3")]),                             # pass
    answer(OTHER),                                                                        # held 1 (not the passed code)
    answer(FULL_WITH_OUTSIDE_EDIT),                                                       # held 2 (static check)
    answer(GOOD_COMMENTED),                                                               # same code -> graded (fail)
    # turn 2
    answer(OTHER),                                                                        # held 1
    answer(OTHER),                                                                        # held 2
    answer(OTHER),                                                                        # held 3
    answer(OTHER),                                                                        # limit reached -> graded (pass)
]
sent = []


def fake_post(base, payload, timeout):
    sent.append(payload)
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
        verdict = "pass" if len(graded) == 2 else "fail"
    return {"verdict": verdict, "criteria": {}, "oracles": [],
            "metrics": {"final_dist_m": 0.2, "collision_count": 0, "collision_partners": [],
                        "first_collision_t": None, "time_to_goal_s": None, "sim_time_s": 30.0}, "_wall_s": 1.0}


A.post = fake_post
A.TL.grade = fake_grade
A.A21 = A.A22 = True
A.A21_BLOCK = None
a = types.SimpleNamespace(model="fake", tag="selftest22", sample=0, base="http://fake", out_dir=str(OUT),
                          prompt_file="PROMPT_A2_2.md", max_turns=2, test_budget=10, query_budget=40,
                          max_tokens=100, temperature=0.7, top_p=0.95, seed_base=1000, budget_chars=400000,
                          request_timeout=10, resume=False, redo_turn=None, redo_result=None, redo_first_call=None)
rc = A.Run(a).main()
for f in A.HERE.glob("selftest22_a0*_carter_run.py"):
    shutil.move(str(f), str(OUT / f.name))
left = sorted(p.name for p in A.HERE.glob("selftest22_a0*"))
log = json.loads((OUT / "selftest22_a0_turns.json").read_text())
res = {p.name: json.loads(p.read_text())["result"] for p in sorted(OUT.glob("selftest22_a0_c*_tool*.json"))}
ev = [json.loads(l) for l in open(OUT / "selftest22_a0_events.jsonl")]
gates = [e["decision"] for e in ev if e["type"] == "submission_gate"]
checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


first = sent[0]["messages"][0]["content"]
ok("scene listing in the first message", '"obstacles":[' in first and "contact_offset_m" in first)
cr = res["selftest22_a0_c01_tool1.json"]
ok("check_route overlaps carry xy and rects", cr.get("overlaps") and "xy" in cr["overlaps"][0]
   and len(cr["overlaps"][0]["objects"][0]) == 4 and cr["overlaps"][0]["objects"][0][3], cr.get("overlaps"))
t1 = res["selftest22_a0_c02_tool1.json"]
ok("rule 1 kept: outside edit is a static DQ, not run", "EDIT REGION 밖" in t1.get("static_check", "")
   and "result" not in t1, t1.get("static_check"))
ok("gate decisions", gates == ["held", "held", "passed_test", "held", "held", "held", "gate_limit"], gates)
ok("gate notes reached the model", any("이 답은 채점하지 않았습니다" in (m.get("content") or "")
                                       for m in sent[4]["messages"] if m["role"] == "user")
   and any("정적 검사에 걸렸습니다" in (m.get("content") or "") for m in sent[5]["messages"] if m["role"] == "user"))
ok("graded submissions: commented same code, then the 4th answer", len(graded) == 2
   and '"""good"""' in graded[0][1] and '"""other"""' in graded[1][1])
ok("submission file is base + region", all(A.TL._outside(g[1]) == A.TL._outside(A.BASE) for g in graded))
ok("passed at turn 2", rc == 0 and log["passed_at_turn"] == 2, log.get("passed_at_turn"))
ok("header says a22", log.get("a22") is True and log.get("query_budget") == 40)
ok("nothing left in the arena tree", not left, left)
for n, good, det in checks:
    print(("PASS " if good else "FAIL ") + n + ("" if good else f"  -> {det}"))
print(f"{sum(g for _, g, _ in checks)}/{len(checks)} checks passed; files in {OUT}")
sys.exit(0 if all(g for _, g, _ in checks) else 1)
