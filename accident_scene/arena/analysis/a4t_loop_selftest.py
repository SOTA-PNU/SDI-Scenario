"""Self-test for prompt A-4T (REPORT.md 실험 18): agent_loop --a4t plumbing, no Isaac, no LLM.

Scripted fake model and fake grader, 2 tests a turn (8 calls), 3 turns:
- the first message is PROMPT_A4T.md + base + the seed EDIT REGION and its numbers;
- a passing test reports completion_s (last trajectory row);
- turn 1 passes and the run goes on; the feedback names the completion and the best so far;
- turn 2 fails; the feedback still names the best;
- turn 3 reaches the call cap with tool calls; one more request with tool_choice none gets an
  answer, which is graded; the best becomes the shorter completion.

    python3 analysis/a4t_loop_selftest.py [out_dir]
"""
import json
import shutil
import sys
import types
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/a4t_loop_selftest").resolve()
if OUT.exists():
    shutil.rmtree(OUT)
GOOD = 'def controller(t, pose, env):\n    """good"""\n    return 0.0, 0.0, True\n'
OTHER = 'def controller(t, pose, env):\n    """other"""\n    return 0.1, 0.0, True\n'
THIRD = 'def controller(t, pose, env):\n    """third"""\n    return 0.2, 0.0, True\n'


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


def run(code, cid):
    return resp(calls=[call("run_isaac_sim", {"code": code, "sim_seconds": 200}, cid)])


SCRIPT = [run(GOOD, "a1"), answer(GOOD),                       # turn 1: test pass, graded pass (60 s)
          answer(OTHER),                                        # turn 2: graded fail
          run(THIRD, "c1"), run(THIRD, "c2")] + [run(THIRD, f"c{k}") for k in range(3, 9)] + [
          answer(THIRD)]                                        # turn 3: cap, forced final, graded pass (40 s)
sent = []


def fake_post(base, payload, timeout):
    sent.append(payload)
    return json.loads(json.dumps(SCRIPT[len(sent) - 1]))


graded = []
LAST_T = {1: 60.0, 2: 30.0, 3: 40.0}


def fake_grade(variant):
    iso = A.TL.ISAAC_LOG_DIR
    runner = A.HERE / f"{variant}_carter_run.py"
    if runner.exists():
        shutil.copy(runner, iso / runner.name)
    (iso / f"{variant}.carter-run.stdout.log").write_text("")
    (iso / f"{variant}.carter-run.stderr.log").write_text("")
    (iso / f"{variant}.kept.json").write_text(json.dumps({"result_files": []}))
    if "_test" in variant:
        verdict, last = "pass", 50.0
    else:
        graded.append((variant, runner.read_text()))
        verdict, last = ("fail" if len(graded) == 2 else "pass"), LAST_T[len(graded)]
    (iso / f"{variant}_trajectory.csv").write_text(
        "t,x,y,yaw\n0.0,-9.0,-10.8,1.0138\n" + f"{last},7.9,16.4,1.0\n")
    return {"verdict": verdict, "criteria": {}, "oracles": [],
            "metrics": {"final_dist_m": 0.2, "collision_count": 0, "collision_partners": [],
                        "first_collision_t": None, "time_to_goal_s": last - 3, "sim_time_s": last + 1}, "_wall_s": 1.0}


A.post = fake_post
A.TL.grade = fake_grade
A.A21 = A.A22 = A.A23 = False
A.A4T = True
A.A4T_SEED = A.load_a4t_seed("npu_gpt_oss_120b_pA4_a0_carter_run.py", "results/npu_gpt_oss_120b_pA4_a0_result.json",
                             "results/npu_gpt_oss_120b_pA4_a0_trajectory.csv")
a = types.SimpleNamespace(model="fake", tag="selftest4t", sample=0, base="http://fake", out_dir=str(OUT),
                          prompt_file="PROMPT_A4T.md", max_turns=3, test_budget=2, query_budget=20,
                          max_tokens=100, temperature=0.7, top_p=0.95, seed_base=1000, budget_chars=400000,
                          request_timeout=10, resume=False, redo_turn=None, redo_result=None, redo_first_call=None)
rc = A.Run(a).main()
for f in A.HERE.glob("selftest4t_a0*_carter_run.py"):
    shutil.move(str(f), str(OUT / f.name))
left = sorted(p.name for p in A.HERE.glob("selftest4t_a0*"))
log = json.loads((OUT / "selftest4t_a0_turns.json").read_text())
t1 = json.loads((OUT / "selftest4t_a0_c01_tool1.json").read_text())["result"]
checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


def user_texts(req):
    return [m.get("content") or "" for m in req["messages"] if m["role"] == "user"]


first = sent[0]["messages"][0]["content"]
ok("first message: A-4T prompt + base + seed", first.startswith("# ARENA 미션 프롬프트 A-4T")
   and "처음 코드 (A-4 에서" in first and "(1.5, 16.5)" in first and "완주 시간 88.067 s" in first)
ok("tool is the A-4 run_isaac_sim only", [t["function"]["name"] for t in sent[0]["tools"]] == ["run_isaac_sim"])
ok("passing test reports completion_s", t1.get("completion_s") == 50.0 and t1["result"]["verdict"] == "pass", t1.get("completion_s"))
fb1 = user_texts(sent[2])[-1]
ok("turn 1 pass does not end the run; feedback names completion and best",
   "통과했고 완주 시간은 60.0 s" in fb1 and "가장 짧은 완주 시간은 60.0 s" in fb1 and "처음 코드 88.067 s" in fb1, fb1[-200:])
fb2 = user_texts(sent[3])[-1]
ok("turn 2 fail feedback keeps the best", "판정 결과입니다" in fb2 and "가장 짧은 완주 시간은 60.0 s" in fb2, fb2[-160:])
last = sent[-1]
ok("turn 3: at the cap one more request, tool_choice none, with the final note",
   last.get("tool_choice") == "none" and user_texts(last)[-1] == A.A4T_FINAL_NOTE and len(sent) == 12, (len(sent), last.get("tool_choice")))
ok("three graded submissions, files are base + region", len(graded) == 3
   and all(A.TL._outside(g[1]) == A.TL._outside(A.BASE) for g in graded))
ok("log: completions, best, first pass", log.get("completions") == [60.0, None, 40.0]
   and log.get("best_completion_s") == 40.0 and log.get("passed_at_turn") == 1 and rc == 0, log.get("completions"))
ok("header says a4t and the seed", log.get("a4t") is True and log.get("seed_completion_s") == 88.067
   and log.get("seed_time_to_goal_s") == 84.133)
ok("nothing left in the arena tree", not left, left)
for n, good, det in checks:
    print(("PASS " if good else "FAIL ") + n + ("" if good else f"  -> {det}"))
print(f"{sum(g for _, g, _ in checks)}/{len(checks)} checks passed; files in {OUT}")
sys.exit(0 if all(g for _, g, _ in checks) else 1)
