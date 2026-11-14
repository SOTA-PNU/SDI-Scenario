"""Self-test for prompt B-1 (project C, REPORT.md 실험 19): agent_loop --b1 plumbing, no Isaac, no LLM.

Scripted fake model and fake grader, 2 turns: the first message is PROMPT_B1.md + base; the tool
is the A-4 run_isaac_sim with the B description (no "same scene as the final grading"); a test
drive runs with ARENA_MOVERS unset; every final grading runs with ARENA_MOVERS=v1 and the
variable is gone afterwards; the run log names the mode and the grading map.

    python3 analysis/b_loop_selftest.py [out_dir]
"""
import json
import os
import shutil
import sys
import types
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b_loop_selftest").resolve()
if OUT.exists():
    shutil.rmtree(OUT)
GOOD = 'def controller(t, pose, env):\n    """good"""\n    return 0.0, 0.0, True\n'
OTHER = 'def controller(t, pose, env):\n    """other"""\n    return 0.1, 0.0, True\n'


def resp(content=None, calls=None):
    msg = {"role": "assistant", "content": content, "reasoning": "r"}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg, "finish_reason": "stop", "logprobs": {"content": []}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 10}}}


def call(code, cid):
    return {"id": cid, "type": "function", "function": {"name": "run_isaac_sim",
                                                        "arguments": json.dumps({"code": code, "sim_seconds": 200})}}


SCRIPT = [resp(calls=[call(GOOD, "a1")]), resp(content="```python\n" + GOOD + "```\n"),     # turn 1: test, graded fail
          resp(content="```python\n" + OTHER + "```\n")]                                       # turn 2: graded pass
sent, seen = [], []


def fake_post(base, payload, timeout):
    sent.append(payload)
    return json.loads(json.dumps(SCRIPT[len(sent) - 1]))


def fake_grade(variant):
    seen.append((variant, os.environ.get("ARENA_MOVERS")))
    iso = A.TL.ISAAC_LOG_DIR
    runner = A.HERE / f"{variant}_carter_run.py"
    if runner.exists():
        shutil.copy(runner, iso / runner.name)
    for suf in (".carter-run.stdout.log", ".carter-run.stderr.log"):
        (iso / f"{variant}{suf}").write_text("")
    (iso / f"{variant}.kept.json").write_text(json.dumps({"result_files": []}))
    graded = [v for v, _ in seen if "_test" not in v]
    verdict = "pass" if "_test" in variant or len(graded) == 2 else "fail"
    return {"verdict": verdict, "criteria": {}, "oracles": [],
            "metrics": {"final_dist_m": 0.2, "collision_count": 0, "collision_partners": [],
                        "first_collision_t": None, "time_to_goal_s": 50.0, "sim_time_s": 52.0}, "_wall_s": 1.0}


A.post = fake_post
A.TL.grade = fake_grade
A.A21 = A.A22 = A.A23 = A.A4T = False
A.B_MODE, A.B_MOVERS = "b1", "v1"
os.environ.pop("ARENA_MOVERS", None)
a = types.SimpleNamespace(model="fake", tag="selftestb1", sample=0, base="http://fake", out_dir=str(OUT),
                          prompt_file="PROMPT_B1.md", max_turns=2, test_budget=10, query_budget=20,
                          max_tokens=100, temperature=0.7, top_p=0.95, seed_base=1000, budget_chars=400000,
                          request_timeout=10, resume=False, redo_turn=None, redo_result=None, redo_first_call=None)
rc = A.Run(a).main()
for f in A.HERE.glob("selftestb1_a0*_carter_run.py"):
    shutil.move(str(f), str(OUT / f.name))
left = sorted(p.name for p in A.HERE.glob("selftestb1_a0*"))
log = json.loads((OUT / "selftestb1_a0_turns.json").read_text())
checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


first = sent[0]["messages"][0]["content"]
desc = sent[0]["tools"][0]["function"]["description"]
ok("first message is PROMPT_B1 + base", first.startswith("# ARENA 미션 프롬프트 B-1") and "`base_carter_run.py`:" in first
   and "로봇(NVIDIA Nova Carter)에 탑재된 센서를 활용하는 코드를" in first)
ok("tool is run_isaac_sim with the B description", [t["function"]["name"] for t in sent[0]["tools"]] == ["run_isaac_sim"]
   and "최종 채점과 같은 장면과 하네스" not in desc and "MISSION 의 상자를 놓은 창고 장면" in desc, desc)
ok("test drive ran without movers", [m for v, m in seen if "_test" in v] == [None], seen)
ok("every final grading ran with movers v1", [m for v, m in seen if "_test" not in v] == ["v1", "v1"], seen)
ok("ARENA_MOVERS cleared after the run", "ARENA_MOVERS" not in os.environ)
ok("log names mode and grading map", log.get("b") == "b1" and log.get("grading_movers") == "v1"
   and [t.get("grading_map") for t in log["turns"]] == ["movers_v1", "movers_v1"], [t.get("grading_map") for t in log["turns"]])
ok("passed at turn 2", rc == 0 and log.get("passed_at_turn") == 2, log.get("passed_at_turn"))
ok("nothing left in the arena tree", not left, left)
for n, good, det in checks:
    print(("PASS " if good else "FAIL ") + n + ("" if good else f"  -> {det}"))
print(f"{sum(g for _, g, _ in checks)}/{len(checks)} checks passed; files in {OUT}")
sys.exit(0 if all(g for _, g, _ in checks) else 1)
