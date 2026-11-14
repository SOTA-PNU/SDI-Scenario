"""Gate G3 for prompt A-2-1 (REPORT.md 실험 15): agent_loop --a21 plumbing, no Isaac, no LLM.

A scripted fake model drives agent_loop.Run through two turns; TL.grade is replaced by a fake
that leaves the files a real run leaves (Kit stdout, kept.json, the telemetry file copied
from a real instrumented Isaac run). Checks: dispatch to the three tools, a recipient-less
call recovered by its argument keys, query errors not using the budget, budget exhaustion and
tool_choice 'none' after two extra calls, the scene attached once per turn (and not after a
region-less scene_obstacles call), the 8k cap, the run header hashes.

    python3 analysis/a21_loop_selftest.py <real a21diag.json> [out_dir]
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
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/a21_loop_selftest").resolve()
if OUT.exists():
    shutil.rmtree(OUT)
REGION = '''def controller(t, pose, env):
    """selftest"""
    return 0.0, 0.0, True
'''


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def resp(content=None, calls=None, raw=None, ct=50):
    lp = {"content": [{"token": raw}]} if raw else {"content": []}
    msg = {"role": "assistant", "content": content, "reasoning": "r"}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg, "finish_reason": "stop", "logprobs": lp}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": ct,
                      "completion_tokens_details": {"reasoning_tokens": 10}}}


SCRIPT = [
    # turn 1
    resp(calls=[call("run_isaac_sim", {"code": REGION, "sim_seconds": 30}, "c1")]),          # scene attached
    resp(calls=[call("scene_obstacles", {"region": [-10, -5, -8, -2]}, "c2")]),               # region: no scene_sent change
    resp(calls=[call("check_route", {"points": "bad"}, "c3")]),                                # error, no budget
    resp(calls=[call("check_route", {"points": [[-4.7, -7.3], [-3.7, 5.1], [8.0, 16.5]]}, "c4")]),
    resp(content="", raw='<|channel|>commentary <|constrain|>json<|message|>{"points": [[-4.7, -7.3]]}<|call|>', ct=80),
    resp(calls=[call("scene_obstacles", {}, "c6")]),                                           # budget 3/3 used
    resp(calls=[call("check_route", {"points": [[0, 0]]}, "c7")]),                             # over budget 1
    resp(calls=[call("check_route", {"points": [[0, 0]]}, "c8")]),                             # over budget 2 -> none
    resp(content="```python\n" + REGION + "```\n"),                                            # submission (fails)
    # turn 2
    resp(calls=[call("scene_obstacles", {}, "c10")]),                                          # scene_sent before test
    resp(calls=[call("run_isaac_sim", {"code": REGION}, "c11")]),                              # no scene attached
    resp(calls=[call("run_isaac_sim", {"code": REGION}, "c12")]),                              # same_as_test
    resp(content="```python\n" + REGION + "```\n"),                                            # submission (passes)
]
seen_choice = []


def fake_post(base, payload, timeout):
    i = len(seen_choice)
    seen_choice.append(payload.get("tool_choice"))
    assert [t["function"]["name"] for t in payload["tools"]] == ["run_isaac_sim", "scene_obstacles", "check_route"]
    return json.loads(json.dumps(SCRIPT[i]))


gradings = []


def fake_grade(variant):
    iso = A.TL.ISAAC_LOG_DIR
    gradings.append(variant)
    runner = A.HERE / f"{variant}_carter_run.py"
    if runner.exists():                      # keep_isaac_logs copies the runner; test() then removes it
        shutil.copy(runner, iso / runner.name)
    (iso / f"{variant}.carter-run.stdout.log").write_text("DBG hello\n[mission] TIMEOUT at t=30.00s\n")
    (iso / f"{variant}.carter-run.stderr.log").write_text("")
    if "_test" in variant:
        shutil.copy(DIAG, iso / f"{variant}_a21diag.json")
    (iso / f"{variant}.kept.json").write_text(json.dumps({"result_files": []}))
    final = len([g for g in gradings if "_test" not in g]) == 2
    return {"verdict": "pass" if final else "fail", "criteria": {}, "oracles": [],
            "metrics": {"final_dist_m": 1.0, "collision_count": 0, "collision_partners": [],
                        "first_collision_t": None, "time_to_goal_s": None, "sim_time_s": 30.0},
            "_wall_s": 1.0}


A.post = fake_post
A.TL.grade = fake_grade
A.A21 = True
a = types.SimpleNamespace(model="fake", tag="selftest", sample=0, base="http://fake", out_dir=str(OUT),
                          prompt_file="PROMPT_A2_1.md", max_turns=2, test_budget=10, query_budget=3,
                          max_tokens=100, temperature=0.7, top_p=0.95, seed_base=1000, budget_chars=300000,
                          request_timeout=10, resume=False, redo_turn=None, redo_result=None, redo_first_call=None)
A.A21_BLOCK = None
rc = A.Run(a).main()
# graded submissions are written next to the arena runners by turn_loop.write_submission
for f in A.HERE.glob("selftest_a0*_carter_run.py"):
    shutil.move(str(f), str(OUT / f.name))
left = sorted(p.name for p in A.HERE.glob("selftest_a0*"))
log = json.loads((OUT / "selftest_a0_turns.json").read_text())
tools = sorted(OUT.glob("selftest_a0_c*_tool*.json"))
res = {p.name: json.loads(p.read_text())["result"] for p in tools}
checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


t1, t2 = log["turns"]
ok("passed at turn 2", rc == 0 and log["passed_at_turn"] == 2, log.get("passed_at_turn"))
ok("header hashes", log.get("scene_sha16") and log.get("tool_sha16") and log.get("input_sha16"), log.get("scene_sha16"))
ok("check_route record keys", all(q.get("min_gap") is not None for q in t1.get("queries", []) if q["tool"] == "check_route"
                                  and not q.get("error")), t1.get("queries"))
ok("scene attached on turn-1 first test", "scene" in res["selftest_a0_c01_tool1.json"])
ok("region query leaves no scene_sent", res["selftest_a0_c02_tool1.json"]["count"] < 20)
ok("bad check_route -> error, no budget", "error" in res["selftest_a0_c03_tool1.json"] and res["selftest_a0_c03_tool1.json"]["query"] == 1)
ok("check_route answered", "segments" in res["selftest_a0_c04_tool1.json"])
rec = t1["calls"][4]
ok("recipient-less call recovered as check_route", rec.get("recovered_tool_call") is not None
   and "segments" in res["selftest_a0_c05_tool1.json"])
ok("budget exhausted -> error", "error" in res["selftest_a0_c07_tool1.json"] and "한도" in res["selftest_a0_c07_tool1.json"]["error"])
ok("tool_choice none after two extra calls", seen_choice[8] == "none", seen_choice)
ok("turn 2 resets tool_choice", seen_choice[9] == "auto", seen_choice)
ok("no scene after a region-less query", "scene" not in res["selftest_a0_c11_tool1.json"])
ok("same_as_test names the first test that ran the code", res["selftest_a0_c11_tool1.json"]["code_ran"]["same_as_test"] == "t1-1"
   and res["selftest_a0_c12_tool1.json"]["code_ran"]["same_as_test"] == "t1-1"
   and res["selftest_a0_c01_tool1.json"]["code_ran"]["same_as_test"] is None)
ok("same_as_base false for a changed region", res["selftest_a0_c11_tool1.json"]["code_ran"]["same_as_base"] is False)
sizes = {k: len(json.dumps({kk: vv for kk, vv in v.items() if kk != "scene"}, ensure_ascii=False)) for k, v in res.items()}
ok("every result within the 8k cap", max(sizes.values()) <= 8000 + 200, sizes)
ok("tests recorded with a21 digest", all(t.get("a21") for t in t1["tests"] + t2["tests"]))
ok("nothing left in the arena tree", not left, left)
ok("call cap = tests + queries + 6", A.call_cap(a) == 10 + 3 + 6, A.call_cap(a))
ok("run header names the A-2-1 parts", all(log.get(k) for k in ("scene_sha16", "listing_sha16", "inject_sha16",
                                                                   "a21_code_sha16")))
ok("queries within budget recorded, errors uncounted", [q["tool"] for q in t1.get("queries", [])]
   == ["scene_obstacles", "check_route", "check_route", "check_route"]
   and [q["query"] for q in t1["queries"]] == [1, 1, 2, 3], t1.get("queries"))
for n, good, det in checks:
    print(("PASS " if good else "FAIL ") + n + ("" if good else f"  -> {det}"))
print(f"{sum(g for _, g, _ in checks)}/{len(checks)} checks passed; files in {OUT}")
sys.exit(0 if all(g for _, g, _ in checks) else 1)
