"""B-5-2 (실험 27) checks for agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_2.md.
No Isaac, no LLM, no files written. B-5-2 is B-5-1 plus a table of what each sensor sees (measured on this
server from the 실험 21 sensor snapshots: results/analysis/c_allsensor_probe*/)."""
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

B51 = (ARENA / "PROMPT_B5_1.md").read_text()
B52 = (ARENA / "PROMPT_B5_2.md").read_text()
BASE = (ARENA / "base_carter_run_ros.py").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


title = B52.splitlines()[0]
check("B-5-2 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
added = [l for l in B52.splitlines() if l not in set(B51.splitlines())]
removed = [l for l in B51.splitlines() if l not in set(B52.splitlines())]
check("only the title changed; the rest is an added block", len(removed) == 1 and removed[0].startswith("# ARENA"))
check("coverage table with the measured numbers",
      all(x in B52 for x in ("−140° ~ +140°", "+40° ~ ±180° ~ −40°", "−11° ~ +13°", "좌우 ±60°", "10 Hz", "60 Hz", "0.05–30 m")))
check("says the two 2D lidars together cover 360°", "두 라이다를 합쳐야 로봇 둘레 360° 를 봅니다" in B52)
check("nothing about the moving obstacles leaks",
      not any(x in B52 for x in ("M1", "arena_mover", "start_zone", "west_corridor", "center", "goal_west", "goal_south")))
check("B-5-1 scene sentences kept (needed by --b-test-movers)",
      "최종 채점은 같은 장면" in B52 and "장면에 없는 물체" not in B52)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = BASE, BASE, "b5", True, True
msg = A.model_input("PROMPT_B5_2.md")
check("first message: no simulator-only sensing, no limit", "raycast" not in msg.lower() and "최대 10번" not in msg)
check("tool text: same scene as grading, no limit",
      "(최종 채점과 같은 장면과 하네스)" in A.tools()[0]["function"]["description"])
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
