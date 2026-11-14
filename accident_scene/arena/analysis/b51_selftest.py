"""B-5-1 (실험 26) checks for agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_1.md.
No Isaac, no LLM, no files written.

B-5-1 is B-5 with the test drives on the grading scene (moving obstacles v1) too. The prompt, the tool
description and the loop must all say the same thing: tests and grading share one scene.
"""
import inspect
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

BASE = (ARENA / "base_carter_run_ros.py").read_text()
B5 = (ARENA / "PROMPT_B5.md").read_text()
B51 = (ARENA / "PROMPT_B5_1.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


title = B51.splitlines()[0]
check("B-5-1 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
check("prompt says tests and grading share one scene",
      "시험 주행(`run_isaac_sim`)과 최종 채점은 같은 장면에서 돕니다" in B51 and "최종 채점 장면과 같습니다" in B51)
check("no sentence left that says the tests lack the objects",
      "장면에 없는 물체" not in B51 and "채점 장면에만" not in B51)
check("only 5 lines differ from B-5 (title, 2 in the instructions, scene line, collision row)",
      len(set(B5.splitlines()) - set(B51.splitlines())) == 5)
check("B-5's requirements and the no-limit text are kept",
      "우회해서 목표로 가는 알고리즘" in B51 and "센서 기본 코드를 참고해" in B51 and "한도가 없습니다" in B51 and "최대 10번" not in B51)

A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED = BASE, BASE, "b5", True
A.TEST_MOVERS = False
d5 = A.tools()[0]["function"]["description"]
A.TEST_MOVERS = True
d51 = A.tools()[0]["function"]["description"]
check("B-5 tool text unchanged (box-warehouse scene, no limit)", "MISSION 의 상자를 놓은 창고 장면" in d5 and "한도가 없다" in d5)
check("B-5-1 tool text: same scene as grading, no limit",
      "(최종 채점과 같은 장면과 하네스)" in d51 and "한도가 없다" in d51 and "최대 10번" not in d51)
src = inspect.getsource(A)
check("main turns the movers on for test drives",
      'os.environ["ARENA_MOVERS"] = B_MOVERS     # B-5-1: test drives on the grading scene' in src)
check("grading leaves the movers on afterwards for B-5-1",
      'os.environ["ARENA_MOVERS"] = B_MOVERS     # B-5-1: the next test drives keep the movers' in src)
msg = A.model_input("PROMPT_B5_1.md")
check("first message: no simulator-only sensing, no limit",
      "raycast" not in msg.lower() and "최대 10번" not in msg)
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
