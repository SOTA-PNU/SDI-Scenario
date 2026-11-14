"""B-5 (실험 25) checks for agent_loop --b5 --test-budget 0 --base-file base_carter_run_ros_noenv.py.
No Isaac, no LLM, no files written.

PROMPT_B5.md is PROMPT_B4.md with the user's two requirements (use the sensor base code above the
EDIT REGION; detour around unexpected obstacles) and the test-drive limit removed. With
--test-budget 0 the tool description, the tool result and the between-turn note must not mention
a limit either, while the B-1..B-4 tool text (limit 10) stays as it was.
"""
import inspect
import sys
from argparse import Namespace
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

NOENV = (ARENA / "base_carter_run_ros.py").read_text()   # the B-5 base: harness-env-free (user 2026-10-28)
B4 = (ARENA / "PROMPT_B4.md").read_text()
B5 = (ARENA / "PROMPT_B5.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


title = B5.splitlines()[0]
check("B-5 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
check("B-5 base base_carter_run_ros.py is the harness-env-free one",
      "return controller(t, pose, None)" in NOENV and "raycast" not in NOENV.lower()
      and NOENV == (ARENA / "base_carter_run_ros_noenv.py").read_text())
check("requirement 1 (sensor base code) is in the prompt",
      "`[EDIT REGION]` 바로 위에 있는 센서 기본 코드를 참고해, 센서 데이터를 활용하도록" in B5)
check("requirement 2 (detour algorithm) is in the prompt",
      "예상치 못한 장애물이 나타날 것에 대비해, 장애물을 만나면 우회해서 목표로 가는 알고리즘으로 구현합니다." in B5)
check("no limit text left in the prompt",
      "최대 10번" not in B5 and "tests_left" not in B5 and "시험 주행 횟수에는 한도가 없습니다" in B5)
b4_lines, b5_lines = set(B4.splitlines()), set(B5.splitlines())
check("every other B-4 line is kept", len(b4_lines - b5_lines) == 7)

A.BASE, A.SP.BASE, A.B_MODE = NOENV, NOENV, "b5"
A.UNLIMITED = False
desc10 = A.tools()[0]["function"]["description"]
check("limit-10 runs (B-1..B-4) keep their tool text", "한 회차에 최대 10번." in desc10 and A.tools()[0] is A.TOOL_B)
A.UNLIMITED = True
desc = A.tools()[0]["function"]["description"]
check("unlimited tool text has no limit", "최대 10번" not in desc and "한도가 없다" in desc)
a = Namespace(test_budget=A.UNLIMITED_TESTS, query_budget=0)
check("between-turn note has no number", A.feedback_note(a) == A.FEEDBACK_NOTE_UNLIMITED)
check("call cap is effectively unlimited", A.call_cap(a) >= A.UNLIMITED_TESTS)
src = inspect.getsource(A)
check("tool result drops tests_left when unlimited",
      '**({} if a.test_budget >= UNLIMITED_TESTS else {"tests_left": a.test_budget - n})' in src)
msg = A.model_input("PROMPT_B5.md")
check("first message names no simulator-only sensing and no limit",
      "raycast" not in msg.lower() and "최대 10번" not in msg)
check("first message shows the noenv base as base_carter_run.py",
      "`base_carter_run.py`:\n\n```python\n" + NOENV + "```" in msg)
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
