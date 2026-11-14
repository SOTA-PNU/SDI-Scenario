"""B-5-3 (실험 28) checks for agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_3.md.
No Isaac, no LLM, no files written. B-5-3 is B-5-2 plus the advice to print sensor values with DBG when a test
drive fails, since the test result itself carries no sensor data."""
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

B52 = (ARENA / "PROMPT_B5_2.md").read_text()
B53 = (ARENA / "PROMPT_B5_3.md").read_text()
BASE = (ARENA / "base_carter_run_ros.py").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


title = B53.splitlines()[0]
check("B-5-3 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
removed = [l for l in B52.splitlines() if l not in set(B53.splitlines())]
check("only the title and the two DBG spots changed",
      len(removed) == 2 and removed[0].startswith("# ARENA") and "때만 출력하십시오" in removed[1])
check("DBG advice present", "시험 결과에는 센서 값이 들어 있지 않습니다" in B53
      and "반응이 켜지고 꺼진 시각" in B53 and "센서 값을 볼 수 있는 곳은 이 출력뿐입니다" in B53)
check("B-5-2 sensor coverage table and same-scene sentences kept",
      "각 센서가 보는 범위" in B53 and "최종 채점은 같은 장면" in B53 and "장면에 없는 물체" not in B53)
check("nothing about the moving obstacles leaks",
      not any(x in B53 for x in ("M1", "arena_mover", "start_zone", "west_corridor", "goal_west", "goal_south")))
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = BASE, BASE, "b5", True, True
msg = A.model_input("PROMPT_B5_3.md")
check("first message: no simulator-only sensing, no limit", "raycast" not in msg.lower() and "최대 10번" not in msg)
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
