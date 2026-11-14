"""B-4 (실험 24) checks for agent_loop --b4 --base-file base_carter_run_ros_noenv.py. No Isaac, no LLM, no files written.

The noenv base is the B-2-1/B-3 ros base with two edits only (the env line of the docstring, and the
controller called with env=None). With it as BASE: an EDIT-REGION-only answer splices in and passes
rule 1; a full file built on the B-3 ros base is disqualified; the first message names no
simulator-only sensing function at all and shows the noenv base as base_carter_run.py.
"""
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

ROS = (ARENA / "base_carter_run_ros_withenv.py").read_text()
NOENV = (ARENA / "base_carter_run_ros_noenv.py").read_text()
A.BASE, A.SP.BASE, A.B_MODE = NOENV, NOENV, "b4"
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def region(text):
    b, e = text.index("# [EDIT REGION]"), text.index("# ========================= [END EDIT REGION]")
    return text[b:e]


check("noenv base keeps the ros base's sensor block and EDIT REGION", region(NOENV) == region(ROS)
      and NOENV[:NOENV.index("# [EDIT REGION]")].endswith(ROS[ROS.index("# ---- Carter"):ROS.index("# [EDIT REGION]")]))
check("noenv base passes env=None to the controller", "return controller(t, pose, None)" in NOENV
      and "ctrl = _robot_only" in NOENV)
answer = A.TL.extract("```python\n" + region((ARENA / "gpu_gpt_oss_120b_pB3_a1_t2_carter_run.py").read_text()) + "```")
built = A.build(answer)
check("EDIT REGION answer splices into the noenv base", "_robot_only" in built and "def lidar2d_points" in built)
check("spliced answer passes rule 1", A.static_check(built)[0])
check("full file on the B-3 ros base is disqualified",
      not A.static_check((ARENA / "gpu_gpt_oss_120b_pB3_a1_t2_carter_run.py").read_text())[0])
msg = A.model_input("PROMPT_B4.md")
check("first message names no simulator-only sensing function", "raycast" not in msg.lower())
check("first message shows the noenv base as base_carter_run.py",
      "`base_carter_run.py`:\n\n```python\n" + NOENV + "```" in msg)
title = (ARENA / "PROMPT_B4.md").read_text().splitlines()[0]
check("B-4 title passes only the b4 gate", all((m == "b4") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4")))
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
