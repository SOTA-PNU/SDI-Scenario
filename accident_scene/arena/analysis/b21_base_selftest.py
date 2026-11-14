"""B-2-1 (실험 22) checks for agent_loop --base-file base_carter_run_ros_withenv.py. No Isaac, no LLM, no files written.

The ros base must be the A-2 base plus the sensor block above the EDIT REGION, and with it as BASE:
an EDIT-REGION-only answer splices into the ros base and passes rule 1; a full ros-based file passes;
a full A-2-based file (no sensor block) and a file with an edited sensor block are disqualified;
the first message carries the sensor block under the name base_carter_run.py.
"""
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

A2 = (ARENA / "base_carter_run_a2.py").read_text()
ROS = (ARENA / "base_carter_run_ros_withenv.py").read_text()
SNIP = (ARENA / "analysis" / "c_sensor_base_all_snippet.py").read_text()
A.BASE, A.SP.BASE, A.B_MODE = ROS, ROS, "b2"
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def region(text):
    b, e = text.index("# [EDIT REGION]"), text.index("# ========================= [END EDIT REGION]")
    return text[b:e]


check("ros base = A-2 base + sensor block before the EDIT REGION", ROS.replace(SNIP + "\n\n", "", 1) == A2)
answer = A.TL.extract("```python\n" + region((ARENA / "gpu_gpt_oss_120b_pB2all_a0_t2_carter_run.py").read_text()) + "```")
built = A.build(answer)
check("EDIT REGION answer splices into the ros base", "def lidar2d_points" in built and region(built).strip())
check("spliced answer passes rule 1", A.static_check(built)[0])
full_ros = ROS.replace(region(ROS), region(built))
check("full ros-based file passes rule 1", A.static_check(full_ros)[0])
full_a2 = (ARENA / "gpu_gpt_oss_120b_pB2all_a0_t2_carter_run.py").read_text()
check("full A-2-based file (no sensor block) is disqualified", not A.static_check(full_a2)[0])
edited = full_ros.replace("_BODY = (-0.70, 0.25, -0.34, 0.34)", "_BODY = (-0.80, 0.25, -0.34, 0.34)", 1)
check("edited sensor block is disqualified", edited != full_ros and not A.static_check(edited)[0])
msg = A.model_input("PROMPT_B2_1.md")
check("first message shows the ros base as base_carter_run.py",
      "`base_carter_run.py`:\n\n```python\n" + ROS + "```" in msg and "def camera_ray" in msg)
check("prompt part carries no sensor code", "def lidar2d_points" not in (ARENA / "PROMPT_B2_1.md").read_text())
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
