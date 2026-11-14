"""B-5-4 (실험 29) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_4.md
--base-file base_carter_run_ros_map.py. No Isaac, no LLM, no files written.

base_carter_run_ros_map.py = base_carter_run_ros.py + one map block (WALLS, the 43 FIXED_OBSTACLES from the scene
listing, map_rects, map_clearance, segment_clearance, unmapped_points) just above the EDIT REGION."""
import math
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

ROS = (ARENA / "base_carter_run_ros.py").read_text()
MAP = (ARENA / "base_carter_run_ros_map.py").read_text()
B53 = (ARENA / "PROMPT_B5_3.md").read_text()
B54 = (ARENA / "PROMPT_B5_4.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


b0, b1 = MAP.index("# ---- 지도 기본 코드 ----"), MAP.index("# [EDIT REGION] mission controller")
check("map base = ros base + the map block only", (MAP[:b0] + MAP[b1:]).replace("\n\n\n\n", "\n\n") == ROS.replace("\n\n\n\n", "\n\n")
      or MAP[:b0].rstrip("\n") + "\n\n\n" + MAP[b1:] == ROS)
ns = {"np": np, "math": math}
exec(MAP[MAP.index("MISSION = dict("):MAP.index("\nSTART = ")], ns)
exec(MAP[b0:b1], ns)
check("map has 14 boxes + 43 fixed obstacles", ns["map_rects"]().shape == (57, 4) and len(ns["FIXED_OBSTACLES"]) == 43)
P = ARENA / "results/analysis/c_allsensor_base_probe"
pts = [np.load(P / f) for f in ("snapA_front2d_world.npy", "snapA_back2d_world.npy")]
p3 = np.load(P / "snapA_lidar3d_world.npy"); p3 = p3[(p3[:, 2] > 0.1) & (p3[:, 2] < 2.0)]
check("real lidar at the start: no unmapped point (2D at margin 0.1, 3D at 0.15), as the prompt says",
      all(len(ns["unmapped_points"](p, 0.1)) == 0 for p in pts) and len(ns["unmapped_points"](p3, 0.15)) == 0
      and "`margin` 0.15 m 에서 남는 점이 없었습니다" in B54)
check("an open-floor point is unmapped", len(ns["unmapped_points"](np.array([[0.0, 0.0]]), 0.3)) == 1)
sc = ns["segment_clearance"]
check("segment_clearance flags B-5-2 a1's squeeze past box 06 (0 m)", sc((9.48, 12.0), (9.48, 16.5)) == 0.0)
check("segment_clearance of the passing B-5 a0 route stays > 0.5 m",
      min(sc(a, b) for a, b in zip([(-9.0, -10.8), (-5.0, -8.0), (-5.0, 6.0), (3.5, 8.0), (5.5, 12.0), (5.0, 14.0)],
                                   [(-5.0, -8.0), (-5.0, 6.0), (3.5, 8.0), (5.5, 12.0), (5.0, 14.0), (8.0, 16.5)])) > 0.5)
title = B54.splitlines()[0]
check("B-5-4 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
check("prompt: map section, map principle, block-only answers, test first",
      all(x in B54 for x in ("## 지도 기본 코드", "센서 반응은 지도에 없는 물체에만", "답은 `[EDIT REGION]` 블록만 내십시오",
                             "첫 응답은 반드시 `run_isaac_sim` 호출이어야 합니다", "(−10.30, 9.35, −12.15, 17.95)")))
check("prompt: no approximate walls, no mover leak, same-scene sentences kept",
      "대략 x" not in B54 and not any(x in B54 for x in ("M1", "arena_mover", "start_zone", "goal_south"))
      and "최종 채점은 같은 장면" in B54 and "장면에 없는 물체" not in B54)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = MAP, MAP, "b5", True, True
region = (ARENA / "results/turns/prompt_b5/isaac_logs/gpu_gpt_oss_120b_pB5_a0_carter_run.py").read_text()
region = region[region.index("# [EDIT REGION]"):region.index("# ========================= [END EDIT REGION]")]
built = A.build(A.TL.extract("```python\n" + region + "```"))
check("EDIT-REGION-only answer splices into the map base and passes rule 1",
      "def unmapped_points" in built and A.static_check(built)[0])
msg = A.model_input("PROMPT_B5_4.md")
check("first message shows the map base, no simulator-only sensing, no limit",
      "def segment_clearance" in msg and "raycast" not in msg.lower() and "최대 10번" not in msg)
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
