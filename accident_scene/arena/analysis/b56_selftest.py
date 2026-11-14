"""B-5-6 (실험 31) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_6.md
--base-file base_carter_run_ros_map.py. No Isaac, no LLM, no files written.

PROMPT_B5_6.md must be exactly PROMPT_B5_5.md with analysis/b56_build_prompt.EDITS applied: no hand-picked waypoints,
the route computed from the map at run time. It must not carry a route or an algorithm name, and the facts it adds
(1.1 m route on the exact map, map_clearance cost, 30 min wall limit, heapq allowed) must hold."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b56_build_prompt as BP  # noqa: E402

MAP = (ARENA / "base_carter_run_ros_map.py").read_text()
B55 = (ARENA / "PROMPT_B5_5.md").read_text()
B56 = (ARENA / "PROMPT_B5_6.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_6.md == PROMPT_B5_5.md + the build EDITS only", BP.build(B55) == B56)
title = B56.splitlines()[0]
check("B-5-6 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
check("no hand-picked waypoint permission left, the computed-route rule present",
      "숫자로 직접 적어도 됩니다" not in B56 and "경유점이나 경로의 x, y 좌표를 손으로 골라 코드에 적지 않습니다" in B56
      and "다음 네 가지를 지켜" in B56 and "## 경로 계산" in B56 and "## 경유점 설계" not in B56)
leaks = ("-4.7", "−4.7", "-3.7, 5.1", "0.7, 11.3", "2.3, 15.1", "A*", "A-star", "Dijkstra", "다익스트라", "BFS",
         "너비 우선", "RRT", "potential field")
check("no route and no algorithm name in the prompt", not any(x in B56 for x in leaks))
ns = {"np": np, "math": math}
exec(MAP[MAP.index("MISSION = dict("):MAP.index("\nSTART = ")], ns)
exec(MAP[MAP.index("# ---- 지도 기본 코드 ----"):MAP.index("# [EDIT REGION] mission controller")], ns)
a3 = [(-9.0, -10.8), (-4.7, -7.3), (-3.7, 5.1), (0.7, 11.3), (2.3, 15.1), (8.0, 16.5)]
check("'1.1 m 이상 떨어진 경로' holds on the exact map (A-3 verified route, 실험 12)",
      min(ns["segment_clearance"](p, q) for p, q in zip(a3, a3[1:])) >= 1.1 and "1.1 m 이상 떨어진 경로" in B56)
t0 = time.perf_counter()
for i in range(3000):
    ns["map_clearance"](-5.0 + i * 1e-3, 0.0)
us = 1e6 * (time.perf_counter() - t0) / 3000
check(f"map_clearance costs about 0.014 ms as the prompt says (now {us:.1f} us)", us < 50 and "약 0.014 ms" in B56)
tl = (ARENA / "turn_loop.py").read_text()
check("one test run is cut at 1800 s wall (turn_loop.grade), the prompt says 30 min",
      "timeout=1800)" in tl and "벽시계로 30분 안에 끝나야 합니다" in B56)
check("prompt: no mover leak, same-scene sentences kept (agent_loop --b-test-movers gate)",
      not any(x in B56 for x in ("M1", "arena_mover", "start_zone", "goal_south"))
      and "최종 채점은 같은 장면" in B56 and "장면에 없는 물체" not in B56)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = MAP, MAP, "b5", True, True
region = ("# [EDIT REGION] mission controller - modify ONLY this block\n"
          "# ===========================================================================\n"
          "import heapq\nfrom collections import deque\n\n\n"
          "def controller(t, pose, env):\n    q = []\n    heapq.heappush(q, (0.0, 0))\n    return 0.0, 0.0, False\n\n\n")
built = A.build(A.TL.extract("```python\n" + region + "```"))
check("an EDIT-REGION answer that imports heapq and collections passes the static check",
      "import heapq" in built and A.static_check(built)[0] and "`heapq`, `collections` 만 허용합니다" in B56)
sha = {}
for p in ("PROMPT_B5_5.md", "PROMPT_B5_6.md"):
    msg = A.model_input(p)
    sha[p] = (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg), msg)
logged = json.loads((ARENA / "results/turns/prompt_b5_5/gpu_gpt_oss_120b_pB55_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-5 input",
      (logged.get("input_sha16"), logged.get("input_chars")) == sha["PROMPT_B5_5.md"][:2])
m56 = sha["PROMPT_B5_6.md"][2]
check("first message shows the map base and the new rule, no simulator-only sensing, no limit",
      "def segment_clearance" in m56 and "## 경로 계산" in m56 and "raycast" not in m56.lower() and "최대 10번" not in m56)
print(f"B-5-6 input sha16 {sha['PROMPT_B5_6.md'][0]} ({sha['PROMPT_B5_6.md'][1]}자), "
      f"B-5-5 {sha['PROMPT_B5_5.md'][0]} ({sha['PROMPT_B5_5.md'][1]}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
