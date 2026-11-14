"""B-5-12 (실험 38) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_12.md
--base-file base_carter_run_ros_map_dbg_heapq_lidartime.py --turn-note-file PROMPT_B5_10_turn_note.md. No Isaac, no LLM.

- PROMPT_B5_12.md is PROMPT_B5_11.md + analysis/b512_build_prompt.EDITS only;
- the follow method says 0.3 m/s and 0.4 rad/s everywhere it gives a speed, and the reason's numbers match
  results/analysis/claude_lidar_rate/speed_false_points.json;
- the new values passed the fixed-map Isaac check on a sparse and a dense path (analysis/b512_follower_check.py);
- no mover position, lane or time is given."""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b512_build_prompt as BP  # noqa: E402

B511 = (ARENA / "PROMPT_B5_11.md").read_text()
B512 = (ARENA / "PROMPT_B5_12.md").read_text()
BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py").read_text()
SPEED = json.loads((ARENA / "results/analysis/claude_lidar_rate/speed_false_points.json").read_text())
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_12.md == PROMPT_B5_11.md + the build EDITS only", BP.build(B511) == B512)
title = B512.splitlines()[0]
check("B-5-12 title passes only the b5 gate", all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
law = B512[B512.index("## 추종 방법"):B512.index("## 센서 기본 코드")]
check("follow method: v cap 0.3, every w clamp +-0.4, no 0.6/0.8 cap left in the law",
      "v = min(0.3, 0.5 * dist + 0.1)" in law and law.count("clamp(1.5 * e, -0.4, 0.4)") == 3
      and "min(0.6" not in law and "-0.8, 0.8" not in law and "-0.6, 0.6" not in law)
check("reason numbers match speed_false_points.json (turning 0.6-0.9: 374, 0.2-0.4: 18 at margin 0.25)",
      round(SPEED["|w| 0.6-0.9 rad/s"]["false_m025"]) == 374 and round(SPEED["|w| 0.2-0.4 rad/s"]["false_m025"]) == 18
      and "호출 하나에 374개" in law and "18개가 남았습니다" in law)
check("five rules, the crossing-object rule and the matching work step are in",
      "다음 다섯 가지를 지켜" in B512 and "그 물체가 지나다니는 길 밖에서 기다렸다가" in B512
      and "경로를 가로지르는 물체는 그 길 밖에서\n   기다렸다 건너는 `controller()`" in B512)
check("no mover position, lane, period or time given",
      not any(x in B512 for x in ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "0.5 m/s", "11.6", "5.54", "M5")))
res = {}
for kind in ("sparse", "dense"):
    f = ARENA / f"results/analysis/b512_follower_check/b512_follower_check_{kind}_result.json"
    res[kind] = json.loads(f.read_text()) if f.exists() else None
check(f"fixed-map Isaac check passed on both paths: { {k: (v or {}).get('verdict') for k, v in res.items()} }",
      all(v and v.get("verdict") == "pass" for v in res.values()))
RUN = (ARENA / "run_b512_sequential.sh").read_text()
CC = {}
for v in ("contact_check_b511_a2_movers", "contact_check_sparse_fixedmap"):
    f = ARENA / f"results/analysis/contact_all_links_check/{v}_result.json"
    CC[v] = json.loads(f.read_text()) if f.exists() else None
check("run script counts contacts on every robot body (export CARTER_CONTACT_ALL_LINKS=1), and the Isaac check saw M1 on a2's wheel "
      "while the fixed-map run stayed at 0 contacts",
      "\nexport CARTER_CONTACT_ALL_LINKS=1\n" in RUN and CC["contact_check_b511_a2_movers"]
      and CC["contact_check_b511_a2_movers"]["verdict"] == "fail"
      and "/World/arena_mover_00" in CC["contact_check_b511_a2_movers"]["metrics"]["collision_partners"]
      and "/World/Nova_Carter_ROS/wheel_right" in CC["contact_check_b511_a2_movers"]["metrics"]["collision_robot_links"]
      and CC["contact_check_sparse_fixedmap"] and CC["contact_check_sparse_fixedmap"]["verdict"] == "pass"
      and CC["contact_check_sparse_fixedmap"]["metrics"]["collision_count"] == 0)
A.BASE = A.SP.BASE = BASE
msg = A.model_input("PROMPT_B5_12.md")
print(f"B-5-12 input sha16 {hashlib.sha256(msg.encode()).hexdigest()[:16]} ({len(msg)}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
