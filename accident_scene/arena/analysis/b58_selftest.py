"""B-5-8 (실험 33) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_8.md
--base-file base_carter_run_ros_map_dbg_heapq.py. No Isaac, no LLM, no files written (reads the b58 smoke summary).

- PROMPT_B5_8.md is PROMPT_B5_7.md + analysis/b58_build_prompt.EDITS only; the new base adds four import lines only;
- the wall limit the prompt states (1,700 s, TimeoutError) is what carter_run_locked.sh does (Isaac-checked);
- every recounted lesson matches the 실험 30-32 records."""
import csv
import hashlib
import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b58_build_prompt as BP  # noqa: E402

DBG = (ARENA / "base_carter_run_ros_map_dbg.py").read_text()
NEW = (ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text()
B57 = (ARENA / "PROMPT_B5_7.md").read_text()
B58 = (ARENA / "PROMPT_B5_8.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_8.md == PROMPT_B5_7.md + the build EDITS only", BP.build(B57) == B58)
title = B58.splitlines()[0]
check("B-5-8 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
added = [ln for ln in NEW.splitlines() if ln not in DBG.splitlines()]
check("new base = B-5-7 base + the heapq/collections/deque import block only",
      NEW.replace("".join(f"{ln}\n" for ln in added), "", 1) == DBG and len(added) == 4
      and any(ln.startswith("import heapq") for ln in added) and any("from collections import deque" in ln for ln in added))
ns = {"np": np, "math": math}
exec(NEW[NEW.index("MISSION = dict("):NEW.index("\nSTART = ")], ns)
exec(NEW[NEW.index("# ---- 지도 기본 코드 ----"):NEW.index("# ---- DBG 기록 ----")], ns)
check("heapq, collections, deque are names in the base namespace", all(k in ns for k in ("heapq", "collections", "deque")))
wrap = (ARENA / "carter_run_locked.sh").read_text()
tl = (ARENA / "turn_loop.py").read_text()
check("wrapper default wall limit 1700 s < turn_loop 1800 s; prompt states 1,700 s and TimeoutError",
      'WALL_LIMIT="${CARTER_WALL_LIMIT_S:-1700}"' in wrap and "timeout=1800)" in tl and "TimeoutError: 시험 주행이 벽시계" in wrap
      and "벽시계 1,700 s 를 넘으면 하네스가 멈추고" in B58 and "`TimeoutError`" in B58)
check("'0.05 s per call -> 600 s drive = 30 min wall' arithmetic", abs(0.05 * 600 * 60 - 1800) < 1e-9 and "0.05 s 가 걸리면 600 s 주행이 벽시계 30분" in B58)
sm = json.loads((ARENA / "results/analysis/b58_smoke/summary.json").read_text())
wl, hb = sm["walllimit"], sm["heapq_base"]
check("Isaac: wall limit stopped the run with a TimeoutError from both extractors, nothing left, lock free",
      wl["result"]["_runtime_error"].startswith("TimeoutError: 시험 주행이 벽시계 70 s") and wl["runtime_error_text"].startswith("TimeoutError")
      and not wl["left"] and wl["lock_free"] and wl["wall_s"] < 120)
check("Isaac: the 실험 32 a1 turn-1 answer (NameError heapq) runs in the new base",
      hb["result"].get("_runtime_error") is None and hb["result"]["metrics"]["first_collision_t"] == 37.817)
turns = {}
for d, tag in (("prompt_b5_5", "pB55"), ("prompt_b5_6", "pB56"), ("prompt_b5_7", "pB57")):
    for s in ("a0", "a1"):
        turns[(tag, s)] = json.loads((ARENA / f"results/turns/{d}/gpu_gpt_oss_120b_{tag}_{s}_turns.json").read_text())["turns"]
graded = [t for v in turns.values() for t in v]
summ = [r for tag in ("pB55", "pB56", "pB57")
        for dd in json.loads((ARENA / f"results/analysis/b5x_method_gpu_gpt_oss_120b_{tag}.json").read_text()).values()
        for r in dd["rows"] if r["kind"] == "graded"]
check("'앞선 세 실험의 채점 12번 가운데 8번은 시험 0번, 나머지 4번은 pass 없이 고친 코드, 12번 모두 실패'",
      len(graded) == 12 and sum(t["tests_used"] == 0 for t in graded) == 8
      and sum(1 for r in summ if r["tests_in_turn"] > 0 and not r["same_code_as_tests"]) == 4
      and all(t["verdict"] != "pass" for t in graded) and "채점 12번 가운데 8번은" in B58)
R = ns["map_rects"]()


def rect_clear(x, y):
    dx = np.maximum(np.maximum(R[:, 0] - x, x - R[:, 1]), 0)
    dy = np.maximum(np.maximum(R[:, 2] - y, y - R[:, 3]), 0)
    return float(np.hypot(dx, dy).min())


drove = []
for (tag, s), tt in turns.items():
    if tag in ("pB56", "pB57"):
        for t in tt:
            f = ARENA / f"results/{t['variant']}_trajectory.csv"
            if t.get("final_dist_m") is not None and f.exists():
                rows = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f.open())]
                drove.append((t, min(rect_clear(x, y) for x, y in rows if not (math.isnan(x) or math.isnan(y)))))
m1 = [t for t, _ in drove if t.get("partners") == ["/World/arena_mover_00"]]
check("'주행한 채점 다섯 번 모두 지도 물체에서 0.8 m 넘게, 네 번은 6–34 s 에 출발 구역, 한 번은 지나서 다른 물체'",
      len(drove) == 5 and all(c > 0.8 for _, c in drove) and len(m1) == 4
      and 6.0 <= min(t["first_collision_t"] for t in m1) and max(t["first_collision_t"] for t in m1) <= 34.0
      and sum(1 for t, _ in drove if t.get("partners") and t["partners"] != ["/World/arena_mover_00"]) == 1
      and "6–34 s" in B58)
waits = [r for r in json.loads((ARENA / "results/analysis/b5x_method_gpu_gpt_oss_120b_pB57.json").read_text())["a0"]["rows"]
         if r["kind"] == "test" and r["turn"] == 2 and r["n"] in (9, 10)]
check("'직전 실험에서 멈춰 기다리기만 한 시험은 부딪히지 않았지만 600 s 안에 목표로 가지 못했다' (실험 32 a0 시험 9, 10)",
      len(waits) == 2 and all(not r["partners"] and r["sim_s"] > 600 and r["final_dist_m"] > 30 for r in waits))
heap = [t for (tag, s), tt in turns.items() if tag in ("pB56", "pB57") for t in tt
        if t["turn"] == 1 and "heapq" in str(t.get("runtime_error") or "")]
ev = [json.loads(l) for tag, s in (("pB56", "a0"), ("pB57", "a1"))
      for l in open(ARENA / f"results/turns/prompt_b5_{tag[-1]}/gpu_gpt_oss_120b_{tag}_{s}_events.jsonl")]
heap_ev = [e for e in ev if e["type"] == "grading" and e["turn"] == 1 and "heapq" in str(e.get("runtime_error"))]
check("'앞선 두 실험에서 1회차 채점 두 번이 heapq NameError'", len(heap_ev) == 2 and "NameError" in str(heap_ev[0]["runtime_error"]))
src_trim = inspect.getsource(A.trim)
check("'대화가 길어지면 오래된 시험 기록부터 정리' is what agent_loop does (trim, budget 300,000 chars)",
      "Drop the oldest exchange" in src_trim and "default=300000" in (ARENA / "agent_loop.py").read_text()
      and B58.rstrip().endswith("답을 내지 말고 계속 시험하십시오.") and B58.count("## 지금 할 일") == 1)
check("prompt: no mover position, same-scene sentences kept",
      not any(x in B58 for x in ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "0.5 m/s", "11.6"))
      and "최종 채점은 같은 장면" in B58 and "장면에 없는 물체" not in B58)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = NEW, NEW, "b5", True, True
reg = (ARENA / "results/turns/prompt_b5_7/isaac_logs/gpu_gpt_oss_120b_pB57_a1_carter_run.py").read_text()
reg = reg[reg.index("# [EDIT REGION]"):reg.index("# ========================= [END EDIT REGION]")]
built = A.build(A.TL.extract("```python\n" + reg + "```"))
check("an EDIT-REGION answer splices into the new base and passes rule 1", "import heapq" in built and A.static_check(built)[0])
sha = {}
for p, base in (("PROMPT_B5_7.md", DBG), ("PROMPT_B5_8.md", NEW)):
    A.BASE = A.SP.BASE = base
    msg = A.model_input(p)
    sha[p] = (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg), msg)
logged = json.loads((ARENA / "results/turns/prompt_b5_7/gpu_gpt_oss_120b_pB57_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-7 input",
      (logged.get("input_sha16"), logged.get("input_chars")) == sha["PROMPT_B5_7.md"][:2])
check("first message shows the import block and ends with '지금 할 일'",
      "import heapq  # noqa" in sha["PROMPT_B5_8.md"][2] and "## 지금 할 일" in sha["PROMPT_B5_8.md"][2])
print(f"B-5-8 input sha16 {sha['PROMPT_B5_8.md'][0]} ({sha['PROMPT_B5_8.md'][1]}자), "
      f"B-5-7 {sha['PROMPT_B5_7.md'][0]} ({sha['PROMPT_B5_7.md'][1]}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
