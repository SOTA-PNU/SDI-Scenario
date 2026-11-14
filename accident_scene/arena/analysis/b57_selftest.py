"""B-5-7 (실험 32) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_7.md
--base-file base_carter_run_ros_map_dbg.py. No Isaac, no LLM, no files written (reads the Isaac smoke summary).

- PROMPT_B5_7.md is PROMPT_B5_6.md + analysis/b57_build_prompt.EDITS only;
- base_carter_run_ros_map_dbg.py differs from base_carter_run_ros_map.py only in the map functions (same values),
  the DBG block above the EDIT REGION and one line in _robot_only; the DBG print keeps what the prompt says;
- every number in the new prompt text matches the code or the 실험 30 / 31 records."""
import contextlib
import csv
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b57_build_prompt as BP  # noqa: E402

OLD = (ARENA / "base_carter_run_ros_map.py").read_text()
NEW = (ARENA / "base_carter_run_ros_map_dbg.py").read_text()
B56 = (ARENA / "PROMPT_B5_6.md").read_text()
B57 = (ARENA / "PROMPT_B5_7.md").read_text()
MARK = "# [EDIT REGION] mission controller - modify ONLY this block\n"
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_7.md == PROMPT_B5_6.md + the build EDITS only", BP.build(B56) == B57)
title = B57.splitlines()[0]
check("B-5-7 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
check("new base = old base outside the map functions, the DBG block and one _robot_only line",
      OLD[:OLD.index("def map_rects():")] == NEW[:NEW.index("_MAP_RECTS = []")]
      and OLD[OLD.index("def unmapped_points("):OLD.index(MARK)] == NEW[NEW.index("def unmapped_points("):NEW.index("# ---- DBG 기록 ----")]
      and OLD[OLD.index(MARK):] == NEW[NEW.index(MARK):].replace('    _BASE_DBG_STATE["t"] = t\n', "", 1))


def load(src):
    ns = {"np": np, "math": math}
    exec(src[src.index("MISSION = dict("):src.index("\nSTART = ")], ns)
    exec(src[src.index("# ---- 지도 기본 코드 ----"):src.index("# ---- DBG 기록 ----") if "# ---- DBG 기록 ----" in src
             else src.index(MARK)], ns)
    return ns


o, n = load(OLD), load(NEW)
rng = np.random.default_rng(1)
P = rng.uniform([-11, -13], [10, 19], size=(2000, 2)); S = rng.uniform([-11, -13, -11, -13], [10, 19, 10, 19], size=(800, 4))
check("map_clearance, segment_clearance, map_rects give the same values as before",
      all(o["map_clearance"](x, y) == n["map_clearance"](x, y) for x, y in P)
      and all(o["segment_clearance"]((a, b), (c, d)) == n["segment_clearance"]((a, b), (c, d)) for a, b, c, d in S)
      and np.array_equal(o["map_rects"](), n["map_rects"]()))
t0 = time.perf_counter()
for i in range(3000):
    n["map_clearance"](-5 + i * 1e-3, 0.0)
t1 = time.perf_counter()
for i in range(200):
    n["segment_clearance"]((-9.0, -10.8), (8.0, 16.5))
t2 = time.perf_counter()
mc_ms, sc_ms = 1e3 * (t1 - t0) / 3000, 1e3 * (t2 - t1) / 200
check(f"timings in the prompt hold (map_clearance {mc_ms:.4f} ms, segment_clearance 32 m {sc_ms:.2f} ms)",
      mc_ms < 0.012 and sc_ms < 1.5 and "약 0.004 ms" in B57 and "약 0.5 ms" in B57)
ns = {}
exec(NEW[NEW.index("# ---- DBG 기록 ----"):NEW.index(MARK)], ns)
exec("def controller(t):\n    print('DBG', round(t, 3))\n    print('plain line')\n", ns)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for k in range(int(20.0 * 60)):
        ns["_BASE_DBG_STATE"]["t"] = k / 60.0
        ns["controller"](k / 60.0)
dbg = [ln for ln in buf.getvalue().splitlines() if ln.startswith("DBG")]
shown, omitted = A.dbg_lines("\n".join(dbg))
check("DBG print: 10 lines in the first 0.5 s, then the first line of every 0.5 s; other lines untouched; "
      "the first 40 shown lines reach ~15 s as the prompt says",
      len(dbg) == 10 + 39 and buf.getvalue().count("plain line") == 1200 and float(shown[39].split()[1]) == 15.0
      and "처음 40줄이 처음 15초쯤을" in B57 and "처음 0.5 s 구간만 열 줄까지 남깁니다" in B57)
sm = json.loads((ARENA / "results/analysis/b57_base_smoke/summary.json").read_text())
check("Isaac smoke run of the new base: same drive as the 실험 31 a0 t2 grading, DBG 953 -> < 80 lines",
      sm["dev_m"] == 0.0 and sm["dbg_ref"] == 953 and sm["dbg_new"] < 80
      and sm["result"]["metrics"]["first_collision_t"] == 10.867)
graded = []
for d in ("prompt_b5_5", "prompt_b5_6"):
    for f in sorted((ARENA / "results/turns" / d).glob("*_turns.json")):
        graded += json.loads(f.read_text())["turns"]
summ = {}
for tag in ("gpu_gpt_oss_120b_pB55", "gpu_gpt_oss_120b_pB56"):
    for s, dd in json.loads((ARENA / f"results/analysis/b5x_method_{tag}.json").read_text()).items():
        summ[(tag, s)] = [r for r in dd["rows"] if r["kind"] == "graded"]
untested0 = sum(1 for t in graded if t["tests_used"] == 0)
tested_not_passed = [r for rows in summ.values() for r in rows if r["tests_in_turn"] > 0 and not r["same_code_as_tests"]]
check("'채점 8번 가운데 5번은 시험 0번, 나머지 3번은 pass 없이 고친 코드, 8번 모두 실패' matches the records",
      len(graded) == 8 and untested0 == 5 and len(tested_not_passed) == 3
      and all(t["verdict"] != "pass" for t in graded) and "8번 가운데 5번은" in B57)
route = json.loads((ARENA / "results/analysis/b56_route_gpu_gpt_oss_120b_pB56.json").read_text())
drove = [r for rows in route.values() for r in rows if r["kind"] == "graded" and r.get("traj_min_clear") is not None]
b56 = [t for t in graded if t["variant"].startswith("gpu_gpt_oss_120b_pB56")]
rows = [(float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
        for r in csv.DictReader(open(ARENA / "results/gpu_gpt_oss_120b_pB56_a0_t2_trajectory.csv"))]
w = [r for r in rows if 9.0 <= r[0] <= 10.8]
turned = sum(abs(math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))) for a, b in zip(w, w[1:]))
spin = max(math.hypot(r[1] - w[0][1], r[2] - w[0][2]) for r in w) < 0.05 and turned > 0.3  # it swung back and forth
check("'주행한 채점 두 번 모두 0.8 m 넘게, 6–11 s 에 움직이는 물체, 한 번은 제자리 회전 중, 뒤쪽 끝 약 0.6 m' matches",
      len(drove) == 2 and all(r["traj_min_clear"] > 0.8 for r in drove)
      and sorted(t["first_collision_t"] for t in b56 if t.get("first_collision_t")) == [6.267, 10.867]
      and all(t["partners"] == ["/World/arena_mover_00"] for t in b56 if t.get("first_collision_t")) and spin
      and abs(math.hypot(0.59, 0.18) - 0.6) < 0.05)
check("prompt: no mover position, same-scene sentences kept",
      not any(x in B57 for x in ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "x = -7", "0.5 m/s"))
      and "최종 채점은 같은 장면" in B57 and "장면에 없는 물체" not in B57)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = NEW, NEW, "b5", True, True
reg = (ARENA / "results/analysis/b57_base_smoke/selftest_b57_base_smoke_carter_run.py").read_text()
reg = reg[reg.index("# [EDIT REGION]"):reg.index("# ========================= [END EDIT REGION]")]
built = A.build(A.TL.extract("```python\n" + reg + "```"))
check("an EDIT-REGION answer splices into the new base and passes rule 1",
      "def print(" in built and "_BASE_DBG_STATE" in built and A.static_check(built)[0])
sha = {}
for p, base in (("PROMPT_B5_6.md", OLD), ("PROMPT_B5_7.md", NEW)):
    A.BASE = A.SP.BASE = base
    msg = A.model_input(p)
    sha[p] = (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg), msg)
logged = json.loads((ARENA / "results/turns/prompt_b5_6/gpu_gpt_oss_120b_pB56_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-6 input",
      (logged.get("input_sha16"), logged.get("input_chars")) == sha["PROMPT_B5_6.md"][:2])
m57 = sha["PROMPT_B5_7.md"][2]
check("first message shows the new base (DBG block, cached map) and the B-5-7 text",
      "_BASE_DBG_STATE" in m57 and "_MAP_RECTS" in m57 and "'DBG 기록' 코드" in m57 and "raycast" not in m57.lower())
print(f"B-5-7 input sha16 {sha['PROMPT_B5_7.md'][0]} ({sha['PROMPT_B5_7.md'][1]}자), "
      f"B-5-6 {sha['PROMPT_B5_6.md'][0]} ({sha['PROMPT_B5_6.md'][1]}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
