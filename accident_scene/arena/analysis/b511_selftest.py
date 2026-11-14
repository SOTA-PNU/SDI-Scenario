"""B-5-11 checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_11.md
--base-file base_carter_run_ros_map_dbg_heapq_lidartime.py --turn-note-file PROMPT_B5_10_turn_note.md.
No Isaac, no LLM, no files written.

- PROMPT_B5_11.md is PROMPT_B5_9.md + analysis/b511_build_prompt.EDITS only, and the new base is the B-5-8..B-5-10
  base + BASE_EDITS only: two docstrings, same code (AST without docstrings), same line count and region line;
- every corrected number matches results/analysis/claude_lidar_rate/timing_facts.json (analysis/c_lidar_timing_check.py)
  and the rate-probe logs;
- 'two of four same-code pairs in the same scene differed, all four kept their verdict' matches the run logs;
- the B-5-9 checks that still apply (answer format line 362, no mover information) hold with the new base."""
import ast
import csv
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b511_build_prompt as BP  # noqa: E402

OLD_BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text()
BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py").read_text()
B59 = (ARENA / "PROMPT_B5_9.md").read_text()
B511 = (ARENA / "PROMPT_B5_11.md").read_text()
F = json.loads((ARENA / "results/analysis/claude_lidar_rate/timing_facts.json").read_text())
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def no_doc(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


check("PROMPT_B5_11.md == PROMPT_B5_9.md + the build EDITS only", BP.build(B59) == B511)
check("new base == B-5-8..B-5-10 base + BASE_EDITS only", BP.build(OLD_BASE, BP.BASE_EDITS) == BASE)
check("new base: same code without docstrings, same line count, [EDIT REGION] on the same line",
      no_doc(OLD_BASE) == no_doc(BASE) and OLD_BASE.count("\n") == BASE.count("\n")
      and OLD_BASE[:OLD_BASE.index("# [EDIT REGION]")].count("\n") == BASE[:BASE.index("# [EDIT REGION]")].count("\n"))
title = B511.splitlines()[0]
check("B-5-11 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))

# lidar timing facts
fr, br = F["front_rep_counts"], F["back_rep_counts"]
check("'두 2D 라이다는 0.1 s(6 스텝)마다 새 스캔, 둘이 같은 스텝에 바뀜': front content changes == back content changes,"
      " all but the first on the same call; each front scan comes back 5 more times",
      len(fr) == 6 and min(fr) >= 0.99 * fr[0] and len(br) == 1 and abs(fr[0] - br[0]) <= 1
      and F["content_change_same_call"] >= br[0] - 1 and F["content_change_back_only"] == 0 and F["content_change_front_only"] <= 1)
check("'처음 들어올 때 약 0.1 s(6 스텝) 전에 잰 것, 다시 올 때마다 한 스텝씩 더 오래된 것': best lags front [6..11], back [6]",
      F["best_lag_steps"]["front"] == [6, 7, 8, 9, 10, 11] and F["best_lag_steps"]["back"] == [6])
rates = {}
for n in ("still_front_first", "still_back_first", "turning_front_first"):
    so = (ARENA / f"results/analysis/claude_lidar_rate/claude_lidar_rate_{n}.carter-run.stdout.log").read_text()
    m = re.findall(r"DBG t=([\d.]+) front=(\d+) back=(\d+)", so)[-1]
    rates[n] = (int(m[1]), int(m[2]))
resend = {n: tuple(c > 300 for c in v) for n, v in rates.items()}
check(f"'그러는 라이다는 실행마다 달랐고, 둘 다 그러지 않은 실행도 있었음': new arrays in 6 s {rates}",
      (False, True) in resend.values() and (True, False) in resend.values() and (False, False) in resend.values()
      and all(min(v) in range(60, 70) for v in rates.values()))
tab = F["false_per_scan_0_9m"]
r = lambda k, lag: tab[k][lag]
check("table '멈춤 0 / 0 | 0 / 0'", r("front standstill", "lag0") == r("back standstill", "lag0") == 0
      and r("front standstill", "lag6") == r("back standstill", "lag6") == 0 and "| 멈춤 | 0 / 0 | 0 / 0 |" in B511)
check("table '곧게 달림 2.1 / 4.9 | 0 / 0'", f"| {r('front driving', 'lag0')} / {r('back driving', 'lag0')} | "
      f"{r('front driving', 'lag6'):g} / {r('back driving', 'lag6'):g} |" in B511)
check("table '돎 156 / 262 | 13 / 45'", f"| {round(r('front turning', 'lag0'))} / {round(r('back turning', 'lag0'))} | "
      f"{round(r('front turning', 'lag6'))} / {round(r('back turning', 'lag6'))} |" in B511)
RR = json.loads((ARENA / "results/analysis/claude_lidar_rate/rule_replay_facts.json").read_text())
check("'새 스캔 하나에서' table, and '다섯 번째로 다시 온 스캔은 곧게 달릴 때 80개, 돌 때 300개' (rule_replay_facts.json)",
      "새 스캔 하나에서 0–9 m" in B511 and RR["front_driving_lag0_by_rep_0_9m"][0] == r("front driving", "lag0")
      and RR["front_turning_lag0_by_rep_0_9m"][0] == r("front turning", "lag0")
      and round(RR["front_driving_lag0_by_rep_0_9m"][5]) == 80 and round(RR["front_turning_lag0_by_rep_0_9m"][5]) == 300
      and "곧게\n  달릴 때 80개, 돌 때 300개가 남았습니다." in B511)
res = F["residual_points_8m_lag6"]
check("'실제 면은 지도 값과 최대 0.21 m, margin 0.1 m 에서 남고 0.25 m 에서 없음, 가장 큰 곳은 상자 08 둘레'",
      round(res["0.1"]["max_map_dist"], 2) == 0.21 and res["0.1"]["n"] > 0 and res["0.25"]["n"] == 0
      and res["0.15"]["share_near_box08"] > 0.5 and abs(res["0.1"]["max_at_xy"][0] - 3.4) < 0.75
      and abs(res["0.1"]["max_at_xy"][1] - 13.0) < 0.75 and "최대 0.21 m 다릅니다" in B511)
check("prompt and base no longer say 10 Hz / 60 Hz for the 2D lidars or 'pose 는 controller() 가 받은'",
      "막힙니다 | 10 Hz |" not in B511 and "막힙니다 | 60 Hz |" not in B511 and "| 60 Hz |" in B511       # camera row stays
      and "pose 는 controller() 가 받은 (x, y, yaw)" not in BASE and "같은 코드는 같은 결과를 냅니다" not in B511)

# same code, same scene, run twice
groups = {}
for d in sorted(glob.glob(str(ARENA / "results/turns/prompt_b5_*/isaac_logs"))) + [str(ARENA / "results/analysis/claude_ros_isaac")]:
    for f in glob.glob(d + "/*_carter_run.py"):
        v = os.path.basename(f)[:-len("_carter_run.py")]
        rj, tr, so = (Path(d) / f"{v}_result.json", Path(d) / f"{v}_trajectory.csv", Path(d) / f"{v}.carter-run.stdout.log")
        if not (rj.exists() and tr.exists() and so.exists()):
            continue
        rows = list(csv.DictReader(open(tr)))
        scene = "movers v1" if "[movers] scenario v1" in so.read_text(errors="replace") else "no movers"
        key = (hashlib.sha1(Path(f).read_bytes()).hexdigest(), scene)
        groups.setdefault(key, []).append((json.loads(rj.read_text())["verdict"], rows[-1]["t"], rows[-1]["x"], rows[-1]["y"],
                                           " ".join(o.get("detail", "") for o in json.loads(rj.read_text())["oracles"])))
pairs = [g for g in groups.values() if len(g) >= 2]
differ = [g for g in pairs if len(set(g)) > 1]
check(f"'같은 코드를 같은 장면에서 두 번 돌린 네 쌍 가운데 두 쌍은 달랐고, 판정은 네 쌍 모두 같았음' (pairs {len(pairs)}, differ {len(differ)})",
      len(pairs) == 4 and all(len(g) == 2 for g in pairs) and len(differ) == 2 and all(len({x[0] for x in g}) == 1 for g in pairs))

# B-5-9 checks that must still hold with the new base
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = BASE, BASE, "b5", True, True
reg = (ARENA / "results/turns/prompt_b5_8/isaac_logs/gpu_gpt_oss_120b_pB58_a0_t2_carter_run.py").read_text()
reg = reg[reg.index("# [EDIT REGION]"):reg.index("# ========================= [END EDIT REGION]")]
body = reg.split("\n", 2)[2]


def verdict(answer):
    built = A.build(A.TL.extract(answer))
    return A.static_check(built), built


(bad3, d3), b3 = verdict("```python\n[EDIT REGION] mission controller - modify ONLY this block\n" + body + "```")
ok1, _ = verdict("설명입니다.\n\n```python\n" + body + "```\n\n위 코드가 답입니다.")
check("answer format unchanged with the new base: a fenced answer passes, a '#'-less header fails at 'line 362' (the lesson)",
      ok1[0] and not bad3 and "line 362" in d3 and b3.splitlines()[361].startswith("[EDIT REGION]"))
check("prompt: no mover position, no winning strategy, same-scene sentence kept",
      not any(x in B511 for x in ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "0.5 m/s", "11.6",
                                  "5.54", "49.7", "M5")) and "최종 채점은 같은 장면" in B511)
A.BASE = A.SP.BASE = OLD_BASE
sha59 = A.model_input("PROMPT_B5_9.md")
A.BASE = A.SP.BASE = BASE
sha511 = A.model_input("PROMPT_B5_11.md")
logged = json.loads((ARENA / "results/turns/prompt_b5_10/gpu_gpt_oss_120b_pB510_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-10 input", (logged["input_sha16"], logged["input_chars"])
      == (hashlib.sha256(sha59.encode()).hexdigest()[:16], len(sha59)))
print(f"B-5-11 input sha16 {hashlib.sha256(sha511.encode()).hexdigest()[:16]} ({len(sha511)}자), B-5-10 {logged['input_sha16']} ({logged['input_chars']}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
