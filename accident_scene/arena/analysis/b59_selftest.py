"""B-5-9 (실험 34) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_9.md
--base-file base_carter_run_ros_map_dbg_heapq.py (the B-5-8 base). No Isaac, no LLM, no files written.

- PROMPT_B5_9.md is PROMPT_B5_8.md + analysis/b59_build_prompt.EDITS only;
- the answer-format text matches what turn_loop.extract and the splice really do (four answer shapes);
- every B-5-8 lesson matches the 실험 33 records (a0-a3)."""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b59_build_prompt as BP  # noqa: E402

BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text()
B58 = (ARENA / "PROMPT_B5_8.md").read_text()
B59 = (ARENA / "PROMPT_B5_9.md").read_text()
D = ARENA / "results/turns/prompt_b5_8"
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_9.md == PROMPT_B5_8.md + the build EDITS only", BP.build(B58) == B59)
title = B59.splitlines()[0]
check("B-5-9 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = BASE, BASE, "b5", True, True
reg = (D / "isaac_logs/gpu_gpt_oss_120b_pB58_a0_t2_carter_run.py").read_text()
reg = reg[reg.index("# [EDIT REGION]"):reg.index("# ========================= [END EDIT REGION]")]
body = reg.split("\n", 2)[2]          # the region without its two header lines
HEAD = "# [EDIT REGION] mission controller - modify ONLY this block\n"


def verdict(answer):
    built = A.build(A.TL.extract(answer))
    return A.static_check(built), built


ok1, _ = verdict("설명입니다.\n\n```python\n" + body + "```\n\n위 코드가 답입니다.")
ok2, _ = verdict("```python\n" + HEAD + "# ====\n" + body + "```")
(bad3, d3), b3 = verdict("```python\n[EDIT REGION] mission controller - modify ONLY this block\n" + body + "```")
(bad4, d4), b4 = verdict("Let's edit code accordingly.\n" + body)
check("answer shapes: prose outside a ```python block passes; the '#' header passes; '#'-less header and fenceless prose fail",
      ok1[0] and ok2[0] and not bad3 and not bad4)
check("…and both failures point at 'line 362', the answer's first line in this base (the B-5-8 lesson)",
      "line 362" in d3 and "line 362" in d4 and b3.splitlines()[361].startswith("[EDIT REGION]")
      and b4.splitlines()[361].startswith("Let's edit") and "'line 362' 는 base 와 합친 파일의 줄 번호로" in B59)
turns, raws = {}, {}
for s in ("a0", "a1", "a2", "a3"):
    turns[s] = json.loads((D / f"gpu_gpt_oss_120b_pB58_{s}_turns.json").read_text())["turns"]
    raws[s] = [json.loads(f.read_text())["choices"][0]["message"] for f in sorted(D.glob(f"gpu_gpt_oss_120b_pB58_{s}_c*_raw.json"))]
first_is_test = [bool(raws[s][0].get("tool_calls")) for s in turns]
answers = [m.get("content") or "" for s in raws for m in raws[s] if not m.get("tool_calls")]
graded = [t for s in turns for t in turns[s]]
dq = [t for t in graded if t["verdict"] == "static_dq"]
dq_ans = [a for a in answers if not A.static_check(A.build(A.TL.extract(a)))[0]]
check("'B-5-8 의 실행 네 번은 모두 1회차 첫 응답으로 시험, 한 번은 첫 시험에서 pass 를 받아 통과'",
      all(first_is_test) and len(first_is_test) == 4 and turns["a0"][1]["verdict"] == "pass"
      and [t for t in turns["a0"][0]["tests"] if t["test"] == 1][0]["verdict"] == "pass"
      and sum(1 for t in graded if t["verdict"] == "pass") == 1)
check("'채점 여덟 번 가운데 네 번은 답 형식 실격: 세 번은 # 없는 [EDIT REGION], 한 번은 블록 없는 설명 문장'",
      len(graded) == 8 and len(dq) == 4 and len(dq_ans) == 4
      and sum(1 for a in dq_ans if a.lstrip().startswith("[EDIT REGION]")) == 3
      and sum(1 for a in dq_ans if "```" not in a and not a.lstrip().startswith(("[EDIT", "def", "#", "import"))) == 1)
tests = [t for s in ("a1", "a2", "a3") for tt in turns[s] for t in tt.get("tests") or []]
waited = [t for t in tests if not t.get("partners") and (t.get("sim_time_s") or 0) > 290 and (t.get("final_dist_m") or 0) > 25]
hit = [t for t in tests if any("arena_mover" in p for p in t.get("partners") or [])]
check("'움직이는 물체와 부딪힌 뒤 반응을 바꾼 코드들은 멈춰 기다리다 못 움직이거나 움직이자마자 다시 부딪힘' (a1-a3 tests)",
      len(waited) >= 5 and len(hit) >= 20)
check("moving-object paragraph present: look with the sensors first, react at run time, no fixed times or coordinates",
      "**움직이는 물체와 부딪혔을 때.**" in B59 and "특정 시각이나 좌표에 맞추지 말고" in B59
      and "그 시험에 보낸 `code` 를 한 글자도 바꾸지 말고" in B59)
check("prompt: no mover position, no winning strategy, same-scene sentences kept",
      not any(x in B59 for x in ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "0.5 m/s", "11.6",
                                 "5.54", "49.7", "M5"))
      and "최종 채점은 같은 장면" in B59 and "장면에 없는 물체" not in B59)
sha = {}
for p in ("PROMPT_B5_8.md", "PROMPT_B5_9.md"):
    msg = A.model_input(p)
    sha[p] = (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg))
logged = json.loads((D / "gpu_gpt_oss_120b_pB58_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-8 input", (logged["input_sha16"], logged["input_chars"]) == sha["PROMPT_B5_8.md"])
print(f"B-5-9 input sha16 {sha['PROMPT_B5_9.md'][0]} ({sha['PROMPT_B5_9.md'][1]}자), B-5-8 {sha['PROMPT_B5_8.md'][0]}")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
