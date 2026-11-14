"""B-5-5 (실험 30) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_5.md
--base-file base_carter_run_ros_map.py. No Isaac, no LLM, no files written.

PROMPT_B5_5.md = PROMPT_B5_4.md with three edits only: the title, work step 4, and a new section
'시험 뒤 작업 방법' (after each test: pick out what decides the cause from the result and the DBG lines,
state the cause, write it as a fixed comment record at the top of the next EDIT REGION). The harness
numbers the prompt quotes are checked against agent_loop and a real B-5-4 grading feedback."""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

MAP = (ARENA / "base_carter_run_ros_map.py").read_text()
B54 = (ARENA / "PROMPT_B5_4.md").read_text()
B55 = (ARENA / "PROMPT_B5_5.md").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


title = B55.splitlines()[0]
check("B-5-5 title passes only the b5 gate",
      all((m == "b5") == (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")))
b4, b5 = B54.split("\n", 1)[1], B55.split("\n", 1)[1]
check("same as B-5-4 up to work step 4", b4[:b4.index("\n4. ")] == b5[:b5.index("\n4. ")])
check("same as B-5-4 from '## 규칙' to the end", b4[b4.index("## 규칙"):] == b5[b5.index("## 규칙"):])
mid = b5[b5.index("\n4. "):b5.index("## 규칙")]
check("between them: new step 4, step 5 kept, the new section only",
      "'시험 뒤 작업 방법' 의 세 단계" in mid and "5. pass 가 나온 그 코드를 그대로 최종 답으로 제출합니다." in mid
      and mid.count("\n## ") == 1 and "\n## 시험 뒤 작업 방법\n" in mid)
check("the DBG facts of the old step 4 are kept in the new section",
      all(x in mid for x in ("시험 결과에는 센서 값이", "`print(\"DBG\", ...)`", "`controller_output` 으로 볼 수 있습니다",
                             "`(v, w)`", "\"verdict\": \"pass\"` 를 확인하기 전에는 제출하지 마십시오")))
check("method: pick out, state the cause, record it in the next code",
      all(x in mid for x in ("**추리기**", "**원인 정하기**", "**반영하기**", "# [시험 N] 결과:", "# [시험 N] 추린 DBG:",
                             "# [시험 N] 원인:", "# [시험 N] 이번 변경:", "짐작으로\n   고치지 말고")))
check("DBG budget and contents: 80 lines, first call segment_clearance, state changes, 5 s",
      all(x in mid for x in ("80줄을\n넘지 않게", "`segment_clearance()`", "상태가 바뀔 때 한 줄", "5초에 한 줄만",
                             "가장 가까운 지도에 없는\n  점의 거리")))
src = (ARENA / "agent_loop.py").read_text()
shown, omitted = A.dbg_lines("\n".join(f"DBG {i} " + "x" * 300 for i in range(100)))
check("prompt numbers match the harness: 40 + 40 lines, 240 chars, omitted-lines key",
      (A.DBG_HEAD, A.DBG_TAIL) == (40, 40) and len(shown) == 80 and omitted == 20
      and all(len(s) == 240 for s in shown) and shown[40].startswith("DBG 60 ")
      and 'out["controller_output_omitted_lines"]' in src and "`controller_output_omitted_lines`" in mid
      and "처음 40줄과 마지막 40줄만 돌아오고" in mid and "240자에서 잘립니다" in mid)
req = json.loads((ARENA / "results/turns/prompt_b5_4/gpu_gpt_oss_120b_pB54_a0_c02_request.json").read_text())
fb = req["messages"][-1]["content"]
check("grading feedback carries no trajectory or controller_output (real B-5-4 turn-2 message), as the prompt says",
      fb.startswith("제출한 코드를 고정 하네스에서 실행했습니다.") and "controller_output" not in fb and "trajectory" not in fb
      and "채점 결과에는 `trajectory` 와 `controller_output` 이 없습니다" in mid)
check("prompt: no mover leak, same-scene sentences kept (agent_loop --b-test-movers gate)",
      not any(x in B55 for x in ("M1", "arena_mover", "start_zone", "goal_south"))
      and "최종 채점은 같은 장면" in B55 and "장면에 없는 물체" not in B55)
A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = MAP, MAP, "b5", True, True
region = (ARENA / "results/turns/prompt_b5/isaac_logs/gpu_gpt_oss_120b_pB5_a0_carter_run.py").read_text()
region = region[region.index("# [EDIT REGION]"):region.index("# ========================= [END EDIT REGION]")]
head, rest = region.split("\n", 2)[:2], region.split("\n", 2)[2]
record = ("# [시험 1] 결과: fail, 충돌 /World/arena_box_00, 첫 충돌 9.6 s, 그때 위치 (-5.78, -3.93)\n"
          "# [시험 1] 추린 DBG: t=9.0 wp=2 react=off front=1.80 left=0.90 v=0.60 w=0.10\n"
          "# [시험 1] 원인: \"따옴표\" 와 '작은따옴표' 가 든 주석\n# [시험 1] 이번 변경: 경유점 2 를 옮김\n")
built = A.build(A.TL.extract("```python\n" + "\n".join(head) + "\n" + record + rest + "```"))
check("an EDIT-REGION answer with the record comments under the two header lines splices and passes rule 1",
      "# [시험 1] 원인:" in built and "def unmapped_points" in built and A.static_check(built)[0])
sha = {}
for p in ("PROMPT_B5_4.md", "PROMPT_B5_5.md"):
    msg = A.model_input(p)
    sha[p] = (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg), msg)
logged = json.loads((ARENA / "results/turns/prompt_b5_4/gpu_gpt_oss_120b_pB54_a0_turns.json").read_text())
check("hash method reproduces the logged B-5-4 input", (logged.get("input_sha16"), logged.get("input_chars"))
      == sha["PROMPT_B5_4.md"][:2])
m55 = sha["PROMPT_B5_5.md"][2]
check("first message shows the map base and the method, no simulator-only sensing, no limit",
      "def segment_clearance" in m55 and "## 시험 뒤 작업 방법" in m55 and "raycast" not in m55.lower()
      and "최대 10번" not in m55)
print(f"B-5-5 input sha16 {sha['PROMPT_B5_5.md'][0]} ({sha['PROMPT_B5_5.md'][1]}자), "
      f"B-5-4 {sha['PROMPT_B5_4.md'][0]} ({sha['PROMPT_B5_4.md'][1]}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
