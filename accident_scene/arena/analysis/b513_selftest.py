"""B-5-13 (실험 39) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_13.md
--base-file base_carter_run_ros_map_lidartime_objects.py --turn-note-file PROMPT_B5_10_turn_note.md, CARTER_CONTACT_ALL_LINKS=1.
No Isaac, no LLM.

- PROMPT_B5_13.md is PROMPT_B5_12.md + analysis/b513_build_prompt.EDITS only;
- the new base is the B-5-11/B-5-12 base + the tracking block + the pose record in _robot_only, nothing else;
- the offline check of observed_objects() (real static scans: no object; synthetic mover: tracked, no false object) holds;
- the lesson's line number is right for the new base; no mover position, lane, period, time or motion form is given;
- the Isaac check of observed_objects() against the real movers passed (results/analysis/b513_objects_isaac.json)."""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA)); sys.path.insert(0, str(ARENA / "analysis"))
import agent_loop as A  # noqa: E402
import b513_build_prompt as BP  # noqa: E402

B512 = (ARENA / "PROMPT_B5_12.md").read_text()
B513 = (ARENA / "PROMPT_B5_13.md").read_text()
OLD = (ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py").read_text()
NEW = (ARENA / "base_carter_run_ros_map_lidartime_objects.py").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("PROMPT_B5_13.md == PROMPT_B5_12.md + the build EDITS only", BP.build(B512) == B513)
check("B-5-13 title passes only the b5 gate", all((m == "b5") == (f"프롬프트 B-{m[1]}" in B513.splitlines()[0]) for m in ("b1", "b2", "b3", "b4", "b5")))
import b513_build_base as BB  # noqa: E402
blk = NEW[NEW.index("# ---- 감지 기본 코드 (물체 추적) ----"):NEW.index("# ---- DBG 기록 ----")]
check("new base == analysis/b513_build_base.build(B-5-11/B-5-12 base): tracking block, every-step update in _robot_only, renamed _S/_BODY",
      BB.build(OLD) == NEW and "_base_track_step(t)" in NEW and "_BODY" not in NEW)
OFF = json.loads((ARENA / "results/analysis/b513_objects_offline.json").read_text())
check(f"offline: static scans give no object, synthetic mover tracked with no false object ({OFF})",
      OFF["static"]["calls_with_object"] == 0 and OFF["synthetic_mover"]["calls_tracked"] >= 0.98 * OFF["synthetic_mover"]["calls_mover_within_7.5m"] > 0
      and OFF["synthetic_mover"]["calls_with_false_object"] == 0
      and "녹화한 실제 스캔 4,338 호출(회전 0.9 rad/s 이하)에 이 함수를 돌렸을 때\n  나온 물체는 없었습니다" in B513)
import math as _m  # noqa: E402
import pickle as _pk  # noqa: E402
_P = [q for q in _pk.load(open(ARENA / "results/analysis/claude_ros_isaac/probe_static_scans.pkl", "rb"))["poses"]]
_w = max(abs(_m.atan2(_m.sin(_P[i + 3][3] - _P[i - 3][3]), _m.cos(_P[i + 3][3] - _P[i - 3][3]))) / (_P[i + 3][0] - _P[i - 3][0])
         for i in range(3, len(_P) - 3))
check(f"the static recording turned at most {_w:.2f} rad/s ('회전 0.9 rad/s 이하')", _w <= 0.9)
A.BASE = A.SP.BASE = NEW
reg = (ARENA / "results/turns/prompt_b5_8/isaac_logs/gpu_gpt_oss_120b_pB58_a0_t2_carter_run.py").read_text()
reg = reg[reg.index("# [EDIT REGION]"):reg.index("# ========================= [END EDIT REGION]")]
body = reg.split("\n", 2)[2]
built = A.build(A.TL.extract("```python\n[EDIT REGION] mission controller - modify ONLY this block\n" + body + "```"))
bad, detail = A.static_check(built)
check(f"lesson line number: a '#'-less header fails at 'line 522', the answer's first line, in this base ({detail.splitlines()[0]})",
      not bad and "line 522" in detail and built.splitlines()[521].startswith("[EDIT REGION]") and "답의 첫 줄이 522행입니다" in B513)
words = ("M1", "arena_mover", "start_zone", "goal_south", "-7.0", "−7.0", "0.5 m/s", "11.6", "왕복", "되돌아", "끝에서",
         "방향을 바꾼", "상자면")
check("no mover position, lane, period, time or motion form in the prompt or the tracking block "
      "('오가' only in the line inherited from B-5-12)",
      not any(w in B513 for w in words) and not any(w in blk for w in words)
      and B513.count("오가") == B512.count("오가") == 1 and "오가" not in blk)
check("the decision method and the tool are described (vel_ok, track_age_s, values at seen_t, every-step update)",
      "다음 여섯 가지를 지켜" in B513 and "계산한 경로의 다음 경유점 쪽으로 가장 많이 가는" in B513
      and "`map_clearance()` 로 확인" in B513 and "제자리에서 돌았다가 짧게 나가기처럼 `(v, w)` 로 낼 수 있는" in B513
      and "옆으로 비키기" not in B513 and "## 감지 기본 코드" in B513
      and "def observed_objects():" in NEW and "`vel_ok`" in B513 and "`track_age_s`" in B513
      and "`seen_t` 때의 값" in B513 and "언제 몇 번 불러도 같은 결과" in B513)
RUN = (ARENA / "run_b513_sequential.sh").read_text() if (ARENA / "run_b513_sequential.sh").exists() else ""
check("run script: B-5-13 prompt and base, all-link contacts", "--prompt-file PROMPT_B5_13.md" in RUN
      and "--base-file base_carter_run_ros_map_lidartime_objects.py" in RUN and "\nexport CARTER_CONTACT_ALL_LINKS=1\n" in RUN)
ISA = ARENA / "results/analysis/b513_objects_isaac.json"
isa = json.loads(ISA.read_text()) if ISA.exists() else None
import hashlib as _h  # noqa: E402
base_sha = _h.sha256(NEW.encode()).hexdigest()[:16]
check(f"Isaac check of observed_objects() against the real movers passed on THIS base ({isa and isa.get('verdict')}, "
      f"base {isa and isa.get('base_sha16')} vs {base_sha})",
      bool(isa) and isa.get("verdict") == "ok" and isa.get("base_sha16") == base_sha)
if isa and "still" in isa and "turning" in isa:
    pm = (isa["still"]["pos_err_median"] + isa["turning"]["pos_err_median"]) / 2
    vm = sorted([isa["still"]["vel_err_median"], isa["turning"]["vel_err_median"]])
    vp = sorted([isa["still"]["vel_err_p95"], isa["turning"]["vel_err_p95"]])
    check(f"prompt's Isaac numbers match the check (pos {pm:.2f}, vel median {vm}, p95 {vp})",
          f"중앙값 약 {pm:.2f} m" in B513 and f"{vm[0]:.2f}–{vm[1]:.2f} m/s" in B513 and f"{vp[0]:.1f}–{vp[1]:.2f} m/s" in B513)
msg = A.model_input("PROMPT_B5_13.md")
print(f"B-5-13 input sha16 {hashlib.sha256(msg.encode()).hexdigest()[:16]} ({len(msg)}자)")
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
