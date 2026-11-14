"""B-5-10 (실험 35) checks: agent_loop --b5 --b-test-movers --test-budget 0 --prompt-file PROMPT_B5_9.md
--base-file base_carter_run_ros_map_dbg_heapq.py --turn-note-file PROMPT_B5_10_turn_note.md. No Isaac, no LLM.

B-5-10 keeps the B-5-9 first message byte for byte and changes only the closing sentence of the turn feedback
('… 고쳐 다시 제출하십시오.' -> test first, then submit the passing code unchanged)."""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402
import turn_loop as TL  # noqa: E402

NOTE = (ARENA / "PROMPT_B5_10_turn_note.md").read_text().strip()
B59 = (ARENA / "PROMPT_B5_9.md").read_text()
BASE = (ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text()
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


A.BASE, A.SP.BASE, A.B_MODE, A.UNLIMITED, A.TEST_MOVERS = BASE, BASE, "b5", True, True
msg = A.model_input("PROMPT_B5_9.md")
logged = json.loads((ARENA / "results/turns/prompt_b5_9/gpu_gpt_oss_120b_pB59_a0_turns.json").read_text())
check("first message is the logged B-5-9 one, byte for byte",
      (hashlib.sha256(msg.encode()).hexdigest()[:16], len(msg)) == (logged["input_sha16"], logged["input_chars"]))
templates = [TL.FEEDBACK.format(result="{}"), TL.RUNTIME_FEEDBACK.format(err="E"), TL.DQ_FEEDBACK.format(detail="D")]
check("each turn feedback template ends with the closing sentence exactly once", all(t.count(A.TURN_CLOSE) == 1 for t in templates))
A.TURN_NOTE = None
check("without --turn-note-file nothing changes", all(A.closing(t) == t for t in templates))
A.TURN_NOTE = NOTE
new = [A.closing(t) for t in templates]
check("with it, the closing sentence is replaced by the note in all three", all(NOTE in t and A.TURN_CLOSE not in t for t in new))
check("the note says what the prompt says: test first, submit the passing code unchanged in one ```python block",
      "'시험 뒤 작업 방법'" in NOTE and "## 시험 뒤 작업 방법" in B59 and "바로 제출하지 마십시오" in NOTE
      and "그 시험에 보낸 `code` 를 한 글자도 바꾸지 말고" in NOTE and "그 시험에 보낸 `code` 를 한 글자도 바꾸지 말고" in B59
      and "```python 코드 블록 하나" in NOTE)
check("the note carries no mover position or strategy",
      not any(x in NOTE for x in ("M1", "arena_mover", "-7.0", "기다", "비키", "좌표")))
print("note sha16", hashlib.sha256(NOTE.encode()).hexdigest()[:16])
print("all passed" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
