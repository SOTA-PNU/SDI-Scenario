"""Build PROMPT_B5_8.md (실험 33) from PROMPT_B5_7.md with the edits below and nothing else.
B-5-8 = B-5-7 + base_carter_run_ros_map_dbg_heapq.py (heapq, collections, deque imported by the base) + the
carter_run_locked.sh wall limit (TimeoutError after 1,700 s) + lessons recounted over 실험 30-32 + a closing
'지금 할 일' section (test first, answer only with a passing code). No route, no algorithm, no mover position.
analysis/b58_selftest.py re-applies EDITS to check the file.

    python3 analysis/b58_build_prompt.py   # refuses to overwrite an existing PROMPT_B5_8.md
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-7 (B-5-6 에 base 가 DBG 줄을 0.5 s 에 한 줄로 줄여 주는 코드와 직전 실험의 교훈을 더함)\n",
     "# ARENA 미션 프롬프트 B-5-8 (B-5-7 에 base 의 경로 계산용 import, 시간 제한 안내, 다시 센 교훈, 끝의 '지금 할 일' 을 더함)\n"),
    ("   `nav_msgs` 등), 경로 계산에 쓸 표준 라이브러리 `heapq`, `collections` 만 허용합니다.\n",
     "   `nav_msgs` 등), 경로 계산에 쓸 표준 라이브러리 `heapq`, `collections` 만 허용합니다. `heapq`, `collections`, `deque` 는\n"
     "   base 에서 이미 import 해 두었습니다.\n"),
    ("  저장하십시오. 다시 계산해야 할 때도 매 호출 하지는 마십시오. 시험 주행 한 번은 벽시계로 30분 안에 끝나야 합니다.\n",
     "  저장하십시오. 다시 계산해야 할 때도 매 호출 하지는 마십시오. `controller()` 는 1/60 s 마다 불리므로 한 번 호출에\n"
     "  0.05 s 가 걸리면 600 s 주행이 벽시계 30분이 됩니다. 시험 주행이 벽시계 1,700 s 를 넘으면 하네스가 멈추고\n"
     "  `runtime_error` 에 `TimeoutError` 를 돌려줍니다.\n"),
    ("- 직전 실험에서 경로를 지도로 계산하자, 주행한 채점 두 번 모두 지도 물체에서 0.8 m 넘게 떨어져 갔습니다. 대신 두 번\n"
     "  모두 출발 뒤 6–11 s 에 출발 구역에서 움직이는 물체와 부딪혔습니다. 한 번은 멈춰 서서 제자리 회전하던 중이었습니다.\n"
     "  제자리 회전에서는 몸체 뒤쪽 끝(`pose` 의 점에서 약 0.6 m)이 원을 그리며 지나갑니다.\n"
     "- 직전 두 실험의 채점 8번 가운데 5번은 그 회차에 시험 주행을 한 번도 하지 않은 답이었고, 나머지 3번도 시험에서\n"
     "  pass 를 받지 못한 코드를 고쳐 바로 낸 것이었습니다. 8번 모두 실패했습니다.\n",
     "- 경로를 지도로 계산하게 한 앞선 두 실험에서, 주행한 채점 다섯 번 모두 지도 물체에서 0.8 m 넘게 떨어져 갔습니다. 대신\n"
     "  네 번은 출발 뒤 6–34 s 에 출발 구역에서 움직이는 물체와 부딪혔고, 출발 구역을 지난 한 번은 더 가서 다른 움직이는\n"
     "  물체와 부딪혔습니다. 한 번은 멈춰 서서 제자리 회전하던 중이었습니다. 제자리 회전에서는 몸체 뒤쪽 끝(`pose` 의 점에서\n"
     "  약 0.6 m)이 원을 그리며 지나갑니다.\n"
     "- 직전 실험에서 멈춰 기다리기만 한 시험은 움직이는 물체와 부딪히지 않았지만, 600 s 안에 목표로 가지 못했습니다.\n"
     "- 앞선 세 실험의 채점 12번 가운데 8번은 그 회차에 시험 주행을 한 번도 하지 않은 답이었고, 나머지 4번도 시험에서\n"
     "  pass 를 받지 못한 코드를 고쳐 바로 낸 것이었습니다. 12번 모두 실패했습니다.\n"
     "- 앞선 두 실험에서 1회차 채점 두 번이 `heapq` 를 import 하지 않아 `NameError` 로 끝났습니다. 이번에는 base 가 `heapq`,\n"
     "  `collections`, `deque` 를 import 해 둡니다.\n"),
    ("실행 속도(벽시계)·코드 길이·주석 언어는 판정에 쓰지 않습니다.\n",
     "실행 속도(벽시계)·코드 길이·주석 언어는 판정에 쓰지 않습니다. 다만 한 번 실행이 벽시계 1,700 s 를 넘으면 하네스가\n"
     "멈추고 실패로 처리합니다.\n\n"
     "## 지금 할 일\n\n"
     "1. 첫 응답으로 `run_isaac_sim` 을 호출해 시험하십시오. 시험하기 전에 답을 내지 마십시오.\n"
     "2. 결과를 '시험 뒤 작업 방법' 대로 읽고 고쳐서 다시 시험하십시오. 시험 횟수에는 한도가 없고, 대화가 길어지면\n"
     "   오래된 시험 기록부터 정리되므로 계속 시험할 수 있습니다.\n"
     "3. 시험에서 `\"verdict\": \"pass\"` 를 받으면 그 코드를 그대로 `[EDIT REGION]` 블록으로 답하십시오. pass 를 받기 전에는\n"
     "   답을 내지 말고 계속 시험하십시오.\n"),
]


def build(src):
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out = ARENA / "PROMPT_B5_8.md"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_7.md").read_text()))
    print("wrote", out)
