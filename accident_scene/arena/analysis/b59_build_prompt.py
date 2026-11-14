"""Build PROMPT_B5_9.md (실험 34) from PROMPT_B5_8.md with the edits below and nothing else.
B-5-9 = B-5-8 (same base base_carter_run_ros_map_dbg_heapq.py) + the exact answer format (one ```python block, code
only, the header comment with its '#'), a test procedure after a collision with a moving object (look at it with
the sensors first; react to what the sensors see at run time, never to fixed times or coordinates), and the lessons
pinned to experiment labels plus two from B-5-8. No route, no algorithm, no mover position, no winning strategy.
analysis/b59_selftest.py re-applies EDITS to check the file.

    python3 analysis/b59_build_prompt.py   # refuses to overwrite an existing PROMPT_B5_9.md
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-8 (B-5-7 에 base 의 경로 계산용 import, 시간 제한 안내, 다시 센 교훈, 끝의 '지금 할 일' 을 더함)\n",
     "# ARENA 미션 프롬프트 B-5-9 (B-5-8 에 답의 정확한 형식, 움직이는 물체와 부딪혔을 때의 시험 방법, 고정한 교훈을 더함)\n"),
    ("블록 밖에서 글자 하나(설명문, 하이픈, 따옴표)만 바뀌어도 실행되지 않고 실격됩니다.\n",
     "블록 밖에서 글자 하나(설명문, 하이픈, 따옴표)만 바뀌어도 실행되지 않고 실격됩니다.\n"
     "답은 ```python 으로 여는 코드 블록 하나에 담고, 블록 안에는 코드만 넣으십시오. 블록 밖의 설명은 채점에 쓰지 않습니다.\n"
     "머리 주석을 쓸 때는 `# [EDIT REGION] mission controller - modify ONLY this block` 처럼 맨 앞의 `#` 까지 쓰십시오.\n"
     "`#` 없는 `[EDIT REGION]` 줄이나, 코드 블록 없이 코드와 섞어 쓴 설명 문장은 구문 오류가 되어 실격됩니다.\n"),
    ("더한 코드는 채점 때와 거의 같은 주행을 다시 보여 줍니다.\n\n## 규칙\n",
     "더한 코드는 채점 때와 거의 같은 주행을 다시 보여 줍니다.\n\n"
     "**움직이는 물체와 부딪혔을 때.** `collision_partners` 에 상자나 고정물이 아닌 물체가 나오면, 고치기 전에 그 물체가 센서에\n"
     "어떻게 보이는지 먼저 확인하십시오. 0.5 s 마다 한 줄에, 지도에 없는 점들 가운데 로봇에서 가장 가까운 점의 월드 좌표와\n"
     "거리, 그 점이 직전 줄에서 움직인 방향을 넣어 찍으면, 그 물체가 언제 어디서 어느 쪽으로 다가왔는지 DBG 에 남습니다.\n"
     "코드는 특정 시각이나 좌표에 맞추지 말고, 실행 중에 센서로 본 물체의 위치와 움직임에 따라 기다리거나 비키도록\n"
     "만드십시오.\n\n## 규칙\n"),
    ("- 경로를 지도로 계산하게 한 앞선 두 실험에서, 주행한 채점 다섯 번 모두",
     "- 경로를 지도로 계산하게 한 앞선 실험 B-5-6, B-5-7 에서, 주행한 채점 다섯 번 모두"),
    ("- 직전 실험에서 멈춰 기다리기만 한 시험은", "- B-5-7 에서 멈춰 기다리기만 한 시험은"),
    ("- 앞선 세 실험의 채점 12번 가운데", "- 앞선 실험 B-5-5 ~ B-5-7 의 채점 12번 가운데"),
    ("- 앞선 두 실험에서 1회차 채점 두 번이 `heapq` 를 import 하지 않아 `NameError` 로 끝났습니다. 이번에는 base 가 `heapq`,\n"
     "  `collections`, `deque` 를 import 해 둡니다.\n",
     "- B-5-6, B-5-7 에서 1회차 채점 두 번이 `heapq` 를 import 하지 않아 `NameError` 로 끝났습니다. 그 뒤로는 base 가 `heapq`,\n"
     "  `collections`, `deque` 를 import 해 둡니다.\n"
     "- 직전 실험 B-5-8 의 실행 네 번은 모두 1회차 첫 응답으로 시험했고, 한 번은 첫 시험에서 pass 를 받아 통과했습니다. 그러나\n"
     "  채점 여덟 번 가운데 네 번은 답 형식 때문에 실격됐습니다. 세 번은 첫 줄에 `#` 없이 `[EDIT REGION] ...` 을 썼고, 한 번은\n"
     "  코드 블록 없이 코드 앞에 설명 문장을 붙였습니다. 이때 하네스가 알려 준 'line 362' 는 base 와 합친 파일의 줄 번호로,\n"
     "  답의 첫 줄이었습니다.\n"
     "- B-5-8 에서 움직이는 물체와 부딪힌 뒤 반응을 바꾼 코드들은, 멈춰 기다리다 다시 움직이지 못하거나 움직이자마자 다시\n"
     "  부딪히기를 되풀이했습니다.\n"),
    ("3. 시험에서 `\"verdict\": \"pass\"` 를 받으면 그 코드를 그대로 `[EDIT REGION]` 블록으로 답하십시오. pass 를 받기 전에는\n"
     "   답을 내지 말고 계속 시험하십시오.\n",
     "3. 시험에서 `\"verdict\": \"pass\"` 를 받으면 그 시험에 보낸 `code` 를 한 글자도 바꾸지 말고 ```python 코드 블록 하나에 담아\n"
     "   답하십시오. pass 를 받기 전에는 답을 내지 말고 계속 시험하십시오.\n"),
]


def build(src):
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out = ARENA / "PROMPT_B5_9.md"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_8.md").read_text()))
    print("wrote", out)
