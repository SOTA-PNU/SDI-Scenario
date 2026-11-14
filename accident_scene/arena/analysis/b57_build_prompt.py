"""Build PROMPT_B5_7.md (실험 32) from PROMPT_B5_6.md with the edits below and nothing else.
B-5-7 = B-5-6 + base_carter_run_ros_map_dbg.py (the base keeps the first DBG line of every 0.5 s of sim time, ten in
the first 0.5 s; map functions cached and vectorised, same values) + the prompt text that describes it + two lessons
from 실험 31. No route, no algorithm, no mover position. analysis/b57_selftest.py re-applies EDITS to check the file.

    python3 analysis/b57_build_prompt.py   # refuses to overwrite an existing PROMPT_B5_7.md
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-6 (B-5-5 에서 경유점을 손으로 고르지 않고, 경로를 실행 중에 지도로 계산하게 함)\n",
     "# ARENA 미션 프롬프트 B-5-7 (B-5-6 에 base 가 DBG 줄을 0.5 s 에 한 줄로 줄여 주는 코드와 직전 실험의 교훈을 더함)\n"),
    ("- `base_carter_run.py` 의 `[EDIT REGION]` 바로 위에 있는 센서 기본 코드를 참고해, 센서 데이터를 활용하도록\n",
     "- `base_carter_run.py` 의 `[EDIT REGION]` 위에 있는 센서 기본 코드를 참고해, 센서 데이터를 활용하도록\n"),
    ("**DBG 찍는 법.** `controller_output` 은 처음 40줄과 마지막 40줄만 돌아오고(생략된 줄 수는\n"
     "`controller_output_omitted_lines`), 한 줄은 240자에서 잘립니다. 실패한 순간이 빠지지 않도록 한 번 주행에 80줄을\n"
     "넘지 않게 찍으십시오.\n\n"
     "- 첫 호출에서 계산한 경로를 몇 줄로 줄여 찍습니다. 점 수, 전체 길이, 구간 여유(`segment_clearance()`)의 최솟값과\n"
     "  그 구간의 두 끝점을 넣습니다. 경로를 찾지 못했으면 그 사실과 이유를 찍습니다.\n"
     "- 그 뒤에는 경유점이 바뀔 때, 센서 반응이 켜지거나 꺼질 때, 정렬 단계에 들어갈 때처럼 상태가 바뀔 때 한 줄을\n"
     "  찍고, 그 밖에는 5초에 한 줄만 찍습니다.\n"
     "- 그 줄에는 시각, `pose`, 지금 향하는 경유점 번호, 반응 상태, 앞 뒤 왼쪽 오른쪽 방향별로 가장 가까운 지도에 없는\n"
     "  점의 거리, 명령 `(v, w)` 를 넣습니다.\n",
     "**DBG 찍는 법.** `controller_output` 은 처음 40줄과 마지막 40줄만 돌아오고(생략된 줄 수는\n"
     "`controller_output_omitted_lines`), 한 줄은 240자에서 잘립니다. 그리고 `base_carter_run.py` 의 `[EDIT REGION]` 바로\n"
     "위 'DBG 기록' 코드가 DBG 줄을 줄여서 내보냅니다. 시뮬레이션 시각 0.5 s 구간마다 그 구간의 첫 DBG 줄만 남기고\n"
     "나머지는 버립니다(처음 0.5 s 구간만 열 줄까지 남깁니다). 그래서 매 호출 찍어도 처음 40줄이 처음 15초쯤을, 마지막\n"
     "40줄이 끝나기 전 20초를 0.5 s 간격으로 보여 줍니다.\n\n"
     "- 첫 호출에서 계산한 경로를 몇 줄로 줄여 찍습니다. 점 수, 전체 길이, 구간 여유(`segment_clearance()`)의 최솟값과\n"
     "  그 구간의 두 끝점을 넣습니다. 경로를 찾지 못했으면 그 사실과 이유를 찍습니다. 첫 0.5 s 에는 열 줄까지만 남습니다.\n"
     "- 그 뒤에는 0.5 s 에 한 줄만 남으므로, 필요한 값을 한 줄에 모두 넣어 찍습니다. 시각, `pose`, 지금 향하는 경유점\n"
     "  번호, 반응 상태, 앞 뒤 왼쪽 오른쪽 방향별로 가장 가까운 지도에 없는 점의 거리와 그 점의 월드 좌표, 명령 `(v, w)`\n"
     "  입니다.\n"),
    ("- `map_clearance()` 는 한 번에 약 0.014 ms 가 걸립니다. 경로 계산은 첫 호출에서 한 번 하고 결과를 전역 변수에\n",
     "- `map_clearance()` 는 한 번에 약 0.004 ms, `segment_clearance()` 는 길이 30 m 구간 하나에 약 0.5 ms 가 걸립니다.\n"
     "  경로 계산은 첫 호출에서 한 번 하고 결과를 전역 변수에\n"),
    ("로봇(NVIDIA Nova Carter)에 탑재된 라이다와 카메라 가운데 메시지가 나오는 것은 앞쪽 2D 라이다, 뒤쪽 2D 라이다, 3D 라이다, "
     "앞쪽 카메라 넷입니다. 이 네 센서를 읽는 기본 코드는 `base_carter_run.py` 의 `[EDIT REGION]` 바로 위에 들어 있습니다.",
     "로봇(NVIDIA Nova Carter)에 탑재된 라이다와 카메라 가운데 메시지가 나오는 것은 앞쪽 2D 라이다, 뒤쪽 2D 라이다, 3D 라이다, "
     "앞쪽 카메라 넷입니다. 이 네 센서를 읽는 기본 코드는 `base_carter_run.py` 의 `[EDIT REGION]` 위에 들어 있습니다."),
    ("`base_carter_run.py` 의 센서 기본 코드 아래, `[EDIT REGION]` 바로 위에 지도와 지도 함수가 들어 있습니다.",
     "`base_carter_run.py` 의 센서 기본 코드 아래에 지도와 지도 함수가 들어 있습니다."),
    ("- DBG 는 처음 40줄과 마지막 40줄만 돌아옵니다. 매 단계(1/60 s) 찍은 시험에서는 15,000줄 가운데 첫 1초와\n"
     "  마지막 1초만 돌아가 실패한 순간을 보지 못했습니다. 반응이 켜지거나 꺼질 때처럼 상태가 바뀔 때만 찍으십시오.\n",
     "- DBG 는 처음 40줄과 마지막 40줄만 돌아옵니다. 앞선 실험에서는 매 단계(1/60 s) 찍어 15,000줄 가운데 첫 1초와\n"
     "  마지막 1초만 돌아가 실패한 순간을 보지 못했습니다. 그래서 이번에는 base 가 DBG 줄을 0.5 s 에 한 줄로 줄입니다.\n"
     "- 직전 실험에서 경로를 지도로 계산하자, 주행한 채점 두 번 모두 지도 물체에서 0.8 m 넘게 떨어져 갔습니다. 대신 두 번\n"
     "  모두 출발 뒤 6–11 s 에 출발 구역에서 움직이는 물체와 부딪혔습니다. 한 번은 멈춰 서서 제자리 회전하던 중이었습니다.\n"
     "  제자리 회전에서는 몸체 뒤쪽 끝(`pose` 의 점에서 약 0.6 m)이 원을 그리며 지나갑니다.\n"
     "- 직전 두 실험의 채점 8번 가운데 5번은 그 회차에 시험 주행을 한 번도 하지 않은 답이었고, 나머지 3번도 시험에서\n"
     "  pass 를 받지 못한 코드를 고쳐 바로 낸 것이었습니다. 8번 모두 실패했습니다.\n"),
    ("  붙여 출력한 줄입니다. 처음 40줄과 마지막 40줄만 돌려주므로 경유점이 바뀔 때처럼 필요한\n"
     "  때만 출력하십시오. 최종 제출 코드에 남아 있어도 판정에는 영향이 없습니다. 시험 결과에서\n",
     "  붙여 출력한 줄입니다. base 의 'DBG 기록' 코드가 0.5 s 에 한 줄로 줄인 뒤, 처음 40줄과 마지막 40줄만\n"
     "  돌려줍니다. 최종 제출 코드에 남아 있어도 판정에는 영향이 없습니다. 시험 결과에서\n"),
]


def build(src):
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out = ARENA / "PROMPT_B5_7.md"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_6.md").read_text()))
    print("wrote", out)
