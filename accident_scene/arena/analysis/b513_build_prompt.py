"""Build PROMPT_B5_13.md (실험 39) from PROMPT_B5_12.md with the edits below and nothing else. Base:
base_carter_run_ros_map_lidartime_objects.py = the B-5-11/B-5-12 base + the tracking tool observed_objects() (and the pose
record it needs in _robot_only). User, 2026-10-29: near-answer hints for B-5-13, "권하는 조합 중 1,2만 적용":
1 = the decision method (predict each object a few seconds ahead from position and velocity, roll each candidate command
forward with the robot's body, choose the one that does not touch and goes furthest toward the goal; standing still is
checked too; backing off and sidestepping are candidates); 2 = the base tool that turns the 2D lidars into tracked objects
with velocities. Not given (item 3): how the moving objects move. Revised after the review wf_d6a4e0ca-94e (base v2, analysis/b513_build_base.py). User, 2026-10-29, on the review's
wording questions: "1. (나) 2. (가)" = the decision sentence names the next waypoint of the computed path as 'toward
the goal', checks candidates against the map with map_clearance(), and replaces 'sidestep' with a (v, w)-expressible move;
the B-5-12 sentence about objects that cross the path back and forth ('오가는 곳') stays as it is. analysis/b513_selftest.py re-applies EDITS.

    python3 analysis/b513_build_prompt.py   # refuses to overwrite an existing PROMPT_B5_13.md
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-12 (B-5-11 에 더 느린 추종 속도와, 경로를 가로지르는 물체 앞에서는 그 길 밖에서 기다렸다 건너라는 지시를 더함)\n",
     "# ARENA 미션 프롬프트 B-5-13 (B-5-12 에 물체 추적 기본 코드 observed_objects() 와, 물체의 몇 초 뒤를 예측해 몸체가 닿지 않는 명령을 고르는 판단 방식을 더함)\n"),
    ("그래서 이번 코드는 다음 다섯 가지를 지켜 구현하십시오.\n", "그래서 이번 코드는 다음 여섯 가지를 지켜 구현하십시오.\n"),
    ("  점에서 뒤로 0.59 m 까지 있습니다). 물체는 비켜 주지 않으므로, 길에 걸친 채 멈추면 멈춰 있어도 부딪힙니다.\n",
     "  점에서 뒤로 0.59 m 까지 있습니다). 물체는 비켜 주지 않으므로, 길에 걸친 채 멈추면 멈춰 있어도 부딪힙니다.\n"
     "- 움직이는 물체에 대한 판단은 이렇게 구현합니다. base 의 `observed_objects()` 가 돌려주는 물체마다 위치와 속도로 몇 초 뒤까지의\n"
     "  위치를 예측합니다. 후보 명령(전진, 감속, 멈춤, 뒤로 물러나기, 제자리에서 돌았다가 짧게 나가기처럼 `(v, w)` 로 낼 수 있는\n"
     "  동작)마다 그동안 로봇이 지날 자리를 굴려 보아, 로봇 몸체(`pose` 의 점에서 앞 0.13 m, 뒤 0.59 m, 좌우 0.25 m)가 예측한 물체와\n"
     "  닿지 않고 지도 물체와도 닿지 않는(`map_clearance()` 로 확인) 명령 가운데, 계산한 경로의 다음 경유점 쪽으로 가장 많이 가는\n"
     "  것을 고릅니다. 멈춤도 이 검사를 거칩니다. 물체는 비켜 주지 않으므로 멈춘 자리도 닿을 수 있습니다.\n"),
    ("   기다렸다 건너는 `controller()` 를 작성합니다.\n",
     "   기다렸다 건너는 `controller()` 를 작성합니다. 움직이는 물체는 `observed_objects()` 로 보고, 위 판단 방식으로 명령을 고릅니다.\n"),
    ("  번호, 반응 상태, 앞 뒤 왼쪽 오른쪽 방향별로 가장 가까운 지도에 없는 점의 거리와 그 점의 월드 좌표, 명령 `(v, w)`\n",
     "  번호, 반응 상태, `observed_objects()` 가 돌려준 가까운 물체의 위치, 속도, 거리, 고른 명령과 그 명령의 예측 여유, 명령 `(v, w)`\n"),
    ("어떻게 보이는지 먼저 확인하십시오. 0.5 s 마다 한 줄에, 지도에 없는 점들 가운데 로봇에서 가장 가까운 점의 월드 좌표와\n"
     "거리, 그 점이 직전 줄에서 움직인 방향을 넣어 찍으면, 그 물체가 언제 어디서 어느 쪽으로 다가왔는지 DBG 에 남습니다.\n",
     "어떻게 보이는지 먼저 확인하십시오. 0.5 s 마다 한 줄에, `observed_objects()` 가 돌려준 가까운 물체의 위치, 속도, 거리를\n"
     "넣어 찍으면, 그 물체가 언제 어디서 어느 쪽으로 다가왔는지 DBG 에 남습니다.\n"),
    ("\n## 지금까지의 주행에서 알게 된 것\n",
     "\n## 감지 기본 코드\n\n"
     "`base_carter_run.py` 의 지도 기본 코드 아래에 물체 추적 함수 `observed_objects()` 가 들어 있습니다. 이 부분도 `[EDIT REGION]`\n"
     "밖이므로 고치지 마십시오.\n\n"
     "- base 가 `controller()` 를 부르기 직전마다 추적을 갱신하므로, `controller()` 안에서 언제 몇 번 불러도 같은 결과를 줍니다.\n"
     "- 갱신: 앞뒤 2D 라이다에 새 스캔이 오면(내용이 직전과 다를 때만), 그 스캔을 잰 때(받기 0.1 s 전)의 로봇 자세로 점을 월드\n"
     "  좌표로 바꿉니다. 로봇에서 8 m 안이면서 지도 물체와 벽에서 0.25 m 넘게 떨어진 점만 남겨 0.1 m 칸으로 솎고, 0.45 m 이하로\n"
     "  떨어진 점끼리 물체 하나로 묶습니다(점이 3개보다 적은 조각은 0.6 m 안의 큰 묶음에 합칩니다). 묶음은 직전 물체의 예측\n"
     "  위치에서 1.0 m 안이면 같은 물체로 이어 붙이고, 최근 0.8 s 의 중심 위치를 직선으로 맞춰 속도를 구합니다.\n"
     "- 돌려주는 값은 가까운 것부터 정렬한 리스트입니다. 물체마다 `id`, `x`, `y`(보인 점들의 중심, 월드 좌표 [m]), `vx`, `vy`,\n"
     "  `speed` [m/s], `vel_ok`(속도를 맞출 자료가 충분했는지), `radius`(중심에서 가장 먼 점까지 [m]), `n_points`, `seen_t`(마지막으로\n"
     "  본 스캔의 시각 [s]), `age_s`(지금 시각 - `seen_t`), `track_age_s`(처음 본 스캔부터 `seen_t` 까지 [s]), `dist`(지금 `pose` 의\n"
     "  점에서 중심까지 [m]) 를 담은 dict 입니다. `x`, `y`, `vx`, `vy` 는 `seen_t` 때의 값이므로, 앞으로의 위치는 `seen_t` 부터\n"
     "  셉니다. 1.0 s 넘게 보이지 않은 물체는 빠집니다.\n"
     "- `vel_ok` 는 최근 0.8 s 안에 4번 넘게, 0.3 s 넘게 보였을 때 True 입니다. 처음 잡힌 직후나 가려졌다 다시 잡힌 직후에는\n"
     "  False 이고 `vx`, `vy` 가 0 으로 나오지만, 멈춰 있다는 뜻이 아닙니다.\n"
     "- 라이다는 로봇 쪽을 향한 면만 보므로 중심은 물체의 실제 중심과 다르고, 보이는 면이 바뀌면 중심이 옮겨 가 속도에 잡음이\n"
     "  섞입니다. 속도는 최근 0.8 s 위치의 직선 맞춤이라 실제 변화보다 늦게 따라갑니다. 지도에 없는 정지 물체도 나오고, 보이는\n"
     "  면이 바뀌면 잠깐 속도가 붙어 나올 수 있습니다.\n"
     "- 이 서버에서 움직이는 물체 없이 경로를 주행하며 녹화한 실제 스캔 4,338 호출(회전 0.9 rad/s 이하)에 이 함수를 돌렸을 때\n"
     "  나온 물체는 없었습니다. 움직이는 물체 하나를 2–3.5 m 거리에서 볼 때 위치 오차는 중앙값 약 0.22 m, 속도 오차는 중앙값\n"
     "  0.04–0.08 m/s, 95 % 가 0.5–0.65 m/s 였습니다(로봇이 멈춰 있을 때와 0.4 rad/s 로 돌 때).\n"
     "\n## 지금까지의 주행에서 알게 된 것\n"),
    ("  코드 블록 없이 코드 앞에 설명 문장을 붙였습니다. 이때 하네스가 알려 준 'line 362' 는 base 와 합친 파일의 줄 번호로,\n"
     "  답의 첫 줄이었습니다.\n",
     "  코드 블록 없이 코드 앞에 설명 문장을 붙였습니다. 이때 하네스가 알려 준 'line 362' 는 그때 base 와 합친 파일의 줄 번호로,\n"
     "  답의 첫 줄이었습니다. 이번 base 에서는 답의 첫 줄이 522행입니다.\n"),
    ("  거르면 지도에 있는 물체의 점이 빠집니다(아래 '라이다 점의 시점' 과 '지도 기본 코드'). 지금 `pose` 로 바꾸면 로봇이\n"
     "  움직이는 동안 지도에 있는 물체의 점도 남습니다.\n",
     "  거르면 지도에 있는 물체의 점이 빠집니다(아래 '라이다 점의 시점' 과 '지도 기본 코드'). 지금 `pose` 로 바꾸면 로봇이\n"
     "  움직이는 동안 지도에 있는 물체의 점도 남습니다. base 의 `observed_objects()` 는 이 변환과 거르기를 이미 합니다(아래\n"
     "  '감지 기본 코드').\n"),
]


def build(src):
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out = ARENA / "PROMPT_B5_13.md"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_12.md").read_text()))
    print("wrote", out)
