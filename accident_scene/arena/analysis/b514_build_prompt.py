"""Build B-5-14 from B-5-13 using only the requested edits and consistency fixes.

The actual model input is this prose plus agent_loop.model_input's verbatim base.
An existing output is archived under results/archive_* before replacement.
"""
from datetime import datetime
from pathlib import Path
import shutil

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_13.md"
OUT = ARENA / "PROMPT_B5_14.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v2.py"
TITLE = "# ARENA 미션 프롬프트 B-5-14 (B-5-13 에 몸체 검사 함수 check_motion() 을 더하고, 판단 문장이 이 함수를 쓰도록 바꿈)"
DECISION = '''- 움직이는 물체에 대한 판단은 base 의 `check_motion(v, w, horizon_s)` 로 합니다. 이 함수는 지금 자세에서 명령 `(v, w)` 를
  `horizon_s` 초 동안 낼 때 로봇 몸체(`pose` 의 점에서 앞 0.13 m, 뒤 0.59 m, 좌우 0.25 m)가 지날 자리를 굴려 보고,
  `observed_objects()` 의 물체를 지금 속도 그대로 움직인 예측 위치와 지도 물체까지의 가장 가까운 간격, 그리고 처음 닿는
  시각을 돌려줍니다. 새 스캔이 들어오는 0.1 s 마다 여러 후보 명령을 이 함수로 재십시오. 후보에는 경로를 따르는 명령과
  그보다 느린 명령, 멈춤 `(0, 0)`, 제자리 회전, 뒤로 물러나기를 넣습니다. `horizon_s` 동안 닿지 않는 후보 가운데 계산한
  경로의 다음 경유점 쪽으로 가장 많이 가는 것을 고릅니다. 멈춤도 다른 후보와 똑같이 이 검사를 거칩니다. 물체는 비켜 주지
  않으므로 멈춘 자리도 닿을 수 있습니다. 모든 후보가 닿는다면 처음 닿는 시각이 가장 늦은 후보를 고릅니다.
'''
LESSONS = '''- 물체의 몇 초 뒤를 예측했지만 `pose` 의 점 하나에 반지름을 더해 거리를 잰 코드는, 몸체의 뒤쪽이나 옆이 물체에 닿았습니다.
- 멈춤을 늘 안전하다고 본 코드는 움직이는 물체가 지나는 길 안에 멈춰 있다가 받혔습니다.
- 예측을 길게 보고 물체 둘레를 크게 잡은 코드는 움직이는 물체를 늘 막힘으로 보아, 600 s 내내 출발하지 못했습니다.
'''
TEST_TIME = "시험 주행은 `sim_seconds=200` 을 넣어 부르십시오. 지금까지 통과한 주행은 모두 140 s 안에 목표에 닿았습니다."
DETECTION = '''- 몸체 검사 함수 `check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2)` 는 가장 최근 `pose` 와 시각에서
  일정한 `(v, w)` 를 낸다고 보고 차동구동 모형으로 `dt` 간격의 자세를 계산합니다. 멈춤 `(0, 0)` 도 같은 방식으로 잽니다.
  `_BASE_BODY_BOX` 는 `pose` 의 점에서 앞 0.13 m, 뒤 0.59 m, 좌우 0.25 m 인 몸체 상자입니다. 물체의 위치는
  `seen_t` 부터 `vel_ok` 가 True 이면 그 속도로 직선 예측하고, False 이면 제자리에 둡니다. `observed_objects()` 의
  추가 필드 `points` 는 `seen_t` 때 물체에서 보인 라이다 점들의 월드 좌표입니다. 이 점들을 예측 속도만큼 옮긴 뒤 몸체
  사각형과의 가장 가까운 간격을 잽니다. 점이 몸체 안에 있으면 간격은 음수입니다. `radius` 는 추적 자료로만 남기고
  간격 계산에는 쓰지 않습니다. 지도 여유는 몸체 둘레를 0.1 m 이하 간격으로 찍은 점의 `map_clearance()`
  최솟값입니다. 돌려주는 dict 는 `min_gap_obj`, `t_hit_obj`, `hit_obj_id`, `min_gap_map`, `t_hit_map`, `t_hit`,
  `end_pose` 입니다. `t_hit_obj` 와 `t_hit_map` 은 지금부터 간격이 `margin_m` 이하가 되는 첫 시각 [s] 이고, 없으면
  None 입니다. `hit_obj_id` 는 처음 닿는 물체의 id(없으면 None), `t_hit` 은 두 시각 중 이른 것(둘 다 없으면 None),
  `end_pose` 는 예측 끝의 (x, y, yaw) 입니다. 이 검사는 센서로 본 물체의 직선 예측과 지도에 대한 계산입니다.
'''


def replace_once(src, old, new):
    assert src.count(old) == 1, old[:100]
    return src.replace(old, new, 1)


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    src = replace_once(src, src.splitlines()[0], TITLE)
    start = src.index("- 움직이는 물체에 대한 판단은 이렇게 구현합니다.")
    end = src.index("\n- ", start + 1) + 1
    src = src[:start] + DECISION + src[end:]
    src = replace_once(src, "\n## Isaac Sim 검증 도구 `run_isaac_sim`\n", "\n" + LESSONS + "\n## Isaac Sim 검증 도구 `run_isaac_sim`\n")
    old_time = "- `sim_seconds` (선택): 이번 시험에서 돌릴 최대 시뮬레이션 시간으로 10–600 s, 기본 600 s\n  입니다. 길이 35 m 안팎의 경로는 위 추종 방법으로 보통 2분 안에 끝나므로 200 정도면\n  충분합니다."
    src = replace_once(src, old_time, "- `sim_seconds` (선택): 이번 시험에서 돌릴 최대 시뮬레이션 시간으로 10–600 s 입니다.\n  " + TEST_TIME)
    src = replace_once(src, "## 지금 할 일\n", "## 지금 할 일\n\n" + TEST_TIME + "\n")
    src = replace_once(src, "지도 기본 코드 아래에 물체 추적 함수 `observed_objects()` 가 들어 있습니다.",
                       "지도 기본 코드 아래에 물체 추적 함수 `observed_objects()` 와 몸체 검사 함수 `check_motion()` 이 들어 있습니다.")
    src = replace_once(src, "\n## 지금까지의 주행에서 알게 된 것\n", "\n" + DETECTION + "\n## 지금까지의 주행에서 알게 된 것\n")
    first_line = base[:base.index("# [EDIT REGION] mission controller")].count("\n") + 4
    src = replace_once(src, "이번 base 에서는 답의 첫 줄이 522행입니다.", f"이번 base 에서는 답의 첫 줄이 {first_line}행입니다.")
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        archive = ARENA / "results" / f"archive_{path.name}_before_b514_{datetime.now():%Y%m%d_%H%M%S_%f}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
