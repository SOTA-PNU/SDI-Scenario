"""Minimal B523 -> B524 edits for v12 endpoint acceptance and planning diagnostics."""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import b523_build_prompt as PREV

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_23.md"
SRC_SHA16 = "89074b58c9815828"
OUT = ARENA / "PROMPT_B5_24.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v12.py"
BASE_SHA16 = "bbe902ad613caa5e"
TITLE = "# ARENA 미션 프롬프트 B-5-24 (B-5-23 에 경로 끝점 이어 붙이기, 거부 이유 표시, validate_plan() 을 더함)"
FIRST_ACTION, CALL_EXAMPLE = PREV.FIRST_ACTION, PREV.CALL_EXAMPLE
SCENE_SENTENCE = "시험 주행(`run_isaac_sim`)과 최종 채점은 같은 장면에서 돕니다."
SCAFFOLD = PREV.SCAFFOLD.replace(
    "# 출력: 실제 start_xy~goal_xy의 점 목록. 찾지 못하면 None 또는 [].",
    "# 출력: 시작/목표 각각0.5m 안에서 끝나는 점 목록. 없으면 None 또는 [].")

VALIDATE_DOC = '''- base의 채택 기준은 `validate_plan(path)`로 미리 확인합니다. `plan_path` 안에서는 현재 호출의
  실제 출발점·GOAL·요청 중심 여유로 검사합니다. 각 끝점 오차가0.5m 이하이면 실제 양끝을 연결하고,
  추가한 연결도 포함하여 전체 `check_path`를 검사합니다. 더 먼 끝점, 빈 경로, 비유한 값이나 지도 검사 실패는 거부합니다.
  반환 dict의 `accepted`, `path`, `reason`, `start_gap_m`, `goal_gap_m`, `check_path`, `clearance_m`을 확인합니다.
  `accepted`이면 `path`는 연결과 검사를 마친 경로입니다. 원 후보나 이 `path`를 반환할 수 있으며 base가 다시 검사합니다.
  함수 밖에서는 최신 pose와 채택 여유(없으면0.5m)를 쓰고, pose 전에는 MISSION의 START를 씁니다.
  `start_source`로 어느 출발점이 사용됐는지 확인합니다.
  이 검사는 경로를 찾거나 제어 상태를 바꾸지 않습니다. 거부 DBG의 끝점 거리·불량 구간·실제 간격과 요구 margin을 보고 표시된 원인을 고칩니다.

'''

CHANGES = (
    "Selected v12 and exact B524 title; preserved B523's literal scene-guard sentence.",
    "Endpoints within0.5m may be connected to actual pose/GOAL, with connector-inclusive path validation.",
    "Public validate_plan result and current planning context documented; reject DBG asks the model to fix the stated cause.",
    "Initial no-path retry every10s and per-reason planning rejection DBG replace incompatible no-retry/per-tick-plan-error wording.",
    "Three hook stubs, v11 target guard, clearance contract, goal/chooser ownership,200s trial and4/4 criterion retained.",
)
SHORTENINGS = CHANGES


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def build(src, base=None):
    assert hashlib.sha256(src.encode()).hexdigest()[:16] == SRC_SHA16
    base = BASE.read_text() if base is None else base
    if BASE_SHA16 is not None:
        assert hashlib.sha256(base.encode()).hexdigest()[:16] == BASE_SHA16
    text = replace_once(src, src.splitlines()[0], TITLE)
    text = replace_once(text, PREV.SCAFFOLD, SCAFFOLD)
    text = replace_once(text,
        "현재 위치부터 GOAL까지 검사에 통과한 첫 경로를 채택합니다. 모두 실패하면 이전 경로를 유지하고, 최초 실패이면\n"
        "정지 입력을 요청합니다. 계획 호출은 최초와 지도 막힘 정지가10초를 넘은 다음 틱에만 수행합니다.",
        "현재 위치부터 GOAL까지 검사에 통과한 첫 경로를 채택합니다. 모두 실패하면 이전 경로를 유지합니다.\n"
        "아직 채택한 경로가 없으면 정지 입력을 요청하고10초마다 현재 위치에서 다시 계획합니다.\n"
        "계획 호출은 최초, 이 경로 없음 재시도, 지도 막힘 정지가10초를 넘은 다음 틱에 수행합니다.")
    text = replace_once(text,
        "모델 함수의 예외·비유한 반환은 해당 틱마다 DBG 한 줄로 알리고 그 틱 `choose_command(0,0,...)` 결과를 사용합니다.",
        "계획의 빈 경로·끝점 거리·지도 검사 실패·예외는 이유별10초에 한 줄 DBG로 알리고 다음 여유를 시도합니다.\n"
        "그 밖의 hook 예외·비유한 반환은 해당 틱마다 DBG 한 줄로 알리고 `choose_command(0,0,...)` 결과를 사용합니다.")
    text = replace_once(text,
        "- `plan_path(start_xy, goal_xy, clearance_m)` 는 실제 입력 양 끝점을 포함하는 유한한2차원 점 목록을 반환합니다.",
        "- `plan_path(start_xy, goal_xy, clearance_m)` 는 유한한2차원 점 목록을 반환합니다. 실제 입력 양 끝점이나\n"
        "  각각0.5m 이내의 격자 끝점을 쓸 수 있으며, 후자는 base가 실제 양끝에 연결하여 검사합니다.")
    text = replace_once(text, "- 최종 검사는 `check_path(points, margin_m=max(0,clearance_m-0.25))`입니다.",
                        "- 선분 지도 검사는 `check_path(points, margin_m=max(0,clearance_m-0.25))`입니다.")
    text = replace_once(text, "## 앞쪽 목표와 추종 명령\n", VALIDATE_DOC + "## 앞쪽 목표와 추종 명령\n")
    text = replace_once(text, "목표 보정 DBG는 이유별10초 간격이며, hook 예외 DBG는 해당 틱마다 별도 한 줄입니다.",
                        "계획 거부와 목표 보정 DBG는 각각 이유별10초 간격이며, 계획 이외의 hook 예외 DBG는 해당 틱마다 별도 한 줄입니다.")
    assert text.count(SCENE_SENTENCE) == 1
    return text


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b524_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT, "chars", len(OUT.read_text()))
