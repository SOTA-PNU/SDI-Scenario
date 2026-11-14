"""B525 -> B526: public map tools and two narrowly scoped base corrections."""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import b525_build_prompt as PREV

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_25.md"
SRC_SHA16 = "8cb0f52ae53da97b"
OUT = ARENA / "PROMPT_B5_26.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v14.py"
BASE_SHA16 = "e0346dc12d4fb3d8"
TITLE = "# ARENA 미션 프롬프트 B-5-26 (B-5-25 에 지도 격자 도구를 더하고 지도 위험 선택과 코너 목표를 보정함)"
FIRST_ACTION, CALL_EXAMPLE = PREV.FIRST_ACTION, PREV.CALL_EXAMPLE
SCENE_SENTENCE = PREV.SCENE_SENTENCE
VALIDATE_DOC = PREV.VALIDATE_DOC
SCAFFOLD = PREV.SCAFFOLD.replace(
    "# TODO: 지도 격자 A*, 모든 연결 검사, 직선 단순화, validate_plan 승인 확인.",
    "# TODO: 아래 지도 도구로 A*·연결 검사·직선 단순화 후 validate_plan 승인 확인.")

MAP_DOC = '''- base의 `occupancy_grid(clearance_m, res)`로 격자를 얻어 **A* 탐색과 직선 단순화는 직접 구현**합니다.
  반환은 `(blocked, origin, res)`이며 `blocked[iy,ix]`는 bool, True가 점유입니다. 요청 여유로 벽과 지도 물체를 팽창합니다.
  `origin=(xmin,ymin)`은0번 칸의 좌하단 바깥 모서리이며, `xy_of((ix,iy),origin,res)`는 칸 중심의 월드 좌표입니다.
  `cell_of((x,y),origin,res)`는 floor로 해당 칸을 구합니다. 범위 보정은 없으므로 배열 접근 전에
  `0<=ix<blocked.shape[1]`, `0<=iy<blocked.shape[0]`를 검사하고 음수 인덱스의 뒤쪽 참조를 허용하지 않습니다.
  격자는 WALLS 안의 완전한 칸만 포함하며 끝점 연결·경로 탐색은 하지 않습니다.
- 칸 중심이 유효해도 연결 선분은 막힐 수 있습니다. 모든8방향 이웃 연결, 실제 출발/목표 좌표와 격자 칸의 연결,
  단순화의 모든 지름길을 `segment_free(p,q,clearance_m)`로 검사합니다. False인 연결은 사용하지 않습니다.
  가까운 칸이 막히면 다른 유효 칸과 연결을 유한 탐색합니다. 실제 출발점을 검사에서 빼지 않습니다.
  이 함수와 격자는 `validate_plan`과 같은 중심 여유 `max(0.25,clearance_m)` 기준을 씁니다.
  선분은 양끝을 포함해 최대0.1m 간격으로 검사하며, 칸 검사만으로 연결을 보장하지 않습니다.
  지도 직사각형의 원래 모서리는 지도 여유가0이므로 양의 여유를 만족하는 경유점으로 쓸 수 없습니다.
'''

CHANGES = (
    "Selected v14/title; retained exact scene preflight sentence, three empty hooks, immutable harness,200s calls and4/4 criterion.",
    "Replaced repeated raw-map/grid prose with occupancy_grid/cell_of/xy_of/segment_free contracts, explicit bounds and per-edge checks; no A* implementation supplied.",
    "Explained map-gap-first emergency selection only when no candidate is map-clear; safe and map-clear branches remain unchanged.",
    "Added adopted-clearance pose-to-target segment check, unsafe_connector fallback and checked-zero if the fallback is also blocked.",
    "Kept validate_plan acceptance, complete batch diagnostics, failed-plan retry, goal completion and exact tested EDIT submission unchanged.",
)
SHORTENINGS = CHANGES


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def build(src, base=None):
    assert hashlib.sha256(src.encode()).hexdigest()[:16] == SRC_SHA16
    base = BASE.read_text() if base is None else base
    assert BASE_SHA16 is not None, "v14 source must be frozen before generation"
    assert hashlib.sha256(base.encode()).hexdigest()[:16] == BASE_SHA16
    text = replace_once(src, src.splitlines()[0], TITLE)
    text = replace_once(text, PREV.SCAFFOLD, SCAFFOLD)
    start = text.index("- 지도는 base의 `WALLS`")
    end = text.index("- 기준점(anchor)", start)
    text = text[:start] + MAP_DOC + text[end:]
    text = replace_once(text,
        "경로를 따른 앞쪽 거리(arc)가0.3~2.0m일 때만 그대로 사용합니다. 두 경계값을 포함합니다.",
        "경로를 따른 앞쪽 거리(arc)가0.3~2.0m이고 현재 pose→target 선분이 채택 여유의 `segment_free`를 통과할 때만 그대로 사용합니다. 두 거리 경계값을 포함합니다.")
    text = replace_once(text,
        "유효한 앞점이 없어 정지 입력을 요청합니다. 보정 이유별 DBG는10초마다 한 줄이며 시각 역행 시 기록을 초기화합니다.",
        "유효한 앞점이 없어 정지 입력을 요청합니다. 연결이 막힌 목표는 `unsafe_connector`로 보정하며, 보정된 앞점도 같은 선분 검사를 통과하지 못하면 정지 입력을 요청합니다.\n"
        "보정 이유별 DBG는10초마다 한 줄이며 시각 역행 시 기록을 초기화합니다.")
    text = replace_once(text,
        "정지 요청이라도 위험하면 탈출 명령이 선택될 수 있습니다. base의 검사는 충돌 회피를 보장하지 않으므로 반드시 실제 시험하십시오.",
        "지도 검사에 안전한 후보가 하나도 없는 긴급 선택은 `min_gap_map`이 큰 후보를 먼저 고르고 그다음 물체 기준을 비교합니다.\n"
        "안전 후보와 지도 안전 후보가 있는 기존 선택은 유지합니다. 정지 요청도 위험하면 탈출 명령이 될 수 있습니다. 검사는 충돌 회피를 보장하지 않으므로 반드시 실제 시험하십시오.")
    text = replace_once(text,
        "pose-target 연결도 지도 여유를 확인합니다. 범위 밖의 반환값은 base의 보정을 받습니다.",
        "pose-target 연결도 `segment_free`로 검사합니다. base는 채택한 경로 여유로 연결을 다시 검사하며, arc 범위 안이어도 코너를 가로지르는 연결은 보정합니다.")
    text = replace_once(text, "계획 호출은 최초, 이 경로 없음 재시도, 지도 막힘 정지가10초를 넘은 다음 틱에 수행합니다.",
        "계획 호출은 최초, 경로 없음 재시도, 지도 막힘 정지가10초를 넘은 다음 틱에 수행합니다. 보정 목표의 연결도 막혀 실제 반환이(0,0)인 상태가 연속10초를 넘어도 다음 틱에 재계획합니다.")
    text = text.replace("보정된 앞점도 같은 선분 검사를 통과하지 못하면 정지 입력을 요청합니다.", "보정된 앞점도 같은 선분 검사를 통과하지 못하면 `unsafe_fallback`으로 정지 입력을 요청합니다.")
    assert text.count(SCENE_SENTENCE) == 1
    assert "segment_clearance(p,q) >= clearance_m" not in text
    return text


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b526_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT, "chars", len(OUT.read_text()))
