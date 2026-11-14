"""B-5-17 prose: map-clear emergency pool and verified route/chooser inputs."""
from datetime import datetime
from pathlib import Path
import shutil

import b516_build_prompt as PREV
from b515_build_prompt import replace_once

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_16.md"
OUT = ARENA / "PROMPT_B5_17.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v6.py"
TITLE = "# ARENA 미션 프롬프트 B-5-17 (B-5-16 에 지도 여유 우선 선택과 경로·추종 입력 검증을 더함)"
CALL_EXAMPLE = PREV.CALL_EXAMPLE
FIRST_ACTION = PREV.FIRST_ACTION
base_config, escape_config, motion_doc = PREV.base_config, PREV.escape_config, PREV.motion_doc
INPUT_DOC = '''- `choose_command()` 에 넣는 `(v_follow, w_follow, target_xy)` 는 지도에서 계산한 경로와 현재 `pose`, 목표 정렬
  상태만으로 계산하십시오. 관측 물체가 앞쪽 원뿔이나 목표까지의 선 가까이에 있다는 이유로 입력 속도를 줄이거나
  `(0, 0)` 으로 덮어쓰지 않습니다. `observed_objects()` 는 진단 출력에 쓸 수 있지만, 입력 명령이나 목표점을 바꾸는
  별도 물체 판단에는 쓰지 않습니다. 물체에 따른 기다림·감속·회피는 `choose_command()` 가 합니다. 입력을 미리 0 으로
  바꾸면 함수가 그 정지를 안전한 follow 로 유지할 수 있으므로, 호출을 거쳤다는 사실만으로 위 규칙을 지킨 것은 아닙니다.
'''
ROUTE_DOC = '''- 단순화할 때는 보존할 현재 기준점(anchor)을 고정하고, 그 점에서 더 먼 점까지의 연결을 `segment_clearance()` 로
  검사하십시오. 지도 여유를 만족하는 가장 먼 점을 새 점으로 추가한 다음 그 새 점을 다음 기준점으로 삼습니다.
  검사할 때마다 기존 마지막 점을 다음 점으로 덮어쓰면 시작점과 중간 경로가 지워질 수 있으므로 그렇게 하지 않습니다.
  최초 계획의 `START` 또는 재계획의 현재 위치와 마지막 `GOAL` 을 보존하고, 완성된 경로의 모든 연결 구간을 다시
  검사하십시오. 시작 위치에서 첫 추종점까지의 연결도 검사 대상입니다. 실패한 구간이 하나라도 있으면 경로를 고칩니다.
- 경유점 간격은 통과 반경 0.35 m 보다 충분히 길게 만드십시오. 짧은 구간은 위 방식으로 지도 여유를 유지하며 묶고,
  코너를 잘라 지도 물체에 가까워지지 않았는지 검사합니다. 추종 목표는 경로 위 진행 위치보다 앞쪽에 잡고, 진행에 따라
  경로 위에서 연속적으로 옮깁니다. 경유점마다 서서 다음 점을 향해 도는 분기를 만들지 않습니다. 앞쪽 목표까지의 연결도
  지도 여유를 확인합니다. 목표점과 경로는 지도에서 계산하며 좌표를 손으로 넣지 않습니다.
- 경로 계산이 실패했을 때 검증하지 않은 `START`–`GOAL` 직선이나 손으로 고른 중간 좌표로 대신하지 마십시오.
  `choose_command()` 는 주어진 명령 후보를 검사하는 함수이며 지도 경로를 대신 찾지 않습니다. 계산 실패 원인을 기록하고
  지도 탐색과 여유 설정을 검토해 다시 계산합니다. 경로 진행이 멈추면 현재 자세와 경로의 진행 위치, 앞쪽 목표, 선택 결과를
  확인하고 지도로 검증한 목표나 경로를 갱신하십시오. 정해 둔 시간만 기다리는 분기나 특정 시각에 출발하는 분기는 넣지 않습니다.
'''


def chooser_doc(config, escape):
    doc = PREV.chooser_doc(config, escape)
    doc = replace_once(doc,
        "  모든 후보가 위험할 때는 `min_gap_map > 0` 인 후보가 하나라도 있으면 `min_gap_map <= 0` 인 후보를 먼저 제외합니다.",
        "  모든 후보가 위험할 때는 먼저 예측 구간 전체에서 지도 여유 조건을 지키는 `t_hit_map is None` 후보만 남깁니다.\n  그런 후보가 없을 때만 `min_gap_map > 0` 인 후보를 남겨 0 이하인 후보를 제외하고, 그것도 없으면 전체 후보를 씁니다.\n  따라서 지도 간격이 양수라도 기본 여유 0.2 m 안에 들어가는 후보는 지도 여유를 지키는 대안보다 앞세우지 않습니다.")
    doc = replace_once(doc,
        "  지도 간격과 물체 간격을 하나의 `min_gap` 으로 합쳐 경쟁시키지 않습니다. 모든 후보의 지도 간격이 0 이하이면\n  제외할 대안이 없으므로 전체 후보에 위 순위를 적용합니다. 이 경우도 충돌 회피를 보장하지는 않습니다.",
        "  지도 간격과 물체 간격을 하나의 `min_gap` 으로 합쳐 경쟁시키지 않습니다. 모든 후보가 위험한 상태에서의 선택은\n  위 순위로 피해를 줄이려는 것이며 충돌 회피를 보장하지는 않습니다.")
    doc = replace_once(doc,
        "  `map_filtered`(모두 위험한 분기에서 지도 침범 후보를 제외했는지), `excluded_map_names`(그때 제외한 이름 목록)를 담습니다.",
        "  `map_filter`(모두 위험한 분기의 지도 기준: clearance, positive, none), `map_filtered`(실제 제외한 후보가 있는지),\n  `excluded_map_names`(실제로 제외한 이름 목록)를 담습니다. clearance 는 지도 여유 조건, positive 는 양수 간격 기준입니다.\n  clearance 기준이어도 모든 후보가 그 기준을 만족하면 제외 목록은 비고 `map_filtered=False` 일 수 있습니다.\n  none 은 안전한 후보를 고른 분기 또는 양수 간격 기준에서 실제로 제외한 후보가 없는 경우도 포함합니다.")
    return replace_once(doc,
        "  `reason` 은 follow_safe, slow_safe, stop_safe, fast_progress, safe_progress, map_filtered_object_gap,\n  emergency_object_gap 중 하나입니다.",
        "  `reason` 은 follow_safe, slow_safe, stop_safe, fast_progress, safe_progress, map_clearance_object_gap,\n  map_filtered_object_gap, emergency_object_gap 중 하나입니다.")


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    config, escape = base_config(base), escape_config(base)
    src = replace_once(src, src.splitlines()[0], TITLE)
    src = replace_once(src,
        "  전진, 제자리 회전, 뒤로 물러나기를 `check_motion()` 으로 재고, 아래에 설명한 안전·진전 규칙으로 고릅니다.\n",
        "  전진, 제자리 회전, 뒤로 물러나기를 `check_motion()` 으로 재고, 아래에 설명한 안전·진전 규칙으로 고릅니다.\n" + INPUT_DOC)
    src = replace_once(src, PREV.ROUTE_DOC, ROUTE_DOC)
    # Update only the versioned chooser paragraph; check_motion is unchanged.
    start = src.index("- 명령 선택 함수 `choose_command(")
    end = src.index("\n## 지금까지의 주행에서 알게 된 것\n", start)
    src = src[:start] + chooser_doc(config, escape) + src[end:]
    src = replace_once(src,
        "- `sim_seconds`: 도구의 허용 범위는 10–600 s 이지만, 이번 시험에서는 항상 200 을 명시하십시오.",
        "- `sim_seconds` (매 호출 필수): 도구 호출 JSON 의 최상위에 `\"sim_seconds\": 200` 을 반드시 넣으십시오.\n  `code` 문자열 안의 주석이나 변수로 적는 것은 도구 인자를 지정한 것이 아닙니다. 허용 범위는 10–600 s 입니다.")
    src = replace_once(src,
        "1. 호출 예시: `" + CALL_EXAMPLE + "`. `code` 에는 시작·끝 주석을 포함한 수정 영역만 넣습니다.\n   `sim_seconds` 는 반드시 200 을 명시합니다. base 전문과 Markdown 코드 펜스를 도구 인자에 넣지 마십시오.",
        "1. 매 호출의 최상위 JSON 에 `code` 와 `sim_seconds` 두 키를 모두 넣습니다. 예시: `" + CALL_EXAMPLE + "`.\n   보내기 전에 `sim_seconds` 값이 200 인지 확인하십시오. `code` 에는 시작·끝 주석을 포함한 수정 영역만 넣고,\n   base 전문과 Markdown 코드 펜스는 도구 인자에 넣지 않습니다.")
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b517_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
