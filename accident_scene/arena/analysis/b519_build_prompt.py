"""B-5-19: deterministic grid planning, executable replanning, shorter map horizon."""
import ast
from datetime import datetime
from pathlib import Path
import shutil

import b518_build_prompt as PREV
from b515_build_prompt import replace_once

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_18.md"
OUT = ARENA / "PROMPT_B5_19.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py"
TITLE = "# ARENA 미션 프롬프트 B-5-19 (B-5-18 에 무작위 없는 경로 계산과 막힘 때 다시 계산하기를 더하고, choose_command() 의 지도 검사 시간을 줄임)"
CALL_EXAMPLE, FIRST_ACTION = PREV.CALL_EXAMPLE, PREV.FIRST_ACTION
base_config, escape_config, motion_doc = PREV.base_config, PREV.escape_config, PREV.motion_doc
INPUT_DOC, ROUTE_DOC = PREV.INPUT_DOC, PREV.ROUTE_DOC
PATH_DOC, PATH_API_DOC = PREV.PATH_DOC, PREV.PATH_API_DOC
GOAL_DOC, SUBMISSION_DOC = PREV.GOAL_DOC, PREV.SUBMISSION_DOC

DETERMINISTIC_DOC = '''- 경로는 `controller()` 안에서 지도 격자 위 A* 로 결정론적으로 계산하십시오. 격자 간격, 이웃을 방문하는 순서,
  비용이 같은 점의 우선순위를 정해 두어 같은 지도, 계획 시작 자세, `GOAL`, 여유 입력이면 항상 같은 경로가 나와야 합니다.
  `random`, 무작위 표본을 뽑는 RRT, 난수 시드나 벽시계에 따라 달라지는 경로는 쓰지 않습니다. 시험에서 통과한 경로가
  같은 입력의 채점에서도 같게 나와야 하기 때문입니다. 재계획의 현재 자세가 달라졌다면 그 자세를 새 입력으로 계산합니다.
  경유점이나 경로의 x, y 좌표를 손으로 골라 숫자로 적지 마십시오. 마지막 점은 `GOAL` 입니다.
  A* 의 점과 이웃 사이 연결을 지도 여유로 검사하고, 얻은 경로는 아래의 고정된 기준점에서 직선으로 줄이는 방식으로
  단순화한 뒤 `check_path()` 로 완성 경로를 검사합니다. 점 수만 늘려 촘촘하게 만든 경로를 그대로 따라가지 않습니다.
'''
REPLAN_DOC = '''- 지도 막힘이 10 s 넘게 이어지면 실제로 경로를 다시 계산하는 분기를 `controller()` 에 넣으십시오.
  DBG 문자열을 읽거나 파싱하지 말고, 매 호출 `choose_command()` 의 반환 `info` 에서 `info['chosen'] == 'stop'`,
  반환한 `(v, w) == (0, 0)`, `info['candidates']['follow']['t_hit_map'] is not None` 이 모두 참인지 확인합니다.
  이 조건이 처음 참인 시각을 제어 시각 `t` 로 저장하고, 끊김 없이 10 s 를 넘으면 현재 `pose` 에서 더 큰 지도 여유로
  A* 를 다시 계산하고 직선 단순화와 `check_path()` 를 거칩니다. 새 경로가 통과하면 경로와 진행 위치, 앞쪽 목표를 갱신해
  다음 제어 주기부터 새 경로를 따릅니다. 조건이 깨지거나 시각이 역행하면 연속 막힘 시각을 초기화합니다.
  재계획을 시도한 뒤에도 그 시각을 초기화하여 매 호출 반복 계산하지 않습니다. 새 경로가 검사를 통과하지 못하면 실패
  원인을 기록하고 검사한 경로를 찾도록 탐색을 고칩니다. 검증하지 않은 직선이나 손으로 고른 점으로 대체하지 않습니다.
  재계획 중에도 반환 명령은 `choose_command()` 의 결과를 그대로 씁니다. 경고를 출력만 하고 기존 경로에 계속 머무는
  구현이나 일정 시각까지 기다리는 분기로 대신하지 마십시오.
'''
BLOCKED_DOC = '''  follow 가 지도 여유를 지키지 못하고 stop 을 10 s 넘게 연속 선택하면, base 는 10 s 마다 별도 DBG 한 줄
  `follow 가 지도에 막힘: 경로 여유 부족` 을 남깁니다. 이 줄은 '경로 계산' 절의 재계획 조건이 이어지고 있다는 진단입니다.
  실제 재계획 조건은 위 `info` 의 follow 지도 TTC와 stop 선택으로 추적하고, DBG 문자열을 파싱하지 않습니다.
  이 경고도 모델의 추가 DBG 제한과 별도로 남습니다.
'''


def map_horizon_config(base):
    values = [ast.literal_eval(n.value) for n in ast.parse(base).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BASE_MAP_HORIZON_CONFIG" for t in n.targets)]
    assert len(values) == 1
    return values[0]


def chooser_doc(config, escape, map_config):
    doc = PREV.chooser_doc(config, escape)
    doc = replace_once(doc, PREV.BLOCKED_DOC, BLOCKED_DOC)
    doc = replace_once(doc,
        "  `e_target` 은 현재 자세에서 `target_xy` 로 향하는 각도 오차를 [-π, π] 로 접은 값입니다. 추가 후보도 모두 같은\n"
        "  `horizon_s` 로 `check_motion()` 을 거칩니다. 안전은 추가 여유를 적용한 물체와 지도를 합친 `t_hit` 이 None 인 것입니다.",
        "  `e_target` 은 현재 자세에서 `target_xy` 로 향하는 각도 오차를 [-π, π] 로 접은 값입니다. 추가 후보도 모두 같은 몸체\n"
        "  검사로 비교하되 물체와 지도의 검사 시간을 나눕니다. 물체와 추가 불확실성 여유는 `horizon_s`(기본 3 s) 전체를\n"
        "  검사하며, 각 후보의 `end_pose` 도 그 전체 시간 끝 자세입니다. 후보 명령 `(v, w)` 의 지도 검사 시간은\n"
        f"  `min(horizon_s, max({map_config['minimum_s']:g}, {map_config['buffer_s']:g} + max(abs(v)/{map_config['linear_decel_m_s2']:g}, abs(w)/{map_config['angular_decel_rad_s2']:g})))` s 입니다.\n"
        "  통상 0.8–1.0 s 이며, 완충 시간과 감속 시간에 기반한 휴리스틱입니다. 실제 제동이나 충돌 회피를 보장하는 값은 아닙니다.\n"
        "  모퉁이에서 현재 명령을 3 s 고정했을 때 경로 바깥 지도에 닿는 것만으로 정지하지 않도록 지도 쪽 시간을 줄인 것입니다.\n"
        "  공개 함수 `check_motion()` 을 직접 부르면 물체와 지도 모두 지정한 `horizon_s` 전체를 검사하는 기존 동작을 유지합니다.\n"
        "  안전은 각각의 검사 시간에서 추가 여유를 적용한 물체와 지도를 합친 `t_hit` 이 None 인 것입니다.")
    doc = replace_once(doc,
        "  모든 후보가 위험할 때는 먼저 예측 구간 전체에서 지도 여유 조건을 지키는 `t_hit_map is None` 후보만 남깁니다.",
        "  모든 후보가 위험할 때는 먼저 각 후보의 짧은 지도 검사 구간 전체에서 지도 여유 조건을 지키는 `t_hit_map is None` 후보만 남깁니다.")
    doc = replace_once(doc,
        "  `info` 는 `chosen`, `reason`, `candidates`, 고른 후보의 `min_gap_obj`, `robust_min_gap_obj`, `min_gap_map`,",
        "  `info` 는 `chosen`, `reason`, `candidates`, 고른 후보의 `min_gap_obj`, `robust_min_gap_obj`, `min_gap_map`,\n"
        "  `object_horizon_s`(물체 검사 시간), `map_horizon_s`(지도 검사 시간),")
    doc = replace_once(doc,
        "  `robust_end_gap_obj` 를 담고, 결합된 `min_gap` 필드는 없습니다. base 는 0.5 s 마다 시각, 선택, 이유, follow/stop 의",
        "  `robust_end_gap_obj`, `object_horizon_s`, `map_horizon_s` 를 담고, 결합된 `min_gap` 필드는 없습니다.\n"
        "  base 는 0.5 s 마다 시각, 선택, 이유, follow/stop 의")
    doc = replace_once(doc,
        "  `t_hit`, 선택한 후보의 원래/추가 여유 적용 물체 간격과 지도 간격을 DBG 한 줄로 기록합니다. 모두 예측 구간의 최솟값이며",
        "  `t_hit`, 선택한 후보의 원래/추가 여유 적용 물체 간격과 지도 간격을 DBG 한 줄로 기록합니다. 물체는 전체 물체 검사 시간,\n"
        "  지도는 해당 후보의 짧은 지도 검사 시간에서 각각 얻은 최솟값이며")
    return doc


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    config, escape, map_config = base_config(base), escape_config(base), map_horizon_config(base)
    src = replace_once(src, src.splitlines()[0], TITLE)
    src = replace_once(src, "시험 주행은 `sim_seconds=200` 을 넣어 부르십시오. 지금까지 통과한 주행은 모두 140 s 안에 목표에 닿았습니다.", "시험 주행은 매번 최상위 도구 인자에 `sim_seconds=200` 을 명시해 부르십시오.")
    src = replace_once(src,
        "- 경로는 `controller()` 안에서 지도로 계산합니다. 경유점이나 경로의 x, y 좌표를 손으로 골라 숫자로 적지\n"
        "  마십시오. 어떤 알고리즘으로 계산할지는 직접 정합니다. 마지막 점은 `GOAL` 입니다.\n",
        DETERMINISTIC_DOC)
    src = replace_once(src,
        "  전진, 제자리 회전, 뒤로 물러나기를 `check_motion()` 으로 재고, 아래에 설명한 안전·진전 규칙으로 고릅니다.",
        "  전진, 제자리 회전, 뒤로 물러나기를 `check_motion()` 과 같은 몸체 기하로 재되, 물체와 지도에는 아래의 서로 다른\n"
        "  검사 시간을 적용합니다. 그 결과로 아래에 설명한 안전·진전 규칙에 따라 고릅니다.")
    src = replace_once(src, ROUTE_DOC, ROUTE_DOC + REPLAN_DOC)
    start = src.index("- 명령 선택 함수 `choose_command(")
    end = src.index("\n## 지금까지의 주행에서 알게 된 것\n", start)
    src = src[:start] + chooser_doc(config, escape, map_config) + src[end:]
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b519_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
