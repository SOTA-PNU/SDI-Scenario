"""B-5-21: four-function EDIT scaffold; no reference implementation in input."""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import b520_build_prompt as PREV
from b515_build_prompt import replace_once

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_20.md"
OUT = ARENA / "PROMPT_B5_21.md"
BASE = PREV.BASE
BASE_SHA16 = PREV.BASE_SHA16
TITLE = "# ARENA 미션 프롬프트 B-5-21 (B-5-20 에 EDIT REGION 뼈대 코드를 더함)"
CALL_EXAMPLE, FIRST_ACTION = PREV.CALL_EXAMPLE, PREV.FIRST_ACTION

SCAFFOLD = '''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
_PLAN_POSE = None
_PLAN_CLEARANCE = 1.0  # 탐색 중심 여유(m); 검사 최소 기준은 0.5 m
_PATH = None
_REPLAN_PENDING = True
_PLAN_FAILED = True
_BLOCK_SINCE = None
_LAST_T = None
_NEXT_PLAN_T = 0.0

def plan_path():
    # 입력: _PLAN_POSE, _PLAN_CLEARANCE, 지도, GOAL. 출력: 검증한 경로 또는 None.
    # TODO: 결정론적 격자 A*, 현재 위치-출발 칸 연결, 직선 단순화, check_path.
    # TODO: 실패 시 여유를 조정해 유한 재시도; 최소 중심 0.5 m 유지, 직선 대체 금지.
    raise NotImplementedError

def lookahead_target(path, pose):
    # 출력: 검증 경로 위 연속 앞쪽 (x, y), 유효 연결이 없으면 None.
    # TODO: 경로 투영/진행 위치로 계산. GOAL 직접 사용 금지; 마지막 구간만 경로 끝점.
    raise NotImplementedError

def follow_cmd(pose, target):
    # 출력: 추종 방법의 (v_follow, w_follow). 물체 판단이나 chooser 호출 없음.
    # TODO: wrap 각도 오차, 0.35 rad 분기, v 최대 0.3, w 최대 절댓값 0.4.
    raise NotImplementedError

def controller(t, pose, env):
    global _PLAN_POSE, _PLAN_CLEARANCE, _PATH, _REPLAN_PENDING, _PLAN_FAILED
    global _BLOCK_SINCE, _LAST_T, _NEXT_PLAN_T
    _PLAN_POSE = pose
    # TODO: 목표 거리/방향과 계획 상태. 목표 0.5 m 안에서는 마무리를 우선.
    goal_mode, goal_stop, replan_now = ...
    if goal_mode:
        # TODO: 방향 오차 0.15 rad 밖이면 정렬, 안이면 정지 입력; target=GOAL[:2].
        v_follow, w_follow, target = ...
    else:
        if replan_now:
            candidate_path = plan_path()
            # TODO: 검사 통과 경로만 _PATH에 채택; 실패/재시도 상태 갱신.
        target = None if _PLAN_FAILED else lookahead_target(_PATH, pose)
        if target is None:
            # TODO: 실패 상태 유지와 재계획 예약. 옛 경로나 직선으로 대체하지 않음.
            v_follow, w_follow, target = 0.0, 0.0, GOAL[:2]
        else:
            v_follow, w_follow = follow_cmd(pose, target)
    v, w, info = choose_command(v_follow, w_follow, target)
    # TODO: info의 연속 지도 stop >10 s면 더 큰 여유로 다음 틱 재계획만 예약.
    # TODO: 조건 해제/시각 역행 시 막힘 시각 초기화. 여기서 chooser를 다시 부르지 않음.
    done = goal_stop and v == 0.0 and w == 0.0
    return v, w, done
# ========================= [END EDIT REGION] ==============================='''

SCAFFOLD_DOC = '''## EDIT REGION 뼈대

**이 뼈대의 구조, 함수 이름, 호출 순서를 바꾸지 않는다. 몸통만 채운다.** 아래 네 함수와 전역 상태를 유지하고,
필요한 보조 함수는 해당 함수 몸통 안에 둡니다. TODO 와 `...`, `raise NotImplementedError` 를 실제 코드로 채운 뒤
시험하십시오. 아래는 완성 제어기나 경로 알고리즘의 구현이 아닙니다. 미완성 뼈대를 그대로 제출하지 않습니다.

- `plan_path()` 는 인자가 없습니다. controller 가 계획 직전에 `_PLAN_POSE` 를 현재 `pose` 로 설정합니다.
  `_PLAN_CLEARANCE` 는 탐색 중심 여유이며 초기 1.0 m, 최소 0.5 m 입니다. 함수는 현재 위치부터 GOAL 까지
  check_path 를 통과한 점 목록 또는 None 을 반환합니다. 전역 경로 채택은 controller 에서만 합니다.
- 접근 모드는 `plan_path()`(필요할 때) → `lookahead_target(path, pose)` → `follow_cmd(pose, target)` 순서입니다.
  target 은 경로 위 앞쪽 점이며, GOAL 을 바로 넣지 않습니다. 마지막 경로 구간에서만 경로의 끝점을 쓸 수 있습니다.
  목표 마무리와 계획 실패 정지 요청만 별도 입력이며, 모든 분기는 마지막 chooser 한 번으로 모읍니다.

```python
''' + SCAFFOLD + '''
```

'''

INPUT_DOC = '''- 추종 입력과 target 은 검증한 지도 경로·현재 pose·목표 상태로만 계산합니다. 관측 부채꼴/거리로 정지나 후진을
  덧붙이지 않습니다. `observed_objects()` 는 추가 진단에만 사용하고 물체 대응은 chooser 에 맡깁니다.
  매 틱 공통 chooser 호출은 정확히 한 번이며, 반환 후 재호출하거나 선택된 속도를 바꾸지 않습니다.
'''

PATH_DOC = '''- 최초와 모든 재계획의 완성 경로는 `check_path(points)` 의 `min_gap_map >= 0.25` 와
  `first_bad_segment is None` 을 모두 만족해야 합니다. 몸체 반폭 0.25 m 를 뺀 값이므로 중심 여유는 최소 0.5 m 입니다.
  실제 현재 위치부터 GOAL 까지를 검사하고, 현재 위치-출발 칸과 모든 이웃 연결도 지도 여유를 확인합니다.
  출발 칸이 막히면 지도에서 다른 유효 칸과 연결을 탐색하며, 현재 위치를 빼거나 최소 기준을 낮추지 않습니다.
  탐색 여유를 조정한 재시도도 매번 검사합니다. 실패한 경로나 검증되지 않은 직선·수동 중간점은 채택하지 않습니다.
'''

PLAN_FAILURE_DOC = '''- 목표 범위 밖에서 계획이나 앞쪽 목표가 실패하면 검증된 새 경로를 얻을 때까지 실패 상태를 유지합니다.
  재시도 대기 틱에도 입력 `(0, 0)`, target `GOAL[:2]`, done=False 로 공통 chooser 에 정지를 요청합니다.
  위험할 때 고른 탈출 명령도 그대로 반환하며 강제로 `(0, 0)` 을 내지 않습니다. GOAL 은 실패 직선 추종 지시가 아닙니다.
  재시도는 제어 시각 t 의 경과 시간과 탐색 상태로 예약하고 매 틱 전체 탐색이나 특정 절대 시각 대기로 대신하지 않습니다.
'''

REPLAN_DOC = '''- chooser 반환 뒤 `info['chosen'] == 'stop'`, `(v, w) == (0, 0)`,
  `info['candidates']['follow']['t_hit_map'] is not None` 이 연속 10 s 를 넘으면 다음 틱 재계획을 예약합니다.
  그때의 현재 pose 와 더 큰 탐색 여유로 plan_path 를 다시 부르고 같은 경로 검사를 거칩니다. 같은 틱 재호출은 금지합니다.
  조건 해제·시각 역행·재계획 시도 뒤에는 막힘 시작 시각을 초기화합니다. DBG 문자열은 파싱하지 않습니다.
'''

GOAL_DOC = '''- 경로 목표와 별개로 `d_goal = hypot(GOAL[0]-x, GOAL[1]-y)`, `e_yaw = wrap(GOAL[2]-yaw)` 를 계산합니다.
  `d_goal <= 0.5` m 이면 예약된 계획보다 목표 마무리를 우선합니다. `|e_yaw| > 0.15` rad 이면 입력은
  `(0, clamp(1.5*e_yaw, -0.4, 0.4))`, 아니면 `(0, 0)` 이며 target 은 `GOAL[:2]` 입니다.
  목표 정지 상태에서 공통 chooser 반환 v,w 가 모두 0 일 때만 done=True 입니다. 실패 정지는 완료가 아닙니다.
  `d_goal > 0.5` m 로 벗어나면 검증 경로 접근으로 돌아갑니다. 이 기준은 성공 조건 0.75 m / 0.26 rad 안에 있습니다.
'''

FORMAT_DOC = '''최종 답은 위 뼈대의 몸통을 채워 시험에서 pass 를 받은 EDIT REGION 을 **python 코드 펜스 하나**에 담습니다.
시험 도구는 `{"code": "...", "sim_seconds": 200}` JSON 을 쓰지만 최종 답은 JSON 객체/문자열이 아닙니다.
도구 code 에는 펜스를 넣지 않고, 최종 답에는 실제 줄바꿈이 있는 코드를 넣습니다. 설명만 내는 답도 실격입니다.
'''

SHORTENINGS = (
    ("input constraints", PREV.INPUT_DOC, INPUT_DOC),
    ("route validation", PREV.PATH_DOC, PATH_DOC),
    ("persistent planning failure", PREV.PLAN_FAILURE_DOC, PLAN_FAILURE_DOC),
    ("deferred replanning", PREV.REPLAN_DOC, REPLAN_DOC),
    ("tick order moved into scaffold", PREV.TICK_DOC, ""),
    ("goal completion", PREV.GOAL_DOC, GOAL_DOC),
    ("format examples replaced by scaffold and compact format rule", PREV.FORMAT_DOC, FORMAT_DOC),
)


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    assert hashlib.sha256(base.encode()).hexdigest()[:16] == BASE_SHA16
    old_length = len(src)
    src = replace_once(src, src.splitlines()[0], TITLE)
    for _, before, after in SHORTENINGS:
        src = replace_once(src, before, after)
    src = replace_once(src, "## 경로 계산\n", SCAFFOLD_DOC + "## 경로 계산\n")
    src = replace_once(src,
        "아래 '제출' 절에 올바른/잘못된 최종 답 예시가 있습니다.",
        "아래 뼈대와 '제출' 절의 형식을 따릅니다.")
    src = replace_once(src,
        "- 위 방법으로 단순화한 경로 위 앞쪽 목표점 `(tx, ty)` 를 매 호출 갱신해 이어서 따라갑니다. 경로 진행 위치와\n"
        "  통과한 경유점은 상태로 관리하되, 경유점마다 멈추는 분기를 넣지 않습니다.",
        "- 위 방법으로 단순화한 경로 위 앞쪽 목표점 `(tx, ty)` 를 매 호출 갱신해 이어서 따라갑니다. 현재 `pose` 로 경로\n"
        "  진행 위치를 계산해 통과한 경유점을 처리하되, 경유점마다 멈추는 분기를 넣지 않습니다.")
    assert len(src) <= old_length, (len(src), old_length)
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b521_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
