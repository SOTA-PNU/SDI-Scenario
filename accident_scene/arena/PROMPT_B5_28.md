# ARENA 미션 프롬프트 B-5-28 (B-5-27 에 시작점 보존과 계획 단계 및 경로 형태 진단을 더함)

당신은 모바일 로봇의 지도 경로 계산과 추종 명령을 작성합니다. 함께 주어진 `base_carter_run.py` 의
`[EDIT REGION]` 안에 아래 세 함수만 구현하십시오. 경로 계산기는 base 에 없으며 모델이 직접 작성합니다.
안정 기준은 a0~a3 네 표본 모두 통과(4/4)입니다. 하네스·splice·EDIT 밖 코드는 변경하지 않습니다.

시험 도구의 `code` 와 최종 답에는 시작·끝 주석을 포함한 EDIT REGION 블록만 넣습니다. base 전문은 반복하지 않습니다.
최종 답은 python 코드 펜스 하나이며, JSON 객체/문자열이나 설명만 제출하면 안 됩니다.

## base 가 맡는 제어 루프

base 는 매 1/60 s 제어 루프를 돌려 필요한 모델 함수를 호출하고, 센서 기반 후보 검사와 최종 명령 선택을 수행합니다.
첫 계획과 지도 막힘 정지가 연속10초를 넘은 재계획에서 중심 여유0.5,0.45,0.4,0.35m 순으로 planner를 부릅니다.
현재 위치부터 GOAL까지 검사에 통과한 첫 경로를 채택합니다. 모두 실패하면 이전 경로를 유지합니다.
아직 채택한 경로가 없으면 정지 입력을 요청하고 재시도 간격을10,20,40,60초로 늘려 최대60초를 유지합니다. 경로를 채택하거나 시각이 역행하면 간격을 초기화합니다.
계획 호출은 최초, 경로 없음 재시도, 지도 막힘 정지가10초를 넘은 다음 틱에 수행합니다. 보정 목표의 연결도 막혀 실제 반환이(0,0)인 상태가 연속10초를 넘어도 다음 틱에 재계획합니다.
모델은 별도 제어 루프·재계획 타이머·거리 정지·후진 분기를 만들지 않습니다.
접근 중 base 는 target의 경로까지 거리가1.0m 이하이고, 로봇과 target을 경로에 사영한 두 점 사이의
경로를 따른 앞쪽 거리(arc)가0.3~2.0m이고 현재 pose→target 선분이 채택 여유의 `segment_free`를 통과할 때만 그대로 사용합니다. 두 거리 경계값을 포함합니다.
그 밖의 목표는 현재 선분 끝을 넘지 않는 최대2.0m 앞점으로 보정합니다. 선분 잔여가0.3m 미만이면
경로를 따라 다음 선분까지 총0.3m 진행한 점을 씁니다. 전체 남은 경로도0.3m 미만이고 목표 마무리 범위 밖이면
유효한 앞점이 없어 정지 입력을 요청합니다. 연결이 막힌 목표는 `unsafe_connector`로 보정하며, 보정된 앞점도 같은 선분 검사를 통과하지 못하면 `unsafe_fallback`으로 정지 입력을 요청합니다.
보정 이유별 DBG는10초마다 한 줄이며 시각 역행 시 기록을 초기화합니다.
추종 반환값을 v∈[0,0.3]m/s, w∈[-0.4,0.4]rad/s로 자른 뒤 `choose_command()` 를 한 번 호출합니다. 목표0.5m 안에서는 base가 방향을 맞추고,
yaw 오차0.15rad 안이면 정지를 요청합니다. 실제 반환 명령 둘 다0일 때만 완료합니다.
새 계획 거부 이유는 DBG로 알리고, 같은 실패가 반복되면 상세 이유를 되풀이하지 않습니다. `DBG plan_summary` 한 줄에서 모든 시도 여유와 결과, 계획별 `ms`, 묶음 `batch_ms`, 반복 횟수를 확인합니다.
`status=accepted/no_path/retained`는 새 경로 채택/경로 없음/이전 경로 유지를 뜻합니다. `no_path`이면 계획부터 고칩니다. 한 여유의 거부 줄만 보고 다른 여유가 승인됐다고 추정하지 않습니다.
그 밖의 hook 예외·비유한 반환은 해당 틱마다 DBG 한 줄로 알리고 `choose_command(0,0,...)` 결과를 사용합니다.
지도 검사에 안전한 후보가 하나도 없는 긴급 선택은 `min_gap_map`이 큰 후보를 먼저 고르고 그다음 물체 기준을 비교합니다.
안전 후보와 지도 안전 후보가 있는 기존 선택은 유지합니다. 정지 요청도 위험하면 탈출 명령이 될 수 있습니다. 검사는 충돌 회피를 보장하지 않으므로 반드시 실제 시험하십시오.
센서 관측, 후보 선택, 목표 마무리와 실패 처리는 base 책임입니다.
base 센서 API 문서의 controller() 호출 예시는 base 내부 루프에 적용합니다. 세 함수에서는 센서나 chooser를 직접 호출하지 않습니다.

아래 한 줄 `controller`는 기존 정적 검사·splice 호환용으로 그대로 둡니다. 실제 dispatcher는 이 함수를 거치지 않고
base 제어 루프를 호출하므로, 이를 바꾸어도 세 함수 호출을 우회할 수 없습니다. base 함수·상태를 재정의하거나 덮어쓰지 마십시오.

## 구현할 세 함수

`pose=(x,y,yaw)`는 월드 좌표[m,rad], target과 경로점은 월드 `(x,y)`[m]입니다. 함수 이름·인자 순서를 지킵니다.
보조 함수는 필요한 함수 몸통 안에 둡니다. 다음은 미완성 입출력 틀이므로 TODO를 채운 뒤 시험합니다.

```python
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
def plan_path(start_xy, goal_xy, clearance_m):
    # TODO: 아래 지도 도구로 A*·연결 검사·직선 단순화 후 validate_plan 승인 확인.
    # 출력: 시작/목표 각각0.5m 안에서 끝나는 점 목록. 없으면 None 또는 [].
    raise NotImplementedError

def lookahead_target(path, pose):
    # TODO: 로봇의 경로 사영점에서 경로를 따라0.3~2.0m 앞인 (x, y).
    raise NotImplementedError

def follow_cmd(pose, target_xy):
    # TODO: 아래 추종 식으로 계산한 (v, w). 물체 대응·chooser 호출 없음.
    raise NotImplementedError

def controller(t, pose, env): return _base_control_loop(t, pose, env)
# ========================= [END EDIT REGION] ===============================
```

## 경로와 지도 계약

- `plan_path(start_xy, goal_xy, clearance_m)` 는 유한한2차원 점 목록을 반환합니다. 실제 입력 양 끝점이나
  각각0.5m 이내의 격자 끝점을 쓸 수 있으며, 후자는 base가 실제 양끝에 연결하여 검사합니다.
  `clearance_m` 는 **로봇 중심의 지도 거리**입니다. 요청값을 임의로 올리거나 내려 무시하지 않습니다.
  같은 지도·시작점·목표점·여유에 같은 경로가 나오도록 격자 A*의 이웃 방문 및 동점 순서를 고정합니다.
  무작위 RRT, 난수·벽시계 의존, 손으로 고른 경유점 좌표, 계획 실패 때의 미검사 직선 대체는 금지합니다.
- base의 `occupancy_grid(clearance_m, res)`로 격자를 얻어 **A* 탐색과 직선 단순화는 직접 구현**합니다.
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
- 권장 해상도는 `res=0.5`m입니다. 처음부터0.1m 격자로 만들지 말고, 계획 함수 한 번이0.5s 안에 끝나는 것을 목표로 합니다.
  이는 성능 권고이며 강제 중단 시간이 아닙니다. `DBG plan_summary`의 `ms`(계획별 ms)와 `batch_ms`(계획 묶음 전체 ms)로 확인하고, 모델 코드에서 벽시계를 읽지 않습니다.
  `occupancy_grid`는 같은 여유·해상도 결과를 base가 캐시하고 호출자 소유 복사본을 반환합니다. 각 계획에서 한 번 얻은 격자를 모든 칸 검사에 재사용합니다.
  탐색은 정수 칸 위에서 수행하고, 이미 확정한 칸이나 오래된 heap 항목을 다시 확장하지 않습니다.
  같은 칸 쌍의 `segment_free` 결과도 계획 안에서 재사용하여 중복 기하 검사를 피합니다.
  빠르게 하려고 이웃 선분 검사를 생략하지 않습니다. 칸 중심이 비어 있어도 대각 연결은 물체를 가로지를 수 있습니다.

다음은 지도 도구 호출 예이며 탐색 코드가 아닙니다. 먼저 범위를 검사한 칸에만 배열을 접근합니다.
```python
blocked, origin, res = occupancy_grid(clearance_m, 0.5)
cell = cell_of(start_xy, origin, res)
in_bounds = 0 <= cell[0] < blocked.shape[1] and 0 <= cell[1] < blocked.shape[0]
p = xy_of(cell, origin, res)
start_connection_ok = in_bounds and not blocked[cell[1], cell[0]] and segment_free(start_xy, p, clearance_m)
```

- 반환 경로는 유한한 월드 좌표의 N×2 배열 또는 좌표쌍 목록이며 N≥2입니다. 출발점에서 목표까지 순서대로 둡니다.
  단순화는 선택 사항입니다. 탐색 원경로가 검사를 통과하면 모델이 그대로 반환해도 됩니다.
  한 점짜리 목록의 마지막 점을 계속 덮어쓰면 첫 점도 사라집니다. 첫 점을 덮어쓰지 말고 확정한 점을 추가합니다.
  단순화 전후의 점 수와 첫·끝 좌표를 비교하고, 출발과 목표가 다르면 결과 한 점은 오류로 봅니다.
  `invalid_path`이면 해상도나 follower를 바꾸기 전에 반환 형태와 점 수부터 확인합니다.
- `validate_plan`은 기존 검사에 더해 `input_shape`, `input_points`, `input_issue`, `repair_hint`를 반환합니다.
  `too_few_points`와 `preserve_start_append`는 시작점 보존과 점 추가 방식을 고치라는 뜻입니다.
  한 점도 유한 좌표쌍이면 실제 출발·목표까지 거리를 알리지만, 승인하지는 않습니다.
  None/빈 목록은 `none_return`/`empty_return`이며 `mark_grid_search_simplify_validate`는 어느 단계에서 끝났는지 표시하라는 뜻입니다.
  `repair_hint`는 진단 문구이며 경로를 자동으로 고치거나 안전 여유를 낮추지 않습니다.

`plan_stage(stage, input_points=None, output_points=None)`로 현재 계획 단계를 남깁니다.
`stage`는 `grid`, `search`, `simplify`, `validate` 중 하나이고 점 수는 비음수 정수 또는 None입니다.
각 단계에 들어갈 때, 결과가 준비됐을 때, None/[]를 반환하기 전에 해당 단계와 알 수 있는 점 수를 기록합니다.
`plan_path` 호출 중에만 사용할 수 있고 밖에서는 RuntimeError입니다. 반환값은 마지막 표시 dict의 복사본입니다.
base는 매 여유 시도마다 표시를 초기화하고 실패 DBG에 붙입니다. 표시는 모델이 남긴 정보일 뿐 실제 단계 수행이나 성공을 증명하지 않습니다.
이 도구는 탐색·직선 단순화·경로 보정을 수행하지 않습니다. 다음은 진단 호출 예이며 경로 계산 코드가 아닙니다. 각 줄을 해당 단계에 배치하며 한곳에 이어 붙이지 않습니다.
```python
plan_stage("grid")
plan_stage("search")
plan_stage("simplify", input_points=len(raw_path))
plan_stage("simplify", input_points=len(raw_path), output_points=len(path))
plan_stage("validate", input_points=len(path))
```

- 단순화에서는 기준점(anchor)에서 연결되는 마지막 유효점을 따로 기억합니다. 다음 점과 연결이 막히면 그 마지막 유효점을 추가하고
  이를 새 기준점으로 삼습니다. 이전 기준점을 중복 추가하거나 막힌 점을 새 기준점으로 바로 지정하지 않습니다.
  시작점을 보존하고 모든 연결과 양 끝점을 다시 검사합니다. 점-직선 편차만으로 중간점을 삭제하지 않으며 코너 여유를 지키고0.35m보다 충분히 긴 구간으로 묶습니다.
- 선분 지도 검사는 `check_path(points, margin_m=max(0,clearance_m-0.25))`입니다. 이 함수는 중심 거리에서 몸체 반폭0.25m를
  뺀 `min_gap_map`, 그 위치 `min_location`, 부족한 첫 구간의0부터 센 번호 `first_bad_segment`를 반환합니다.
  선분 양 끝을 포함해 최대0.1m 간격으로 검사합니다. `first_bad_segment is None`이고
  `min_gap_map >= max(0,clearance_m-0.25)`여야 하며 수치 경계에는1e-12m 오차를 허용합니다.
  중심 여유0.5/0.45/0.4/0.35m는 각각 검사 margin0.25/0.20/0.15/0.10m입니다. 낮은 요청에도 기본margin0.25를 고집하지 않습니다.

- 반환하기 전에 `validate_plan(path)`를 호출하여 `accepted`를 확인합니다. `plan_path` 안에서는 현재 호출의
  실제 출발점·GOAL·요청 중심 여유로 검사합니다. 각 끝점 오차가0.5m 이하이면 실제 양끝을 연결하고,
  추가한 연결도 포함하여 전체 `check_path`를 검사합니다. 더 먼 끝점, 빈 경로, 비유한 값이나 지도 검사 실패는 거부합니다.
  승인된 `path`를 반환합니다. 거부되면 원인을 고쳐 다시 검사하고, 찾지 못해 None 또는 []를 반환하기 전에
  `reason`, `first_bad_segment_points`, `check_path`의 최소 간격과 요구 margin(`min_gap_map`, `margin_m`)을 한 줄 DBG로 남깁니다. 240자 안에 핵심값을 넣습니다.
  `validate_plan`의 거부 정보를 버리고 None만 반환하면 base는 빈 경로와 마지막으로 남긴 단계 표시만 알 수 있습니다.
  `reason`, 끝점 거리, `check_path`의 `first_bad_segment`, `min_location`, 실제 간격과 요구 margin을 확인합니다.
  `first_bad_segment_points`는 실제 양끝 연결 후 불량 구간의 두 점입니다. 이 점과0부터 센 구간 번호를 짧은 DBG로 확인해 고칩니다.
  `min_location`은 경로 전체의 최소 간격 위치이므로 첫 불량 구간과 다를 수 있습니다.
  함수 밖에서는 최신 pose와 채택 여유(없으면0.5m), pose 전에는 MISSION의 START를 쓰며 `start_source`로 구분합니다.
  이 검사는 경로를 찾거나 제어 상태를 바꾸지 않습니다. base는 반환된 경로를 같은 기준으로 다시 검사합니다.

## 앞쪽 목표와 추종 명령

`lookahead_target(path, pose)`는 현재 pose의 경로 사영 위치에서 경로를 따라0.3~2.0m 앞인 경로 위 점을 반환합니다.
이 범위는 직선 거리가 아니라 선분을 이어 따라간 거리이며 경계값을 포함합니다. 경로에서1.0m 넘게 떨어진 점은 허용되지 않습니다.
현재 선분 끝을 넘겨 코너를 가로지르지 말고, 통과 반경0.35m 안에서 다음 선분으로 전환합니다.
처음부터 거리0.2m 초과인 첫 점을 찾으면 지나온 점을 다시 쫓을 수 있으므로 그렇게 하지 않습니다.
멀리 있는 GOAL을 바로 반환하지 않습니다. 마지막 구간에서 GOAL이 같은 앞거리0.3~2.0m 범위 안이면 사용할 수 있습니다.
pose-target 연결도 `segment_free`로 검사합니다. base는 채택한 경로 여유로 연결을 다시 검사하며, arc 범위 안이어도 코너를 가로지르는 연결은 보정합니다.

`follow_cmd(pose, target_xy)`는 다음 추종 식으로 `(v,w)`를 반환합니다.
`dist=hypot(tx-x,ty-y)`, `e=wrap(atan2(ty-y,tx-x)-yaw)`, `wrap(a)=atan2(sin(a),cos(a))`입니다.
`abs(e)>0.35`rad이면 v=0, 아니면 `v=min(0.3,0.5*dist+0.1)`m/s입니다.
항상 `w=clamp(1.5*e,-0.4,0.4)`rad/s입니다. 이 입력을 물체 거리로0으로 덮거나 별도 정지·후진·기다림을 더하지 않습니다.
빠른 전진·감속·회피·정지는 base chooser가 선택합니다. 세 함수의 전역 상태가 base 루프를 바꾸게 만들지 마십시오.

## 구현 규칙과 시험 뒤 작업

- EDIT 밖의 docstring, MISSION, START/GOAL, dispatcher와 마지막 실행 코드는 한 글자도 바꾸지 않습니다.
  참조 자료는 이 프롬프트와 base뿐입니다. 파일 입출력·스레드·벽시계·난수에 의존하지 않습니다.
  `math`, `numpy`, `heapq`, `collections`, `deque`는 base에서 제공됩니다. 필요한 계획·추종 계산만 작성합니다.
- 경로 알고리즘 전체를 base나 이 프롬프트가 제공하지 않습니다. 지도에서 계산하고, 관측 물체의 위치·레인·시각을 추측해
  경유점이나 대기 시각을 코드에 넣지 않습니다.
- 먼저 `plan_summary`의 채택 결과와 `plan_reject`의 형태·점 수·단계·수리 힌트를 확인합니다. `no_path`이면 `plan_path`의 반환 형태 또는 불량 연결을 고치고,
  경로 채택을 확인하기 전에는 정체를 lookahead/follow 문제로 추정해 그 함수를 바꾸지 않습니다.
  채택 뒤에는 `collision_partners`, `first_collision_t`, `final_pose`, 거리와 trajectory·DBG로 원인을 좁힙니다.
  chooser의 예측 최소간격은 현재의 실제 접촉 측정값이 아닙니다.
- 근거로 실패 원인을 정한 뒤 해당 함수만 고쳐 다시 시험합니다. EDIT 맨 위에 `# [시험 N] 결과/원인/변경: ...`를 남깁니다.
  관측이 부족하면 진단부터 보완하며 추측한 센서 정지 분기를 넣지 않습니다. 함수 예외는 base DBG에서 확인합니다.

## Isaac Sim 검증 도구 `run_isaac_sim`

시험 주행(`run_isaac_sim`)과 최종 채점은 같은 장면에서 돕니다.
첫 응답은 이 도구 호출이어야 합니다. 서버의 동일 장면·고정 하네스에서 주행합니다. 시험은 판정에 직접 합산되지 않으며,
같은 코드도 센서 메시지 시점과 물리 반영 차이로 실행 결과가 달라질 수 있습니다.
매 호출 최상위 JSON에 `code`와`sim_seconds`를 둘 다 넣습니다: `{"code": "...", "sim_seconds": 200}`.
`code`는 시작·끝 주석을 포함한 EDIT 블록만, 코드 펜스 없이 넣습니다. JSON 문자열의 줄바꿈·따옴표를 올바르게 escape합니다.
`sim_seconds`는200입니다. 코드 안 주석에 적는 것은 도구 인자가 아니며, 생략하면600초가 적용되므로 생략하지 않습니다.

결과에는 `static_check`, `result`(verdict,최종 pose·거리·yaw오차,접촉수·상대·최초시각,목표도달시간),
`trajectory`(최대80개 `[t,x,y,yaw]`), `harness_events`, `controller_output`, `runtime_error`가 있습니다.
추가 DBG는0.5초 구간마다 한 줄(첫 구간은10줄), chooser 자동 DBG는 별도로0.5초마다 한 줄입니다.
목표 보정 DBG는 이유별10초 간격입니다. 계획 묶음 요약은 시도 여유와 승인/거부 이유, 소요 시간을 앞부분에 보여주고 같은 실패 반복은 한 줄로 합칩니다. 도구의 한 줄 전체240자 제한으로 `details` 상세JSON이 잘릴 수 있으며,
계획 이외의 hook 예외 DBG는 해당 틱마다 별도 한 줄입니다.
출력은 처음40/마지막40줄로 줄어들 수 있으며 생략량은 `controller_output_omitted_lines`에 나옵니다.
runtime_error가 없어도 hook 실패 DBG와 정체를 확인합니다. 한 번의 실행이 벽시계1,700초를 넘으면 중단됩니다.
시험 횟수에는 한도가 없습니다. `verdict: pass`가 나오기 전에는 최종 답을 제출하지 말고 원인을 고쳐 다시 시험합니다.

## 제출

pass를 받은 시험의 `code`에 보낸 EDIT 블록을 한 글자도 바꾸지 않고 python 코드 펜스 하나로 제출합니다.
도구 호출은 JSON이지만 **최종 답은 JSON 객체/문자열이 아닙니다.** 실제 줄바꿈이 있는 코드와 두 경계 주석을 포함합니다.
base 전문이나 펜스 밖 설명을 내지 않습니다. 주석·정리도 변경이므로 바꾸려면 다시 시험해 pass를 확인합니다.

## 미션

창고 전체가 실험장입니다. 로봇은 남서쪽 모서리 `START` 자세로 스폰되고, 반대편
북동쪽 모서리의 `GOAL` 까지 약 34 m 를 주행해야 합니다. 사이에는 바닥에 놓인
정적 박스 14개가 있고, 그 위치·크기·높이는 base 의 `MISSION["boxes"]` 에
주어집니다(x, y = 중심, width = X 방향, depth = Y 방향, height = 높이, 단위 m).
창고에는 벽·기둥·선반·지게차 같은 고정 구조물도 있습니다.

목표 위치에 도달해 `GOAL` 의 yaw 로 정렬한 뒤 정지하십시오.

## 성공 조건

고정 하네스는 MISSION의 기준으로 판정합니다: 최종 위치가 GOAL에서0.75m 이내, yaw 오차0.26rad 이내, 접촉0회입니다.
base가 목표 근처 정렬·완료를 처리합니다. 시험은200초를 명시하며 최종 채점은 MISSION의 timeout(기본600초)을 따릅니다.
충돌 뒤에는 하네스가5초 후 실행을 중단합니다. 멈춰 있어도 물체가 닿으면 충돌입니다.

## 지금 할 일

첫 응답에서 바로 `run_isaac_sim` 을 호출하십시오. 시험하기 전에 최종 답을 내지 마십시오.

1. 세 TODO를 구현한 EDIT 블록으로 `{"code": "...", "sim_seconds": 200}`을 호출합니다.
2. plan_summary의 채택 여부부터 확인하고 해당 함수만 고쳐 다시 시험합니다.
3. pass를 받은 바로 그 블록을 python 코드 펜스 하나로 제출합니다.
