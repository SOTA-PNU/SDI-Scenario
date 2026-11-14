# 실험 49 B-5-23 a2 분석

**직접 원인은 A* 탐색 실패가 아니라 반환 경로의 끝점 계약 위반이다.** 두 회차 모두 실제 출발점과 GOAL 대신 격자점을 반환해 base가 경로를 채택하지 못했다.
a2 완료 자료만 분석했다. 진행 중인 a3, 공용 실행 로그와 프로세스는 조회하지 않았고 하네스, base, 프롬프트 및 모델 코드는 수정하지 않았다.

| 실행 | 시험 수/성격 | 결과 | sim 종료 s | 최종 거리 m | 접촉 |
|---|---|---|---:|---:|---:|
| 1회차 제출 | 시험 없이 직접 제출 | 시간 초과 FAIL | 601.0 | 31.6777 | 0 |
| 2회차 test01 | 유일한 시험 | 시간 초과 FAIL | 601.0 | 31.6702 | 0 |
| 2회차 제출 | 실패 시험 뒤 제출 | 시간 초과 FAIL | 601.0 | 31.6702 | 0 |
시험 호출의 `sim_seconds`가 누락돼 200 s 지시 대신 기본 600 s가 적용됐다. 실제 TIMEOUT은 600 s, 결과 종료 시각은 601 s다.
2회차 시험과 제출은 공백과 docstring을 제외한 실행 AST가 같고 trajectory CSV도 바이트 단위로 같다. [실행 원자료 대조](b523_a2_run_evidence.json)

**DBG와 원함수 재현을 구분했다.** 각 실행의 DBG 1,197줄은 모두 choose이며 planner 실패, 빈 경로, check_path 거부, hook 예외, target 보정 진단은 없다.
첫 줄은 `DBG choose t=0.00 pick=follow why=follow_safe f=- s=- ... map=0.72`다. 이 `follow`는 정지 입력도 포함하므로 주행했다는 뜻이 아니다.
1회차는 follow 1,192/turn_r 4/back 1, 2회차는 follow 1,191/turn_r 4/back 2로 간헐적 회피는 있었다. 출발점 기준 최대 변위는 각각 0.518/0.511 m로 완전 무이동은 아니다.
controller runtime_error는 null이며 hook 예외 DBG도 없다. 시작 시 `isaacsim.asset.browser`의 requests import 오류는 600 s 임무를 중단한 controller 예외와 구분한다.
로그만으로 숨은 거부 사유를 단정하지 않고, 저장된 원 `plan_path`의 AST와 지도로 두 회차 × 4개 clearance를 그대로 재현했다. [재현 코드](../../analysis/b523_a2_planner_probe.py), [7/7 확인과 수치](b523_a2_planner_evidence.json)
각각 0.5/0.45/0.4/0.35 m에서 A*는 8/8 목표 칸에 도달하고 비어 있지 않은 경로를 반환했다. 별도로 호출한 `check_path`도 8/8 통과했다.
저장 첫 pose `(-8.9998,-10.7997)`은 칸 `(7,7)`의 `(-8.9,-10.75)`로 바뀌어 약 0.11149 m 차이가 난다. GOAL `(8,16.5)`은 칸 `(92,143)`의 `(8.1,16.45)`로 바뀌어 0.11180 m 차이다.
CSV pose는 소수 넷째 자리까지만 저장되지만, 고정 GOAL 오차만으로도 base 허용치 `1e-6 m` 초과가 독립적으로 확정된다.
격자는 0.2 m이며 `round((xy-origin)/step)`와 `origin+index*step`은 서로 일관된다. 출발/목표 칸 지도 여유는 1.40/1.25 m로 최대 요청 0.5 m보다 크고, 이중 팽창이나 막힌 양 끝 칸이 원인이 아니다.
8방향 이웃과 목표 칸을 꺼낼 때 끝내는 종료 조건도 정상 동작했다. 이웃 선분 전체를 검사하지 않는 일반적 위험은 있지만 이번 8개 경로 거부의 원인은 아니다.
1회차의 “ensure start and goal”도 실제 인자 대신 `raw_path`의 격자 끝점을 다시 넣는다. 2회차는 단순화만 제거해 점 수가 6/6/5/4에서 141/139/138/138로 바뀌었고 끝점 오류는 남았다.
base는 끝점 오차에서 먼저 `None`을 반환하므로 실제 validator 내부 `check_path` 호출은 0회였다. 공개 지도 검사 통과와 base 경로 채택을 혼동하면 원인을 찾지 못한다.
4개 경로가 모두 거부되면 `path=None`으로 target/follow 함수에 도달하지 않고 nominal `(0,0)`을 chooser에 준다. `follow_safe`는 `chosen=='stop'`이 아니므로 “지도 정지 10 s” 재계획 조건도 성립하지 않는다.
빈 관측과 고정 첫 pose를 쓴 분리 시험에서 계획 4회, target/follow 0회, t=0/1/10/600의 정지 입력과 재계획 없음이 확인됐다. 이는 실제 센서나 600 s 물리를 재생한 시험은 아니다.

핵심 원문 10줄: [2회차 실행 snapshot](../turns/prompt_b5_23/isaac_logs/gpu_gpt_oss_120b_pB523_a2_t2_carter_run.py)의 planner와 같은 파일의 base 검사다.
```text
1120: step = 0.2  # metres – coarse enough for speed, fine enough for navigation
1125: return (int(round((pt[0] - wx0) / step)),
1126: int(round((pt[1] - wy0) / step)))
1130: return (wx0 + idx[0] * step,
1131: wy0 + idx[1] * step)
1201: path.append(to_world(cur))
1203: path.reverse()
1204: return path
885: if math.dist(path[0], start_xy) > 1e-6 or math.dist(path[-1], goal_xy) > 1e-6:
886: return None
```

| 이전 a2 | 확인된 결함과 현재 사례의 차이 |
|---|---|
| [B516](b516_a2_analysis.md) | A* 단순화에서 기준점(anchor) 소실, 이후 수동 경유점과 센서 정지 분기. 현재는 단순화 제거 뒤에도 실제 끝점 누락이 남았다. |
| [B518](b518_a2_analysis.md) | 시드 없는 RRT의 경로 변동과 지도 정지 뒤 재계획 부재. 성공 시험 뒤 같은 코드의 제출은 정체했다. 현재 A*는 결정론적이며 탐색 자체는 성공했다. |
| [B519](b519_a2_analysis.md) | 막힌 재계획 시작 칸에서 None 이후 미검사 직선 경로 사용. 2회차 시험들은 M1 접촉, 최종 제출은 JSON 형식 실격이었다. 현재 시작 칸은 열려 있고 base가 잘못된 끝점을 거부했다. |
공통점은 유효 경로의 검증과 전달, 실패 진단이 어긋났다는 점이다. 같은 a2라는 이유로 동일한 격자 결함이나 시드의 인과 효과라고 묶지 않는다.
시드 확인: 네 실험의 실제 첫 요청은 모두 **1201**이다(`seed_base 1000 + sample 2×100 + call 1`); 1002가 아니다. 후속 요청 번호와 대화는 다르며 mover v1 및 simulator seed 42와도 구분한다. [비교 근거](b523_a2_history_comparison.json)

**개선안과 기대 범위 — 하네스 불변, A*와 경로 선택은 모델 소유를 유지한다.**
- base 우선안: 실제 채택 조건과 같은 `validate_plan(path,start,goal,clearance)`를 공개해 점 수, 양 끝 오차, 거부 이유와 지도 검사 결과를 반환하고 계획 시도마다 DBG로 남긴다. 이번 silent endpoint 거부를 단순화 실패로 오인하는 여지를 줄인다.
- base 보조안: 중심 clearance 기준 점유 격자, origin/resolution/축 순서, world↔cell 변환과 실제 끝점↔격자점 연결 여유만 제공한다. 탐색, 경로 재구성, 단순화는 모델에 남긴다. 현재 변환은 이미 일관되므로 **격자 도구만으로 이번 결함을 제거할 수는 없다.**
- 프롬프트: 실제 `start_xy/goal_xy`와 snapped cell을 구분하고 양 끝 연결 및 재구성 뒤 끝점 오차 검사를 요구한다. 실패 이유를 고친 뒤 시험하며 최상위 `sim_seconds=200`과 PASS 코드의 동일 제출을 다시 점검하게 한다.
- 평가 방법: 비격자 정렬 끝점, 막힌 최근접 칸, 대각 연결과 변환 왕복의 계약 시험으로 도구가 줄이는 오류 범위만 검증한다. 이번에는 개선 API를 구현하거나 a2 planner를 고쳐 재주행하지 않았으므로 성공률 개선을 주장하지 않는다.
