# carter-scenario-levels

Nova Carter **난이도 레벨 사다리 시나리오 모음**. [`yongjunshin/cv-infra-user`](https://github.com/yongjunshin/cv-infra-user)의
정본 시나리오(`scenarios/nova_carter_warehouse_goal.yaml`)를 기준으로, **"로봇이 무엇을 할 줄 알아야
통과하는가"** 를 L0 → L3 4단계로 쌓아 올린 CV-Infra 표준 입력 인스턴스다.

> **현 상태**: 4개 레벨 YAML + 설계/튜닝 문서가 채워져 있다. 값은 cv-infra-user 정본의 **실측 fill을
> 승계**했고(토픽·타입·프레임·readiness 무변경), 레벨별로 새로 도입한 값(`debug_obstacle` 3값,
> `max_time_to_goal_s`, yaw 활성화)은 **아직 GPU 실측 전**이다 — `scenarios/LEVELS.md`의 "튜닝 절차"
> 순서대로 probe 하며 수렴시킨다. 스키마는 플랫폼 소유이며 본 저장소는 **인스턴스화만** 한다.

## 저장소 레이아웃

- `scenarios/` — 표준 입력 인스턴스(`*.yaml`) + 커스텀 oracle 플러그인(`*.py`).
- `scenarios/LEVELS.md` — 사다리 설계 근거 · 레벨별 성공/실패 정의 · **probe 튜닝 절차** · 스키마 한계.

## 레벨 사다리

각 레벨은 **바로 아래 레벨의 조건을 전부 포함**하고 하나씩만 더 요구한다(단조 누적). 그래서
"어느 레벨에서 처음 깨지는가"가 곧 진단 결과다.

| 레벨 | 파일 | 세계 | 새로 요구하는 능력 | 판정 |
|---|---|---|---|---|
| **L0** 직진 도달 | [`level0`](scenarios/nova_carter_warehouse_level0.yaml) | 빈 통로 2.0 m | 전진 주행 | `reached_goal` (pos 1.0, yaw 미판정) |
| **L1** 주행 + 자세 정렬 | [`level1`](scenarios/nova_carter_warehouse_level1.yaml) | 빈 통로 6.0 m | 회전 제어 · 도착 자세 수렴 · 무접촉 | + yaw 0.26 활성, pos 0.75, `no_collision` |
| **L2** 장애물 회피 | [`level2`](scenarios/nova_carter_warehouse_level2.yaml) | 인지 가능한 1.0 m 박스, 경로 중반(y=2.0) | 인지 → 우회 → 복귀 | L1과 동일 (난이도는 세계가 올림) |
| **L3** 시간 제약 회피 | [`level3`](scenarios/nova_carter_warehouse_level3.yaml) | 같은 박스를 목표 직전(y=3.5)으로 | 효율적 플래닝 (짧은 복귀 구간 + 시간 예산) | + `max_time_to_goal` 45 s |

**L3에서만 "도달했는데 실패"(너무 느림)가 가능**하다. L0~L2는 도달·무접촉이 곧 성공이다.

레벨 간 차이는 `scenario.goal` · `scenario.debug_obstacle` · `scenario.timeout_s` ·
`acceptance_criteria` **네 곳에만** 있다. `interface.adapter_config`는 정본의 실측 fill과 완전히 동일하다.

## 값의 출처

레벨 설계에 쓴 수치는 전부 cv-infra-user 정본/변형 파일에 기록된 실측치에 앵커했다.

- 주행선 `x=-6.0`, `y=-1.0 → 5.0` 은 통과 실적 경로(cycle-5 run2/run6, cycle-6 N=10) → L0의 `y=1.0`은
  그 위의 점이라 자유공간이 보장된다.
- `position_tolerance_m` 하한 **0.75** — 0.5 이하는 AMCL+nav2 xy-tol 스택이 간헐 초과해 flaky
  (run5 FAIL >0.5, run6 0.414, run7 0.245).
- 무장애물 `time_to_goal` 8.4~10.5 s (N=10 산포 1.017 s) → L3 시간 상한 45 s (≈4.3x, mission
  `timeout_s` 120 대비 2.7x 타이트).
- L2/L3 장애물 높이 **1.0 m** — 정본 변형 `nova_carter_warehouse_obstacle_fail.yaml`의 0.15 m 저상 박스가
  *nav이 못 보고 긁는* 의도적 fail(run7 collision_count 3603)이었던 것의 **반대 짝**이다. 두 파일을 같이
  돌리면 "인지 실패"와 "회피 실패"가 분리 진단된다.

## 실행 전 확인할 것 (1건)

⚠ yaw 판정은 `goal_orientation_wxyz` 가 설정되어야 활성화된다(`scenario.goal.yaw` 만으로는 비활성).
L1/L3 params에 yaw=1.5708의 쿼터니언 `[0.7071068, 0, 0, 0.7071068]`을 넣어뒀는데, criteria params는
미지 키를 **loud-reject 하지 않고 조용히 무시**한다(정본의 draft `goal_tolerance_m` 사례). 키 이름이
틀리면 L1이 "위치만 판정"으로 **조용히 퇴화**하므로, 첫 런에서 `result.json`의 `reached_goal` detail에
yaw 항목이 실제로 잡히는지 확인할 것. 미지원이면 `scenarios/max_time_to_goal.py`를 본뜬 yaw 판정
커스텀 oracle이 대안이다.

## 커스텀 oracle

`scenarios/max_time_to_goal.py`는 cv-infra-user의 플러그인 예시를 **그대로 가져온 사본**이다(L3의
`oracle: "max_time_to_goal:MaxTimeToGoalOracle"` 참조가 이 저장소만으로 동작하려면 시나리오 YAML과
같은 디렉토리에 있어야 한다). 원본·최신본은 항상
[cv-infra-user/scenarios/max_time_to_goal.py](https://github.com/yongjunshin/cv-infra-user/blob/main/scenarios/max_time_to_goal.py)를 따른다.

작성 규칙 요약(원본 README 기준): `cv_infra.oracles.base.OracleBase` 서브클래스, **결정적 순수 파이썬**
(시계·랜덤·네트워크 금지), 모듈 스코프 import는 stdlib + `cv_infra.*` 만, 파일명은 러너 이미지의 설치
패키지명과 **비충돌**하는 고유한 이름, ⚠ 모듈 스코프에서 `omni.*`/`isaacsim.*` **import 금지**
(러너가 시뮬레이터 부팅 전에 평가 엔진을 구성하므로 크래시).

## 스키마 한계 (L4 이상은 플랫폼 변경 필요)

현재 계약으로는 표현할 수 없다 — `scenario.goal` 단일(다중 웨이포인트 불가) · `scenario.debug_obstacle`
단수(장애물 2개 이상 불가) · 동적/이동 장애물 불가 · 시작 pose 지정 불가(항상 AMCL 시작
pose `(-6.0, -1.0, yaw=pi)`).
