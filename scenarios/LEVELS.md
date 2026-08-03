# Carter 시나리오 레벨 사다리 (L0 → L3)

정본 `nova_carter_warehouse_goal.yaml` 의 `acceptance_criteria` 를 기준으로, **"로봇이 무엇을 할 줄
알아야 통과하는가"** 를 단계별로 쌓아 올린 4개 시나리오. 각 레벨은 **바로 아래 레벨의 조건을 전부
포함**하고 하나씩만 더 요구한다(단조 누적) — 그래서 어느 레벨에서 처음 깨지는지가 곧 진단 결과다.

## 사다리 요약

| 레벨 | 이름 | 세계 | 새로 요구하는 능력 | 판정 (acceptance_criteria) |
|---|---|---|---|---|
| **L0** | 직진 도달 | 빈 통로, 2.0 m | 전진 주행 | `reached_goal` (pos 1.0, yaw 미판정) |
| **L1** | 주행 + 자세 정렬 | 빈 통로, 6.0 m | 회전 제어 · 목표 자세 수렴 · 무접촉 | + yaw 0.26 활성, pos 0.75, `no_collision` |
| **L2** | 장애물 회피 | 인지 가능한 박스(1.0 m) 경로 중반 | 인지 → 우회 → 복귀 | L1과 동일 판정 (난이도는 세계가 올림) |
| **L3** | 시간 제약 회피 + 정밀 도달 | 같은 박스를 목표 직전(y=3.5)으로 | 효율적 플래닝 (짧은 복귀 구간 + 시간 예산) | + `max_time_to_goal` 45 s |

핵심: **L3에서만 "도달했는데 실패"가 가능**하다(너무 느림). L0~L2는 도달·무접촉이 곧 성공이다.

## 레벨별 성공/실패 정의

### L0 — `nova_carter_warehouse_level0.yaml`
- **성공**: 60 s 안에 (-6.0, 1.0) 의 1.0 m 반경 안에 들어옴.
- **실패**: 미도달 / 타임아웃. (충돌·자세는 채점하지 않음 — 의도적)
- **경로 안전성**: 목표점이 정본 통과 실적 주행선(x=-6.0, y=-1.0→5.0) 위라 자유공간 보장.
- **명시적 단서**: 시작 pose 가 yaw=pi(-x 향)이라 nav2 는 출발 시 제자리 회전 1회를 한다.
  L0의 "앞으로만"은 *주행 구간이 단일 직진이고 조향/우회가 불필요*하다는 뜻이며 초기 정렬
  회전은 채점 대상이 아니다. 시작 헤딩과 완전히 일치하는 목표(예: `(-8.0, -1.0, yaw=pi)`)를
  원하면 그 지점은 통과 실적이 없으므로 **자유공간 probe 먼저**.

### L1 — `nova_carter_warehouse_level1.yaml`
- **성공**: 120 s 안에 (-6.0, 5.0) 0.75 m 안 + yaw 오차 ≤ 0.26 rad + 접촉 0.
- **실패**: 미도달 / 자세 미정렬 / 접촉 발생 / 타임아웃.
- **tolerance 하한**: `position_tolerance_m` 을 0.5 이하로 내리지 말 것. 실측상 AMCL+nav2
  xy-tol 스택이 간헐적으로 0.5 를 넘는다(run5 FAIL >0.5, run6 0.414, run7 0.245) → flaky 테스트가 된다.

### L2 — `nova_carter_warehouse_level2.yaml`
- **성공**: L1 조건 전부 + 경로를 막은 박스를 **건드리지 않고** 우회.
- **실패**: 박스 접촉(회피 실패) / 우회 실패로 인한 타임아웃 / L1 실패 조건.
- **negative control 이 이미 있음**: 기존 `nova_carter_warehouse_obstacle_fail.yaml` 이 같은 자리에
  height 0.15 m 저상 박스를 두어 *nav 이 못 보고 긁는* 의도적 fail 케이스다(run7 실측
  collision_count 3603, 도달은 성공). L2는 그 박스를 **1.0 m 로 키워 "보이게" 만든 짝**이다 —
  두 파일을 같이 돌리면 "인지 실패"와 "회피 실패"가 분리 진단된다.

### L3 — `nova_carter_warehouse_level3.yaml`
- **성공**: L2 조건 전부 + `time_to_goal ≤ 45.0 s`.
- **실패**: L2 실패 조건 + **도달은 했으나 45 s 초과**.
- **45 s 근거**: 무장애물 실측 8.4~10.5 s (N=10 산포 1.017 s), 기존 커스텀 oracle 예시가 무장애물
  기준 30 s(≈3x)를 씀 → 회피·재계획·재정렬 비용을 감안해 4.3x 로 잡되 mission `timeout_s` 120
  대비 2.7x 타이트. 첫 probe 에서 실제 회피 포함 time_to_goal 을 재고 조정할 것.
- **의존**: `scenarios/max_time_to_goal.py` (이미 저장소에 존재). YAML을 같은 디렉토리에 두면 끝.

## 튜닝 절차 (probe 순서)

레벨을 올릴 때마다 **한 번에 한 개 값만** 바꾸고 재측정한다.

1. **L0 → 통과 확인.** 실패하면 시나리오가 아니라 브링업 문제다(먼저 정본 재현부터).
2. **L1 의 yaw 판정이 실제로 켜지는지 확인.** ⚠ 정본 주석상 yaw 체크는
   `goal_orientation_wxyz` 가 설정되어야 활성화된다(`scenario.goal.yaw` 만으로는 비활성).
   L1/L3 params 에 넣어둔 `[0.7071068, 0, 0, 0.7071068]` 은 yaw=1.5708 의 (w,x,y,z) 변환이다.
   criteria params 는 미지 키를 **loud-reject 하지 않고 조용히 무시**하므로(draft
   `goal_tolerance_m` 사례) 키 이름이 틀리면 L1이 "위치만 판정"으로 **조용히 퇴화**한다.
   첫 런에서 `result.json` 의 `reached_goal` detail 에 yaw 항목이 실제로 잡히는지 확인할 것.
   미지원이면 → `max_time_to_goal.py` 를 본떠 yaw 판정 커스텀 oracle 을 하나 추가하는 게 대안.
   yaw 로 FAIL 하면 tolerance 를 0.35~0.5 로 올려 재측정(0.26 은 정본에 적힌 값일 뿐 미실측).
3. **L2 장애물 3값(`height`/`width`/`x`) 수렴 루프:**
   - 로봇이 전혀 비켜가지 않음(회피 미발생) → `width` 0.8 → 1.0
   - 도달 실패/타임아웃(통로가 막힘) → `width` 0.6, `x` -6.5 로 밀어 틈 확보
   - 충돌 발생 → 2D lidar flow 를 보고 "인지 실패"인지 "회피 실패"인지 먼저 구분
   - 기준 회랑: lane 중심 x=-6.0, 섀시 주행 회랑 대략 x∈[-6.3, -5.7]
     (obstacle_fail 의 x=-6.35·width 0.5 박스가 "half-lane graze" 였다는 실측에서 역산)
4. **L3 는 L2 가 안정적으로 통과한 뒤에만.** 회피 포함 실제 time_to_goal 을 먼저 재고
   `max_time_to_goal_s` 를 그 값의 3~4x 로 재설정한다.
5. 각 레벨 확정 후 `seed` 를 바꿔 2~3회 더 돌려 flaky 여부 확인(현재 전부 `seed: 42` 고정).

## 스키마 한계 (레벨 설계 시 알고 있어야 할 것)

현재 플랫폼 계약으로 **표현 불가능**한 것들 — L4 이상을 만들려면 플랫폼 변경이 필요하다:

- **다중 웨이포인트**: `scenario.goal` 은 단일 목표. 순차 경유 미션은 불가.
- **장애물 2개 이상**: `scenario.debug_obstacle` 은 단수 매핑(x/y/height/width/depth).
- **동적 장애물 / 이동 방해물**: 월드 상태 슬롯이 정적 박스만 지원.
- **시작 pose 지정**: 스키마에 없음. 항상 AMCL 시작 pose (-6.0, -1.0, yaw=pi) 에서 출발.
- 커스텀 oracle 은 **결정적 순수 파이썬**이어야 하고 모듈 스코프에서 `omni.*`/`isaacsim.*`
  import 금지(러너가 시뮬레이터 부팅 전에 평가 엔진을 구성하므로 크래시).

## 파일 목록

```
scenarios/nova_carter_warehouse_level0.yaml   # L0 직진 도달
scenarios/nova_carter_warehouse_level1.yaml   # L1 주행 + 자세 정렬 + 무접촉
scenarios/nova_carter_warehouse_level2.yaml   # L2 장애물 회피
scenarios/nova_carter_warehouse_level3.yaml   # L3 시간 제약 회피 + 정밀 도달
scenarios/max_time_to_goal.py                 # (기존) L3 가 참조하는 커스텀 oracle
scenarios/nova_carter_warehouse_obstacle_fail.yaml  # (기존) L2 의 negative control
```

모든 파일은 정본의 `interface.adapter_config` 실측 fill 을 **그대로** 승계한다(토픽/타입/프레임/
readiness 무변경). 레벨 간 차이는 `scenario.goal` · `scenario.debug_obstacle` · `timeout_s` ·
`acceptance_criteria` 에만 있다.
