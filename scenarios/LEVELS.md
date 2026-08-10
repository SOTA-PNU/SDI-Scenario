# Carter 시나리오 레벨 사다리 (L0 → L3)

정본 `nova_carter_warehouse_goal.yaml` 의 `acceptance_criteria` 를 기준으로, **"로봇이 무엇을 할 줄
알아야 통과하는가"** 를 단계별로 쌓아 올린 4개 시나리오. 각 레벨은 **바로 아래 레벨의 조건을 전부
포함**하고 하나씩만 더 요구합니다(단조 누적) — 그래서 어느 레벨에서 처음 깨지는지가 곧 진단 결과입니다.

## 사다리 요약

| 레벨 | 이름 | 세계 | 새로 요구하는 능력 | 판정 (acceptance_criteria) |
|---|---|---|---|---|
| **L0** | 직진 도달 | 빈 통로, 전방 2.0 m | 전진 주행 (회전 불필요) | `reached_goal` (pos 1.0, yaw 미판정) |
| **L1** | 주행 + 자세 정렬 | 빈 통로, 6.0 m | 회전 제어 · 목표 자세 수렴 · 무접촉 | + yaw 0.26 활성, pos 0.75, `no_collision` |
| **L2** | 장애물 회피 | 인지 가능한 박스(1.0 m) 경로 중반 | 인지 → 우회 → 복귀 | L1과 동일 판정 (난이도는 세계가 올림) |
| **L3** | 시간 제약 회피 + 정밀 도달 | 같은 박스를 목표 직전(y=3.5)으로 | 효율적 플래닝 (짧은 복귀 구간 + 시간 예산) | + `max_time_to_goal` 12 s (실측 재조정, 초판 45 s) |

핵심: **L3에서만 "도달했는데 실패"가 가능**합니다(원인: 속도). L0~L2는 도달·무접촉을 성공기준으로 정했습니다.

## 레벨별 성공/실패 정의

### L0 — `nova_carter_warehouse_level0.yaml`
- **성공**: 60 s 안에 (-8.0, -1.0) 의 1.0 m 반경 안에 들어옴.
- **실패**: 미도달 / 타임아웃. (충돌·자세는 채점하지 않음 — 의도적)
- **경로 안전성**: goal 은 스폰 헤딩(-x) **정면 2.0 m**. 2026-08-03 워크스테이션 실측
  (`levels/common/probe_scene.py`, PhysX overlap): 전방 2.0 m 까지 FREE, 2.5 m 부터
  팔레트(SM_PaletteA_358/359)가 통로를 막음 → tolerance 1.0 이면 안전 여유 충분.
- **초판(a07c868)과의 차이**: goal (-6.0, 1.0, yaw=1.5708) → (-8.0, -1.0, yaw=pi).
  초판은 전방 자유공간이 미확인이라 "+y lane 위의 통과 실적 점"으로 절충하고 출발 시
  제자리 회전 1회를 허용했습니다. probe 로 전방 자유공간이 확정되어 레벨 원 정의
  ("**앞으로만** 가서 **바로 앞** 목표 도달, 회전 불필요")를 문자 그대로 복원했습니다.

### L1 — `nova_carter_warehouse_level1.yaml`
- **성공**: 120 s 안에 (-6.0, 5.0) 0.75 m 안 + yaw 오차 ≤ 0.26 rad + 접촉 0.
- **실패**: 미도달 / 자세 미정렬 / 접촉 발생 / 타임아웃.
- **tolerance 하한**: `position_tolerance_m` 을 0.5 이하로 내리지 마십시오. 실측상 AMCL+nav2
  xy-tol 스택이 간헐적으로 0.5 를 넘습니다(run5 FAIL >0.5, run6 0.414, run7 0.245) → flaky 테스트가 됩니다.

### L2 — `nova_carter_warehouse_level2.yaml`
- **성공**: L1 조건 전부 + 경로를 막은 박스를 **건드리지 않고** 우회.
- **실패**: 박스 접촉(회피 실패) / 우회 실패로 인한 타임아웃 / L1 실패 조건.
- **negative control 이 이미 있음**: 기존 `nova_carter_warehouse_obstacle_fail.yaml` 이 같은 자리에
  height 0.15 m 저상 박스를 두어 *nav 이 못 보고 긁는* 의도적 fail 케이스입니다(run7 실측
  collision_count 3603, 도달은 성공). L2는 그 박스를 **1.0 m 로 키워 "보이게" 만든 짝**입니다 —
  두 파일을 같이 돌리면 "인지 실패"와 "회피 실패"가 분리 진단됩니다.

### L3 — `nova_carter_warehouse_level3.yaml`
- **성공**: L2 조건 전부 + `time_to_goal ≤ 12.0 s`.
- **실패**: L2 실패 조건 + **도달은 했으나 12 s 초과**.
- **12 s 근거 (2026-08-03 실측 재조정 — 초판 45 s)**: 초판 45 s 는 nav2 스택 가정의 미실측
  추정치였고, `levels/` 스탠드얼론 실측에서 L2 회피 컨트롤러(=L3 base)가 ttg 15.433 s 로
  여유 통과해 시간축이 안 물렸습니다. 속도 튜닝 정답(CRUISE 1.0 / W_MAX 1.5 / AVOID 0.5)의
  실측 ttg 8.3 s 와의 중간값 12.0 으로 설정(마진 −3.4 s / +3.7 s, seed 고정 결정적 환경).
  ⚠ cv-infra + nav2 SUT 용도로는 미실측 — 그 용도면 먼저 재고 재조정하십시오.
- **의존**: `scenarios/max_time_to_goal.py` (이미 저장소에 존재). YAML을 같은 디렉토리에 두면 끝.

## 튜닝 절차 (probe 순서)

레벨을 올릴 때마다 **한 번에 한 개 값만** 바꾸고 재측정합니다.

> 참고: 아래 2번(yaw 판정 활성화 caveat)은 **cv-infra 러너로 돌릴 때**의 주의사항입니다.
> `levels/` 의 스탠드얼론 하네스는 yaw 체크를 `yaw_tolerance_rad` 만으로 직접 구현하므로
> 해당 caveat 이 적용되지 않습니다 (L1/L3 실측에서 yaw 판정 활성 확인됨).

1. **L0 → 통과 확인.** 실패하면 시나리오가 아니라 브링업 문제입니다(먼저 정본 재현부터).
2. **L1 의 yaw 판정이 실제로 켜지는지 확인.** ⚠ 정본 주석상 yaw 체크는
   `goal_orientation_wxyz` 가 설정되어야 활성화됩니다(`scenario.goal.yaw` 만으로는 비활성).
   L1/L3 params 에 넣어둔 `[0.7071068, 0, 0, 0.7071068]` 은 yaw=1.5708 의 (w,x,y,z) 변환입니다.
   criteria params 는 미지 키를 **loud-reject 하지 않고 조용히 무시**하므로(draft
   `goal_tolerance_m` 사례) 키 이름이 틀리면 L1이 "위치만 판정"으로 **조용히 퇴화**합니다.
   첫 런에서 `result.json` 의 `reached_goal` detail 에 yaw 항목이 실제로 잡히는지 확인하십시오.
   미지원이면 → `max_time_to_goal.py` 를 본떠 yaw 판정 커스텀 oracle 을 하나 추가하는 게 대안.
   yaw 로 FAIL 하면 tolerance 를 0.35~0.5 로 올려 재측정(0.26 은 정본에 적힌 값일 뿐 미실측).
3. **L2 장애물 3값(`height`/`width`/`x`) 수렴 루프:**
   - 로봇이 전혀 비켜가지 않음(회피 미발생) → `width` 0.8 → 1.0
   - 도달 실패/타임아웃(통로가 막힘) → `width` 0.6, `x` -6.5 로 밀어 틈 확보
   - 충돌 발생 → 2D lidar flow 를 보고 "인지 실패"인지 "회피 실패"인지 먼저 구분
   - 기준 회랑: lane 중심 x=-6.0, 섀시 주행 회랑 대략 x∈[-6.3, -5.7]
     (obstacle_fail 의 x=-6.35·width 0.5 박스가 "half-lane graze" 였다는 실측에서 역산)
4. **L3 는 L2 가 안정적으로 통과한 뒤에만.** 회피 포함 실제 time_to_goal 을 먼저 재고
   `max_time_to_goal_s` 를 그 값의 3~4x 로 재설정합니다.
5. 각 레벨 확정 후 `seed` 를 바꿔 2~3회 더 돌려 flaky 여부 확인(현재 전부 `seed: 42` 고정).

## 스키마 한계 (레벨 설계 시 알고 있어야 할 것)

현재 플랫폼 계약으로 **표현 불가능**한 것들 — L4 이상을 만들려면 플랫폼 변경이 필요합니다:

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
levels/                                       # 각 레벨의 base/정답 코드 + 실측 리포트
```

모든 파일은 정본의 `interface.adapter_config` 실측 fill 을 **그대로** 승계합니다(토픽/타입/프레임/
readiness 무변경). 레벨 간 차이는 `scenario.goal` · `scenario.debug_obstacle` · `timeout_s` ·
`acceptance_criteria` 에만 있습니다.

## 실측 구현 (levels/)

이 YAML들은 2026-08-03 워크스테이션(Isaac Sim 4.5, A100)에서 **cv_infra 없이 직접 실행·검증**
되었습니다. `levels/levelN/` 마다 base 코드(레벨 시작점)와 solution 코드(목표 성공 정답)가 있고,
둘은 `[EDIT REGION]` 블록 외에는 문자 단위로 동일합니다 — base→solution diff 가 곧 그 레벨의
정답 키입니다(NPU 로컬 LLM 벤치마크 픽스처). 실행 결과·과정·시나리오 YAML 변경 내역은
`levels/levelN/REPORT.md` 와 `levels/README.md` 에 있습니다.
