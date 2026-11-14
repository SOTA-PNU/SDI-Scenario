# Level 3 실측 리포트 — 시간 제약 회피 + 정밀 도달

2026-08-03, Isaac Sim 4.5.0 / A100 80GB (GPU0) / seed 42 / dt=1/60.
판정 원본: `results/base_result.json`, `results/solution_result.json`
(바운드 재조정 측정 기록: `results/measure_bound45_{base,solution}_*.{json,csv}`).

## 1. 미션과 판정

`scenarios/nova_carter_warehouse_level3.yaml`:

| 항목 | 값 |
|---|---|
| 스폰 / goal | L2 와 동일: `(-6.0,-1.0,π)` → `(-6.0, 5.0, +1.5708)` |
| 세계 상태 | 같은 1.0 m 박스를 **goal 직전 y=3.5** 로 이동 (goal 까지 1.5 m — 우회 직후 곧바로 자세 수렴) |
| **신규 판정 축** | `max_time_to_goal`: goal 디스크 **최초 진입 ≤ 12.0 s** (재조정 — 아래 §3) |
| 기타 판정 | L2 와 동일 (pos 0.75 / yaw 0.26 / 무접촉 / timeout 120 s) |
| L2 대비 신규 요구 | **효율** — 같은 일을 시간 예산 안에. 유일하게 "도달했지만 실패"가 가능한 레벨 |

## 2. base 코드와 실행 결과 — 왜 실패하는가

base 의 `[EDIT REGION]` == **레벨 2 의 solution** (사다리 규칙): bearing-팽창 회피를
갖춘 안전-저속 컨트롤러 (CRUISE 0.5 / W_MAX 1.2 / AVOID 0.25).

**실측 (base, bound 12 s): `verdict: fail` — 오직 시간 축만 실패**

| 지표 | 값 |
|---|---|
| oracle 상세 | reached_goal **✓** (0.2746 m / yaw 0.0688) · no_collision **✓** (0건) · max_time_to_goal **✗** (`too_slow`) |
| time_to_goal | **15.433 s** > bound 12.0 (바운드 45 측정 런과 소수 3자리까지 동일 — 결정성) |
| 궤적 | 박스(y=3.5)를 동쪽으로 무접촉 우회 → 남동쪽에서 디스크 진입 → 정렬까지 완료 |
| 최종 pose | (-5.8654, 4.7607, 1.6396) |
| sim / wall / frames | 18.633 s / 62.3 s / 1118 |

이 실패는 사다리에서 유일하게 **행동 결함이 아니라 효율 결함**입니다: 도달·무접촉·정렬
전부 성공하고도 늦어서 집니다. trajectory.png 에서 base 와 solution 의 경로가 공간상
거의 겹칩니다 — 차이는 오직 시간입니다.

## 3. 풀이 과정 — 시간 상한 재조정 (45 s → 12 s)

초판(a07c868)의 45 s 는 nav2 스택 가정의 미실측 추정치였습니다. LEVELS.md 튜닝 절차
("L3 는 실제 time_to_goal 을 재고 재설정")대로 진행:

1. **L3 base 를 bound 45 로 측정** → `verdict: pass`, **ttg 15.433 s** (col 0,
   final_d 0.2746, yaw_err 0.0688 — `measure_bound45_base_result.json`).
   45 s 바운드는 전혀 안 물립니다 → 이대로면 base 가 이미 통과하는 무의미한 레벨.
2. **속도 튜닝 solution 을 bound 45 로 측정** → `verdict: pass`, **ttg 8.300 s**
   (col 0, final_d 0.2762, yaw_err 0.0690 — `measure_bound45_solution_result.json`).
   같은 회피 알고리즘이 속도 상수만으로 미션 시간을 거의 반감.
3. **바운드 = 두 실측의 중간 12.0 s** 로 확정 (base 대비 −3.4 s, solution 대비 +3.7 s
   의 대칭 마진; seed 고정 결정적 하네스라 이 마진이면 flaky 하지 않음 — L0 재실행
   실측에서 소수 4자리 재현 확인). YAML·LEVELS.md·runner given(MAX_TTG) 동기화 후
   base/solution 을 **최종 재실측** (§2, §5 의 기록이 그 결과입니다).

## 4. 수정 내용 (base → solution, 정답 키)

`[EDIT REGION]` 블록만 수정 (`check_edit_region.py` 검증 통과). **속도 상수 3개가
정답의 전부입니다** — 알고리즘(bearing-팽창 회피 + 3상 점-컨트롤러)은 그대로:

```diff
--- levels/level3/base_carter_run.py
+++ levels/level3/solution_carter_run.py
@@ (EDIT REGION 내부만)
-CRUISE_V = 0.5     # m/s cruise speed toward the goal
+CRUISE_V = 1.0     # m/s cruise: raised so the mission fits the time budget
 STOP_DIST = 0.30   # m, well inside the 0.75 m position tolerance
 K_HEADING = 1.8    # P gain: bearing error -> yaw rate
-W_MAX = 1.2        # rad/s yaw rate limit
+W_MAX = 1.5        # rad/s yaw rate limit: snappier turns save seconds
@@
-AVOID_V = 0.25      # m/s while maneuvering around an obstacle
+AVOID_V = 0.5       # m/s while maneuvering: budget-conscious but careful
```

핵심 아이디어: L3 가 요구하는 능력은 새 알고리즘이 아니라 **시간 예산 인식**입니다.
안전 마진을 깎지 않는 선(SIDE_MARGIN·THREAT_DIST·감속 로직 불변)에서 순항·회전·우회
속도만 예산에 맞게 올립니다. 셋 중 하나라도 안 올리면 12 s 를 못 맞춥니다(순항 구간,
초기 회전 π/2, 우회 구간이 각각 병목).

## 5. solution 실행 결과 — 어떻게 성공했는가

**실측 (solution, bound 12 s): `verdict: pass` — 3개 oracle 전부 통과**

| 지표 | 값 |
|---|---|
| time_to_goal (디스크 최초 진입) | **8.283 s** (bound 12.0 — 여유 3.7 s; 측정 런 8.300 s 와 1 프레임 차) |
| min / final 거리 | 0.2794 / **0.2798 m** (tol 0.75) |
| yaw 오차 (최종) | **0.0711 rad** (tol 0.26) |
| 최종 pose | (-5.8174, 4.788, 1.6419) |
| 충돌 | **0** |
| sim / wall / frames | 11.717 s / 38.3 s / 703 |

전개: base 와 같은 모양의 우회(동쪽, 측면 여유 유지)를 **거의 두 배 속도로** 수행 —
순항 1.0 m/s, 초기 π/2 회전 1.5 rad/s, 우회 0.5 m/s. 안전 지표는 그대로입니다
(충돌 0, 최종 오차도 base 와 사실상 동일). 시간만 15.433 → 8.283 s 로 단축.

## 6. 초판(a07c868) 대비 시나리오/저장소 변경 내역

| 파일 | 변경 | 사유 |
|---|---|---|
| `scenarios/nova_carter_warehouse_level3.yaml` | **`max_time_to_goal_s: 45.0 → 12.0`** + 근거 주석 블록 재작성 + EXPECTED 갱신 (goal/obstacle/tolerance/timeout 은 초판 그대로) | 45 s 는 실측상 안 물림(base 15.4 s 통과). §3 절차로 재조정 — cv-infra+nav2 용도로는 미실측이라는 caveat 명기 |
| `scenarios/LEVELS.md` | L3 표·섹션의 45 s → 12 s + 실측 근거 | 위와 동일 |
| `levels/level3/` (신규) | base/solution/REPORT/results | 이 리포트 |

## 7. 재현

```bash
bash levels/run_isaac.sh levels/level3/base_carter_run.py       # FAIL 재현 (시간 초과만)
bash levels/run_isaac.sh levels/level3/solution_carter_run.py   # PASS 재현
python3 levels/common/check_edit_region.py levels/level3        # 불변식 검증
```
