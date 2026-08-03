# Level 1 실측 리포트 — 전 구간 주행 + 도착 자세 정렬

2026-08-03, Isaac Sim 4.5.0 / A100 80GB (GPU0) / seed 42 / dt=1/60.
판정 원본: `results/base_result.json`, `results/solution_result.json`.

## 1. 미션과 판정

`scenarios/nova_carter_warehouse_level1.yaml`:

| 항목 | 값 |
|---|---|
| 스폰 | 월드 `(-6.0, -1.0, yaw=π)` |
| goal | `(-6.0, 5.0, yaw=+1.5708)` — +y lane 6.0 m (전 구간 FREE, 프로브 실측) |
| 판정 | `reached_goal`: 최종 거리 ≤ **0.75 m** AND yaw 오차 ≤ **0.26 rad** + `no_collision`: 섀시 접촉 0 |
| timeout | 120 s (sim) |
| L0 대비 신규 요구 | **회전 제어** · **도착 자세 수렴** · **무접촉** |

## 2. base 코드와 실행 결과 — 왜 실패하는가

base 의 `[EDIT REGION]` == **레벨 0 의 solution** (사다리 규칙: L(n) base = L(n-1) solution).
전진만 할 줄 아는 로봇에게 "뒤쪽" 목표를 준 상황이다.

**실측 (base): `verdict: fail` — 두 oracle 모두 실패**

| 지표 | 값 |
|---|---|
| 궤적 | 스폰 헤딩(-x)으로 직진 → goal 반대 방향 |
| min / final 거리 | 6.0 / **6.5635 m** (goal 에 접근한 적 없음) |
| yaw 오차 (최종) | 1.5778 rad |
| 충돌 | **360건**, first t=6.917 s, partner=`.../BoxSetA/SM_CardBoxA_05` |
| 종료 | 충돌 5 s 후 조기 종료 (wall 절약; 판정 무영향 — 두 oracle 이미 확정 실패) |

전개: 전방 2.0 m 자유공간을 지나 2.5 m 의 팔레트 지대를 (바퀴 높이로) 밀고 들어가다
x≈-8.7 에서 카드박스 더미에 섀시가 정면 접촉, 그대로 정지. **이 런이 충돌 oracle 의
양성 대조이기도 하다** — 섀시 contact report 가 실제 충돌에서 발화함을 확인.

## 3. 수정 내용 (base → solution, 정답 키)

`[EDIT REGION]` 블록만 수정 (`check_edit_region.py` 검증 통과).

```diff
--- levels/level1/base_carter_run.py
+++ levels/level1/solution_carter_run.py
@@ -47,21 +47,49 @@
 # [EDIT REGION] mission controller - modify ONLY this block
 # ===========================================================================
 
-CRUISE_V = 0.4   # m/s, forward only (L0 rule: w stays 0)
-STOP_DIST = 0.45  # m, stop well inside the 1.0 m position tolerance
+CRUISE_V = 0.5     # m/s cruise speed toward the goal
+STOP_DIST = 0.30   # m, well inside the 0.75 m position tolerance
+K_HEADING = 1.8    # P gain: bearing error -> yaw rate
+W_MAX = 1.2        # rad/s yaw rate limit
+BEARING_GATE = 0.5  # rad: rotate in place while badly misaligned
+K_ALIGN = 2.0      # P gain for the final in-place alignment
+ALIGN_W_MAX = 0.8  # rad/s limit during final alignment
+YAW_DONE = 0.08    # rad, well inside the 0.26 rad yaw tolerance
+
+
+def _wrap(a):
+    while a > math.pi:
+        a -= 2.0 * math.pi
+    while a < -math.pi:
+        a += 2.0 * math.pi
+    return a
 
 
 def controller(t, pose, env):
-    """Forward-only drive: v >= 0 and w == 0 for the entire mission.
+    """Three-phase point controller: rotate -> drive -> final alignment.
 
-    The goal is dead ahead of the spawn heading, so distance-to-goal falls
-    monotonically while driving straight; stop once safely inside the disc.
+    L1 adds two capabilities over the L0 forward-only controller:
+    (1) steering - the goal is NOT ahead of the spawn heading, so rotate
+        toward the goal bearing and keep correcting it while driving;
+    (2) arrival pose - once inside the goal disc, rotate in place until the
+        commanded goal yaw is met, then finish.
     """
     x, y, yaw = pose
-    d = math.hypot(GOAL[0] - x, GOAL[1] - y)
-    if d <= STOP_DIST:
-        return 0.0, 0.0, True  # arrived - stop and finish
-    return CRUISE_V, 0.0, False
+    gx, gy, gyaw = GOAL
+    d = math.hypot(gx - x, gy - y)
+
+    if d > STOP_DIST:
+        bearing = math.atan2(gy - y, gx - x)
+        err = _wrap(bearing - yaw)
+        w = max(-W_MAX, min(W_MAX, K_HEADING * err))
+        v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
+        return v, w, False
+
+    # inside the goal disc: align to the commanded arrival yaw, then finish
+    yerr = _wrap(gyaw - yaw)
+    if abs(yerr) > YAW_DONE:
+        return 0.0, max(-ALIGN_W_MAX, min(ALIGN_W_MAX, K_ALIGN * yerr)), False
+    return 0.0, 0.0, True
```

핵심 아이디어 — L1 이 요구하는 두 능력을 정확히 추가:
1. **조향**: goal 방위(bearing)와 현재 yaw 의 오차에 P 제어(`K_HEADING`)로 회전 명령.
   오차가 크면(`BEARING_GATE` 초과) 제자리 회전만 — 출발 시 π→π/2 정렬이 여기서 일어난다.
2. **도착 자세**: goal 디스크(`STOP_DIST`) 진입 후 제자리 회전으로 goal yaw 에 수렴
   (`YAW_DONE` = 0.08 rad, 판정 tol 0.26 의 1/3 지점에서 종료 선언).

## 4. solution 실행 결과 — 어떻게 성공했는가

**실측 (solution): `verdict: pass`**

| 지표 | 값 |
|---|---|
| time_to_goal (디스크 최초 진입) | **11.85 s** |
| min / final 거리 | 0.293 / **0.2748 m** (tol 0.75) |
| yaw 오차 (최종) | **0.0147 rad** (tol 0.26 — YAW_DONE 0.08 안쪽) |
| 최종 pose | (-6.0039, 4.7252, 1.5561) |
| 충돌 | **0** |
| sim / wall / frames | 14.0 s / 46.6 s / 840 |

전개: 출발 시 π→π/2 제자리 회전(`BEARING_GATE` 초과 구간, v=0) → +y lane 을 6 m 순항
(P 조향으로 lane 중심 유지, 접촉 0) → 11.85 s 에 goal 디스크 진입 → `STOP_DIST` 안에서
정지 후 최종 정렬(K_ALIGN) → yaw 오차 0.0147 rad 에서 종료 선언. 세 판정(위치 0.27/0.75,
yaw 0.015/0.26, 무접촉 0건) 모두 큰 여유로 통과.

## 5. 초판(a07c868) 대비 시나리오/저장소 변경 내역

| 파일 | 변경 | 사유 |
|---|---|---|
| `scenarios/nova_carter_warehouse_level1.yaml` | **값 변경 없음** (goal/tolerance/timeout 초판 그대로) | 초판 값이 실측으로 검증됨 |
| — 참고 | 초판 주석의 `goal_orientation_wxyz` caveat 은 **cv-infra 러너 전용**이다. `levels/` 하네스는 yaw 판정을 `yaw_tolerance_rad` 로 직접 구현 → L1 실측에서 yaw 판정 활성 확인 (base 의 yaw_err 1.578 rad 이 실패 사유에 포함됨) | LEVELS.md 튜닝절차에 노트 추가 |
| `levels/level1/` (신규) | base/solution/REPORT/results | 이 리포트 |

## 6. 재현

```bash
bash levels/run_isaac.sh levels/level1/base_carter_run.py       # FAIL 재현 (충돌+역주행)
bash levels/run_isaac.sh levels/level1/solution_carter_run.py   # PASS 재현
python3 levels/common/check_edit_region.py levels/level1        # 불변식 검증
```
