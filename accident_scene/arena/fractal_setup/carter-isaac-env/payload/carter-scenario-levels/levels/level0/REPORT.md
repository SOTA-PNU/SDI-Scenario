# Level 0 실측 리포트 — 직진 도달 (forward-only reach)

2026-08-03, Isaac Sim 4.5.0 / A100 80GB (GPU0) / seed 42 / dt=1/60.
판정 원본: `results/base_result.json`, `results/solution_result.json` (trajectory CSV 동봉).

## 1. 미션과 판정

`scenarios/nova_carter_warehouse_level0.yaml`:

| 항목 | 값 |
|---|---|
| 스폰 | 월드 `(-6.0, -1.0, yaw=π)` — 프로브 실측, 맵 프레임과 일치 |
| goal | `(-8.0, -1.0)` = 스폰 헤딩(-x) **정면 2.0 m** |
| 판정 | `reached_goal` 만: 최종 거리 ≤ **1.0 m** (yaw 미채점, 충돌 미채점) |
| timeout | 60 s (sim) |
| 요구 능력 | **전진 주행 하나** — 회전/조향/후진/우회 불필요 |

## 2. base 코드와 실행 결과 — 왜 실패하는가

`base_carter_run.py` 의 `[EDIT REGION]`:

```python
def controller(t, pose, env):
    """No control code yet: the robot never moves."""
    return 0.0, 0.0, False
```

제어 코드가 없습니다. 로봇은 ROS 그래프에 살아 있지만 `/cmd_vel` 이 0 → 60 s 내내 스폰에 정지.

**실측 (base): `verdict: fail`**

| 지표 | 값 |
|---|---|
| time_to_goal | — (goal 디스크 진입 없음) |
| min/final 거리 | 1.9359 / **1.9349 m** (tol 1.0 초과) |
| 최종 pose | (-6.0651, -1.0, 3.1414) — 사실상 스폰 그대로 |
| 충돌 | 0 |
| 실패 사유 | `reached_goal: timeout` |

(-0.065 m 의 미세 표류는 물리 정착(settling)이며 판정에 무의미. 동일 조건 2회 실행에서
final_d 가 소수 4자리까지 일치 — **결정성 확인**.)

## 3. 수정 내용 (base → solution, 정답 키)

`[EDIT REGION]` 블록만 수정. 그 외는 문자 단위 동일(`levels/common/check_edit_region.py` 로 검증).

```diff
--- levels/level0/base_carter_run.py
+++ levels/level0/solution_carter_run.py
@@ -46,10 +46,21 @@
 # [EDIT REGION] mission controller - modify ONLY this block
 # ===========================================================================
 
+CRUISE_V = 0.4   # m/s, forward only (L0 rule: w stays 0)
+STOP_DIST = 0.45  # m, stop well inside the 1.0 m position tolerance
+
 
 def controller(t, pose, env):
-    """No control code yet: the robot never moves."""
-    return 0.0, 0.0, False
+    """Forward-only drive: v >= 0 and w == 0 for the entire mission.
+
+    The goal is dead ahead of the spawn heading, so distance-to-goal falls
+    monotonically while driving straight; stop once safely inside the disc.
+    """
+    x, y, yaw = pose
+    d = math.hypot(GOAL[0] - x, GOAL[1] - y)
+    if d <= STOP_DIST:
+        return 0.0, 0.0, True  # arrived - stop and finish
+    return CRUISE_V, 0.0, False
```

핵심 아이디어: goal 이 헤딩 정면이므로 **w=0 고정, v=0.4 전진**만으로 거리가 단조 감소.
`d ≤ 0.45 m` 에서 정지 선언(허용 오차 1.0 m 안쪽 깊숙이).

## 4. solution 실행 결과 — 어떻게 성공했는가

**실측 (solution): `verdict: pass`**

| 지표 | 값 |
|---|---|
| time_to_goal (디스크 최초 진입) | **2.633 s** |
| final 거리 | **0.4008 m** (tol 1.0) |
| 최종 pose | (-7.5992, -0.9989, 3.1402) |
| 충돌 | 0 |
| sim / wall / frames | 5.017 s / 16.9 s / 301 |

물리 해석: 디스크 경계(반경 1.0)까지 직선 1.0 m — 0.4 m/s 순항 + 가속 램프 ≈ 2.6 s. 실측
2.633 s 로 일치. 이후 d=0.45 까지 전진해 정지, 최종 0.40 m.

## 5. 초판(a07c868) 대비 시나리오/저장소 변경 내역

| 파일 | 변경 | 사유 |
|---|---|---|
| `scenarios/nova_carter_warehouse_level0.yaml` | **goal `(-6.0, 1.0, yaw=1.5708)` → `(-8.0, -1.0, yaw=π)`**, timeout 60 유지, 주석 재작성 | 초판은 전방 자유공간이 미확인이라 "+y lane 위 통과 실적 점"으로 절충(출발 시 제자리 회전 1회 허용). `probe_scene.py` 실측으로 전방(-x) 2.0 m FREE / 2.5 m 팔레트가 확정되어, 레벨 원 정의("**앞으로만** 가서 **바로 앞** 목표")를 문자 그대로 복원 |
| `scenarios/LEVELS.md` | L0 섹션 재작성 (goal/근거/초판 차이), levels/ 참조 추가 | 위와 동일 + 실측 구현 링크 |
| `levels/level0/` (신규) | base/solution/REPORT/results | 이 리포트 |

판정 파라미터(`position_tolerance_m: 1.0`, yaw 비활성, timeout 60)는 초판 그대로입니다.

## 6. 재현

```bash
bash levels/run_isaac.sh levels/level0/base_carter_run.py       # FAIL 재현
bash levels/run_isaac.sh levels/level0/solution_carter_run.py   # PASS 재현
python3 levels/common/check_edit_region.py levels/level0        # 불변식 검증
```

주의: verdict 는 `results/*_result.json` 이 정본 (Kit 이 exit code 를 덮는 경우 있음).
GPU1 은 이 호스트에서 부팅 crash — GPU0 사용 (`levels/README.md` 참고).
