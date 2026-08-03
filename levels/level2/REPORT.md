# Level 2 실측 리포트 — 장애물 인지·회피

2026-08-03, Isaac Sim 4.5.0 / A100 80GB (GPU0) / seed 42 / dt=1/60.
판정 원본: `results/base_result.json`, `results/solution_result.json`
(1차 시도 실패 기록: `results/attempt1_gapfollow_result.json`).

## 1. 미션과 판정

`scenarios/nova_carter_warehouse_level2.yaml`:

| 항목 | 값 |
|---|---|
| 스폰 / goal | L1 과 동일: `(-6.0,-1.0,π)` → `(-6.0, 5.0, +1.5708)` |
| **신규 세계 상태** | `debug_obstacle` 박스 `(x=-6.2, y=2.0)`, 1.0 m 높이, 0.8×0.6 m — lane 중반을 막음 (x∈[-6.6,-5.8], y∈[1.7,2.3]) |
| 판정 | L1 과 동일 (pos 0.75 / yaw 0.26 / 무접촉 / 120 s) — **난이도는 세계가 올림** |
| L1 대비 신규 요구 | **인지 → 우회 → 복귀** (주입 박스는 충돌 제외 목록에 없음 — 스치면 실패) |

## 2. base 코드와 실행 결과 — 왜 실패하는가

base 의 `[EDIT REGION]` == **레벨 1 의 solution** (사다리 규칙). 3상 점-컨트롤러는
pose 만 보고 조향한다 — 센서가 없으니 lane 위에 나타난 박스를 알 길이 없다.

**실측 (base): `verdict: fail` — 두 oracle 모두 실패**

| 지표 | 값 |
|---|---|
| 궤적 | +y lane 정상 순항 → y≈1.56 에서 박스 앞면(y=1.7)에 정면 충돌 |
| 충돌 | **360건**, first t=6.283 s, partner=**`/World/debug_obstacle`** (주입 박스 그 자체) |
| min / final 거리 | 3.4401 / **3.4401 m** (tol 0.75) |
| 최종 pose | (-6.0496, 1.5602, 1.5708) — yaw 오차 0.000 (heading 은 완벽, 위치가 막힘) |
| 종료 | 충돌 5 s 후 조기 종료 (판정 무영향 — 두 oracle 확정 실패) |

의미: L1 솔루션의 결함이 아니라 **L1 스킬셋의 한계**를 정확히 노출한다. yaw 오차 0.000
이 보여주듯 주행·정렬은 완벽한데, "보는 능력"이 없어서 실패한다. 또한 충돌 파트너가
정확히 `/World/debug_obstacle` 로 잡혀 **장애물 주입 + contact oracle 경로가 이 레벨에서도
검증**됐다 (L1 base 의 카드박스 충돌에 이은 두 번째 양성 대조).

## 3. 풀이 과정 — 1차 시도(gap-follow)의 실패와 진단

정답에 이르기까지의 실측 루프를 그대로 기록한다 (`attempt1_gapfollow_*.{json,csv}` 보존).

**1차 시도**: 전방 콘(±0.6 rad) 최근접이 1.4 m 안이면, "2.2 m 이상 열려 있고 이웃
레이(±atan2(0.45, 2.2)≈0.20 rad)도 열린" 방위 중 goal 에 가장 가까운 쪽으로 0.25 m/s
주행하는 gap-follow.

**실측 (attempt 1): `verdict: fail`** — col=360, first t=8.617 s, 최종 (-5.6482, 1.5011).

궤적이 말해 주는 것: 회피 자체는 **정상 발동했다**. y≈0.39 (박스 앞면까지 1.31 m)에서
열린 동쪽으로 조향을 시작했고, 스캔 기하·조향 부호·발동 거리 모두 의도대로였다.
실패 원인은 둘:

1. **코너 여유 부족** — 이웃-레이 각도 마진 ±0.20 rad 은 장애물이 2.2 m 거리일 때
   0.45 m 의 측면 여유라는 가정인데, 박스에 1.0 m 까지 접근한 시점엔 같은 각도가
   **0.2 m** 로 줄어든다. 로봇 반폭 수준이라 동쪽 앞모서리 (-5.8, 1.7) 를 스쳤다.
2. **복귀가 너무 이름** — 모서리가 전방 콘을 벗어나는 순간 위협이 사라져 L1 동작으로
   복귀, goal 쪽(왼쪽)으로 되꺾었다. 섀시는 아직 박스 옆면을 지나는 중이었고
   t=8.617 s, (-5.77, 1.44) 에서 박스에 걸렸다.

**재설계**: 두 결함 모두 "마진이 거리와 무관하게 고정"인 데서 왔다 → 마진을 거리의
함수로 만드는 **bearing 팽창(VFH-lite)** 으로 교체했다. 각 히트 레이 `(b, r)` 가
`b ± atan2(SIDE_MARGIN, r)` 방위 쐐기를 차단한다: 가까울수록 넓게 막히므로 (1)이
해결되고, 박스 옆을 지나는 동안 측면 히트가 goal 방향을 계속 차단하므로 명시적
상태(state) 없이 (2)의 조기 복귀도 사라진다.

## 4. 수정 내용 (base → solution, 정답 키)

`[EDIT REGION]` 블록만 수정 (`check_edit_region.py` 검증 통과). 추가 상수 5개 +
`_clamp` 헬퍼 + 컨트롤러 본문:

```diff
--- levels/level2/base_carter_run.py
+++ levels/level2/solution_carter_run.py
@@ (EDIT REGION 내부만)
 YAW_DONE = 0.08    # rad, well inside the 0.26 rad yaw tolerance
+# --- L2 additions: obstacle avoidance on the planar ray scan ---
+BLOCK_RANGE = 2.2   # m: a hit nearer than this blocks bearings around it
+SIDE_MARGIN = 0.55  # m: lateral clearance to keep from any hit point
+THREAT_DIST = 1.4   # m: front-cone hit closer than this -> creep speed
+FRONT_CONE = 0.6    # rad: half-angle of the "in my way" cone
+AVOID_V = 0.25      # m/s while maneuvering around an obstacle
```

```python
    err = _wrap(math.atan2(gy - y, gx - x) - yaw)
    scan = env.raycast_scan()
    # hits at or beyond the goal distance cannot be in the way
    blocks = [(b - math.atan2(SIDE_MARGIN, r), b + math.atan2(SIDE_MARGIN, r))
              for b, r in scan if r < min(BLOCK_RANGE, d)]

    def blocked(rb):
        return any(lo <= rb <= hi for lo, hi in blocks)

    front = [r for b, r in scan if abs(b) <= FRONT_CONE]
    threat = min(front) if front else 99.0

    if not blocked(err) and threat >= THREAT_DIST:
        # goal direction clear: L1 behaviour unchanged
        w = _clamp(K_HEADING * err, W_MAX)
        v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
        return v, w, False

    # goal direction blocked: steer to the clear bearing nearest it
    cands = [b for b, r in scan if r >= BLOCK_RANGE and not blocked(b)]
    if not cands:
        return 0.0, W_MAX * 0.6, False  # boxed in: rotate in place and rescan
    best = min(cands, key=lambda b: abs(_wrap(b - err)))
    w = _clamp(K_HEADING * best, W_MAX)
    v = AVOID_V if threat < THREAT_DIST or abs(best) > BEARING_GATE else 0.8 * CRUISE_V
    return v, w, False
```

핵심 아이디어:
1. **거리-반비례 팽창**: `BLOCK_RANGE`(2.2 m) 안의 모든 히트가 자기 주변 방위를
   `±atan2(SIDE_MARGIN, r)` 만큼 차단 — 어느 거리에서든 측면 여유가 일정하게
   `SIDE_MARGIN`(0.55 m)으로 유지된다.
2. **goal 방향이 안 막혔으면 L1 그대로** — 회피 코드는 필요할 때만 개입한다
   (`r < min(BLOCK_RANGE, d)`: goal 보다 먼 히트는 애초에 장애물이 아님).
3. **막혔으면 가장 goal 쪽의 열린 방위로** 감속 주행. 우회 중엔 박스 옆면 히트가
   goal 방향을 계속 차단하므로, 박스를 실제로 벗어나야만 복귀가 시작된다.

## 5. solution 실행 결과 — 어떻게 성공했는가

**실측 (solution, 2차 시도): `verdict: pass`**

| 지표 | 값 |
|---|---|
| time_to_goal (디스크 최초 진입) | **15.25 s** (L1 무장애물 11.85 s 대비 우회 비용 +3.4 s) |
| min / final 거리 | 0.2748 / **0.2741 m** (tol 0.75) |
| yaw 오차 (최종) | **0.0678 rad** (tol 0.26) |
| 최종 pose | (-5.9391, 4.7328, 1.6386) |
| 충돌 | **0** |
| sim / wall / frames | 18.0 s / 113.9 s / 1080 |

전개 (trajectory.csv / trajectory.png): 스폰 회전(t<1.5 s) → goal 방향이 박스 쐐기에
막힌 것을 **원거리에서 감지**, y≈-0.25 부터 미리 동쪽으로 사선 진입 → 박스 동면
(x=-5.8) 옆을 중심선 x≈-5.37~-5.47 로 통과(면과의 측면 여유 ≈0.4 m — `SIDE_MARGIN`
설계값 그대로) → 통과 중엔 측면 히트의 쐐기가 goal 방향을 계속 차단해 조기 복귀 없음
→ 박스를 벗어난 y≈2.4 부터 완만한 복귀 곡선 → 남동쪽에서 디스크 진입(15.25 s) →
도착 정렬 후 종료. 1차 시도의 두 실패 원인(코너 여유·조기 복귀)이 정확히 사라졌다.

## 6. 초판(a07c868) 대비 시나리오/저장소 변경 내역

| 파일 | 변경 | 사유 |
|---|---|---|
| `scenarios/nova_carter_warehouse_level2.yaml` | **값 변경 없음** (obstacle x/y/치수, goal, tolerance, timeout 초판 그대로) | 초판 장애물 값이 실측으로 유효 확인 — base 가 정확히 그 박스에 충돌했고 solution 이 우회 |
| `levels/level2/` (신규) | base/solution/REPORT/results | 이 리포트 |

## 7. 재현

```bash
bash levels/run_isaac.sh levels/level2/base_carter_run.py       # FAIL 재현 (박스 정면 충돌)
bash levels/run_isaac.sh levels/level2/solution_carter_run.py   # PASS 재현
python3 levels/common/check_edit_region.py levels/level2        # 불변식 검증
```
