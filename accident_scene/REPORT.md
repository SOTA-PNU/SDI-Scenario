# accident_scene — 인지 실패로 인한 충돌 (실측 리포트)

레벨 사다리(L0-L3)와 달리 **단일 장면**이다. 레벨별로 나누지 않았다.
base 러너는 `levels_zero2level` 의 **디폴트 base**(빈 컨트롤러)로 시작한다.

측정: `fractal` (Ubuntu 24.04, RTX PRO 6000 Blackwell, 드라이버 580.173.02, Isaac Sim 4.5.0.0), seed 42.

> **두 번째 장면 [`arena/`](arena/REPORT.md)** — 같은 원리를 창고 전체로 넓혔다. 출발과 목표가
> 반대편 모서리이고 박스 14개(보이는 10 + 숨은 4)가 그 사이에 있다.

---

## 한 줄 요약

주행 회랑 한가운데에 **높이 0.25 m** 박스를 둔다. 컨트롤러가 쓸 수 있는 유일한 외부 감각인
`env.raycast_scan()` 은 **z=0.35 m 평면**의 광선 부채꼴이라 이 박스를 한 빔도 맞히지 못한다.
그래서 기본 스캔으로 회피를 구현한 컨트롤러는 박스를 못 보고 **그대로 박는다**.

## base — levels_zero2level 디폴트

`base_carter_run.py` 의 `[EDIT REGION]` 은 `levels_zero2level/level*/base_carter_run.py` 의
블록을 **바이트 단위로 그대로** 가져온 것이다 — 제어 코드가 없어 로봇이 주행하지 않는다.

```python
def controller(t, pose, env):
    """No control code yet: the robot never moves."""
    return 0.0, 0.0, False
```

파일 형식도 zero2level 을 따른다: givens 는 `MISSION` 테이블, docstring 에는 PASS 기준과
ROBOT API 만 있고 접근 지침·힌트는 없다. `[EDIT REGION]` 밖은 고정 배관이다.

```
carter-run: accident_scene/base -> FAIL  ttg=None  final_dist=6.0014m  collisions=0
  [FAIL] reached_goal   final_dist=6.001m (tol 0.75)
  [PASS] no_collision   collision_count=0
  TIMEOUT 120 s — 로봇이 스폰 (-6.0, -1.0) 에서 주행하지 않음
```

실측 결과: [`results/base_result.json`](results/base_result.json), 궤적 `results/base_trajectory.csv`.

> 명령이 0 이어도 로봇은 **약 1.06 mm/s 로 일정하게 전방으로 미끄러진다** — 120 s 에 0.127 m
> (최종 자세 x = −6.129, 방향은 헤딩 그대로). 원본 `Harness` 를 그대로 쓰는 이 장면과
> arena 가 1% 안으로 같은 속도(1.061 / 1.053 mm/s)라서
> 로봇/물리 쪽 성질이지 이 폴더 코드 탓이 아니다(원인은 조사하지 않았다). `final_dist` 가
> 6.000 이 아니라 6.001 인 이유다.

## 왜 이게 새로운가

`levels/level1/base` 와 `levels/level2/base` 는 이미 충돌한다(각 360회). 하지만 그건
*순진한 직진 컨트롤러*가 *보이는* 장애물에 부딪히는 것이다. 회피 능력만 넣으면 해결된다.

이 장면은 반대다. 회피를 구현해도 부딪힌다. 세계가 감각을 속이기 때문이다.

README와 `scenarios/LEVELS.md` 는 L2를 상류 정본의 `nova_carter_warehouse_obstacle_fail.yaml`
(저상 박스를 nav이 못 보고 긁는 의도적 fail)의 **반대 짝**으로 설명하면서, 둘을 같이 돌려야
"인지 실패"와 "회피 실패"가 분리 진단된다고 적어 두었다. 그런데 그 negative control 은 이
저장소에 없다. 이 폴더가 그 빈자리를 메운다.

| | 장애물이 | 실패하면 그건 |
|---|---|---|
| L2 / L3 | **보인다** (height 1.0 m) | *회피* 능력의 문제 |
| **accident_scene** | **안 보인다** (height 0.25 m) | *인지* 의 문제 |

---

## 장면

| | 값 |
|---|---|
| 스폰 | `(-6.0, -1.0)`, yaw = π (L1-L3와 동일) |
| 목표 | `(-6.0, 5.0)`, yaw = 1.5708 |
| 허용 오차 | 위치 0.75 m, yaw 0.26 rad |
| timeout | 120 s |
| 장애물 | `x=-6.0, y=3.8, height=0.25, width=1.2, depth=0.6` |

박스는 `x ∈ [-6.6, -5.4]`, `y ∈ [3.5, 4.1]` 을 덮는다. 주행 회랑(`x ∈ [-6.3, -5.7]`)을
완전히 가로막고, 목표 **1.2 m 앞** 최종 접근 구간에 있다.

---

## 설계 검증 1 — 왜 못 보는가 (스캔 평면별 전방 원뿔 최소 판독)

| 로봇 y | 박스 앞면까지 | z = 0.02 ~ 0.25 | z = 0.30 / **0.35 (기본값)** | 바닥 포화 빔 |
|---|---|---|---|---|
| +1.01 | 2.49 m | **2.49 m** | 5.53 m | 0 / 61 |
| +2.01 | 1.49 m | **1.49 m** | 5.33 m | 0 / 61 |
| +2.50 | 1.00 m | **1.00 m** | — | 0 / 61 |

기본 평면은 5.3~5.5 m, 즉 **먼 벽만** 본다. 박스가 아예 없는 경우와 구별이 불가능하다.
낮은 평면은 전부 정확한 거리를 주고, 바닥에 포화된 빔은 하나도 없다.

## 설계 검증 2 — 회피를 구현해도 부딪힌다

이 장면이 "쉽게 부딪히는" 세계인지는 base 로는 확인할 수 없다(주행하지 않으니까). 그래서
기본 스캔으로 회피하는 **참조 컨트롤러** — `levels/level3/solution_carter_run.py` 의 반응형
회피기, L2·L3 통과 — 를 이 장면에 넣어 쟀다 (base 아님, 설계 검증용):

```
verdict=FAIL  collision_count=360  partner=/World/debug_obstacle  final_dist=1.662m
```

2회 반복이 0.001 m 이내로 같다. 박스에 막혀 정지하는 결정론적 사고다.

## 설계 검증 3 — 풀 수 있다 (함정이 아니다)

같은 참조 컨트롤러에서 **세 줄**만 바꾸면 통과한다:

```python
BLOCK_RANGE = 3.0            # 2.2 -> 3.0   (더 일찍 반응)
SIDE_MARGIN = 0.95           # 0.55 -> 0.95 (박스가 1.2 m 폭이라 여유를 넓힘)
scan = env.raycast_scan(z=0.15)   # 기본 평면 아래를 본다
```

```
verdict=PASS  collision_count=0  time_to_goal=11.783s  final_dist=0.2764m
```

`raycast_scan` 의 `z` 는 인자다. **"내 센서가 못 보는 영역이 있다"를 인지하고 감각 자체를
바꾸는 것**이 이 장면이 요구하는 능력이다.

---

## 파라미터를 이렇게 고른 이유 (전부 실측 스윕, 참조 컨트롤러 기준)

### 높이 — 0.25 m

| height | 결과 |
|---|---|
| 0.05 | 로봇이 **타고 넘어가며 긁는다**. 충돌 87~166회, 목표는 도달(ttg 7.25 s, final_dist 0.28) |
| 0.10 | 걸려서 정지. 충돌 3322회 |
| 0.15 | 걸려서 정지. 충돌 360회 |
| **0.25** | 걸려서 정지. 충돌 360회. **z ≤ 0.25 평면에서 보이므로 풀 수 있다** |
| 0.30 | 걸려서 정지하지만 기본 평면에 **걸리기 시작**한다 |

0.25 를 고른 이유는 풀 수 있어서다. 0.05 는 `z ≤ 0.05` 에서만 보여 여유가 거의 없다.
**0.35 가 경계다** — 그 이상은 기본 스캔에 잡혀서 이 시나리오가 그냥 L2 변형이 된다.

0.05 변형이 필요하면(= `reached_goal` 은 통과하고 `no_collision` 만 실패하는, 오라클 하나만
분리하는 형태) `height` 만 0.05 로 바꾸면 된다. 상류 `obstacle_fail` 과 같은 의미론이다.

### 위치 — y = 3.8

스캔 평면을 낮춘 참조 컨트롤러 기준:

| 박스 y | 결과 |
|---|---|
| 3.5 | PASS, 충돌 0 |
| **3.8** | **PASS, 충돌 0, ttg 11.783 s** |
| 4.1 | FAIL, 충돌 102 |
| 4.5 | FAIL, 충돌 170 |

4.1 이상은 박스와 목표 사이가 너무 좁아 **우회 후 목표 중심으로 복귀하는 것 자체가 불가능**해진다.
그건 인지 문제가 아니라 기하 문제라서 이 장면이 시험하려는 능력이 흐려진다.
3.8 은 최종 접근 구간이면서 복귀 여유(0.9 m)가 남는 최대값이다.

### 폭 — 1.2 m

L2 의 0.8 보다 넓혔다. 회랑을 완전히 덮어 **우연히 옆으로 비켜가는 경우를 없앤다**.

---

## 실행

```bash
CARTER_REPO=<repo> $CARTER_WS/tools/bin/carter-run accident_scene/base_carter_run.py
```

`base` 는 **FAIL(exit 1)이 정상이다** (빈 컨트롤러, 120 s TIMEOUT — 실행에 약 2분).
`set -e` 나 `&&` 체인으로 묶지 마라 — 정상 동작이 에러로 보인다. 판정의 진실은
`accident_scene/results/base_result.json` 의 `verdict` 다.

zero2level 과 같은 방식으로 쓰려면 base 를 `<모델이름>_carter_run.py` 로 복사해 그 파일의
`[EDIT REGION]` 만 고친다. 결과는 `results/<모델이름>_result.json` 에 쌓인다.

레벨 러너들과 달리 이 폴더는 사다리 밖이라 `levels/common/check_ladder.py` ·
`check_edit_region.py` · `check_yaml_sync.py` 의 검사 대상이 아니다(세 체커 모두
`levels/level0..3` 와 `scenarios/nova_carter_warehouse_level{0..3}.yaml` 만 본다 — 확인함).
`check_edit_region.py` 는 인자로 파일을 주면 이 폴더에서도 쓸 수 있다.

## 파일

```
accident_scene/
├── nova_carter_warehouse_accident.yaml   시나리오 (설계 근거 주석 포함)
├── base_carter_run.py                    base 러너 ([EDIT REGION] = levels_zero2level 디폴트, 빈 컨트롤러)
├── REPORT.md                             이 문서
└── results/
    ├── base_result.json                  base 실측 결과 (FAIL, TIMEOUT)
    ├── base_trajectory.csv               base 궤적 (정지)
    └── accident_base_isaac_1280.mp4      base 실행 Isaac Sim 녹화 (경로 추적)
```

---

## 영상

`results/accident_base_isaac_1280.mp4` — **디폴트 base 실행의 Isaac Sim 렌더**
(1280×720, 15 fps 실시간, 123 s = 실행 121 s + 끝 2 s 정지, 2.2 MiB). 하네스가 120 s
TIMEOUT 뒤 1 s 더 돌리므로 HUD 시각은 121.0 s 에서 끝난다. 기존 `levels/level*/results/l*_solution_1280.mp4` 와 **같은 카메라**다.
빈 컨트롤러라 로봇은 120 초 내내 스폰 자리에 서 있고(위의 0.13 m 크리프뿐), 마지막 3 초에
TIMEOUT 배너가 뜬다.
왼쪽 위 HUD(시각·접촉 수)와 배너는 실행 결과로 덧그린 것이다.

녹화할 때만(`CARTER_RECORD=1`) 러너가 렌더러를 **경로 추적 8 spp + OptiX 디노이저**로 바꾼다
(이유와 측정은 [`arena/REPORT.md`](arena/REPORT.md) 의 "화질" 절):

```bash
CARTER_RECORD=1 CARTER_RECORD_EVERY=4 CARTER_RECORD_SIZE=1920x1080 CARTER_RECORD_DIR=/tmp/frames \
  CARTER_REPO=<repo> $CARTER_WS/tools/bin/carter-run accident_scene/base_carter_run.py
ANNOT_EVERY=4 ANNOT_MODE=hud python3 accident_scene/arena/annotate_video.py /tmp/frames \
  accident_scene/results/base_result.json accident_scene/results/base_trajectory.csv out.mp4
```

녹화 실행은 판정용 결과 파일을 덮어쓰므로(REMOTE_VIEWING.md 의 "녹화 런은 시각화 전용")
비녹화 실측으로 복원해 두었다. 이번 녹화 실행 결과는 FAIL(TIMEOUT), 충돌 0, final_dist
6.0014 m 로 비녹화 실측과 같고, 궤적 CSV 는 **바이트 단위로 동일**했다(명령이 없으니 ROS 2
지연이 끼어들 곳이 없다). 녹화 벽시계는 22 분(1343 s, 물리 스텝당 약 0.18 s — 매 스텝 렌더).

화면의 주황 박스는 **실제 충돌체와 같은 크기**(`1.2 × 0.6 × 0.25 m`)로 렌더된다.
(스테이지의 바운딩박스 API 는 다른 값을 보고하는데, 그건 `extent` 속성 문제다 — 아래 부록.)

---

## 부록 — `_spawn_obstacle` 의 `extent` 속성 (실측 중 발견)

`levels/common/carter_env.py:_spawn_obstacle` 이 만드는 `FixedCuboid` 를 스테이지에서 읽으면:

```
size          = 1.0
xformOp:scale = (1.2, 0.6, 0.25)                               # 지정값
extent        = [(-0.6, -0.3, -0.125), (0.6, 0.3, 0.125)]      # ← 잘못됨
```

**실제 기하는 `size × scale` = 지정값 그대로다.** 렌더러와 PhysX 충돌체가 모두 이 값을 쓴다.
레이캐스트가 확정한다 — 이 박스는 z=0.25 평면에서 맞고 z=0.30 에서 빗나가며, 비정사각
검증 박스(`1.0 × 0.5 × 0.6`)도 높이·폭·깊이 모두 지정값 경계에서 정확히 맞고 빗나갔다.
Isaac 화면 영상에 찍힌 박스도 지정 크기다.

잘못된 것은 **`extent` 한 가지**다. 단위 큐브(`size=1`)의 extent 는 ±0.5 여야 하는데 월드 크기의
절반이 기록돼 있어서, extent 를 쓰는 곳 — `UsdGeom.BBoxCache`, 뷰포트의 frame-selected,
컬링 — 은 여기에 scale 을 한 번 더 곱해 **제곱 크기**를 본다:

| 지정값 (= 렌더 = 충돌체) | `BBoxCache` 가 보고하는 값 |
|---|---|
| L2/L3 `0.8 × 0.6 × 1.0` | `0.640 × 0.360 × 1.000` |
| accident `1.2 × 0.6 × 0.25` | `1.440 × 0.360 × 0.0625` |

판정·충돌·센싱·렌더 모두 올바른 기하를 쓰므로 이 저장소의 실측 결과, 이 장면의 설계,
영상은 모두 유효하다. 바운딩박스 API 로 장애물 크기를 재는 도구를 만들 때만 조심하면 된다.

> 이 부록의 첫 판(2026-09-11 오전)은 "시각 메시가 제곱 크기로 렌더된다"고 적었다. 틀렸다.
> `BBoxCache` 값을 렌더 크기로 오해한 것이고, prim 속성을 직접 읽어 바로잡았다.
