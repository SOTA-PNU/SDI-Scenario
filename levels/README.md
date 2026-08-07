# 레벨 별 사다리 방식 시나리오 구현

> LLM 에게 `base_carter_run.py`와 PROMPT.md를 주고 베이스 코드의 `[EDIT REGION]` 블록만 수정해 레벨을 통과시키게
> 합니다. 여기 있는 `prompted_carter_run.py` 가 Claude 가 만든 **정답(레퍼런스) 코드**입니다.

## 구조

```
levels/
  run_isaac.sh                  # 이 호스트의 Isaac Sim 환경으로 python 실행 (진입점)
  common/
    carter_env.py               # 공통 하네스: 부팅·장애물 주입·GT pose·충돌 리포트·판정·JSON
    probe_scene.py              # 정찰 프로브 (스폰 pose/자유공간/rclpy/충돌 API 검증)
    check_edit_region.py        # base==solution (수정영역 외) 불변식 체커
    check_ladder.py             # 사다리 규칙 체커: levelN/base == levelN-1/solution (수정영역)
    check_yaml_sync.py          # scenarios/*.yaml 본문 == runner givens 동기화 체커
    plot_traj.py                # base vs solution 궤적 PNG 렌더러
  level{0..3}/
    base_carter_run.py          # 레벨 시작점 (실행하면 FAIL — 그 레벨의 능력이 없음)
    solution_carter_run.py      # 정답 코드 (실행하면 PASS)
    PROMPT.md                   # 고정 프롬프트 (로컬 LLM 벤치마크 입력 — 아래 프로토콜)
    prompted_carter_run.py      # 위 프롬프트만 보고 작성한 기준 구현 (PASS 실측)
    REPORT.md                   # 과정·결과·diff·시나리오 YAML 변경 내역
    results/                    # {base,solution,prompted}_result.json + trajectory.csv
```

## 벤치마크 규칙

1. **base 와 prompted 는 `[EDIT REGION]` 블록 외에 문자 단위로 동일**합니다.
2. 수정 대상은 `controller(t, pose, env) -> (v, w, done)` 와 그 보조 상수/함수 뿐.
3. **레벨 N 의 base == 레벨 N-1 의 solution**. 사다리의 "능력 누적"이
   코드로도 성립합니다: 이전 레벨을 풀던 코드가 다음 레벨에서 왜 실패하고 어떻게 성공하도록 하는지에 대한 실측.
4. 판정(verdict)의 진실은 `results/*_result.json` 의 `verdict` 필드입니다.

## 고정 프롬프트 프로토콜 (로컬 LLM 평가용)

각 레벨의 `PROMPT.md` 에는 **고정 프롬프트**가 있습니다 — 로컬 LLM 을 평가할 때 이 프롬프트
본문과 `base_carter_run.py` 전문을 그대로 입력하고, 출력된 `[EDIT REGION]` 블록을
base 사본에 끼워 실행·채점합니다.

`prompted_carter_run.py` 는 그 프롬프트만 보고 작성한 **기준 구현**입니다.

로컬 LLM 평가 시 모델에게는 **본문+base 입력 → 블록 출력**만 시키십시오 (파일 생성·
실행·채점까지 시키면 절차 실패가 능력 측정을 오염시킵니다). 출력 블록의 채점은 한
명령으로 자동화되어 있습니다:

```bash
python3 levels/common/grade_block.py <레벨 0-3> <모델이름> <블록파일|->
# 예: python3 levels/common/grade_block.py 0 llama-3.1-8b block.txt
# 이식 → 규칙검사 → Isaac 실측 → results/<모델이름>_result.json 채점 (PASS=exit 0)
```

첫 로컬 LLM 실측 (2026-08-07): **llama-3.1-8b — L0 PASS** (ttg 1.3 s, final_d 0.71 m,
충돌 0). 정지 판단을 목표 반경 1.0 m 경계에서 내려 기준 구현(final_d 0.39 m)보다
여유가 얇습니다. 코드는 `level0/llama-3.1-8b_carter_run.py`.

**prompted 실측 (2026-08-05, 개정판 프롬프트로 재도출·재실측, seed 42)**

| 레벨 | verdict | time_to_goal | final_d | yaw_err | 충돌 | solution 대비 |
|---|---|---|---|---|---|---|
| L0 | **PASS** | 2.17 s | 0.39 m | — | 0 | 같은 구조, 상수만 다름 |
| L1 | **PASS** | 12.2 s | 0.32 m | 0.007 rad | 0 | 같은 3단계 구조, 게인·게이트 다름 |
| L2 | **PASS** | 17.43 s | 0.28 m | 0.069 rad | 0 | 같은 bearing-팽창 계열 (margin 0.55) |
| L3 | **PASS** | 8.95 s (≤12) | 0.28 m | 0.071 rad | 0 | 같은 속도 3상수 계열 (0.9/1.6/0.45) |

프롬프트 반복 이력: L2 1차 재도출은 측면 여유를 0.5 m 로 잡아 박스 모서리 스침 10회로
FAIL — 프롬프트에 여유 하한(0.55 m 이상)과 전방 원뿔 반각(0.6 rad 수준) 권고를 보강한
2차 재도출이 무충돌 PASS 했습니다. 회피 여유가 프롬프트 지침의 민감 지점임을 실측으로
확인한 반복입니다.

## 컨트롤러 API (LLM 프롬프트에 포함될 계약)

`controller(t, pose, env)` 는 sim 1/60 s 마다 호출됩니다:

| 인자/반환 | 의미 |
|---|---|
| `t` | 미션 시작 후 sim 초 |
| `pose` | `(x, y, yaw)` — 섀시의 GT 월드 pose (localization 은 스코프 밖, GT 제공) |
| `env.raycast_scan(n_beams=61, fov_deg=180, z=0.35, max_range=6.0)` | 현재 헤딩 기준 평면 레이 팬 → `[(상대 bearing rad, 거리 m), ...]`, 거리==max_range 는 무히트 |
| 반환 `(v, w, done)` | `/cmd_vel` 의 linear.x [m/s], angular.z [rad/s], 미션 종료 선언 |

`done=True` 를 반환하면 하네스가 로봇을 정지시키고(1 s settle) 최종 pose 로 판정합니다.

## 판정 의미론 (cv-infra oracle 의 스탠드얼론 구현)

| oracle | 통과 조건 | 비고 |
|---|---|---|
| `reached_goal` | **최종** pose 가 goal 의 `position_tolerance_m` 안 AND (`yaw_tolerance_rad` 설정 시) yaw 오차 tol 안 | 최종 상태 기준 — 스치고 지나가면 불통 |
| `no_collision` | 섀시 contact report 이벤트 0건 | 제외: 자기 몸체(`/World/Nova_Carter_ROS`)·바닥(`GroundPlane`)·창고 구조물 바닥면. **주입한 박스는 제외 안 됨** |
| `max_time_to_goal` | goal 디스크 **최초 진입** 시각 ≤ bound | cv-infra `max_time_to_goal.py` 와 동일한 도달 정의 |
| `timeout_s` | sim 시간 예산 | 초과 시 미션 종료 후 판정 |

## 실행 방법

```bash
bash levels/run_isaac.sh levels/level0/solution_carter_run.py   # 레벨0 정답 실행
bash levels/run_isaac.sh levels/level0/base_carter_run.py       # 레벨0 base (FAIL 재현)
```

전제: `/home/jun/carter_ws` (Isaac Sim 4.5 micromamba env — `nova_carter_sim/README.md` 의
브링업). 환경변수 `CARTER_WS` 로 위치 변경 가능.

## 실측 결과 요약

Isaac Sim 4.5.0, A100 80GB (GPU0), seed 42, dt=1/60. 상세는 각 REPORT.md.

| 레벨 | base 결과 (왜 실패) | solution 결과 |
|---|---|---|
| L0 | FAIL — 제어 코드 없음, 60 s 타임아웃, 스폰에서 정지 (final_d 1.93 m) | PASS — ttg 2.63 s, final_d 0.40 m, 충돌 0 |
| L1 | FAIL — L0의 전진-only 코드가 뒤쪽 goal 반대 방향으로 직진, 팔레트 지대 관통 후 카드박스에 섀시 충돌 360건 (final_d 6.56 m, yaw_err 1.58) | PASS — ttg 11.85 s, final_d 0.27 m, yaw_err 0.015 rad, 충돌 0 |
| L2 | FAIL — L1 컨트롤러가 박스를 못 보고 정면 충돌 360건 (partner=`/World/debug_obstacle`) | PASS — ttg 15.25 s, final_d 0.27 m, yaw_err 0.068 rad, 충돌 0. 1차 gap-follow 시도는 코너 스침으로 실패 → bearing-팽창(VFH-lite)으로 재설계 (REPORT §3) |
| L3 | FAIL — 회피·도달·정렬 전부 성공하고도 **ttg 15.433 s > bound 12 s** (유일하게 시간축만 실패 — 효율 결함) | PASS — ttg 8.283 s (여유 3.7 s), final_d 0.28 m, 충돌 0. 정답 키 = 속도 상수 3개 (CRUISE 1.0 / W_MAX 1.5 / AVOID 0.5) |

## 특이사항 (재현 시 알아야 할 것)
- `/cmd_vel` 제어는 브리지 내장 rclpy 를 **같은 프로세스**에서 import 해 수행
  (`carter_env.py` 참고).
- variant(파일명 유래)가 ROS 2 노드 이름에 들어가므로 하네스가 `[^A-Za-z0-9_]` 를 `_` 로
  치환합니다 — 점·하이픈이 든 모델 이름(`llama-3.1-8b` 등)도 그대로 파일명으로 쓸 수 있고,
  결과 JSON 의 variant 는 원래 이름을 유지합니다.
