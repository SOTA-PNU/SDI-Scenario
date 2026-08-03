# levels/ — 레벨 사다리 실측 구현 (LLM 벤치마크 정답 코드)

`scenarios/nova_carter_warehouse_level{0..3}.yaml` 을 **cv_infra 없이** Isaac Sim 4.5 에서
직접 성공시킨 코드와 그 실측 기록. 목적은 **NPU 로컬 LLM 벤치마크 픽스처**다:

> LLM 에게 `base_carter_run.py` 를 주고 `[EDIT REGION]` 블록만 수정해 레벨을 통과시키게
> 한다. 여기 있는 `solution_carter_run.py` 가 Claude 가 만든 **정답(레퍼런스) 코드**이고,
> base→solution diff 가 곧 그 레벨의 정답 키다.

## 구조

```
levels/
  run_isaac.sh                  # 이 호스트의 Isaac Sim 환경으로 python 실행 (진입점)
  common/
    carter_env.py               # 공통 하네스: 부팅·장애물 주입·GT pose·충돌 리포트·판정·JSON
    probe_scene.py              # 정찰 프로브 (스폰 pose/자유공간/rclpy/충돌 API 검증)
    check_edit_region.py        # base==solution (수정영역 외) 불변식 체커
  level{0..3}/
    base_carter_run.py          # 레벨 시작점 (실행하면 FAIL — 그 레벨의 능력이 없음)
    solution_carter_run.py      # 정답 코드 (실행하면 PASS)
    REPORT.md                   # 과정·결과·diff·시나리오 YAML 변경 내역
    results/                    # {base,solution}_result.json + trajectory.csv (실측 원본)
```

## 벤치마크 규칙 (파일 구조가 강제하는 것)

1. **base 와 solution 은 `[EDIT REGION]` 블록 외에 문자 단위로 동일**하다.
   `python levels/common/check_edit_region.py levels/level0 ... levels/level3` 으로 기계 검증.
2. 수정 대상은 `controller(t, pose, env) -> (v, w, done)` 와 그 보조 상수/함수뿐이다.
   하네스(부팅·판정·센서 API)는 고정 — LLM 이 판정을 조작할 수 없다.
3. **레벨 N 의 base == 레벨 N-1 의 solution** (수정영역 기준). 사다리의 "능력 누적"이
   코드로도 성립한다: 이전 레벨을 풀던 코드가 다음 레벨에서 왜 실패하는지가 base 실측이다.
4. 판정(verdict)의 진실은 `results/*_result.json` 의 `verdict` 필드다.
   (Kit 종료 핸들러가 프로세스 exit code 를 0 으로 덮는 경우가 있어 exit code 는 신뢰 불가)

## 컨트롤러 API (LLM 프롬프트에 포함될 계약)

`controller(t, pose, env)` 는 sim 1/60 s 마다 호출된다:

| 인자/반환 | 의미 |
|---|---|
| `t` | 미션 시작 후 sim 초 |
| `pose` | `(x, y, yaw)` — 섀시의 GT 월드 pose (localization 은 스코프 밖, GT 제공) |
| `env.raycast_scan(n_beams=61, fov_deg=180, z=0.35, max_range=6.0)` | 현재 헤딩 기준 평면 레이 팬 → `[(상대 bearing rad, 거리 m), ...]`, 거리==max_range 는 무히트 |
| 반환 `(v, w, done)` | `/cmd_vel` 의 linear.x [m/s], angular.z [rad/s], 미션 종료 선언 |

`done=True` 를 반환하면 하네스가 로봇을 정지시키고(1 s settle) 최종 pose 로 판정한다.

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

2026-08-03, Isaac Sim 4.5.0, A100 80GB (GPU0), seed 42, dt=1/60. 상세는 각 REPORT.md.

| 레벨 | base 결과 (왜 실패) | solution 결과 |
|---|---|---|
| L0 | FAIL — 제어 코드 없음, 60 s 타임아웃, 스폰에서 정지 (final_d 1.93 m) | PASS — ttg 2.63 s, final_d 0.40 m, 충돌 0 |
| L1 | (실측 후 기입) | (실측 후 기입) |
| L2 | (실측 후 기입) | (실측 후 기입) |
| L3 | (실측 후 기입) | (실측 후 기입) |

## 호스트 특이사항 (재현 시 알아야 할 것)

- **GPU1 부팅 크래시**: 이 호스트의 GPU1(CUDA_VISIBLE_DEVICES=1)로는 SimulationApp 부팅이
  URDF importer 확장 시작 중 segfault 로 죽는다(단독 실행 포함 재현 2회, GPU0 는 정상).
  모든 실측은 GPU0. 병렬 실행 불가.
- 스텝 속도 ~17 fps (A100 은 RT 코어 없음 — `nova_carter_sim/README.md` 참고). sim 60 s ≈ wall 3.5 분.
- 프로브 실측: 스폰 = 월드 `(-6.0, -1.0, yaw=π)` == 맵 좌표 (두 프레임 일치),
  전방(-x) 자유공간 2.0 m (2.5 m 부터 팔레트), +y lane 은 goal 까지 FREE.
- `/cmd_vel` 제어는 브리지 내장 rclpy 를 **같은 프로세스**에서 import 해 수행
  (`carter_env.py` — RoboStack 을 섞으면 ABI 가 깨진다, `isaac_python.sh` 주석 참고).
