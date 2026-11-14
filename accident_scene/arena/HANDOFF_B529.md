# 인계: B-5-29 (B-5-28 + 로봇 속도 5배) — fractal에서 이어 하기

작성 2026-11-14, 브랜치 `b529-speedx5`. 다른 서버(`SOTA-Furiosa-RNGD`, RTX PRO 6000, 드라이버 595)에서
준비했고, **그 서버에서는 라이다가 동작하지 않아 끝까지 검증하지 못했다**(아래 3절). 드라이버 580인 fractal에서 이어 한다.

> **이 문서를 읽는 것이 Claude라면:** 4절 순서대로 진행한다. 각 단계의 기대값과 다르면 다음 단계로 가지 말고 사용자에게 보고한다.
> 기존 `pB528` 결과 파일은 실험 기록이므로 덮어쓰지 않는다(러너도 거부한다).

## 1. 무엇을 바꿨나

사용자 요청: "카터의 기본 속도가 너무 느렸다. 모든 속도를 5배로" → 이어서 "탈출은 직진 속도와 일치",
"회전 속도가 문제였다면 문제된 것만 수정".

| 위치 | v16 / B-5-28 | v17 / B-5-29 |
|---|---|---|
| `_BASE_LOOP_CFG` 추종 상한 `v_max` / `w_max` | 0.3 m/s / 0.4 rad/s | **1.5 / 2.0** |
| 프롬프트 추종 식 (`PROMPT_B5_29.md` 24, 154, 155행) | `v=min(0.3,0.5*dist+0.1)`, `w=clamp(1.5e,±0.4)` | `v=min(1.5,2.5*dist+0.5)`, `w=clamp(7.5e,±2.0)` |
| fast 후보 `fast_min` / 상한 | 1.0 / 1.0 | **5.0 / 5.0** |
| 후진 `back` | -0.2 | **-1.0** |
| 탈출 `_BASE_ESCAPE_CONFIG` slow / fast / curve v | 0.15 / 1.0 / 1.0 | 0.75 / **1.5 / 1.5** (직진 속도에 맞춤) |
| 탈출 회전 이득 / 상한 | 1.5 / 0.4 | 7.5 / 2.0 |
| 목표 정렬 이득 (`1.5*yaw_error`) | 1.5 | 7.5 |
| 제자리 회전 `turn_l`/`turn_r` | ±0.4 | **±0.4 (유지)** — 2.0이면 끼임, 3절 |

거리 여유, 예측 시간(`_BASE_MAP_HORIZON_CONFIG`, `linear_decel_m_s2: 2.0`), 물체 추적 설정은 그대로다.
프롬프트는 세 줄만 바뀌었고 길이는 B-5-28과 같은 11,887자다.

새 파일:
- `PROMPT_B5_29.md`, `base_carter_run_ros_map_lidartime_objects_check_v17.py`(머리 주석에 변경 요약)
- `run_b529_sequential.sh` — `run_b528_sequential.sh`에서 프롬프트/base/TAG(`gpu_gpt_oss_120b_pB529`)/폴더(`results/turns/prompt_b5_29`)만 바꿈
- `traj_video.py` — 궤적 CSV로 위에서 본 MP4(H.264)를 그리는 도구(Isaac 녹화가 안 될 때용)
- 시험용 제출 사본 (채점에만 씀, 모델 산출물 아님):
  - `speedx5_test_pB528_a1_carter_run.py` — B-5-28 a1 코드 + v17 변경 + follow 식 5배, **제자리 회전 2.0**
  - `speedx5t04_test_pB528_a1_carter_run.py` — 위와 같고 제자리 회전만 0.4 (= 현재 v17)
  - `regrade_crojjang_pB528_a1_carter_run.py`, `regrade_movers_pB528_a1_carter_run.py` — 원래 a1 그대로(재채점용)
  - `probe_lidar_pB528_a1_carter_run.py` — a1 + `_robot_only`에서 2 s마다 라이다 점 개수 출력(진단용)

`levels/common/carter_env.py` 변경은 fractal에 원래 있던 미커밋 수정(`CARTER_CONTACT_ALL_LINKS`)을 커밋한 것이다.
뷰포트 우회 패치(`enable_no_viewport.sh`)는 **들어 있지 않다**.

## 2. 이 브랜치의 나머지

`accident_scene/` 전체는 원래 `main`에 없었다(fractal의 로컬 작업). 2026-11-10 백업본(`pro6000-bak`)을 그대로 올렸다.
백업에서 빠진 큰 결과 묶음(`results/turns/`, `frames_*`, `results/analysis` 일부)은 없다. fractal의
`/home/jun/SDI-Scenario`에 그 이후 변경이 있을 수 있으니 4절 0단계에서 비교한다.

## 3. 다른 서버에서 확인한 것 (드라이버 595 + 뷰포트 끔)

모든 채점은 `CARTER_CONTACT_ALL_LINKS=1`, 결과 파일은 `results/<variant>_result.json`.

| variant | movers | 판정 | 도달 | 최종 거리 | 충돌 | 비고 |
|---|---|---|---|---|---|---|
| `regrade_crojjang_pB528_a1` | 끔 | pass | 58.4 s | 0.4625 | 0 | fractal 공식 61.55 s |
| `regrade_movers_pB528_a1` | v1 | pass | 58.4 s | 0.4625 | 0 | 위와 소수점까지 같음 → 라이다 의심 |
| `speedx5_test_pB528_a1` | 끔 | **fail** | — | 12.2 | 0 | 15 s에 박스 05 오른쪽에 붙어 600 s까지 정지 |
| `speedx5t04_test_pB528_a1` | v1 | pass | **25.2 s** | 0.097 | 0 | 평균 1.33, 최고 1.50 m/s |
| `probe_lidar_pB528_a1` | v1 | pass | 58.65 s | 0.469 | 0 | **앞·뒤 2D 라이다 0점, 추적 물체 0** |

해석:
- **제자리 회전 2.0 rad/s 끼임:** chooser는 후보를 `0.5 s + 정지시간`(감속 가정 2.0 m/s², 2.68 rad/s²) 동안 굴려
  0.2 m 여유를 본다. 2.0 rad/s 제자리 회전은 1.25 s 동안 약 140° 회전으로 예측돼 박스 옆에서 늘 위험 판정 →
  turn/back/follow 전부 막혀 `stop`만 1,155회. 0.4로 되돌리니(0.8 s, 약 0.3 rad) 해소.
- **fast 5 m/s는 한 번도 선택되지 않았다**(정지시간 2.5 s → 약 15 m 앞까지 비어야 함). 실제 최고 속도는 추종 1.5 m/s.
- **라이다:** 뷰포트를 끄면 RTX 2D 라이다가 아무것도 내보내지 않는다. 로봇은 움직이는 장애물을 못 본다.
  위 표의 movers=v1 pass는 "안 마주쳤다"는 뜻일 뿐이다. 박스/벽은 base가 지도로 피하므로 영향이 없다.
- 실측 감속은 재지 못했다(급정거 장면이 없었음). 2.0 m/s² 가정이 1.5 m/s에서 맞는지 미확인.

## 4. fractal에서 할 일

```bash
# 0. 작업 트리: /home/jun/SDI-Scenario 의 로컬 상태를 건드리지 않게 별도 worktree 로 받는다
cd /home/jun/SDI-Scenario && git status --short | head      # 로컬 변경 확인만
git fetch origin b529-speedx5
git worktree add ~/SDI-Scenario-b529 origin/b529-speedx5
cd ~/SDI-Scenario-b529
diff -rq accident_scene /home/jun/SDI-Scenario/accident_scene -x results -x __pycache__ | head   # 11-10 이후 차이 확인
export CARTER_REPO=$PWD                                      # carter-run 이 이 트리를 쓰게

# 1. 전제 확인
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # 580.x 여야 한다 (595 면 라이다가 죽는다)
git diff origin/main -- levels/common/probe_scene.py | head    # 비어 있어야 함 (뷰포트 패치 없음)
curl -s localhost:8420/v1/models | head -c 200                 # gpt-oss-120b. 없으면 ~/models/serve_gpt-oss-120b.sh

# 2. 라이다가 사는지 (가장 먼저)
cd accident_scene/arena
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 bash grade_batch.sh 'probe_lidar_pB528_a1_carter_run.py' --force
grep '^PROBE' results/probe_lidar_pB528_a1.isaac.log | head    # front_pts/back_pts > 0, objs 가 가끔 > 0 이어야 함

# 3. 기준선: 원래 a1 을 공식 조건(movers v1)으로 — 공식 기록 pass 61.55 s 와 비교
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 bash grade_batch.sh 'regrade_movers_pB528_a1_carter_run.py' --force

# 4. 속도 5배 시험 (모델 없이 a1 코드로) — 영상 녹화 켬
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 CARTER_RECORD=1 bash grade_batch.sh 'speedx5t04_test_pB528_a1_carter_run.py' --force
#    필요하면 회전 2.0 판도: 'speedx5_test_pB528_a1_carter_run.py'

# 5. 4 가 pass 면 모델 실험 B-5-29 (a0,a1 → 둘 다 pass 면 a2,a3). tmux 권장
tmux new -d -s b529 'bash run_b529_sequential.sh'
cat results/turns/prompt_b5_29/sequential.log
```

- `--force`는 grade_batch가 기존 결과를 건너뛰지 않게 한다. 3절의 결과는 라이다 없는 서버의 것이므로 덮어써도 된다
  (기존 `gpu_gpt_oss_120b_pB528_*` 결과는 이 명령들과 이름이 달라 안전하다).
- 영상: `CARTER_RECORD=1` 이면 Isaac 녹화(README/REMOTE_VIEWING.md 참고). 궤적만으로 그리려면
  `$CARTER_WS/mamba/envs/isaacsim/bin/python traj_video.py <variant>_carter_run.py --movers v1 --speedup 4`
  → `results/<variant>_topdown.mp4` (시스템 ffmpeg 사용).
- a4~a7이 필요하면 `run_b528_a4a7.sh`를 같은 방식(sed로 B5_28→B5_29, pB528→pB529, v16→v17)으로 복사한다.

## 5. 열린 질문 (사용자 결정 필요)

1. fast 후보 상한 5.0 m/s는 실질적으로 쓰이지 않는다. 탈출처럼 1.5로 맞출지.
2. 라이다가 살아난 상태(fractal)에서 1.5 m/s가 움직이는 장애물(0.5~0.6 m/s)에 대해 충분히 안전한지 — 4단계 결과로 판단.
3. 감속 가정 `linear_decel_m_s2: 2.0`이 실제 Carter와 맞는지. 4단계 궤적에서 급정거 구간이 있으면 잰다.
