# 원본 서버(A100) 특이사항 — 어디까지가 요구사항이고 어디부터가 그 서버 사정인가

새 서버에서 재현할 때 "원본이 이렇게 했으니 나도 그래야 한다"와 "그건 그 서버 사정이다"를
구분하기 위한 문서다. 전부 실측으로 확인된 내용이다.

## 그 서버 사정 — 따라 하지 않아도 되는 것

| 원본의 선택 | 이유 | 새 서버에서는 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=0` 고정 | **GPU1이 Isaac 부팅 중 segfault** (URDF importer 확장) | 아무 GPU나 써도 된다. 단 **한 장만** 쓸 것 |
| WebRTC 라이브 스트리밍 포기, MJPEG 사용 | A100은 **NVENC가 0개**라 네이티브 WebRTC가 원리적으로 불가 | RTX/L40/A40 계열이면 WebRTC가 더 나은 경로다 |
| 약 17 fps 스테핑 | A100은 **RT 코어가 없어** RTX 레이트레이싱이 SM에서 돈다 | RT 코어가 있으면 훨씬 빠르다. `wall_time_s`가 달라지는 건 정상 |
| 경로가 `/home/jun/carter_ws` | 그 서버 계정 이름 | `CARTER_WS`로 아무 데나. 세 파일(`run_isaac.sh`, `carter_env.py`, `probe_scene.py`)이 이 변수를 각자 읽는다 |
| `~/.local`에 vLLM/torch 2.8 스택 | 같은 서버에서 돌린 **무관한 실험** | 없는 게 낫다. 있으면 `PYTHONNOUSERSITE=1`로 차단 (스크립트가 자동 처리) |
| 발표자료 pptx, 데모 mp4 커밋 | 공유 목적 | 환경 재현과 무관 |

## 진짜 요구사항 — 반드시 지켜야 하는 것

| 항목 | 근거 |
|---|---|
| `--/rtx/verifyDriverVersion/enabled=false` | Vulkan이 driverVersion의 minor를 8비트에 담아서 minor >= 256인 드라이버를 오독한다. 535.**309**.01 이 "535.53"으로 읽혀 금지 범위 `[0.0, 535.129)`에 걸린다. 드라이버가 새것이어도 **지우면 안 된다** |
| 시스템 ROS / RoboStack을 source하지 않기 | 그 Humble ABI와 Isaac 번들 Humble ABI가 섞이면 브리지가 죽는다. `run_isaac.sh`에도 같은 경고가 주석으로 있다 |
| 직렬 실행 | 동시 실행 시 원본 실측 9.2 -> 4.5 fps. 측정값이 달라진다 |
| env 디렉터리 쓰기 권한 | Kit이 cooked-mesh/material 캐시를 `site-packages/omni/cache/` **안에** 쓴다. 읽기전용이면 매 부팅 재컴파일하거나 실패한다 |
| 아웃바운드 네트워크 | 씬 USD를 매 실행 스트리밍. `~/.cache/ov`를 복사해 가도 오프라인에서는 `ERROR_CONNECTION`이다 (실측) |

## 측정된 시간

| 구간 | 값 |
|---|---|
| 최초 부팅 (셰이더 콜드) | 127.1 초 |
| 이후 부팅 (31회 관측) | 51.2 - 56.5 초 |
| 부팅 -> 씬 로드 완료 | 약 75 초 |
| 부팅 -> 브리지/rclpy 준비 | 약 80 초 |
| 텍스처 캐시 최초 생성 | 약 7.7 분 |

셰이더 캐시는 GPU+드라이버 해시로 키가 잡히므로 **새 GPU에서는 반드시 한 번 재컴파일**한다.
텍스처 캐시(`~/.cache/ov/texturecache`, 1.3 GB)는 GPU 무관한 BCn 데이터라 옮겨도 유효하다.

## 결과 판정에서 쓰는 값 / 안 쓰는 값

`results/*_result.json`의 `metrics` 중:

- **비교한다** — `time_to_goal_s`, `sim_time_s`, `final_dist_m`, `final_yaw_err_rad`,
  `final_pose`, `min_goal_dist_m`, `collision_count`. 전부 물리 시뮬레이션 값이라 PhysX
  결정론에 의해 재현된다.
- **비교하지 않는다** — `wall_time_s`, `frames`. 벽시계/렌더 의존값이라 GPU가 다르면 반드시 달라진다.

`seed: 42`는 결과 JSON에 기록만 되고 실제로 어떤 난수도 시드하지 않는다. 재현성은 전적으로
PhysX 결정론에서 온다.

## 실측 재현 지터 (2026-08-27, 원본 서버에서 확인)

| 실행 | 1회차 | 2회차 | 차이 |
|---|---|---|---|
| `level0/solution` `time_to_goal_s` | 2.633 | 2.633 | 0 (완전 일치) |
| `level3/base` `time_to_goal_s` | 15.433 | 15.417 | 0.0167 s = 물리 스텝 1개 |
| `level3/base` `final_dist_m` | 0.2745 | 0.2796 | 0.005 m |

판정(`verdict`)은 두 번 다 동일했다. `verify_parity.py`의 기본 허용 오차
(±0.05 s / ±0.02 m)는 이 지터를 덮도록 잡은 값이다. **완전 일치를 기대하지 마라.**
