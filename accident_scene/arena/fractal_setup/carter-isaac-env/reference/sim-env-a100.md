> **이 문서는 원본 리포의 orphan 브랜치 `origin/sim-env-a100`에서 그대로 가져온 것이며,
> 일부 내용이 낡았다.** 참고용으로만 읽어라. 확인된 오류:
>
> - "이 환경에서 레벨 시나리오를 실행해 본 적은 없다" — **틀렸다.** 이후 L0~L3 전부 실측
>   완료되었고 그 결과가 이 패키지의 `reference/expected_results.json`이다.
> - `sim_env/scripts/ros2_env.sh` — 그 브랜치에도, 디스크에도 **존재하지 않는다.**
> - `build_ws.sh`는 `opt/cuda` 심볼릭 링크 팜과 VPI를 전제하는데, 그걸 만드는 코드는
>   어디에도 없다. 어차피 L0~L3 벤치마크는 colcon 워크스페이스를 쓰지 않는다.
> - 설치 명령이 없다. 실제 재현 절차는 이 패키지의 `README.md` 3절이다.

---

# sim_env — Isaac Sim 실행 환경 (A100 호스트)

이 저장소의 레벨 시나리오를 **실제로 돌려볼 GPU 호스트 환경**. 루트 README 가 말하는
"아직 GPU 실측 전 — probe 하며 수렴시킨다"의 그 probe 를 실행할 수 있는 상태까지 세팅한 것이다.

Isaac Sim 4.5.0 이 Nova Carter 를 구동하고, ROS 2 브리지가 로봇을 ROS 그래프에 올리며,
`/cmd_vel` 로 제어 루프가 닫힌다. 전부 **root 권한 없이** 설치된다.

- 설치 위치: `$CARTER_WS` (기본 `/home/jun/carter_ws`, 약 24 GB). 저장소에는 스크립트만 둔다.
- 호스트: Ubuntu 22.04.5 / NVIDIA A100 80GB x2 / 드라이버 535.309.01 / RAM 251 GB

## 시나리오 인터페이스 대조표

`scenarios/*.yaml` 의 `interface.adapter_config` 요구사항 중 **시뮬레이터가 제공해야 하는 쪽**을
이 환경에서 실측한 결과. (2026-08-03, `carter_warehouse_navigation.usd`, 전 센서 on)

| adapter_config 요구 | 이 환경 | 실측 |
|---|---|---|
| `cmd_vel: /cmd_vel` (Twist) | 있음 (구독) | 10 s 주행 시 **+0.747 m** 이동 |
| `clock_topic: /clock` | 있음 | 9.2 Hz |
| `odom_topics: [/odom, /chassis/odom]` | `/chassis/odom` | 9.2 Hz (`odom` → `base_link`) |
| `/front_3d_lidar/lidar_points` (PointCloud2) | 있음 | 9.2 Hz |
| `/front_2d_lidar/scan` (LaserScan) | 있음 — **⚠ 아래** | 1.6 Hz |
| `/back_2d_lidar/scan` (LaserScan) | 있음 — **⚠ 아래** | 1.6 Hz |
| `frames: odom, base_link` | 있음 | `/tf` 36.6 Hz |
| `frames: map` | **없음** | AMCL(nav2) 이 떠야 생김 = SUT 쪽 |
| `goal_interface: /navigate_to_pose` | **없음** | nav2 액션 = SUT 쪽 |
| `readiness: /lifecycle_manager_navigation/is_active` | **없음** | SUT 쪽 |

⚠ **2D 라이다는 스톡 상태에서 나오지 않는다.** `carter_warehouse_navigation.usd` 는
`publish_{front,back}_2d_lidar_scan` 헬퍼를 `enabled=True` 로 두면서, 정작 데이터를 넣어주는
`IsaacCreateRenderProduct` 노드는 `inputs:enabled=False` 로 배포한다. 그래서 그냥 띄우면
`/front_2d_lidar/scan` 과 `/back_2d_lidar/scan` 이 **토픽 목록에 아예 없다.** 3D 라이다는 해당 속성이
없어 정상 동작하므로 눈치채기 어렵다.

L0~L3 네 레벨 전부가 이 두 토픽을 `adapter_config.sensors` 에 요구한다.
`02_run_carter_sim.py` 가 `play()` 전에 둘 다 `True` 로 켠다. 스톡 동작을 보려면 `--no-2d-lidar`.

## 아직 확인되지 않은 것

이 환경에서 **레벨 시나리오를 실행해 본 적은 없다.** 위 표는 토픽·프레임 수준의 대조일 뿐이고,
아래는 확인하지 않은 채로 남아 있다.

- **CV-Infra 러너가 이 호스트에 없다.** `sut.image_ref: carter-sut:p2` 를 띄우려면 Docker 가 필요한데
  root 권한이 없어 설치하지 못했다. `reached_goal` · `no_collision` · `max_time_to_goal` 판정은
  한 번도 돌지 않았다.
- **`ros_distro: jazzy` 와 이 환경(Humble)이 다르다.** Isaac Sim 4.5 의 `isaacsim.ros2.bridge` 는
  Humble 라이브러리만 번들한다 (`humble/lib` 238개, `jazzy` 디렉토리 없음). adapter_config 의
  `ros_distro` 가 SUT 컨테이너를 가리키는 것이라면 표준 메시지 타입은 DDS 상에서 배포판을 넘어
  통신되므로 문제되지 않을 가능성이 높다. 다만 **첫 런에서 확인이 필요하다.** 시뮬레이터 쪽을
  Jazzy 로 올리려면 Isaac Sim 5.x 가 필요하고, 그건 이 호스트 드라이버(535.309.01)보다 최신
  드라이버를 요구한다.
- **`scene: nova_carter_warehouse` 가 여기서 여는 씬과 같은지 미확인.** 이 환경은
  `Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd` 를 연다. scene 이름을 어떤 USD 로
  해석하는지는 CV-Infra 플랫폼 소유다.
- **`debug_obstacle` 주입 경로 미확인.** L2/L3 의 박스를 어느 계층이 스테이지에 넣는지 보지 않았다.
- **시작 pose `(-6.0, -1.0, yaw=pi)` 미확인.** 이 환경은 USD 기본 pose 로 로봇을 띄운다.

## 실행

```bash
# 터미널 1 — 시뮬레이터 (부팅 1~2분)
sim_env/scripts/isaac_python.sh sim_env/scripts/02_run_carter_sim.py
#   "[sim] timeline playing" 이 뜨면 준비 완료

# 터미널 2 — 토픽 확인 / 주행 검증
sim_env/scripts/03_verify_ros_topics.sh
sim_env/scripts/04_drive_test.sh 10

# 대화형 ROS 2 셸 (Nav2 · rviz2 · teleop 설치돼 있음)
source sim_env/scripts/ros2_env.sh
```

`--seconds N` 으로 자동 종료, `--webrtc` 로 헤드리스 화면 스트리밍(포트 8211).
시뮬레이터를 **동시에 2개 띄우지 말 것** — GPU 를 공유해 9.2 → 4.5 fps 로 떨어진다(실측).

| 스크립트 | 용도 |
|---|---|
| `isaac_python.sh` | Isaac Sim 환경변수 세팅 후 python 실행 |
| `ros2_env.sh` | `source` 하면 ROS 2 Humble 셸 (colcon 워크스페이스 오버레이 포함) |
| `build_ws.sh` | colcon 빌드 (CUDA/VPI 배선 포함) |
| `isaac_common.py` | 호스트별 launch config + 검증된 에셋 URL |
| `00_smoke_test.py` | 부팅 / 물리 / 렌더를 각각 따로 판정 |
| `01_inspect_carter_scene.py` | 프림 · 조인트 · 카메라 · ROS 2 그래프 노드 덤프 |
| `02_run_carter_sim.py` | **시뮬레이터 본체** |
| `03_verify_ros_topics.sh` | 토픽 목록 + 발행 주기 + odom 샘플 |
| `04_drive_test.sh` | `/cmd_vel` 발행 후 오도메트리가 실제로 움직였는지 |
| `05_inspect_lidar_nodes.py` | `ros_lidars` 그래프 상태 (2D 라이다 문제 추적용) |

## 환경 구성

- **Isaac Sim 4.5.0** — micromamba env(`isaacsim`, Python 3.10)에 pip 설치. 이 호스트 드라이버와
  Isaac ROS 3.2 / ROS 2 Humble 조합에 맞는 버전이 4.5 다.
- **ROS 2 Humble** — [RoboStack](https://robostack.github.io/) 으로 `ros2` env 에 설치.
  `ros-humble-desktop` + Nav2(31개 패키지) + xacro + colcon + 컴파일러. `apt` · root 불필요.
- **소스 워크스페이스** `$CARTER_WS/src`: `nova_carter`, `isaac_ros_common`, `isaac_ros_nova`
  (전부 `release-3.2`).
- **빌드 완료**: `nova_carter_description`(xacro → 32링크 URDF), `hawk_description`,
  `owl_description`, `pandar_xt32_description`, `rplidar_s2e_description`, `isaac_ros_common`,
  `isaac_ros_launch_utils`, `isaac_ros_test`.
- 씬 에셋은 공개 Omniverse S3 버킷에서 스트리밍. Nucleus 서버 불필요.

## 호스트 특유의 우회 4건

세팅하면서 실제로 막혔던 것들. 재발견 방지용으로 남긴다.

### 1. A100 은 공식 미지원인데 동작한다

NVIDIA 요구사항은 *"GPUs without RT Cores (A100, H100) are not supported"* 라고 명시한다. 그런데
드라이버는 A100 에서 `VK_KHR_ray_tracing_pipeline` · `VK_KHR_acceleration_structure` ·
`VK_NV_ray_tracing` 을 노출한다(`vulkaninfo` 디바이스별 확인, `VK_KHR_ray_query` 는 없음). 즉 RTX
렌더러가 레이트레이싱을 SM 에서 돌린다. 느리지만 — 전 센서 켜고 **9.2 fps** — 정상 동작하며,
Isaac Sim 은 `DLSS-RR is not supported ... with this GPU` 를 남기고 계속 진행한다.

프레임레이트가 낮은 건 버그가 아니라 RT 코어가 없는 비용이다. 시나리오의 `timeout_s` /
`max_time_to_goal_s` 를 이 호스트에서 튜닝할 때 이 점을 감안해야 한다.

### 2. Kit 이 충분히 최신인 드라이버를 거부했다

```
Installed driver: 535.53
The unsupported driver range: [0.0, 535.129)
```

실제 드라이버는 **535.309.01** 이다. Vulkan 의 `driverVersion` 은 minor 를 8비트에 담는데
`309 & 0xff == 53` 이라 Kit 이 "535.53" 으로 읽고 미달로 판단한다. 드라이버가 아니라 검사가 틀린
것이므로 모든 실행이 `--/rtx/verifyDriverVersion/enabled=false` 를 넘긴다 (`isaac_common.py`).

### 3. ROS 2 브리지가 조용히 죽는다

`isaacsim.ros2.bridge` 가 *"ROS_DISTRO env var not found"* → *"ROS2 Bridge startup failed"* 로
실패하면, USD 안의 모든 ROS 노드가 `Could not find node type interface for 'ROS2CameraHelper'` 로
죽는다. **이때 시뮬레이션은 멀쩡해 보이면서 아무것도 발행하지 않는다.**

`isaac_python.sh` 가 `ROS_DISTRO=humble` 을 설정하고 브리지의 *번들* Humble 라이브러리를
`LD_LIBRARY_PATH` 에 넣는다. RoboStack 설치본은 **일부러 source 하지 않는다** — 그래야 브리지가
내부 `rclpy` 를 로드하고(`Attempting to load internal rclpy` → `rclpy loaded`) 두 ABI 가 섞이지 않는다.
RoboStack 쪽은 DDS 로 통신하므로 `ROS_DOMAIN_ID` 와 `RMW_IMPLEMENTATION` 만 맞으면 된다.

`02_run_carter_sim.py` 는 브리지가 안 올라오면 **hard-fail** 한다. 조용히 퇴화하면 시나리오가
"센서 없음"으로 오판정될 수 있기 때문이다.

### 4. 2D 라이다가 꺼진 채 배포된다

위 "시나리오 인터페이스 대조표" 의 ⚠ 참고.

### 빌드 배선: CUDA 와 VPI

`isaac_ros_common-extras.cmake` 가 `find_package(CUDA REQUIRED)` 를 하고 모든 Isaac ROS 패키지가
이를 상속한다. 우분투의 CUDA 11.5 는 `/usr/bin/nvcc` + `/usr/lib/x86_64-linux-gnu` 로 흩어져 있어
레거시 `FindCUDA` 모듈이 못 찾는다 — `CUDA_TOOLKIT_ROOT_DIR` 가 바뀌면 캐시된 결과를(명령줄로 넘긴
`-DCUDA_CUDART_LIBRARY=` 까지) **초기화한 뒤** `$ROOT/lib64` 만 뒤지기 때문이다.
`$CARTER_WS/opt/cuda` 가 그 레이아웃을 흉내낸 심볼릭 링크 모음이다.

`isaac_ros_common` 은 NVIDIA VPI 에 링크하는 `vpi_utilities` 도 빌드하는데 VPI 는 conda 에 없다.
NVIDIA `jetson/x86_64/jammy` 아카이브의 `libnvvpi3` / `vpi3-dev` `.deb` 를 root 없이
`$CARTER_WS/opt/vpi` 에 풀어 쓴다.

`build_ws.sh` 가 둘 다 담고 있다.

## 빌드하지 않은 것

`nova_carter.repos` 의 인지(perception) 절반 — `isaac_ros_visual_slam`, `isaac_ros_nvblox`,
`isaac_ros_dnn_stereo_depth`, `custom_nitros_image`, `nvblox_nav2` 등 — 은 **빌드하지 않았다.**
NITROS/GXF 와 TensorRT 스택이 필요한데 Isaac ROS 는 이를 Docker 이미지나 NVIDIA apt 저장소로
배포하고, 이 호스트에는 root 가 없다. 그래서 `nova_carter_navigation`, `nova_carter_bringup`,
`nova_carter_docking` 은 클론만 되어 있고 빌드되지 않았다.

전체 스택이 필요하면 Docker 설치 후 `isaac_ros_common/scripts/run_dev.sh` 를 같은 `src/` 로 실행한다.
CV-Infra 러너를 이 호스트에서 돌리려면 어차피 Docker 가 필요하므로, 그 시점에 함께 해결하면 된다.
