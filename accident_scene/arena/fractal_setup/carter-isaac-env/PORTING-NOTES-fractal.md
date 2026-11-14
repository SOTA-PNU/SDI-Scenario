# 이식 기록 — `fractal` (Ubuntu 24.04 / RTX PRO 6000 Blackwell)

원본 A100 서버(Ubuntu 22.04, glibc 2.35, 드라이버 535.309.01)용으로 만들어진 이 패키지를
`fractal`에 올리면서 **호스트 차이 때문에** 걸린 것들과, 그 해결을 기록한다.
전부 패키지 README의 어떤 항목에도 해당하지 않는 새 함정이다.

| | 원본 서버 | fractal |
|---|---|---|
| OS / glibc | Ubuntu 22.04.5 / 2.35 | Ubuntu 24.04.4 / **2.39** |
| 커널 | 6.8.0-124 | **7.0.0-28** |
| GPU | A100 80GB PCIe ×2 (sm_80) | RTX PRO 6000 Blackwell 97GB (**sm_120**) |
| 드라이버 | 535.309.01 (R535) | 595.84 (R590) → **580.173.02 (R580)**, 2026-09-11 교체 |

> ### 현재 상태 (2026-09-11)
>
> 드라이버를 **580.173.02** 로 바꿨다. 이 호스트에서 확인한 것:
>
> - 595 에서 SIGSEGV 로 죽던 스톡 부팅이 **RC=0** 으로 뜬다. RTX 씬DB가 생성되고 살아남는다
>   (새 드라이버 첫 부팅 셰이더 컴파일 201 s).
> - 그래서 **1~3절의 뷰포트 우회는 이 호스트에서 더 이상 필요 없고, 되돌렸다.**
>   하네스는 원본 커밋과 바이트 단위로 같다. 1~3절은 R590/R610 드라이버 호스트를 위한 기록으로 남긴다.
> - **4절(qdldl)은 여전히 필요하다** — glibc 문제라 드라이버와 무관하다.
> - `CARTER_RECORD=1` 로 Isaac 뷰포트 녹화가 된다
>   (`SDI-Scenario/accident_scene/results/accident_base_isaac_1280.mp4`).
> - 벤치마크 재대조: **판정 8/8 일치, 17/20 matched** (595+패치 때 13/20). 남은 3건은 물리 3~4 스텝.
>   `level2/solution` 은 원본과 정확히 같은 15.25 s. 상세는 7절.

`setup/00_preflight.sh`는 전부 통과한다. 아래 셋은 프리플라이트가 볼 수 없는 층위의 문제다.

---

## 1. RTX 렌더러가 부팅 직후 SIGSEGV — 드라이버 브랜치 문제

### 증상

`SimulationApp(...)`이 `[21.502s] app ready`까지 간 뒤 `_wait_for_viewport`에서 죽는다.

```
Thread 20 "carb.tasking16" received signal SIGSEGV
#0-#3  librtx.scenedb.plugin.so
#4-#6  libcarb.scenerenderer-rtx.plugin.so
#7     libomni.hydra.rtx.plugin.so
```

`site-packages/omni/logs/Kit/Isaac-Sim Python/4.5/` 의 **로그 9개가 전부** 동일한
`SceneDbContext : TLAS instance limit` 줄에서 끝난다. 죽기 1ms 전 줄은 언제나
`HydraTexture 'omni_kit_widget_viewport_ViewportTexture_0'` 생성이다.

### 원인

GPU 세대가 아니라 **드라이버 브랜치**다.

- 같은 크래시가 **RTX 4090**(Blackwell 아님)에서도 595.71.05로 재현되고, NVIDIA 메인테이너가
  "R590(595.x) 브랜치는 Omniverse RTX 렌더러와 알려진 비호환이 있다"고 답했다
  ([IsaacSim discussion 648](https://github.com/isaac-sim/IsaacSim/discussions/648)).
- 반대로 Isaac Sim **4.5.0**이 **같은 RTX PRO 6000 Blackwell**에서 드라이버 590.48.01로
  정상 구동된 보고가 있다
  ([NVIDIA forum 366278](https://forums.developer.nvidia.com/t/366278)) — 저쪽 불만은 FPS이지 크래시가 아니다.

### 해결 — 드라이버를 바꾸지 않는다

RTX **확장 로딩은 무해하다**. Kit 로그상 `omni.hydra.rtx`는 1,219ms에 플러그인을 *등록*만 하고,
*초기화*는 1.8초 뒤 `omni.kit.widget.viewport`가 뷰포트 텍스처를 만들 때 일어난다.
그러니 **뷰포트를 만들지 않으면** RTX 씬DB가 초기화되지 않는다.

```bash
bash tools/enable_no_viewport.sh          # 적용
bash tools/enable_no_viewport.sh --revert # 원복
```

`tools/enable_offline_assets.sh`와 동일한 방식으로 `levels/common/` 의 SimulationApp 3곳을 패치한다.

**주입하는 Kit 인자**

| 인자 | 왜 |
|---|---|
| `--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true` | ViewportWindow 자체를 안 만든다 |
| `--/exts/omni.kit.viewport.window/windowMenu/entryCount=0` | `create_viewport_window_menu`가 None을 반환 |
| `--/exts/omni.kit.viewport.menubar.render/autoManage/enabledList=` | **빈 문자열이어야** auto-manage가 꺼진다. `none`은 truthy라 통과해버리고 `/renderer/enabled`를 다시 쓴다 |
| `--/renderer/enabled=none` | 백스톱. **비우면 안 된다** — `texture.py:554`가 빈 목록을 "아무 렌더러나 허용"으로 읽어 rtx를 고른다 |
| `--/rtx/hydra/mdlMaterialWarmup=false` | 쓰지 않을 MDL 워밍업 생략 |
| `--/app/extensions/excluded/0=omni.physx.ui` | 아래 2절 |

`--/renderer/active=none`은 **넣지 않는다**. `texture.py:574`가
`set_extension_enabled_immediate("omni.hydra.none", True)`로 빠진다.

**시도했다가 버린 것**

- 뷰포트 창은 두고 렌더러만 끄기 → `_wait_for_viewport`가 `viewport_handle`을 영원히 기다려 행업(RC=124 실측)
- `--/app/extensions/disableStartup/0=omni.physx.ui` → SIGSEGV(RC=139 실측)
- scenedb 튜닝 6종(`maxInstancesLimit`, `forceMaxTLASInstancesLimit`, `cudaInterop`,
  `compaction`, `useGeometryValidation`, `instanceBudget`) → 전부 SIGSEGV
- 컨테이너/다른 드라이버 유저스페이스 → 커널 모듈이 버전 일치를 강제한다. 이 호스트에서 실측:
  580 libcuda는 `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, 580 Vulkan ICD는 `VK_ERROR_INCOMPATIBLE_DRIVER`
- lavapipe 등 소프트웨어 Vulkan → `libgpu.foundation.plugin.so`에 `cmpl $0x10de` PCI 벤더 게이트가
  하드코딩돼 비-NVIDIA 디바이스를 전부 거부
- RTX 플러그인만 최신 Kit로 교체 → 5.1 바이너리가 libpython3.11 / boost_python311 / 다른 USD를
  요구한다. 링커 수준에서 불가
- isaacsim 6.0.1(595 브랜치가 검증된 유일 버전) → Kit 110.1 / omni_physics 110.1이라 물리 결과가
  구조적으로 달라진다. 대조가 목적인 벤치마크에서는 자해다

### 패리티

`sim.step(render=True)`는 **손대지 않았다**. 그건 `app.update()` 한 번으로 환원되고(`simulation_context.py:648-661`),
app.update()는 hydra 엔진 유무와 무관하게 omni.physx와 OmniGraph를 tick한다.
비교 대상 지표는 전부 물리 상태에서 읽는다(`get_world_pose`, PhysX 접촉 리포트, PhysX 레이캐스트).
기준 결과 20건 모두 `sim_time_s / frames == 1/60`이라 루프는 고정 스텝이고 벽시계가 물리에
새어 들어갈 수 없다.

부작용: `CARTER_RECORD` / `CARTER_RECORD_DIR` / `CARTER_LIVESTREAM` / `live_mjpeg_sim.py`가
동작하지 않는다. 어차피 이 호스트에서는 패치 전에도 프로세스가 먼저 죽어서 못 썼다.

---

## 2. `omni.physx.ui`가 종료를 막는다 — 뷰포트가 없을 때

### 증상

뷰포트를 없앤 뒤, 스크립트가 마지막 줄을 출력한 **뒤에** 프로세스가 안 끝나고
동일 트레이스백을 쏟아낸다. 실측: 로그 **1,130만 줄**, 20분 행업.

```
omni/physxui/scripts/extension.py:83 in on_stage_update
  camera = ViewportCameraState()
RuntimeError: No default or provided Viewport
```

`on_stage_update`는 아무 가드 없이 매 tick 호출된다.

### 원인과 해결

`/app/extensions/excluded`만으로는 못 막는다. 부팅 시점엔 제외되지만
`enable_extension("isaacsim.ros2.bridge")`가 `omni.isaac.dynamic_control` ·
`isaacsim.core.simulation_manager`를 통해 다시 끌어오고, 그 경로는 제외 목록을 보지 않는다.

그래서 브리지를 켠 **직후** 명시적으로 끈다 (패치 스크립트가 삽입):

```python
import omni.kit.app as _kit_app
_kit_app.get_app().get_extension_manager() \
    .set_extension_enabled_immediate("omni.physx.ui", False)
```

UI 전용 확장이라 `omni.physx`(시뮬레이션, 접촉 리포트)는 그대로다.
실측: 뷰포트 에러 1,898,124건 → **2건**(끄기 전 2 tick), 로그 5,690,494줄 → **841줄**.

---

## 3. `SimulationApp.close()`가 영원히 대기한다

### 증상

`PROBE COMPLETE` 출력 후 12코어를 태우며 `libomni.fabric.endframe.plugin.so`에서 돈다.
faulthandler 스택:

```
simulation_app.py:570 close()
  -> omni.replicator.core orchestrator.wait_until_complete()
    -> async_engine.run_once -> asyncio is_running
```

렌더 프로덕트가 없으니 Replicator 오케스트레이터가 완료 상태에 도달하지 못한다.

### 해결

`close()`에는 이미 파라미터가 있다 — `def close(self, wait_for_replicator=True)`.
패치 스크립트가 SimulationApp의 close 호출 4곳을 `close(wait_for_replicator=False)`로 바꾼다.
모든 레벨 러너는 `carter_env.py:576` 한 곳을 거친다.

**패리티 영향 없음**: 결과 JSON은 전부 close() 이전에 쓰인다. 순수 teardown이다.

실측: 스모크 테스트 1200초 타임아웃 → **10초, exit 0**.

---

## 4. `qdldl`이 glibc 2.39에서 로드되지 않는다 — 드라이버와 무관

### 증상

프로브가 `VERDICT: cmd_vel drive FAILED, odom sub WORKS`, GT 변위 **0.000 m**를 낸다.
odom은 오는데 로봇이 안 움직인다.

### 원인

```
Failed to import python module isaacsim.robot.wheeled_robots.
  qdldl.cpython-310-x86_64-linux-gnu.so: cannot dynamically load position-independent executable
```

`isaacsim.robot.wheeled_robots` → `holonomic_controller` → `osqp` → `qdldl`.
그 확장이 **DifferentialController OmniGraph 노드**를 제공하므로, 로딩이 실패하면
`/cmd_vel`은 구독되지만 바퀴 명령으로 변환되지 않는다.

PyPI의 `qdldl==0.1.9.post1` 휠은 `PT_INTERP`를 가진 **PIE 실행파일**로 빌드돼 있다.
glibc 2.39는 그런 오브젝트의 `dlopen`을 거부한다. 원본 서버의 glibc 2.35는 허용했다.
**버전 문제가 아니다** — `locks/requirements-isaacsim.env-only.txt`와 정확히 같은 버전이다.

### 해결

같은 버전을 소스에서 빌드한다 (gcc + cmake 필요, 둘 다 이 호스트에 있음):

```bash
MAMBA_ROOT_PREFIX=$CARTER_WS/mamba PYTHONNOUSERSITE=1 \
  $CARTER_WS/tools/bin/micromamba run -n isaacsim \
  python -m pip install --no-binary qdldl --force-reinstall --no-deps "qdldl==0.1.9.post1"
```

검증:

```
readelf -h qdldl.cpython-310-x86_64-linux-gnu.so   ->  Type: DYN (Shared object file)
readelf -l qdldl.cpython-310-x86_64-linux-gnu.so   ->  INTERP 없음
```

실측 결과: GT 변위 0.000 m → **0.628~0.638 m**, `VERDICT: cmd_vel drive WORKS`.

**env 전체 1,883개 .so를 훑어 같은 부류를 더 찾았다**: `opencv_python.libs/libQt5Core`와
CAD 컨버터의 `HEPBMP.so` 둘뿐이고, 둘 다 벤치마크 실행 경로 밖이다.

`requests` 미설치로 `isaacsim.asset.browser`도 실패하지만 UI 전용이고,
`locks/requirements-isaacsim.env-only.txt`에도 없다 — **원본 서버에서도 똑같이 실패했다**.
일부러 그대로 둔다.

---

## 새 호스트에서 다시 세팅하는 순서

```bash
tar xzf carter-isaac-env-20260827.tar.gz && cd carter-isaac-env
bash setup/install.sh
```

`setup/10_install_env.sh` 7단계를 이 이식에서 고쳐 두었기 때문에, 드라이버가 R590/R610인
호스트에서도 이 한 줄이 4단계를 끝까지 통과한다. 스톡 부팅을 먼저 시도하고 실패하면
뷰포트를 없앤 인자로 재시도한 뒤, `enable_no_viewport.sh`를 실행하라고 안내한다.

glibc가 2.36 이상이면 그 안내를 따르기 전에 qdldl을 먼저 고쳐야 한다(4절). 정리하면:

```bash
bash setup/install.sh                       # 2단계에서 "viewport suppressed" 안내가 나오면 아래를 따른다

MAMBA_ROOT_PREFIX=$HOME/carter_ws/mamba PYTHONNOUSERSITE=1 \
  $HOME/carter_ws/tools/bin/micromamba run -n isaacsim \
  python -m pip install --no-binary qdldl --force-reinstall --no-deps "qdldl==0.1.9.post1"

bash tools/enable_no_viewport.sh            # R590/R610 드라이버 호스트에서만
bash setup/install.sh                       # 다시 돌리면 3·4단계까지 통과 (멱등, 실측 23초)
bash setup/40_run_benchmark.sh
```

실측 통과 기준:

- `setup/install.sh` → `INSTALL_RC=0`, 4단계 전부 OK
- `setup/30_smoke_test.sh` → 10초 안에 exit 0, `cmd_vel drive WORKS`
- `setup/40_run_benchmark.sh` → 8회 1분 42초, **verdict 8/8 일치**, 타이밍은 5절 참고

## 이 호스트에서 건드린 것 / 안 건드린 것

**건드린 것 (2026-09-11, 사용자 요청)**

- NVIDIA 드라이버 595.84 → **580.173.02**. Isaac 화면 녹화(`CARTER_RECORD`)가 필요해서다.
  595 는 다른 관리자(jinmok)가 2026-07-05 에 설치하고 `apt-mark hold` 로 고정해 둔 것이었다.
  사용자가 직접 실행한 명령 (한 트랜잭션으로 595 16개 제거 + 580 18개 설치):

  ```bash
  sudo apt-get install -y --allow-change-held-packages \
    nvidia-driver-580-open \
    linux-modules-nvidia-580-open-7.0.0-28-generic \
    linux-modules-nvidia-580-open-7.0.0-29-generic \
    <595 패키지 16개 뒤에 '-' 를 붙여 제거>
  ```

  커널 7.0.0-28 / 7.0.0-29 **양쪽 모두** Ubuntu 사전빌드 모듈을 같이 설치했으므로 DKMS 빌드
  실패로 부팅 후 GPU 가 죽는 경우가 없다. 580 패키지에는 hold 를 걸지 **않았다**.
  작업 전 패키지 목록: `~/carter-isaac-env/nvidia-packages-before-580.txt`
  되돌리기: `sudo apt-get install -y --allow-change-held-packages nvidia-driver-595-open` 후 재부팅.

**안 건드린 것**

- 커널, X 서버, 다른 사용자 환경
- 러너 로직, 오라클, 시나리오 YAML, 기준 결과 JSON
- 드라이버 교체 후 **뷰포트 우회 패치는 되돌렸다** — 하네스(`levels/common/`)는 원본 커밋과
  바이트 단위로 같다. `enable_no_viewport.sh` 는 R590/R610 드라이버 호스트용으로 남겨 둔다

---

## 5. 대조 결과 — 판정 20/20 일치, 타이밍은 일정한 +0.23초

`bash setup/40_run_benchmark.sh` (8회, **1분 42초**. 원본은 약 20분)

| 실행 | 판정 | `time_to_goal_s` (원본 → 여기) |
|---|---|---|
| level0 / base | fail = fail | — |
| level0 / solution | pass = pass | 2.633 → 2.917 |
| level1 / base | fail = fail | — |
| level1 / solution | pass = pass | 11.85 → 12.067 |
| level2 / base | fail = fail | — |
| level2 / solution | pass = pass | 15.25 → 15.45 |
| level3 / base | fail = fail | 15.433 → 15.667 |
| level3 / solution | pass = pass | 8.283 → 8.5 |

`verify_parity.py`: **13 matched, 7 mismatched, 0 not run**.
불일치는 전부 `time_to_goal_s` / `sim_time_s`이고, `verdict` 불일치는 **0건**이다.
`collision_count`(정확 일치 필드), `final_dist_m`, `final_yaw_err_rad`는 전부 허용 오차 안이다.

### 원인 — ROS 2 전송 지연이 벽시계 기준이기 때문

델타가 지속시간에 비례하지 않고 **일정한 상수**(+0.183 ~ +0.284초)다. 프레임으로 환산하면:

| 실행 | 프레임 차 | 여기 ms/프레임 | 차 × ms/프레임 |
|---|---|---|---|
| level0 / base | **+0** | 2.35 | — |
| level0 / solution | +17 | 2.52 | 43 ms |
| level1 / base | +17 | 2.40 | 41 ms |
| level1 / solution | +14 | 2.46 | 34 ms |
| level2 / base | +14 | 2.53 | 35 ms |
| level2 / solution | +14 | 2.65 | 37 ms |
| level3 / base | +14 | 2.65 | 37 ms |
| level3 / solution | +11 | 2.66 | 29 ms |

일정한 **약 35ms의 실시간 지연**이다. `/cmd_vel`은 DDS를 거치고 그 지연은 벽시계 기준인데,
물리는 고정 스텝(1/60)이다. 원본은 프레임당 56ms라 그 35ms가 0.6프레임이었고,
여기는 프레임당 2.5ms(RT 코어 + 렌더러 없음, 21배 빠름)라 같은 35ms가 14프레임이 된다.

**`level0/base`가 대조군이다**: 목표에 도달하지 않고 60초 타임아웃까지 돌기 때문에
프레임 수가 로봇 진행과 무관하고, 그래서 차이가 **정확히 +0**이다.

즉 이건 이식 결함이 아니라, ROS 2를 루프 안에 둔 하네스 설계가 훨씬 빠른 호스트에서 드러낸
성질이다. 뷰포트 패치 때문이 아니다 — RT 코어가 있는 어떤 호스트에서든 정도만 다르게 나타난다.

`level0/solution`을 원본과 자세히 비교하면:

```
final_pose        [-7.5992, -0.9989, 3.1402] -> [-7.6062, -0.9988, 3.1402]   (7mm)
final_yaw_err_rad 0.0014 -> 0.0014
min_goal_dist_m   0.4393 -> 0.439
collision_count   0 -> 0
frames            301 -> 318      (+17)
sim_time_s        5.017 -> 5.3    (+0.283 = 17/60)
wall_time_s       16.9 -> 0.8     (비교 제외)
```

궤적의 도착점은 7mm 안에서 재현된다. 늦게 출발해서 늦게 도착할 뿐이다.

### 재현 안정성 (같은 호스트, 반복)

```
level0/solution : 2.917 / 2.900 / 2.867 / 2.883   (편차 0.050 = 3 물리 스텝)
level3/base     : 15.667 / 15.650 / 15.633        (편차 0.034 = 2 물리 스텝)
```

원본 서버 자체도 `level3/base`에서 15.433과 15.417로 1스텝 흔들렸다(README 5절).
같은 자릿수이고, 판정은 모든 반복에서 동일했다.

### 이걸로 벤치마크가 여전히 유효한가

유효하다. 사다리의 변별력은 그대로다 — L3의 12초 제한에서 base는 15.667초로 여전히 넘고
solution은 8.5초로 여전히 안 넘는다. 양쪽 마진이 3.5초씩 남는다.
`time_to_goal_s`의 절대값까지 원본과 맞춰야 한다면 실행 속도를 원본 프레임 레이트로
스로틀해야 하는데, 그건 하네스 변경이라 하지 않았다.

이 실행의 결과 원본은 `$CARTER_WS/fractal_results_20260909/` 에 보존했고
(`benchmark_run.log` 포함), `levels/*/results/` 는 동결 baseline에서 원복해 두었다.
`git status`에는 패치한 하네스 3개 파일만 뜬다.

---

## 6. 남은 자잘한 것

- `10_install_env.sh` 5단계의 `pip check` WARN은 오탐이다. 실제 출력은
  `No broken requirements found.` 인데 스크립트가 비어있지 않은 출력을 전부 경고로 찍는다.
- `10_install_env.sh` 7단계는 이 이식에서 수정했다. 스톡 부팅을 먼저 시도하고,
  실패하면 뷰포트를 없앤 인자로 재시도한 뒤 `enable_no_viewport.sh`를 안내한다.
  덕분에 `bash setup/install.sh` 한 줄이 이 호스트에서 4단계 전부 통과한다(실측 23초, RC=0).

---

## 7. 드라이버 580 재대조 (2026-09-11) — 판정 8/8, 17/20 matched

드라이버 580.173.02 + **원본 하네스**(뷰포트 우회 패치 없음, 렌더러 켜짐)로 `40_run_benchmark.sh` 재실행.
8회 5분 29초.

| 실행 | 판정 | `time_to_goal_s` 원본 → 595+패치 → **580** |
|---|---|---|
| level0 / base | fail = fail | — |
| level0 / solution | pass = pass | 2.633 → 2.917 → **2.700** |
| level1 / base | fail = fail | — |
| level1 / solution | pass = pass | 11.85 → 12.067 → **11.867** |
| level2 / base | fail = fail | — |
| level2 / solution | pass = pass | 15.25 → 15.45 → **15.25** (정확히 일치) |
| level3 / base | fail = fail | 15.433 → 15.667 → **15.45** |
| level3 / solution | pass = pass | 8.283 → 8.5 → **8.333** |

`verify_parity.py`: **17 matched, 3 mismatched** (595+패치 때 13 / 7). verdict 불일치 0건.
남은 3건은 전부 물리 스텝 3~4개 이내다:

- level0/solution `time_to_goal_s` +0.067 (4 스텝)
- level3/base `sim_time_s` +0.050, level3/solution `time_to_goal_s` +0.050 — 둘 다 **정확히 3 스텝**이고
  허용 오차 0.05 의 경계에서 부동소수점 반올림으로 `>` 에 걸린 것이다

### 5절의 설명이 그대로 맞았다

렌더러가 켜지면서 프레임당 벽시계 시간이 늘었고, 그만큼 DDS 지연이 차지하는 프레임 수가 줄었다:

| | 프레임당 벽시계 | 원본 대비 프레임 차 |
|---|---|---|
| 원본 A100 (렌더러 on, RT 코어 없음) | ~56 ms | 0 (기준) |
| fractal 595 + 우회 패치 (렌더러 off) | ~2.5 ms | +11 ~ +17 |
| **fractal 580 + 원본 하네스 (렌더러 on)** | **~15 ms** | **+1 ~ +4** |

프레임 차 × ms/프레임 은 여전히 수십 ms 대의 일정한 지연으로 환산되고, 타임아웃까지 도는
`level0/base` 는 이번에도 **정확히 +0** 이다. 즉 남은 차이도 이식 결함이 아니라 벽시계 기준
ROS 2 전송 지연이 고정 스텝 물리에 새어 드는 하네스 성질이다. 원본(56 ms)보다 이 호스트가
3.7배 빠르므로 완전히 0 이 되지는 않는다.
