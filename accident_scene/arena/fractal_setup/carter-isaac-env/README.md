# Carter / Isaac Sim 4.5.0 시나리오 검증 환경 — 이식 패키지

Nova Carter L0–L3 시나리오 벤치마크를 **원본 A100 서버에서 돌리던 그대로** 다른 GPU 서버에서
재현하기 위한 패키지다. 설치 스크립트, 버전 잠금 파일, 벤치마크 소스, 그리고 원본 서버의
**기준 측정값**이 들어 있어서 새 서버가 같은 결과를 내는지 기계적으로 대조할 수 있다.

> **이 문서를 읽는 것이 Claude(또는 다른 에이전트)라면**: 아래 "3. 설치" 순서를 그대로 따르면 된다.
> 각 단계는 멱등(idempotent)하고 실패 시 원인을 출력한다. 사용자에게 되묻지 말고 진행하되,
> **7. 반드시 지킬 것**에 적힌 규칙은 어기지 마라 — 전부 실측으로 확인된 함정이다.

---

## 1. 무엇을 재현하는가

| | |
|---|---|
| 시뮬레이터 | Isaac Sim **4.5.0.0** (micromamba env에 pip 설치, Python 3.10) |
| 로봇 / 씬 | Nova Carter, `Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd` |
| ROS 2 | Isaac Sim에 **번들된 Humble rclpy**를 in-process로 사용 (별도 ROS 설치 불필요) |
| 벤치마크 | `levels/level0..3`의 `base` / `solution` / `prompted` 러너 |
| 판정 | `reached_goal`, `no_collision`, `max_time_to_goal` 오라클 → `results/*_result.json` |
| 원본 호스트 | Ubuntu 22.04.5, A100 80GB PCIe ×2, 드라이버 535.309.01, glibc 2.35 |
| 벤치마크 소스 커밋 | `e184e99` (github.com/hyunjun1234/carter-scenario-levels, main) |

핵심 성질: **`time_to_goal_s`는 벽시계가 아니라 물리 시뮬레이션 시간**이다. PhysX가 (거의)
결정론적이므로 GPU가 달라도 사실상 같은 값이 나온다 — 다만 완전히 같지는 않다(5절 참고). 반대로 `wall_time_s`와 `frames`는 GPU에 따라 반드시 달라지므로
대조에서 제외한다 (`tools/verify_parity.py`가 알아서 처리한다).

---

## 2. 하드 요구사항

설치 전에 `setup/00_preflight.sh`가 전부 자동 점검한다. 하나라도 어기면 Isaac Sim이 아예 뜨지 않는다.

| 항목 | 요구 | 왜 |
|---|---|---|
| OS / glibc | **glibc ≥ 2.34** (Ubuntu 22.04+) | Isaac Sim 4.5 휠이 `manylinux_2_34`. Ubuntu 20.04(2.31)는 설치 자체가 불가 |
| Python | **정확히 3.10** | 4.5.0.0은 Linux용 `cp310` 휠만 배포. 3.11/3.12는 후보 없음 |
| NVIDIA 드라이버 | **≥ 535.129**, 반드시 **GL/Vulkan 유저스페이스 포함** | 아래 경고 참조 |
| Vulkan ICD | `/usr/share/vulkan/icd.d/nvidia_icd.json` 존재 | Kit의 RTX 렌더러는 Vulkan 전용이고 GPU를 이 파일로만 찾는다 |
| 시스템 라이브러리 | `libX11.so.6`, `libXext.so.6`, `libvulkan.so.1`, `libGL/libGLX/libGLdispatch` 등 | env에는 X11/GL 라이브러리가 **하나도** 없다 |
| `nvidia-smi` | PATH에 존재 | Kit이 부팅 중 직접 실행한다 |
| 디스크 | **~30 GB** (env 15 GB + 휠 캐시 7 GB) | |
| 네트워크 | S3 **HTTPS 443** + CloudFront **HTTP 80** 아웃바운드 | 씬 USD를 매 실행마다 스트리밍한다 |

> ### 가장 흔한 실패: 드라이버는 있는데 Isaac Sim이 GPU를 못 찾는 경우
> `nvidia-headless-*`, 데이터센터 모드의 `nvidia-driver-*-server`, 또는
> `NVIDIA_DRIVER_CAPABILITIES=graphics` 없이 띄운 컨테이너는 `libnvidia-gl-*`가 빠져 있다.
> 그러면 Vulkan ICD가 없고 Kit이 **디바이스를 0개**로 인식한다 — 에러는 "GPU 없음"처럼 보여서
> 원인 파악이 어렵다. `cat /usr/share/vulkan/icd.d/nvidia_icd.json`이 되는지 **먼저** 확인하라.
> Docker면 `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`로 실행할 것.
>
> 그리고 `apt install nvidia-driver-535`는 지금 Ubuntu 22.04에서 **535를 설치하지 않는다**
> (transitional 스텁이라 580을 끌어온다). 요구사항은 "535.129 이상 + GL 유저스페이스"이지
> 특정 버전이 아니다. `nvidia-driver-580` 등 실제로 존재하는 걸 설치하면 된다.

---

## 3. 설치

```bash
tar xzf carter-isaac-env-<날짜>.tar.gz
cd carter-isaac-env

# 설치 위치를 바꾸려면 여기서 export (기본: ~/carter_ws, ~/carter-scenario-levels)
# export CARTER_WS=/data/carter_ws
# export CARTER_REPO=/data/carter-scenario-levels

bash setup/install.sh
```

> 벤치마크 소스는 커밋 `e184e99` 그대로이되 **이식용 변경 2건**이 들어 있다: `CARTER_WS` 기본값을
> `/home/jun/carter_ws` → `$HOME/carter_ws`로 바꾼 것과, 원격 뷰잉 문서의 호스트 주소를 플레이스홀더로
> 바꾼 것. 둘 다 원본 서버에서는 동작이 동일하다. 상세는 `payload/carter-scenario-levels/PORTABILITY-PATCHES.md`.
> **러너 로직과 기준 결과 JSON은 손대지 않았다.**

`install.sh`는 아래 4단계를 순서대로 돌리고, 실패하면 그 지점에서 원인을 출력하고 멈춘다.
고친 뒤 그냥 다시 실행하면 된다 (완료된 단계는 건너뛴다).

| 단계 | 스크립트 | 하는 일 | 소요 |
|---|---|---|---|
| 1 | `00_preflight.sh` | 위 요구사항 전부 점검. 아무것도 바꾸지 않음 | 10초 |
| 2 | `10_install_env.sh` | micromamba + Isaac Sim env 구축, 헤드리스 부팅 확인 | 20–50분 |
| 3 | `20_deploy_repo.sh` | 벤치마크 리포 배치, **기준 결과 동결**, 래퍼 설치 | 10초 |
| 4 | `30_smoke_test.sh` | Isaac 부팅 + ROS 2 브리지 + `/cmd_vel` 주행 실증 | 3–6분 |

단계별로 따로 돌려도 된다: `bash setup/00_preflight.sh` 처럼.

### 환경이 정확히 어떻게 만들어지는가

원본 서버의 `conda-meta/history`와 설치 로그에서 그대로 복원한 명령이다:

```bash
micromamba create -y -n isaacsim -c conda-forge python=3.10 pip
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
# + 원본 설치가 빠뜨린 3개 보정 (아래 설명)
pip install typing_extensions==4.15.0 filelock==3.29.1 fsspec==2026.4.0
```

세 부분 모두 필수다:

- `--extra-index-url https://pypi.nvidia.com` — pypi.org에는 isaacsim **4.5.x가 아예 없고**
  `isaacsim-extscache-*`는 릴리스가 0개다.
- `[all,extscache]` — `[all]`만 쓰면 extscache 3개(약 4.0 GB, 용량 대부분)가 빠진다.
- `==4.5.0` — 핀이 없으면 pypi.org의 **isaacsim 6.x**가 조용히 설치된다. API가 달라서
  이 하네스는 동작하지 않는다.

**보정 3개가 왜 필요한가**: 원본 서버는 설치 시점에 `~/.local`에 그 패키지들이 이미 있어서
pip이 "Requirement already satisfied"로 건너뛰었다. 그래서 env 자체는 자기완결적이지 않고,
깨끗한 머신에서는 env의 torch가 import조차 안 된다. 이 패키지는 그 결함을 고쳐서 설치한다.

---

## 4. 실행

```bash
# 단일 시나리오
$CARTER_WS/tools/bin/carter-run levels/level0/solution_carter_run.py

# 전체 매트릭스 + 원본 대조 (base+solution 8회, 약 20분)
bash setup/40_run_benchmark.sh

# 빠른 확인 1회 (약 2분)
bash setup/40_run_benchmark.sh --quick

# prompted 변형까지 12회 (약 30분)
bash setup/40_run_benchmark.sh --full
```

`carter-run`은 리포의 `levels/run_isaac.sh`를 감싸기만 한다. 하네스를 수정하지 않으므로
`CARTER_WS`만 맞으면 기준 결과가 나온 그 호출과 동일하다. 추가로 해주는 것은
경로 주입, `PYTHONNOUSERSITE=1` 격리, 직렬 실행 강제, 결과 JSON 기반 판정 출력뿐이다.

---

## 5. 동일성 검증

```bash
python3 tools/verify_parity.py              # 전체 대조
python3 tools/verify_parity.py --level 0 3  # 특정 레벨만
python3 tools/verify_parity.py --strict     # 소수 3자리까지 일치 요구
```

비교 대상:

- **정확히 일치해야 함** — `verdict`, 각 오라클의 pass/fail, `collision_count`
- **허용 오차 내** (기본 ±0.05 s / ±0.02 m / ±0.02 rad) — `time_to_goal_s`, `sim_time_s`,
  `final_dist_m`, `final_yaw_err_rad`. 물리 스텝이 1/60초라 기본 오차는 약 3스텝이다.
- **무시** — `wall_time_s`, `frames`. GPU 의존이라 비교하면 정상 재현도 실패로 뜬다.

> **같은 호스트에서 두 번 돌려도 완전히 같은 값이 나오지는 않는다.** 원본 서버에서 실측:
> `level3/base`의 `time_to_goal_s`가 **15.433**과 **15.417**로 갈렸다 — 정확히 물리 스텝
> 1개(1/60 s) 차이다. 같은 조건에서 `level0/solution`은 두 번 모두 **2.633**으로 동일했다.
> 즉 PhysX는 "대체로" 결정론적이고, 판정이 바뀌지 않는 한 이 정도 흔들림은 정상이다.
> 기본 허용 오차 ±0.05 s는 3스텝이라 이 흔들림은 덮으면서 진짜 회귀(초 단위 차이)는 잡는다.
> **`--strict`는 이 때문에 정상 재현에서도 실패할 수 있다 — 진단용으로만 써라.**
> 판정(`verdict`)이 다르면 그건 흔들림이 아니라 실제 문제다.

### 원본 서버 기준값 (seed 42)

| 실행 | 판정 | `time_to_goal_s` | 비고 |
|---|---|---|---|
| level0 / base | **fail** | — | 60초 타임아웃까지 목표 미도달 |
| level0 / solution | pass | 2.633 | |
| level1 / base | **fail** | — | |
| level1 / solution | pass | 11.85 | |
| level2 / base | **fail** | — | |
| level2 / solution | pass | 15.25 | |
| level3 / base | **fail** | 15.433 | 목표엔 도달하나 12.0초 제한 초과 |
| level3 / solution | pass | 8.283 | |
| level0..3 / prompted | pass | 2.167 / 12.2 / 17.433 / 8.95 | `--full`에서만 실행 |

전체 20개 실행의 기준값은 `reference/expected_results.json`에 있다.

**`base`가 FAIL하는 것이 정상이다.** 사다리 구조상 `L(n).base == L(n-1).solution`이고,
`carter-run`은 판정이 FAIL이면 exit 1로 끝난다. `set -e`나 `&&` 체인으로 묶으면 정상 동작이
에러로 보인다. (러너를 `carter-run` 없이 직접 부르면 FAIL이어도 0이 나온다 — 7절 6번 참고.)

---

## 6. 이 패키지에 **일부러 안 넣은 것**

| 제외 | 이유 |
|---|---|
| `carter_ws/mamba/envs/ros2` (6.4 GB, RoboStack Humble) | L0–L3가 **전혀 쓰지 않는다**. 리포 전체에 `envs/ros2`, `/opt/ros`, `setup.bash` 참조가 0건. ROS 2는 Isaac 휠에 번들된 Humble rclpy를 in-process로 쓴다 |
| colcon 워크스페이스 (`src/`, `build/`, `install/`), `opt/cuda`, `opt/vpi` | 위와 같음. nova_carter_sim bringup 전용이고 벤치마크 실행 경로에 없다 |
| vLLM / transformers / torch 2.8 | 원본 서버 `~/.local`에 있던 **무관한 실험**이다. 벤치마크 코드에 import가 0건 |
| HuggingFace 캐시 (475 GB) | 다른 실험 것이다. 이 벤치마크는 모델 가중치를 쓰지 않는다 |
| NPU(FuriosaAI RNGD) 서빙 스택 | **이 서버에 없다.** LLM은 루프 밖에서 코드 블록만 만들어 준다 (아래) |
| 데모 영상 8개, 발표자료 pptx (약 190 MB) | 환경 재현과 무관. 필요하면 GitHub 리포에서 받으면 된다 |
| 15 GB env 통째 tarball | conda env는 재배치가 안 된다 — `bin/` 아래 26개 파일이 `/home/jun/carter_ws` 접두사를 하드코딩한다. 재설치가 더 안전하고 결과도 같다 |

### LLM / NPU는 어떻게 되는가

LLM은 **벤치마크 실행 루프 안에 없다.** 모델이 `[EDIT REGION]` 파이썬 블록을 텍스트로 만들어 주면,
`levels/common/grade_block.py`가 그걸 러너에 끼워 넣고 Isaac Sim이 실행한다. 원본에서는 별도 서버의
FuriosaAI RNGD를 OpenAI 호환 HTTP 엔드포인트로 붙여 썼다. **아무 OpenAI 호환 서버(vLLM, Ollama, TGI)로
대체 가능하고**, 시뮬레이터 재현과는 완전히 독립이다. 이 패키지는 시뮬레이터 쪽만 다룬다.

---

## 7. 반드시 지킬 것

전부 실측으로 확인된 함정이다.

1. **`PYTHONNOUSERSITE=1` 없이 실행하지 마라.** Ubuntu 22.04의 시스템 python도 3.10이라
   `~/.local/lib/python3.10/site-packages`가 env보다 **앞서** sys.path에 올라간다. 원본 서버에서
   실측: env의 numpy 1.26.4 대신 유저사이트의 2.2.6이 로드된다 (isaacsim-core는 `numpy<2.0.0` 요구).
   이 패키지의 모든 스크립트가 자동으로 설정하지만, 직접 `run_isaac.sh`를 부를 때는 직접 넣어야 한다.

2. **실행 전에 기준 결과를 백업하라.** 모든 실행이 같은 `results/` 디렉터리에
   `<variant>_result.json`을 **덮어쓴다.** `20_deploy_repo.sh`가 `$CARTER_WS/reference_baseline`에
   읽기전용 사본을 만들고 로컬 git 커밋도 남긴다. 복구: `cd $CARTER_REPO && git checkout -- levels/`

3. **동시에 두 개 이상 돌리지 마라.** GPU를 나눠 쓰면 측정값이 달라진다 (원본 실측 9.2 → 4.5 fps).
   `carter-run`이 막아 준다.

4. **`--/rtx/verifyDriverVersion/enabled=false`를 지우지 마라.** Vulkan이 드라이버 minor를 8비트에
   담기 때문에 minor ≥ 256인 드라이버(예: 535.**309**.01 → 535.53으로 읽힘)가 금지 범위에 걸린다.
   "드라이버가 최신이니까 괜찮겠지"라고 지우면 부팅이 실패한다.

5. **시스템 ROS나 RoboStack을 source한 셸에서 실행하지 마라.** 그 Humble ABI와 브리지 번들 ABI가
   섞이면 브리지가 죽는다. 증상이 고약한데, 시뮬레이터는 멀쩡해 보이면서 토픽을 아무것도 발행하지
   않는다. 하네스는 이 경우 **exit 2**로 즉시 죽도록 되어 있다.

6. **exit code는 `carter-run`을 통해서만 믿어라.** 래퍼는 결과 JSON의 판정에서 종료 코드를 다시
   만든다: `0` = PASS, `1` = FAIL(정상적인 벤치마크 결과), `2` = ROS 2 브리지 기동 실패(환경 결함),
   `70` = 결과 JSON이 이번 실행으로 갱신되지 않음(도중에 죽음 — 이전 실행 결과를 오독하지 않도록 막는다),
   `75` = 다른 실행이 이미 돌고 있음.
   러너 스크립트를 직접 부르면 **종료 코드를 믿을 수 없다.** 러너는 `sys.exit(0 if pass else 1)`로
   끝나지만 `SimulationApp.close()`가 Kit을 내리면서 프로세스를 0으로 만들어 버린다 — 원본 서버에서
   level3/base가 판정 `fail`인데도 0으로 끝나는 것을 실측했다. `micromamba run`의 문제가 아니다
   (그건 1도 7도 그대로 전달한다). **판정의 진실은 언제나 `results/*_result.json`의 `verdict`다.**

7. **`wall_time_s`로 판정하지 마라.** 원본은 RT 코어가 없는 A100이라 약 17 fps다. RT 코어가 있는
   GPU에서는 훨씬 빨라지고, 그러면 `wall_time_s`는 반드시 달라진다. 물리 시간(`time_to_goal_s`)만 본다.

8. **첫 부팅은 느리다.** 셰이더 컴파일 때문에 최초 1회는 약 127초, 이후는 51–56초다. 셰이더 캐시는
   GPU+드라이버 해시로 키가 잡히므로 새 GPU에서는 반드시 한 번 다시 컴파일한다.

---

## 8. 오프라인 / 네트워크 제한 서버

기본 경로는 매 실행마다 NVIDIA S3에서 씬 USD를 스트리밍한다. `get_assets_root_path()`는
`<root>/Isaac`과 `<root>/NVIDIA`를 둘 다 stat한 뒤에야 값을 돌려주고, 실패하면
`RuntimeError("Could not find assets root folder")`로 죽는다. 폴백은 없다.
`~/.cache/ov`를 복사해 가도 소용없다 — 그건 콘텐츠 주소 기반 HTTP 캐시라 오프라인에서
`ERROR_CONNECTION`을 낸다 (실측 확인).

인터넷이 막힌 서버라면 로컬 미러를 만들어야 한다 (약 2.3 GB, 전체 54 GB 중 필요한 부분만):

```bash
# boto3는 이미 env 안에 있으므로 추가 설치가 없다
$CARTER_WS/tools/bin/micromamba run -n isaacsim \
    python tools/mirror_assets.py /data/isaac-assets

bash tools/enable_offline_assets.sh /data/isaac-assets
export CARTER_ASSET_ROOT=/data/isaac-assets
```

`enable_offline_assets.sh`는 `SimulationApp`을 만드는 3개 파일에
`--/persistent/isaac/asset_root/default=...`를 추가한다 (Kit은 sys.argv를 읽지 않아서 이 경로뿐이다).
`CARTER_ASSET_ROOT`가 비어 있으면 패치 전과 동작이 같고, `--revert`로 되돌릴 수 있다.

---

## 9. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| 부팅이 GPU를 못 찾음 / Vulkan 디바이스 0개 | 드라이버에 GL 유저스페이스 없음 | `00_preflight.sh` 4번 항목. 풀 드라이버 설치 |
| `EOFError: EOF when reading a line` | EULA 프롬프트 (`OMNI_KIT_ACCEPT_EULA` 미설정) | `carter-run` 또는 `run_isaac.sh`로 실행하라 |
| `RuntimeError: Could not find assets root folder` | 에셋 서버 도달 불가 | 8절 오프라인 미러 |
| exit 2 | ROS 2 브리지 기동 실패 | 깨끗한 셸에서 재실행 (시스템 ROS source 금지) |
| exit 70 | 결과 JSON이 이번 실행으로 갱신되지 않음 (실행이 도중에 죽음) | 위쪽 Kit 로그를 봐라. 이전 실행 결과를 오독하지 않도록 일부러 막은 것이다 |
| exit 75 | 다른 Isaac 실행이 이미 돌고 있음 | 끝날 때까지 기다려라. 동시 실행은 측정값을 바꾼다 |
| numpy 2.x가 로드됨 | 유저사이트 shadowing | `PYTHONNOUSERSITE=1` |
| `micromamba: command not found` | PATH에 없음 | 항상 `$CARTER_WS/tools/bin/micromamba` 절대경로 사용 |
| 기준 결과가 사라짐 | 실행이 덮어씀 | `git checkout -- levels/` 또는 `$CARTER_WS/reference_baseline` |
| `No installation candidate` (isaacsim) | python이 3.10이 아니거나 glibc < 2.34 | `00_preflight.sh` 1번 항목 |
| 첫 실행이 유난히 느림 | 셰이더 컴파일 | 정상. 2회차부터 빨라진다 |

Kit 로그 위치: `$CARTER_WS/mamba/envs/isaacsim/lib/python3.10/site-packages/omni/logs/Kit/`

---

## 10. 패키지 구성

```
carter-isaac-env/
├── README.md                     이 문서
├── CLAUDE.md                     에이전트용 요약 규칙
├── bin/micromamba                micromamba 2.8.1 (원본과 동일 바이너리)
├── setup/
│   ├── install.sh                전체 설치 (1→4단계)
│   ├── lib.sh                    공통 경로·버전 정의
│   ├── 00_preflight.sh           호스트 점검
│   ├── 10_install_env.sh         Isaac Sim env 구축
│   ├── 20_deploy_repo.sh         리포 배치 + 기준 결과 동결
│   ├── 30_smoke_test.sh          부팅/브리지/주행 실증
│   └── 40_run_benchmark.sh       매트릭스 실행 + 대조
├── tools/
│   ├── carter-run                실행 래퍼 (설치 시 $CARTER_WS/tools/bin으로 복사)
│   ├── verify_parity.py          원본 대조
│   ├── mirror_assets.py          오프라인 에셋 미러
│   └── enable_offline_assets.sh  로컬 에셋 루트로 전환
├── locks/
│   ├── requirements-isaacsim.env-only.txt   env 실제 내용 80개 (유저사이트 오염 제외)
│   ├── conda-isaacsim.explicit.txt          conda 28개 explicit 잠금
│   ├── conda-ros2.explicit.txt              (선택) RoboStack bringup용 812개
│   └── conda-*.yaml
├── payload/carter-scenario-levels/          벤치마크 소스 (커밋 e184e99, 미디어 제외)
├── payload/COMMIT                           소스 커밋 SHA
├── payload/carter-scenario-levels/PORTABILITY-PATCHES.md   원본 커밋 대비 변경 2건 (아래)
├── reference/
│   ├── expected_results.json     원본 서버 기준값 20건
│   ├── host-quirks.md            "요구사항 vs 그 서버 사정" 구분표
│   ├── source-host-snapshot.txt  원본 하드웨어/OS 스냅샷
│   └── sim-env-a100.md           원본 환경 문서 (낡음 — 보정 주석 포함)
└── logs/
    ├── isaacsim_install.log          원본 pip 설치 로그 (출처 증빙)
    └── isaacsim_conda_history.txt    원본 env 생성 명령 원문
```
