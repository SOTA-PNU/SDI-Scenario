# 인계: B-5-29 (B-5-28 + 로봇 속도 5배) — fractal 새 계정에서 처음부터

작성 2026-11-14, 브랜치 `b529-speedx5`. 다른 서버(`SOTA-Furiosa-RNGD`, RTX PRO 6000, 드라이버 595)에서 준비했지만
**그 서버에서는 Isaac의 라이다가 동작하지 않아 끝까지 검증하지 못했다**(2절). 드라이버 580인 fractal의
**아무것도 설치되지 않은 새 계정**에서 이어 한다. jun 계정의 파일·서비스는 남의 것이므로 읽기만 하고 바꾸지 않는다.

> **이 문서를 읽는 것이 Claude라면:** 3절(환경 구축) → 4절(실험) 순서로 진행한다. 각 단계의 기대값과 다르면 다음 단계로
> 가지 말고 사용자에게 보고한다. sudo는 쓰지 않는다(필요해 보이면 사용자에게 넘긴다). 남의 서비스(jun의 vLLM 등)를
> 쓰거나 65 GB 모델을 받기 전에는 사용자에게 묻는다. 긴 작업(설치, 다운로드, 모델 기동, Isaac)은 tmux 또는 백그라운드로
> 돌리고 로그는 `~/logs/`에 둔다.
>
> 사용자가 시작할 때 쓸 문장: "SDI-Scenario `b529-speedx5` 브랜치의 `accident_scene/arena/HANDOFF_B529.md`를 읽고 3절부터 진행해 줘"

## 1. 무엇을 바꿨나

사용자 요청: "카터의 기본 속도가 너무 느렸다. 모든 속도를 5배로" → "탈출은 직진 속도와 일치" →
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
| 제자리 회전 `turn_l`/`turn_r` | ±0.4 | **±0.4 (유지)** — 2.0이면 끼임, 2절 |

거리 여유, 예측 시간(`_BASE_MAP_HORIZON_CONFIG`, `linear_decel_m_s2: 2.0`), 물체 추적 설정은 그대로다.
프롬프트는 세 줄만 바뀌었고 길이는 B-5-28과 같은 11,887자다.

이 브랜치의 파일:
- `PROMPT_B5_29.md`, `base_carter_run_ros_map_lidartime_objects_check_v17.py`(머리 주석에 변경 요약), `run_b529_sequential.sh`
  (`run_b528_sequential.sh`에서 프롬프트/base/TAG `gpu_gpt_oss_120b_pB529`/폴더 `results/turns/prompt_b5_29`만 바꿈)
- `traj_video.py` — 궤적 CSV로 위에서 본 MP4(H.264)를 그린다(Isaac 녹화가 안 될 때)
- `fractal_setup/` — 환경 구축용(3절): `carter-isaac-env/`(fractal 이식 수정이 반영된 Isaac 설치 패키지, 저장소 루트의
  `portable-env/*.tar.gz`(08-27)보다 최신), `setup_vllm.sh`, `serve_gpt-oss-120b_gpu.sh`
- 시험용 제출 사본(채점에만 씀, 모델 산출물 아님):
  `speedx5_test_pB528_a1`(a1 코드 + v17 + follow 식 5배, 제자리 회전 2.0), `speedx5t04_test_pB528_a1`(같고 제자리 회전 0.4 = 현재 v17),
  `regrade_crojjang_pB528_a1`·`regrade_movers_pB528_a1`(원래 a1 그대로), `probe_lidar_pB528_a1`(a1 + 2 s마다 라이다 점 개수 출력)
- `accident_scene/` 전체는 원래 `main`에 없던 fractal 로컬 작업의 2026-11-10 백업본이다(큰 결과 묶음 `results/turns/`,
  `frames_*` 등은 없음). `levels/common/carter_env.py`는 fractal의 미커밋 수정(`CARTER_CONTACT_ALL_LINKS`)이다.
  뷰포트 우회 패치(`enable_no_viewport.sh`)는 **들어 있지 않다** — fractal(580)에서는 쓰지 않는다.

## 2. 다른 서버에서 확인한 것 (드라이버 595 + 뷰포트 끔)

모든 채점은 `CARTER_CONTACT_ALL_LINKS=1`. 결과는 `results/<variant>_result.json`.

| variant | movers | 판정 | 도달 | 최종 거리 | 충돌 | 비고 |
|---|---|---|---|---|---|---|
| `regrade_crojjang_pB528_a1` | 끔 | pass | 58.4 s | 0.4625 | 0 | fractal 공식 기록 61.55 s |
| `regrade_movers_pB528_a1` | v1 | pass | 58.4 s | 0.4625 | 0 | 위와 소수점까지 같음 → 라이다 의심 |
| `speedx5_test_pB528_a1` | 끔 | **fail** | — | 12.2 | 0 | 15 s에 박스 05 오른쪽에 붙어 600 s까지 정지 |
| `speedx5t04_test_pB528_a1` | v1 | pass | **25.2 s** | 0.097 | 0 | 평균 1.33, 최고 1.50 m/s |
| `probe_lidar_pB528_a1` | v1 | pass | 58.65 s | 0.469 | 0 | **앞·뒤 2D 라이다 0점, 추적 물체 0** |

- **제자리 회전 2.0 rad/s 끼임:** chooser는 후보를 `0.5 s + 정지시간`(감속 가정 2.0 m/s², 2.68 rad/s²) 동안 굴려 0.2 m 여유를 본다.
  2.0 rad/s 제자리 회전은 1.25 s 동안 약 140° 회전으로 예측돼 박스 옆에서 늘 위험 → turn/back/follow가 다 막혀 `stop`만 1,155회.
  0.4로 되돌리니(0.8 s, 약 0.3 rad) 해소.
- **fast 5 m/s는 한 번도 선택되지 않았다**(정지시간 2.5 s → 약 15 m 앞까지 비어야 함). 실제 최고 속도는 추종 1.5 m/s.
- **라이다:** 뷰포트를 끄면 RTX 2D 라이다가 아무것도 내보내지 않는다. 그 서버의 movers=v1 pass는 "안 마주쳤다"는 뜻일 뿐이다.
  박스/벽은 base가 지도로 피하므로 영향 없음.
- 실측 감속은 재지 못했다(급정거 장면 없음).

## 3. fractal 새 계정 환경 구축

fractal 사양(2026-11-10 스냅샷): Ubuntu 24.04(glibc 2.39), RTX PRO 6000 97 GB, 드라이버 580.173.02, `/` 여유 약 639 GB,
`/snap/bin/uv`, `/usr/bin/python3.12`, apt `ffmpeg`, apt `nvidia-cuda-toolkit`(CUDA **12.0** — sm_120 불가라 쓰지 않음).
jun은 같은 호스트에서 vLLM(127.0.0.1:8420, gpt-oss-120b)과 20b 서비스(8410, 라우터 8400)를 돌리곤 했다.

### 3-A. 사전 확인 (읽기만)

```bash
id; echo $HOME; df -h $HOME | tail -1                       # 필요: 코드 1 GB + Isaac 25 GB + (vLLM env 10 GB) + (모델 65 GB)
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader   # 580.x 여야 함. 595 면 멈추고 보고
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ps -o user=,pid=,args= -p $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd,) 2>/dev/null | cut -c1-150
ss -ltn | grep -E ':(8400|8410|8420) '                      # 누가 이미 서빙 중인지
for t in git uv python3.12 tmux ffmpeg gcc cmake curl; do printf '%-11s %s\n' $t "$(command -v $t || echo MISSING)"; done
ls /usr/share/vulkan/icd.d/nvidia_icd.json
```

### 3-B. 코드 받기 (공개 저장소라 토큰 불필요)

```bash
mkdir -p ~/logs
git clone -b b529-speedx5 https://github.com/SOTA-PNU/SDI-Scenario.git ~/SDI-Scenario
cd ~/SDI-Scenario && git log --oneline -3
```

### 3-C. LLM 서버 (gpt-oss-120b, OpenAI 호환, 127.0.0.1:8420)

실험 스크립트는 `http://127.0.0.1:8420/v1`, 모델 이름 `gpt-oss-120b`를 부른다. 같은 호스트의 localhost 포트는 계정과 무관하게 접근된다.

1. **이미 떠 있는지:** `curl -s 127.0.0.1:8420/v1/models` 가 `gpt-oss-120b`를 돌려주면(jun의 서버일 가능성이 큼) **사용자에게
   그 서버를 써도 되는지 묻는다.** 허락되면 3-C는 끝(GPU에 같은 모델을 두 번 올릴 메모리는 없다).
2. **모델 파일 찾기**(직접 띄워야 할 때):
   ```bash
   ls -l /home/jun/models/gpt-oss-120b/config.json 2>&1          # fractal 원본 위치. 읽히면 그걸 쓴다
   find / -xdev -maxdepth 6 \( -name 'models--openai--gpt-oss-120b' -o -name 'gpt-oss-120b' \) -type d 2>/dev/null | head
   ```
   읽을 수 있는 디렉터리가 있으면 `mkdir -p ~/models && ln -s <그 경로> ~/models/gpt-oss-120b`
   (HF 캐시 형식이면 `.../snapshots/<rev>`를 가리킨다. 안에 `config.json`과 `model-0000*-of-00014.safetensors` 15개가 있어야 한다).
   못 찾으면 **사용자 승인 후** 받는다(65 GB, 공개·토큰 불필요, 연구실 망 보호를 위해 동시 1개):
   ```bash
   ~/venvs/vllm/bin/hf download openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
     --local-dir ~/models/gpt-oss-120b --max-workers 1 \
     --include 'model-*-of-00014.safetensors' --include model.safetensors.index.json --include config.json \
     --include generation_config.json --include chat_template.jinja --include special_tokens_map.json \
     --include tokenizer.json --include tokenizer_config.json      # 22 파일, 65,276,851,099 B (3. 다음에 실행)
   ```
3. **vLLM 환경:** `bash accident_scene/arena/fractal_setup/setup_vllm.sh > ~/logs/setup_vllm.log 2>&1` (몇 분, 약 8 GB).
   끝에 `cuda True (12, 0)`와 `OK`가 나와야 한다. 이 스크립트가 왜 nvcc를 13.0.88로 고정하고 CUDA_HOME 링크 트리를 만드는지는 머리 주석 참고.
4. **기동:**
   ```bash
   tmux new -d -s vllm120b 'bash ~/SDI-Scenario/accident_scene/arena/fractal_setup/serve_gpt-oss-120b_gpu.sh > ~/logs/vllm120b.log 2>&1'
   until curl -sf 127.0.0.1:8420/v1/models >/dev/null || grep -q Error ~/logs/vllm120b.log; do sleep 5; done; tail -3 ~/logs/vllm120b.log
   ```
   1~2분. 8420이 다른 용도로 막혀 있으면 `PORT=8421`로 띄우고 `run_b529_sequential.sh`의 `--base` 포트를 바꾼다.

### 3-D. Isaac Sim 4.5 (`~/carter_ws`, 약 25 GB, sudo 불필요)

```bash
cd ~/SDI-Scenario/accident_scene/arena/fractal_setup/carter-isaac-env
CARTER_WS=$HOME/carter_ws CARTER_REPO=$HOME/SDI-Scenario bash setup/install.sh > ~/logs/isaac_install.log 2>&1; tail -5 ~/logs/isaac_install.log
# glibc 2.39 에서는 qdldl 휠이 로드되지 않아 로봇이 안 움직인다(PORTING-NOTES-fractal.md 4절). 소스에서 다시 빌드:
MAMBA_ROOT_PREFIX=$HOME/carter_ws/mamba PYTHONNOUSERSITE=1 $HOME/carter_ws/tools/bin/micromamba run -n isaacsim \
  python -m pip install --no-binary qdldl --force-reinstall --no-deps "qdldl==0.1.9.post1"
CARTER_WS=$HOME/carter_ws CARTER_REPO=$HOME/SDI-Scenario bash setup/install.sh > ~/logs/isaac_install2.log 2>&1; tail -5 ~/logs/isaac_install2.log
```

- 첫 `install.sh`는 qdldl 때문에 4단계(스모크 테스트, `cmd_vel drive FAILED`)에서 실패할 수 있다. 정상이다 — qdldl을 고친 뒤 두 번째 실행.
- 기대값(두 번째 실행): 4단계 모두 OK, 스모크 테스트 `VERDICT: cmd_vel drive WORKS, odom sub WORKS`.
- 3단계는 `already deployed: ~/SDI-Scenario`로 지나간다(payload를 복사하지 않는다).
- **`tools/enable_no_viewport.sh`는 실행하지 않는다.** 2단계에서 "viewport suppressed" 안내가 나오면 드라이버가 580이 아니라는
  뜻이니 멈추고 보고한다(그 패치를 쓰면 라이다가 죽는다 — 2절).
- 첫 부팅은 셰이더 컴파일로 수 분 걸릴 수 있다(fractal 580 첫 부팅 201 s).

### 3-E. 공용 서버에서 지킬 것

```bash
export ROS_DOMAIN_ID=37        # carter-run 기본값 0. jun 등 다른 Isaac/ROS 프로세스와 DDS 가 섞이지 않게 0 이 아닌 값
```
- `agent_loop.py`는 `carter_run_locked.sh`로 Isaac을 돌리고, 잠금 파일 `/tmp/carter_run_serial.lock`으로 **호스트 전체에서 한 번에
  하나만** 돌게 한다(동시 실행은 타이밍과 판정을 바꾼다). 그 파일이 다른 계정 소유라 열리지 않으면(`Permission denied`)
  `pgrep -af carter-run`으로 다른 사람이 Isaac을 돌리는지 확인하고, 아무도 없을 때만 `export CARTER_LOCK=/tmp/carter_run_serial_$USER.lock`.
- 시작 전 GPU 여유가 10,000 MiB 미만이면 러너가 거부한다(exit 75). vLLM 0.80 + Isaac 약 6 GB 조합은 맞는다.
- `traj_video.py`는 시스템 `ffmpeg`(libx264)를 쓴다.

## 4. 실험 순서

```bash
cd ~/SDI-Scenario/accident_scene/arena
export ROS_DOMAIN_ID=37

# 1. 라이다가 사는지 (가장 먼저) — front_pts/back_pts > 0, objs 가 가끔 > 0 이어야 한다
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 bash grade_batch.sh 'probe_lidar_pB528_a1_carter_run.py' --force
grep '^PROBE' results/probe_lidar_pB528_a1.isaac.log | head

# 2. 기준선: 원래 a1 을 공식 조건(movers v1)으로 — 공식 기록 pass 61.55 s, 0.467 m 와 비교
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 bash grade_batch.sh 'regrade_movers_pB528_a1_carter_run.py' --force

# 3. 속도 5배 시험 (모델 없이 a1 코드로), Isaac 녹화 켬 (REMOTE_VIEWING.md 참고)
ARENA_MOVERS=v1 CARTER_CONTACT_ALL_LINKS=1 CARTER_RECORD=1 bash grade_batch.sh 'speedx5t04_test_pB528_a1_carter_run.py' --force
$HOME/carter_ws/mamba/envs/isaacsim/bin/python traj_video.py speedx5t04_test_pB528_a1_carter_run.py --movers v1 --speedup 4   # 궤적 영상

# 4. 3 이 pass 면 모델 실험 B-5-29 (a0,a1 → 둘 다 pass 면 a2,a3). 3-C 의 서버가 떠 있어야 한다
tmux new -d -s b529 'cd ~/SDI-Scenario/accident_scene/arena && ROS_DOMAIN_ID=37 bash run_b529_sequential.sh'
cat results/turns/prompt_b5_29/sequential.log
```

- `--force`: grade_batch가 기존 결과를 건너뛰지 않게 한다. 2절 결과는 라이다 없는 서버의 것이라 덮어써도 된다
  (`gpu_gpt_oss_120b_pB528_*` 공식 결과는 이름이 달라 안전).
- 결과는 2절 표와 같은 형식으로 정리해 이 문서(또는 `REPORT.md` "실험 55")에 추가하고 사용자에게 보고한다.
- a4~a7이 필요하면 `run_b528_a4a7.sh`를 sed로(B5_28→B5_29, pB528→pB529, b5_28→b5_29, v16→v17) 복사해 쓴다.
- 결과를 push할 때 이 저장소는 **공개**다. 토큰·IP·개인정보가 없는지 확인하고 `main`이 아닌 이 브랜치에 올린다.

## 5. 열린 질문 (사용자 결정 필요)

1. fast 후보 상한 5.0 m/s는 실질적으로 쓰이지 않는다. 탈출처럼 1.5로 맞출지.
2. 라이다가 살아난 상태에서 1.5 m/s가 움직이는 장애물(0.5~0.6 m/s)에 대해 충분히 안전한지 — 4절 3단계로 판단.
3. 감속 가정 `linear_decel_m_s2: 2.0`이 실제 Carter와 맞는지 — 4절 3단계 궤적에 급정거 구간이 있으면 잰다.
