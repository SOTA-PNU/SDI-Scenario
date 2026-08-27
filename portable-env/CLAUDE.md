# 에이전트용 지침 — Carter / Isaac Sim 4.5 이식 패키지

이 디렉터리는 Nova Carter L0–L3 시나리오 벤치마크를 새 GPU 서버에 세팅하기 위한 패키지다.
전체 설명은 `README.md`에 있다. 아래는 반드시 지켜야 할 것만 추린 것이다.

## 설치를 요청받았을 때

```bash
bash setup/install.sh
```

이거 하나면 된다. 4단계(preflight → env → repo → smoke)를 순서대로 돌리고 실패 지점에서 멈춘다.
전부 멱등이므로 원인을 고친 뒤 그대로 다시 실행하면 된다. 사용자에게 되묻지 말고 진행하라.

설치 위치를 바꾸려면 실행 전에 `CARTER_WS` / `CARTER_REPO`를 export 한다.
**기본값은 `$HOME/carter_ws`이지 `/home/jun/carter_ws`가 아니다.**

## 검증을 요청받았을 때

```bash
bash setup/40_run_benchmark.sh      # 8회 실행 + 원본 대조, 약 20분
python3 tools/verify_parity.py      # 대조만 다시
```

## 절대 하지 말 것

- `PYTHONNOUSERSITE=1` 없이 벤치마크를 실행하지 마라. `~/.local`이 Isaac env를 가려서
  numpy가 2.x로 바뀐다. 이 패키지 스크립트들은 자동으로 설정한다.
- 벤치마크 실행 전에 `levels/*/results/`를 백업하지 않은 채로 돌리지 마라. 실행이 덮어쓴다.
  (`20_deploy_repo.sh`가 `$CARTER_WS/reference_baseline`에 읽기전용 사본을 만든다.)
- `--/rtx/verifyDriverVersion/enabled=false` 플래그를 지우지 마라. 드라이버 minor가 256 이상이면
  Vulkan이 오독해서 부팅이 막힌다.
- 시스템 ROS나 RoboStack을 source한 셸에서 실행하지 마라. ROS 2 브리지가 죽는다.
- 두 개 이상 동시에 실행하지 마라. 측정값이 달라진다.
- `wall_time_s`나 `frames`로 결과를 비교하지 마라. GPU 의존값이라 정상 재현도 실패로 보인다.
- 15 GB env를 tar로 옮기려 하지 마라. conda env는 재배치가 안 된다(경로 하드코딩). 재설치가 정답이다.

## 결과를 해석할 때

- `base_*` 실행이 **FAIL하는 것이 정상**이다. `carter-run`은 FAIL 시 exit 1로 끝나므로
  `set -e`나 `&&` 체인으로 묶지 마라.
- 판정의 진실은 항상 `results/*_result.json`의 `verdict`다. 러너를 `carter-run` 없이 직접 부르면
  **FAIL이어도 프로세스가 0으로 끝난다** (`SimulationApp.close()`가 Kit을 내리면서 코드를 덮어쓴다 —
  실측 확인). `carter-run`이 JSON에서 종료 코드를 다시 만들어 주니 항상 래퍼를 써라.
- `carter-run` 종료 코드: 0=PASS, 1=FAIL, 2=ROS 2 브리지 기동 실패(환경 결함),
  70=결과 JSON이 갱신 안 됨(실행이 도중에 죽음), 75=다른 실행이 이미 돌고 있음.

## 범위

이 패키지는 **시뮬레이터 쪽만** 다룬다. LLM/NPU는 벤치마크 루프 밖에 있고(모델이 코드 블록만
텍스트로 만들어 준다), RoboStack ros2 env와 colcon 워크스페이스는 L0–L3가 전혀 쓰지 않으므로
일부러 제외했다. 이유는 README 6절에 있다.
