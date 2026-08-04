# QUICKSTART — 직접 돌려보고, 직접 풀어보기

레벨 사다리(L0→L3)를 손으로 재현하고, LLM 대신 직접 `[EDIT REGION]`을 고쳐 레벨을
통과시켜 보는 가이드. 모든 명령은 **레포 루트**에서 실행한다.

## 0. 전제

- Isaac Sim 4.5 micromamba 환경이 `~/carter_ws` 에 있는 호스트 (이 A100 서버 기준.
  다른 위치면 `export CARTER_WS=<경로>`).
- 서버에는 클론이 이미 있다: `/tmp/cv-infra-carter-levels`. 없으면:
  `git clone https://github.com/SOTA-PNU/SDI-Scenario.git`
- **GPU0만 사용** (이 호스트의 GPU1은 Isaac 부팅 crash — `run_isaac.sh` 기본값이 GPU0).
  **동시에 한 런만** 실행.
- ⚠ **판정의 정본은 `results/*_result.json` 의 `verdict`** — exit code 가 아니다
  (Kit 종료 핸들러가 exit code 를 0 으로 덮는 경우가 있음).

## 1. FAIL/PASS 재현 (레벨 0부터)

```bash
cd /tmp/cv-infra-carter-levels
export ROS_DOMAIN_ID=17     # 다른 ROS 프로세스와 격리용 (아무 값)

bash levels/run_isaac.sh levels/level0/base_carter_run.py       # ~1.5분 → FAIL
bash levels/run_isaac.sh levels/level0/solution_carter_run.py   # ~1.5분 → PASS
```

긴 부팅 로그가 흐른 뒤 마지막에 이렇게 끝나면 성공:

```
===== RESULT =====
{ ... "verdict": "pass" ... }
===== VERDICT: PASS =====
```

결과 원본: `levels/level0/results/{base,solution}_result.json` (+ 궤적 CSV).
같은 방식으로 level1~3 을 돌리면 각 REPORT.md 의 표와 같은 숫자가 나온다
(seed 42 고정 — 소수 3~4자리까지 재현됨).

## 2. 직접 풀어보기 (LLM 역할 체험)

레벨 1을 예로:

```bash
# 1) base 를 내 답안 파일로 복사 — 파일명 규칙: <이름>_carter_run.py
#    (결과가 results/<이름>_result.json 으로 저장된다)
cp levels/level1/base_carter_run.py levels/level1/my_carter_run.py

# 2) my_carter_run.py 의 [EDIT REGION] ~ [END EDIT REGION] 블록"만" 수정한다.
#    - 계약: controller(t, pose, env) -> (v, w, done)
#      t: sim 초 / pose: (x, y, yaw) GT / v: 전진 m/s / w: yaw rad/s / done: 종료 선언
#    - 장애물 레벨(L2+)에선 env.raycast_scan() 으로 61빔 평면 스캔을 쓸 수 있다.
#    - GOAL 등 상단 givens 와 하단 하네스는 수정 금지.

# 3) 규칙 위반 검사 (수정영역 밖을 건드렸는지 — GPU 불필요, 즉시)
python3 levels/common/check_edit_region.py levels/level1/my_carter_run.py

# 4) 실행 + 채점
bash levels/run_isaac.sh levels/level1/my_carter_run.py
python3 -c "import json; print(json.load(open('levels/level1/results/my_result.json'))['verdict'])"
```

막히면 그 레벨의 `REPORT.md` 에 정답 키(base→solution diff)와 실패 원인 분석이 있다.
스포일러 없이 힌트만 원하면 각 레벨 YAML 상단 주석(미션 정의)까지만 볼 것.

## 3. 검증 스크립트 3종 (GPU 불필요, 수 초)

```bash
python3 levels/common/check_edit_region.py levels/level0 levels/level1 levels/level2 levels/level3
python3 levels/common/check_ladder.py      # levelN/base == levelN-1/solution (사다리 규칙)
python3 levels/common/check_yaml_sync.py   # scenarios/*.yaml 본문 == runner givens
```

## 4. 궤적 그림 다시 그리기

```bash
MAMBA_ROOT_PREFIX=$HOME/carter_ws/mamba $HOME/carter_ws/tools/bin/micromamba run -n isaacsim \
  python levels/common/plot_traj.py levels/level2
# -> levels/level2/results/trajectory.png (base vs solution, goal 디스크/장애물 포함)
```

## 5. 함정 모음

| 증상 | 원인/대처 |
|---|---|
| exit code 0 인데 FAIL | 정상 — verdict 는 JSON 이 정본 (§0) |
| GPU1 에서 부팅 중 segfault | 이 호스트 고유 문제 — GPU0 사용 |
| 런이 몇 분씩 걸림 | 정상 — A100 은 RT 코어가 없어 ~17 fps (sim 60 s ≈ wall 3.5 분) |
| `micromamba: prefix does not exist` | `MAMBA_ROOT_PREFIX=$HOME/carter_ws/mamba` 누락 |
| 같은 코드인데 결과가 다름 | 다른 Isaac/ROS 프로세스와 동시 실행했는지 확인 (직렬만 지원) |
