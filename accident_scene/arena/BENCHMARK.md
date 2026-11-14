# arena 코딩 성능 비교 — 실측 기록

모델은 `base_carter_run.py`(levels_zero2level 디폴트, 빈 컨트롤러)를 복사해
`[EDIT REGION]` 만 고칩니다. 판정은 고정 하네스가 합니다.

트랙이 둘이고, **재는 능력이 다르므로 서로 비교하지 않습니다.**

| 트랙 | 프롬프트 | 파일 이름 | 장애물을 아는 법 |
|---|---|---|---|
| **A — 계획** | [`PROMPT.md`](PROMPT.md) | `<모델이름>_carter_run.py` | base 의 `MISSION["boxes"]` 표를 읽어도 된다 |
| **B — 반응** | [`PROMPT_B.md`](PROMPT_B.md) | `<모델이름>_b_carter_run.py` | 스캔으로만. 좌표·배치 하드코딩 금지 |

B 는 `levels_zero2level/PROMPT_universal.md` 의 규칙 6 을 그대로 가져온 것이고, 이 장면이
원래 노린 능력(센서가 못 보는 영역을 인지하고 감각을 바꾸는 것)을 잽니다 — 낮은 박스 4개는
기본 스캔 평면 z=0.35 m 아래라 표를 못 쓰면 평면을 내리지 않고는 피할 수 없습니다.

> **환경 v2 (2026-10-08):** 박스 04·05·06·13 의 높이를 0.6 m 로 올려 모든 박스가 스캔 평면과
> 로봇의 모든 라이다 높이에 걸리게 했다. 이 문서의 기록은 모두 **v1**(낮은 박스 4개) 기준이다.
> v1 원본은 `env_v1/`, 변경과 실측은 `REPORT.md` '환경 v2' 절.

측정 호스트: `fractal` (RTX PRO 6000 Blackwell, 드라이버 580.173.02, Isaac Sim 4.5.0.0).

## ⚠ 지금까지의 Claude 기록은 프롬프트 동등이 아니다 (2026-09-14 트랜스크립트 감사)

두 Claude 제출(트랙 A, 트랙 B)은 "프롬프트와 base 만 본 무맥락 세션"으로 만들었다고 적었지만,
실제 트랜스크립트를 감사해 보니 **스폰 메시지에 프롬프트 밖의 유도 문장이 들어 있었다.**

- 트랙 A: "the mission table and the sensor description are the whole specification, and details
  there that seem incidental may not be" — 이 장면의 함정은 정확히 `MISSION` 표의 height 열과
  docstring 의 z=0.35 스캔 평면의 상호작용이다. 그 두 축을 짚고 "사소해 보이는 항목이 결정적"
  이라고 경고한 문장이다. 제출 보고서의 첫 문장이 바로 그 통찰이었다.
- 트랙 B: "the defaults are not necessarily the right choice for every part of the drive" —
  스캔 평면을 바꾸라는 지시나 다름없다. 1회차의 핵심 기능이 다층 평면 스캔이었다.
- 트랙 B: 규칙 6 검사가 **기계적 텍스트 검사**라는 사실을 알려줬다(프롬프트에는 없는 정보).
- 두 트랙 모두: 한국어 프롬프트의 미션·규칙을 영어로 미리 요약해 줬다.
- 하네스가 서브에이전트에 **사용자 auto-memory(MEMORY.md)를 주입**했고, 그 안에 이 벤치마크
  장면에 대한 한 줄이 들어 있다. "무맥락"은 엄밀히 사실이 아니다.
- 작업량도 다르다. 트랙 A 세션은 26분 동안 도구를 28번 쓰며 자체 2D 재현기를 만들어 103회
  섭동 시험까지 했다. 프롬프트만 받고 코드 한 덩어리를 내는 모델과 같은 조건이 아니다.

**따라서 아래 표의 Claude 줄은 로컬 NPU 모델과 직접 비교하면 안 된다.** 비교하려면
`PROMPT_*.md` 만 입력으로 주는 조건으로 다시 받아야 한다 — 스폰 메시지는 경로와 금지 규칙만
담은 포인터여야 하고, 메모리·스킬 주입을 끄고, 파일명도 모델이 스스로 정하게 해야 한다.

### 피드백 스키마의 누설 (트랙 B 2회차)

`obstacle` 블록을 뺐는데도 `collision_partners: ["/World/arena_box_00"]` 가 남아 있었다.
프림 이름의 인덱스 00 은 base 의 박스 표 첫 행이고, 그게 없더라도 START + `first_collision_t`
+ 평균 속도로 접촉 지점을 1 m 이내로 역산할 수 있다(그 세션은 실제로 그 계산을 했다).
재시도를 허용하려면 피드백은 verdict, 실패한 오라클, `final_dist_m`, `final_yaw_err_rad`,
`collision_count`, 중단 플래그까지만 줘야 한다.

## 결과

### 트랙 A — 계획

| 모델 | 제출 | verdict | 도달 시각 | 최종 거리 | yaw 오차 | 충돌 | 파일 |
|---|---|---|---|---|---|---|---|
| base (제어 코드 없음) | — | **FAIL** (120 s 초과) | — | 32.033 m | — | 0 | `base_carter_run.py` |
| Claude Opus 5 | **1회차(블라인드)** | **PASS** | 37.417 s | 0.195 m | 0.070 rad | 0 | (파일 삭제됨, 이력: `git show worktree-accident-scene:accident_scene/arena/claude_carter_run.py`) |

### 트랙 B — 반응

| 모델 | 제출 | verdict | 도달 시각 | 최종 거리 | yaw 오차 | 충돌 | 파일 |
|---|---|---|---|---|---|---|---|
| Claude Opus 5 † | 2회차 | **PASS** | 90.033 s | 0.243 m | — | 0 | `claude_b_carter_run.py` |

† 이 제출은 **15° 바퀴 축 결함 하네스를 목표로** 만들어졌다(사용자 요청 실험). 위 숫자는 같은
코드를 정상 하네스에서 돌린 대조값이고, 결함 하네스에서는 46.350 s 다. 1회차는 FAIL(보이는
상자와 충돌, t=5.7 s)이었고 `result.json`(레이아웃 블록 제외)을 한 번 피드백해 2회차가 통과했다.
정상 하네스만 보고 만든 깨끗한 트랙 B 기록은 아직 없다. 전체 로그:
[`results/toe15_attempt_log.md`](results/toe15_attempt_log.md), 제출본은 `results/submissions/`.

허용치: 거리 0.75 m, yaw 0.26 rad, 충돌 0, 제한 시간 120 s.
표의 값은 모두 **정상 로봇**(바퀴 축 토우 0°) 기준이다. 기계적 결함을 주입한 실행은
비교 대상이 아니며 `REPORT.md` 의 "기계적 결함 주입" 절에 따로 적는다.

### Claude Opus 5 (2026-09-12)

- 프롬프트와 base 파일만 읽은 **컨텍스트 없는 세션**이 작성했습니다. 이 저장소의
  `layout.json`·`REPORT.md`·`sim2d.py`·`levels/` 해답은 열지 않았습니다(프롬프트 규칙 6).
  이미 배치와 해법을 아는 세션이 직접 짜면 비교가 성립하지 않습니다.
- 첫 제출이 그대로 통과해 재제출은 없었습니다.
- 같은 파일을 2회 실행해 결과가 같았습니다(도달 37.417 s, 최종 거리 0.1947 / 0.1948 m).
- 접근: 스폰 yaw 1.0138 rad 이 목표를 향한 직선인데 그 선이 박스 00·03·06 을 지난다는 것을
  먼저 짚고, **박스 04·05·06·13 의 height(0.18~0.25 m)가 스캔 평면 z=0.35 m 보다 낮아
  광선에 안 잡힌다**는 점을 `MISSION` 표에서 추론했습니다. 그래서 경로는 표로 계획하고
  (10점 폴리라인, 박스 면에서 0.85 m 이상, 서쪽 벽에서 2.5 m 이상), 표에 없는 것(벽·기둥·
  선반·지게차)만 스캔으로 피하는 혼합 방식입니다. 추종은 pure pursuit, 마지막에 제자리
  회전으로 목표 yaw 정렬. `raycast_scan()` 은 기본 평면 그대로 씁니다 — 낮은 박스를
  "센서로 보는" 대신 지도상 장애물로 취급했습니다.
- 영상: [`results/arena_claude_isaac.mp4`](results/arena_claude_isaac.mp4)


## 1회차(블라인드) 실측 — 오픈 모델 3종 × 2트랙 × k=5 × 가속기 2종

같은 가중치, 같은 양자화, 같은 두 파일 입력. 다른 것은 백엔드뿐이다
(NPU: furiosa-llm 라우터, RNGD 4장 · GPU: vLLM 0.29.0, RTX PRO 6000 Blackwell 1장).
furiosa-ai 가 재배포한 체크포인트는 상류 safetensors 와 sha256 이 같아서,
"같은 가중치" 는 가정이 아니라 검증된 사실이다.

| 모델 | 가속기 | n | 통과 | 함정 충돌 | 보이는 충돌 | 정지 | 런타임 오류 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b (MoE 120B) | GPU | 10 | 0 | 3 | 6 | 1 | 0 |
| gpt-oss-120b (MoE 120B) | NPU | 10 | 0 | 1 | 9 | 0 | 0 |
| Qwen3-32B-FP8 (dense) | GPU | 10 | 0 | 3 | 5 | 2 | 0 |
| Qwen3-32B-FP8 (dense) | NPU | 10 | 0 | 3 | 4 | 3 | 0 |
| EXAONE-4.0-32B-FP8 (dense) | GPU | 10 | 0 | 0 | 6 | 4 | 0 |
| EXAONE-4.0-32B-FP8 (dense) | NPU | 10 | 0 | 0 | 5 | 3 | 2 |
| **합계** | **GPU** | **30** | **0** | **6** | **17** | **7** | **0** |
| **합계** | **NPU** | **30** | **0** | **4** | **18** | **6** | **2** |

**통과 0/60.** 가속기를 바꿔도 통과 수는 같고 실패 분포도 사실상 겹친다 — 이 벤치마크에서
코딩 성능 차이는 모델에서 오지 서빙 백엔드에서 오지 않는다. 가속기 비교는 주장이 아니라
통제 조건으로 읽어야 한다.

`trap_collision` 은 스캔 평면 z=0.35 m 아래의 박스(4·5·6·13, 높이 0.18~0.25 m)와 부딪힌
것이다. 이 장면이 노린 인지 실패가 실제로 발동했다는 증거다 — 이전 라운드에서는 모델 코드가
함정 박스 2 m 안으로 들어간 적조차 없었다. `runtime_error` 는 컨트롤러가 예외를 던져 하네스가
채점하지 못한 실행이고(둘 다 `UnboundLocalError`), 누락이 아니라 실패 유형으로 센다.

서빙 지표는 [`results/generated/serving.csv`](results/generated/serving.csv),
판정은 [`results/runs.csv`](results/runs.csv) 가 원본이다. 표는
`python3 score_runs.py` 와 `python3 score_serving.py` 가 만든다.

### 2회차 이후 — 턴 프로토콜

실패한 제출에 판정을 돌려주고 다시 받으면 몇 턴 만에 통과하는지는 [`TURNS.md`](TURNS.md) 에
따로 적었다. 요약: gpt-oss-120b 를 트랙 A 5샘플 × 8회차로 돌려 **GPU 1/5 통과(5회차),**
**NPU 0/5.** 도구는 [`turn_loop.py`](turn_loop.py) 이고 회차별 요청·응답·판정이
[`results/turns/`](results/turns/) 에 전부 남는다.

## 채점 절차 (채점자용 — 모든 모델 동일)

프롬프트에는 절차를 넣지 않았다. 모델은 다른 서버(예: 로컬 NPU)에서 코드를 만들고,
파일만 이 저장소로 옮겨 와 여기서 수동으로 채점한다.

```bash
# 1) 받은 파일을 accident_scene/arena/<변이>_carter_run.py 로 저장
#    트랙 A: <모델이름>          트랙 B: <모델이름>_b
# 2) EDIT REGION 밖이 base 와 같은지 검사 (다르면 그 제출은 무효)
python3 levels/common/check_edit_region.py accident_scene/arena/<변이>_carter_run.py
# 2-B) 트랙 B 만: 배치 하드코딩 검사 (걸리면 무효, 통과해도 코드는 읽어볼 것)
python3 accident_scene/arena/check_b_track.py accident_scene/arena/<변이>_carter_run.py
# 3) 실행 (직렬로만 — 동시 실행은 타이밍이 달라진다)
CARTER_REPO=<repo> $CARTER_WS/tools/bin/carter-run accident_scene/arena/<변이>_carter_run.py
# 4) 판정: accident_scene/arena/results/<변이>_result.json 의 verdict
```

- **제출은 최대 3회**. 1회차는 프롬프트와 base 만 보고 작성한다(블라인드).
- 실패하면 모델에게 **그 실행의 `<변이>_result.json` 만** 돌려주고 다시 받는다.
  씬 내부 해설, 배치 의도, 다른 모델의 코드는 주지 않는다.
  **트랙 B 는 `obstacle` 블록을 반드시 빼고 준다** — 그 블록에 박스 14개의 좌표·크기·높이가
  그대로 들어 있어서, 전문을 주면 규칙 6(장애물은 스캔으로만)이 피드백 한 번에 무너진다.
  나머지(verdict, oracles, metrics, criteria)는 그대로 준다: 충돌 상대 프림 이름과 시각은
  "못 본 것에 부딪혔다"를 알아차릴 단서이고, 그게 이 트랙에서 재려는 능력이다.
- 기록 항목: verdict, 제출 횟수, `time_to_goal_s`, `final_dist_m`, `final_yaw_err_rad`,
  `collision_count`, 첫 충돌 상대와 시각.
- 판정의 진실은 `results/<모델이름>_result.json` 이다. 표의 숫자는 그 파일에서 옮긴다.
- 실패 제출도 기록에 남긴다.
- 녹화 실행은 시각화 전용이라 판정에 쓰지 않는다(`REPORT.md` 의 "화질" 절).
- 첫 장면(`accident_scene/base_carter_run.py`)도 같은 절차로 비교할 수 있다.
  프롬프트의 미션 문단만 그 장면의 `MISSION` 에 맞게 바꾸면 된다.
