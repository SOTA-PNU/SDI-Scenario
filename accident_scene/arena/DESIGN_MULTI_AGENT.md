# 설계안: 다중 에이전트 루프 M-1 (B-5-29 기반)

작성 2026-11-15, 브랜치 `b529-speedx5`. 구현 전 설계 문서다. 결정이 필요한 항목은 8절에 모았다.

## 1. 왜 바꾸나 — B-5-29에서 관찰한 것

| 문제 | 근거 (results/turns/prompt_b5_29) |
|---|---|
| pass 전 제출·시험 없이 제출 | a0 t1: 시험 7회 모두 fail 뒤 코드 대신 `[No answer provided ...]` 제출 → 1543행 구문 오류로 정적 실격. a1 t1·t2: 시험 0회로 제출 |
| `sim_seconds` 누락 | 프롬프트는 200을 요구하지만 a0 시험 8회 모두 기본값 600 s로 실행(회당 약 10분) |
| 맥락 누적 | 입력 토큰이 c01 31,811 → c08 86,532. 시험 1회당 약 +7,500 토큰. 131,072 한도까지 약 13회 |
| 디버깅 정보 부족 | 시험 1회 DBG 약 1,370줄 중 앞 40 + 뒤 40줄만 전달. 로봇이 멈춘 50~60 s 이후 구간 대부분이 잘림. 턴 사이 채점 피드백에는 결과 JSON만 있고 DBG·궤적 없음 |
| 차단 장치 꺼짐 | `gate()`(`agent_loop.py:892`)는 `--a22/--a23` 전용이고 `--b5`와 함께 쓸 수 없다(`agent_loop.py:1448`) |

한 모델이 한 대화 안에서 규칙 준수, 긴 로그 해석, 코드 수정을 모두 맡는 구조가 원인이다.

## 2. 구조

```
            ┌────────────────────── 오케스트레이터 (코드, LLM 아님) ──────────────────────┐
            │  규칙 강제 · 반복 관리 · 시험/채점 실행 · 시도 기록 · 산출물 저장             │
            └───┬───────────────────────────┬───────────────────────────┬───────────────┘
                │ ① 수정 요청                │ ③ 실행 요약 + DBG 원본     │
                ▼                            ▼                            │
        수정 에이전트 (LLM)            분석 에이전트 (LLM)                 │
        매 반복 새 맥락               매 반복 새 맥락                      │
        → EDIT 블록 하나              → 진단 JSON                          │
                │ ② 코드                     │ ④ 진단 ─────────────────────┘
                ▼
        정적 검사 → Isaac 시험 200 s → 결정적 요약(run_digest)
```

- **오케스트레이터 (코드):** 규칙 관리 역할을 맡는다. LLM에게 맡기지 않는 이유는 규칙이 모두 기계적으로 판단되기 때문이다.
- **분석 에이전트 (LLM):** 디버깅 정보를 해석하고 수정 방향을 정한다. 긴 원본은 이 에이전트의 맥락 안에서만 읽는다.
- **수정 에이전트 (LLM):** 진단을 받아 코드를 고친다. 맥락 크기가 반복 횟수와 무관하게 일정하다.
- 수정 방향을 정하는 에이전트를 따로 두지 않는다. 분석 결과에 "어느 함수를 왜"가 이미 들어 있고, 한 단계를 더 거치면 전달 손실과 호출 시간만 늘어난다. 필요하면 9절의 확장으로 시험한다.
- 두 LLM 모두 같은 서버의 gpt-oss-120b를 쓴다(B-5-29와 같은 모델, 같은 샘플링).

## 3. 반복 루프 (한 시료)

```
iter = 1
code = None, diagnosis = None, attempts = []
while iter <= MAX_ITERS and 벽시계 < MAX_WALL:
    code_new = 수정 에이전트(프롬프트, base, code, diagnosis, attempts)
    if 코드 펜스 없음 / EDIT 경계 없음:   재요청 (같은 반복 안에서 최대 2회) → 그래도 없으면 반복 실패로 기록
    정적 검사 실패:                       분석 에이전트 생략, 오류 문구를 진단으로 바로 사용
    시험 주행 (sim_seconds = 200 고정, movers v1, 하네스가 설정)
    digest = run_digest(시험 결과)
    if verdict == pass:
        채점 (같은 code_new, 600 s, movers v1)
        if 채점 pass: 종료 (성공)
        digest = run_digest(채점 결과)       # 채점 실패도 같은 요약으로 분석에 넘긴다
        graded += 1; if graded >= MAX_GRADES: 종료 (실패)
    diagnosis = 분석 에이전트(프롬프트 요약, base, code_new, digest, DBG 원본 구간, attempts)
    attempts.append(한 줄 기록)
    code = code_new; iter += 1
```

규칙 강제(오케스트레이터가 구조로 보장):
- **제출은 하네스가 한다.** 시험에서 pass를 받은 바로 그 EDIT 블록만 채점에 보낸다. 모델의 "포기 문구"나 시험 안 한 코드가 채점될 경로가 없다.
- `sim_seconds`는 모델 인자가 아니라 하네스 상수(200)다.
- 수정 에이전트에는 도구가 없다. 출력은 python 코드 펜스 하나뿐이고, 형식이 틀리면 하네스가 재요청한다.
- 정적 검사(구문, EDIT 밖 변경, `controller` 존재)는 지금의 `static_check()`(`agent_loop.py:415`)를 그대로 쓴다.

B-5 계열과의 대응: "턴"은 **채점 횟수**로 바꾼다(`MAX_GRADES` 기본 2 = 기존 `--max-turns 2`). 시험 횟수 무제한은 `MAX_ITERS`/`MAX_WALL`로 대신한다.

## 4. 결정적 요약 `run_digest.py` (새 파일)

LLM 없이 시험/채점 한 번의 산출물에서 뽑는다. 입력: `<variant>_result.json`, `<variant>_trajectory.csv`(약 60 Hz), `carter-run.stdout.log`.
출력 상한 약 8,000자.

| 항목 | 내용 |
|---|---|
| 결과 | verdict, 최종 pose·거리·yaw 오차, 충돌 수·상대·최초 시각, time_to_goal |
| 진행 | 출발 후 5 s 간격 위치 대신, **10 s마다 (t, x, y, 목표까지 거리)** — 최대 60행 |
| 정체 구간 | 1 s 이동 평균 속력 < 0.05 m/s가 5 s 이상 이어진 구간마다: 시작·끝 시각, 위치, 길이 |
| 정체 중 chooser | 구간별 `pick`·`why` 빈도 상위 5개, `map`·`obj_min`·`rob`의 시작/최소/끝 값 |
| 전체 chooser | 전 구간 `pick`·`why` 빈도 |
| 계획 | `plan_summary`·`plan_reject` 전 줄(같은 줄 반복은 횟수로 합침) |
| 예외 | hook 예외 DBG 전 줄(중복 합침), runtime_error 요약 |
| 충돌 | 최초 충돌 시각 ±3 s의 DBG |

정체 판정 임계값(0.05 m/s, 5 s)은 상수로 두고 산출물에 기록한다.

## 5. 분석 에이전트

- **입력 (매 반복 새 대화, 약 55K 토큰 예상):**
  1. 역할 지시 `PROMPT_M1_ANALYST.md` (새 파일): "원인 진단만 한다, 코드를 쓰지 않는다, 아래 JSON으로만 답한다"
  2. 과제 프롬프트 `PROMPT_B5_29.md` 본문 중 API·계약 부분(도구·제출 절 제외)
  3. base 코드 (chooser·계획 재시도 동작을 이해하는 데 필요)
  4. 이번 EDIT 블록
  5. `run_digest` 출력
  6. DBG 원본 구간: 첫 정체 시작 −10 s ~ +20 s, 최초 충돌 −5 s ~ +2 s의 모든 DBG 줄(`dbg_lines_full`, 상한 60,000자)
  7. 시도 기록(최근 10줄)
- **출력 (JSON 하나, 하네스가 스키마 검사, 실패 시 1회 재요청):**
  ```json
  {"failure_mode": "stall|collision|no_path|runtime_error|static|timeout_moving|other",
   "evidence": ["t=57.9 (5.28,4.45)부터 600 s까지 정체, pick=stop 1139회 why=stop_safe", "..."],
   "function": "plan_path|lookahead_target|follow_cmd|none",
   "direction": "무엇을 어떻게 바꿀지 3문장 이내",
   "not_to_change": "이번에 건드리면 안 되는 것",
   "confidence": "high|medium|low"}
  ```
- `function: none`은 "모델 코드로는 고칠 수 없고 base 한계로 보인다"는 뜻이다. 이 판정이 2회 연속이면 오케스트레이터가 시료를 끝내고 사람이 보도록 기록한다(base 결함을 모델에 떠넘기지 않기 위함).

## 6. 수정 에이전트

- **입력 (매 반복 새 대화, 약 40K 토큰으로 일정):**
  1. 과제 프롬프트 `PROMPT_M1_CODER.md` (새 파일): `PROMPT_B5_29.md`에서 `run_isaac_sim` 절, "지금 할 일", "제출" 절만 바꾼다 → "EDIT 블록 하나를 python 코드 펜스로 출력". 과제·API·추종 식 문장은 한 글자도 바꾸지 않는다(비교 가능성).
  2. base 코드 (지금의 `model_input()`과 같은 방식으로 붙임)
  3. 현재 EDIT 블록 (첫 반복은 없음)
  4. 최신 진단 JSON
  5. 시도 기록(최근 10줄)
- 시도 기록 한 줄 형식(하네스가 작성):
  `#3 test fail dist=21.8 stall@57.9s(5.28,4.45) fn=plan_path dir="코너 여유..." 변경="[시험 3] ..."`
  "변경"은 프롬프트가 이미 요구하는 EDIT 맨 위 `# [시험 N] 결과/원인/변경:` 주석에서 하네스가 뽑는다.
- 같은 EDIT 블록(정규화 후 sha)이 다시 나오면 시험하지 않고 "이전 시도 #k와 같은 코드"라고 기록해 재요청한다.

## 7. 구현 위치와 재사용

- 새 스크립트 `multi_agent_loop.py`. `agent_loop.py`의 턴 루프에는 손대지 않는다(B-5 결과 재현성 유지).
- 재사용: `build`, `static_check`, `region_of`, `code_key`, `dbg_lines_full`, `traj_sample`, `post`, `raw_text`, `recover_call`(`agent_loop.py`), `TL.grade`, `TL.feedback_json`, `TL.keep_isaac_logs`(`turn_loop.py`), `carter_run_locked.sh`.
  `Run.test()`는 도구 호출 인자 처리와 결과 조립이 섞여 있어, Isaac 실행부만 함수로 떼어 두 스크립트가 같이 쓰게 한다(동작 변경 없음, B-5 재채점으로 확인).
- 새 파일: `multi_agent_loop.py`, `run_digest.py`, `PROMPT_M1_CODER.md`, `PROMPT_M1_ANALYST.md`, `run_m1_sequential.sh`.
- 산출물(`--out-dir results/turns/prompt_m1`): 기존과 같은 이름 규칙에 역할 접두사를 붙인다.
  `<stem>_i<NN>_coder_request.json` / `_raw.json`, `<stem>_i<NN>_analyst_request.json` / `_raw.json`,
  `<stem>_i<NN>_digest.json`, `<stem>_attempts.jsonl`, `<stem>_events.jsonl`, `<stem>_turns.json`(채점별 요약), `isaac_logs/`.
- 인자: `--model`, `--tag`, `--sample`, `--base`, `--seed-base`, `--max-iters`(기본 20), `--max-wall-h`(기본 3), `--max-grades`(기본 2), `--test-sim-s`(200), `--dbg-window-chars`(60000), `--base-file`, `--movers`(v1).

## 8. 결정 (2026-11-15 사용자)

1. **비교 조건:** M-1과 B-5-29G를 함께 돌린다. B-5-29G = B-5-29 + `agent_loop.py --b-gate`
   (시험에서 pass를 받은 EDIT 블록만 채점, 코드 없는 답도 붙잡음, 회차당 안내 10회 뒤에는 채점, 시험 주행은 항상 200 s).
   두 조건 모두 a0~a3를 조건 없이 다 돌린다(`run_b529g_sequential.sh`, `run_m1_sequential.sh`).
   B-5-29G의 시험 주행은 회차당 최대 10번(`--test-budget 10`, 사용자 2026-11-15). 프롬프트는 `PROMPT_B5_29G.md`로,
   `PROMPT_B5_29.md` 188행의 "시험 횟수에는 한도가 없습니다"만 10번 한도 문장으로 바꿨다(그 밖 동일).
   첫 시작(09:33)은 한도 없이 시작했다가 멈췄고 산출물은 `results/turns/prompt_b5_29g_aborted_20261115_093603/`에 있다.
2. **반복 상한:** `--max-iters 20`, `--max-wall-h 3`, `--max-grades 2`.
3. **분석 에이전트 입력:** base 코드 전문과 수정 에이전트가 만든 이번 코드를 모두 받는다(5절).
4. **이어 고칠 코드:** 직전 시도의 코드.
5. 시료 a0~a3, `--seed-base 1000`, 속도 설정 v17 그대로.

### 구현에서 설계와 달라진 점

- `Run.test()`를 리팩터링하지 않고 `multi_agent_loop.py`가 `agent_loop.Run` 인스턴스를 만들어 `test()`를 그대로 부른다
  (B-5 경로 무변경). 시험 변형 이름은 `<stem>_t<채점회차>_test<반복번호>`.
- 수정 에이전트 답에서 EDIT 시작·끝 주석이 모두 있어야 코드로 인정한다(`splice()`는 아무 글이나 구역에 끼워 넣기 때문).
- 분석 에이전트가 형식에 맞는 JSON을 2회 안에 내지 못하면 하네스가 실행 요약의 핵심 값으로 대체 진단을 만든다
  (`function: unknown`, `_fallback: true`).
- 프롬프트: `PROMPT_M1_CODER.md`는 `PROMPT_B5_29.md`의 2~171행(과제·API·추종 식)과 미션·성공 조건을 그대로 쓰고,
  도구·제출·지금 할 일 절만 "하네스가 시험·진단" / "출력" 절로 바꿨다. `PROMPT_M1_ANALYST.md`는 새로 썼다.

## 8-1. M-1b (2026-11-15 10:52~)

사용자 결정: B-5-29G는 "희망이 없다"고 보고 중단(10:50, 시험 8회 모두 fail·붙잡힘 7회, 산출물
`results/turns/prompt_b5_29g_stopped_20261115_105052/`). M-1 v1도 중단(반복 12회 모두 fail, 최소 거리 11.8 m,
`results/turns/prompt_m1_stopped_20261115_105052/`)하고 다음을 바꿔 M-1b로 다시 시작(`--tag gpu_gpt_oss_120b_pM1b`,
`results/turns/prompt_m1b/`).

- **코드가 바뀔 때마다 분석:** 정적 검사(컴파일) 실패도 분석 에이전트가 평가한다(v1은 하네스 문구만 전달).
- **변경 기록은 하네스가 계산:** v1에서 수정 에이전트는 `# [시험 N] ... 변경:` 주석을 한 번도 쓰지 않아 시도 기록의 변경 칸이
  모두 비었고, 진단이 같은 방향을 오갔다(예: `lookahead_target`의 `segment_free` 검사를 3회차 제거 → 7회차 추가 → 9회차 제거).
  이제 직전 코드와 비교한 함수별 증감(예: `plan_path +107/-1`)을 시도 기록에 넣고, 분석 에이전트에는 diff(상한 8,000자)도 준다.
  v1 코드에 적용해 보면 4회차에 `plan_path` 94줄 삭제 → 5회차 107줄 재작성 같은 통째 재작성이 반복됐다.
- **분석 프롬프트:** 먼저 이번 수정이 직전 진단을 반영했고 결과가 좋아졌는지 평가하게 했고, 세 함수 안에서 센서·관측 함수나
  `choose_command`를 부르는 제안을 금지 목록에 명시했다(v1 2회차 진단이 `observed_objects()` 호출을 제안).

## 8-2. M-1c (2026-11-15 11:52~): base v18

- M-1b i05(10:56 시작 기준 5회차)가 GOAL에서 0.29 m, 충돌 0이었지만 `yaw_misaligned`(0.546 rad)로 fail. 시간 한도가 아니라
  base 목표 정렬의 회전 초과였다: v17은 정렬 회전 상한이 2.0 rad/s라 |yaw 오차| <= 0.15에서 정지 명령을 낼 때 약 1.9 rad/s로
  돌고 있었고, 관성으로 0.55 rad를 지나쳤다(최종 yaw 1.559, GOAL 1.014). 세 함수로는 고칠 수 없는 base 결함인데 분석 에이전트는
  `plan_path`에서 GOAL을 옮기라고 진단했다.
- 사용자 결정으로 정렬 회전 상한만 0.4 rad/s(v16 값)로 되돌린 `base_..._v18.py`를 만들었다(이득 7.5 유지, 그 밖 v17과 동일).
  같은 i05 코드를 v18에 넣어 최종 채점 조건(600 s, movers v1)으로 돌리면 **pass**: 도달 540.45 s, 0.255 m, yaw 오차 0.108, 충돌 0
  (`results/check_m1b_i05_v18_result.json`). 다만 정지가 369.8 s로 길다.
- `run_digest.py`에 `ended_by`(time_limit / collision_abort / pass / other)와 `time_budget`(목표 원 진입 시각, 정지 시간, 같은 자리
  반복 정지 `stall_spots`)를 추가했다. 하네스 판정 이유만으로는 시간 한도 종료와 정렬 실패를 구분할 수 없었기 때문이다.
- 분석 프롬프트에 시간 예산 해석(속도 상한·추종 식은 고정이므로 정지·재계획 대기를 줄이는 쪽으로 지시)과
  "목표 원 안 정렬 실패는 base 책임 → `function: none`"을 추가했다.
- M-1b 산출물: `results/turns/prompt_m1b_stopped_20261115_114231/`(12회 반복, i05가 최고). M-1c: `results/turns/prompt_m1c/`.

## 8-3. M-1c a0 복기 (2026-11-15 11:52~13:08, 19회 반복, 모두 fail)

사용자 지시로 a1을 시작하지 않고 멈췄다(19회차 시험 직후, 20회차 미실행). 산출물 `results/turns/prompt_m1c/`.

| 회차 | 결과 | 비고 |
|---|---|---|
| 1 | 충돌 27회(움직이는 물체 `arena_mover_00`) | 진단: `lookahead_target`, "plan_path·follow_cmd는 건드리지 말 것" |
| 2 | 경로 없음(예외) | 수정 에이전트가 `plan_path`를 프롬프트의 빈 틀 `raise NotImplementedError`로 되돌림(-83줄) |
| 3~18 | 경로 없음 16회 연속 | 매번 `plan_path` 수십~백여 줄 재작성. 로봇이 출발하지 못함(31.95 m) |
| 19 | 경로 채택, 18.25 m | 처음으로 출발 |

- **시간:** 반복당 약 4분, 그중 Isaac 시험이 약 3.5분. 모델 호출은 수정 12~26 s, 분석 5~33 s. 경로가 없어 출발조차 못 한
  16회의 Isaac 시험에 약 1시간을 썼다.
- **분석의 진동:** 경로 없음의 실제 원인을 볼 자료가 `reason=empty_path stage=search`뿐이라, 진단이 격자 해상도 0.25↔0.5,
  이웃 4↔8방향을 번갈아 지시했다(i05~i18).
- **규칙 위반 진단 3건:** i04·i15 "A* 실패 시 시작-목표 직선 반환"(프롬프트가 금지한 미검사 직선 대체), i18 "이웃 이동 전
  `segment_free` 검사 제거"(프롬프트가 모든 연결 검사를 요구). i15를 따른 i16은 허용되지 않은 `plan_stage('fallback')`로 예외.
- **진단 무시:** i01 진단이 `plan_path`를 건드리지 말라고 했는데 i02가 `plan_path`를 지웠다. 수정 에이전트는 매 반복 새 맥락에서
  블록 전체를 다시 쓰므로, 진단 대상이 아닌 함수도 망가질 수 있다.
- **오프라인 경로 검사 가능:** base는 하네스를 1565행에서야 띄우고 지도 도구는 순수 Python이라, 그 앞까지만 실행하면 Isaac 없이
  `plan_path`+`validate_plan`을 1초 안에 돌릴 수 있다. M-1b i05(채택), M-1c i18(empty_path, search), i19(채택) 모두 Isaac 결과와 일치.

## 8-4. M-1d (2026-11-15 14:00~): 함수 단위 수정, 변경 이유, 진단 규칙 검사, 코드 추적

사용자 결정으로 다음만 적용했다(오프라인 경로 검사는 하지 않음).
- 수정 에이전트는 `변경 이유:` + 바꾼 함수의 완전한 def 만 낸다. 하네스가 직전 코드의 같은 이름 함수만 바꿔 끼우고, 첫 시도는
  세 함수 모두, `NotImplementedError`가 남으면 거절. 진단의 `not_to_change`는 없앴다(어느 함수든 고칠 수 있음, 진단 대상 밖
  변경은 기록만).
- 진단 규칙 검사: 직선 대체, `segment_free` 검사 제거, 센서·`choose_command` 직접 호출, GOAL 이동, EDIT 밖·base 변경,
  `clearance_m` 임의 하향을 담은 진단은 거절하고 다시 쓰게 한다.
- 분석 프롬프트에 코드 추적 지시(실패 단계의 코드를 실제 입력값으로 한 줄씩 따라가고, 틀린 줄을 인용).

1회 반복 시험(빈 틀에서, base v18):
- trial 1(코드 추적 지시 전, `results/turns/prompt_m1d_trial/`): 첫 코드가 M-1c와 같은 `h` 덮어쓰기 버그. 분석은 시작·목표 칸과
  격자 크기까지 맞게 계산했지만 탐색 루프를 따라가지 않고 "Given the difficulty"로 포기, 직선 대체를 제안(1차 규칙 검사 거절, 2차는
  `segment_free` 검사 직선이라 통과했지만 효과 없음). `h`→`hh`만 바꾸면 모든 여유에서 채택됨을 확인.
- trial 2(코드 추적 지시 후, `results/turns/prompt_m1d_trial2/`): 첫 코드가 달랐다. A*는 58점을 찾았지만 단순화 단계에서
  `UnboundLocalError: prev_pt`. 분석은 예외·위치·변수를 정확히 짚었고(로그에 메시지가 그대로 있어 쉬운 경우) 두 가지를 제안:
  (a) 단순화를 빼고 raw 경로 반환 → 모든 여유에서 채택(효과 있음), (b) `prev_pt` 초기화 → 그것만으로는 모든 여유에서 `check_path`
  거부. 실제 결함은 연결이 이어질 때 `continue` 전에 `prev_pt`를 갱신하지 않는 것(초기화+갱신이면 채택, 7점).
  코드 추적 지시가 `h` 같은 숨은 버그에 효과가 있는지는 이번 코드로는 확인하지 못했다.

## 9. 이후 확장 (이번 범위 밖)

- 분석 결과를 받아 수정 지시를 단계별로 쪼개는 "지휘 에이전트" 추가 → M-1 대비 효과를 별도 비교.
- 분석·수정에 서로 다른 모델 사용(예: 분석만 큰 모델).
- `run_digest`를 B-5 계열의 턴 사이 피드백에도 붙이는 조건(B-5-29D).
