# 실험 41 B-5-15 a0 실패 분석

대상은 2026-11-05 21:07:45–21:13:55의 a0 자료다. a1, 실행 중인 프로세스, base, 프롬프트, 하네스와 실행 스크립트는 변경하지 않았다.
**두 제출은 모두 choose_command를 사용했다. 실패는 우회 분기가 아니라 경로와 후보 구성, 모든 후보가 위험할 때의 선택에서 발생했다. c02는 빈 인자가 아니라 잘못된 JSON이었다.**

| 실행 | 실제 시험 | 제출 채점 |
|---|---:|---|
| a0 1회차 | 0회, c01 도구 호출 0 | box_06 첫 접촉 97.333 s, 접촉 360건, 목표까지 4.5644 m |
| a0 2회차 | 0회, c02 인자 파싱 실패 뒤 c03 제출 | M3 첫 접촉 79.583 s, 접촉 51건, 목표까지 12.5526 m |
| 같은 v4의 관문 | 별도 단순 추종기 | 접촉 0, 목표 도착 62.25 s, 호출 중앙값 1.2677 ms |

근거: [회차 기록](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/gpu_gpt_oss_120b_pB515_a0_turns.json), [관문 결과](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_gate_isaac/summary.json).

## 1. 명령 경유와 경로 차이

두 코드의 choose_command, check_motion, 후보 생성과 선택 함수는 v4와 AST가 같다. 정상 반환 3곳씩 모두 chooser가 돌려준 v,w를 그대로 반환한다.
[1회차 코드](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0_carter_run.py:943)의 943/949/961행, [2회차 코드](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0_t2_carter_run.py:942)의 942/948/960행에서 각각 종료 정지, 목표 정렬, 일반 추종 명령을 검사한다. 우회 반환은 없다.
핵심 발췌는 아래 12줄이다. 위쪽 3개는 서로 다른 반환 분기이고, 마지막 2줄은 base의 후보 생성 조건이다.

```python
v_cmd, w_cmd, info = choose_command(0.0, 0.0, GOAL[:2])
done = (v_cmd == 0.0 and w_cmd == 0.0)
return v_cmd, w_cmd, done
v_cmd, w_cmd, info = choose_command(v_follow, w_follow, GOAL[:2])
return v_cmd, w_cmd, False
if abs(e) > 0.35:
    v_follow = 0.0
    w_follow = max(-0.4, min(0.4, 1.5 * e))
v_cmd, w_cmd, info = choose_command(v_follow, w_follow, target_xy)
return v_cmd, w_cmd, False
if v_follow > 0.0:
    candidates.append(("fast", min(max(cfg["fast_scale"] * v_follow, cfg["fast_min"]), 1.0), w_follow))
```

| 경로 조건, 저장 코드의 경로 계산 재실행 | 1회차 | 2회차 | 관문 |
|---|---:|---:|---:|
| 격자 간격 / 요구 지도 여유 m | 0.5 / 0.6 | 0.2 / 1.2 | 0.5 / 1.0 |
| 경유점 수 / 길이 m | 12 / 35.506 | 34 / 37.053 | 7 / 34.776 |
| 구간 길이 최소 / 중앙값 m | 0.500 / 1.500 | 0.200 / 0.283 | 0.707 / 5.539 |
| 계산 경로의 최소 지도 여유 m | 0.600 | 1.200 | 1.018 |

두 제출은 일직선의 중간 점만 지우고, 관문은 지도 선분 여유를 확인하며 먼 점까지 직선으로 연결해 단순화한다. 2회차의 33구간 중 17개가 경유점 통과 반경 0.35 m보다 짧다.
정상 추종의 v≤0.3, |w|≤0.4, 방향 오차 0.35 rad 초과 시 제자리 회전은 관문과 같다. 하지만 경유점 위치와 전환 시점이 달라 chooser에 들어가는 명령과 fast의 진전 비교가 달라진다.
제출은 경유점 인덱스 소진 뒤 목표 위치를 다시 확인하지 않는 정렬 분기도 갖는다. 관문은 매번 위치를 확인한다. 이번 두 충돌은 목표 정렬 전이라 이 차이를 직접 원인으로 보지는 않는다.
근거와 재계산 경로: [제어기 증거](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_a0_controller_evidence.json), [관문 추종기](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_gate_isaac/b515_gate_probe_carter_run.py:733).

## 2. box_06: 검출한 지도 위험을 선택 단계에서 허용

접촉 전 5초인 92.333–97.333 s에 남은 DBG 10줄을 묶었다. `-`는 None이며 간격은 선택 명령의 3초 예측 최솟값이다.

| DBG 시각 s | 선택 / 이유 | follow / stop t_hit s | 원래 물체 / 여유 적용 간격 m |
|---|---|---|---|
| 92.5–95.0, 6줄 | stop / stop_safe | 0.4 / - | 1.10–1.45 / 0.35–0.79 |
| 95.5 | stop / largest_gap | 0.4 / 3.0 | 1.05 / 0.20 |
| 96.0 | stop / stop_safe | 0.3 / - | 0.99 / 0.22 |
| 96.5 | back / largest_gap | 0.2 / 2.1 | 1.39 / 0.54 |
| 97.0 | back / largest_gap | 0.0 / 0.0 | 1.30 / 0.56 |

이 구간은 fast=1.0 m/s 주행이 아니다. 명령은 정지 또는 후진 -0.2 m/s이고, 궤적 차분의 최대 속력도 약 0.200 m/s다.
접촉 부근 계획 경로의 중심점과 상자 사이 여유는 0.8 m였다. 북향 몸체라면 0.55 m지만, 회피로 서향이 된 실제 몸체는 긴 뒤쪽 때문에 96.5 s 현재 지도 여유가 0.2175 m로 줄었다.
저장 자세와 동일 helper로 **지도 성분만 재계산**하면 96.5 s stop은 min_gap_map=0.2175, t_hit_map=None이나 back은 min_gap_map=0, t_hit_map=0.1 s다. 회전 후보도 최소 0.1753 m로 0.2 m 미만이다.
그런데 all-unsafe의 largest_gap은 지도 침범 후보를 제외하지 않고 `min(robust_min_gap_obj, min_gap_map)`이 가장 큰 것을 고른다. 지도 거리는 내부에서 0으로 막히지만 물체의 여유 적용 간격은 음수가 될 수 있어, 지도 침범 예측 0이 음수인 동적 위험보다 높은 순위가 될 수 있다.
97.0 s에는 현재 지도 여유도 0.1559 m로 기준보다 작았는데 back을 계속 골랐다. 즉 **0.2 m 검사가 놓친 것이 아니라 검출 후 선택 규칙이 허용했다.**
접촉 시 보간한 차체 기하 간격 약 0.099 m는 기존 측정의 상자 contactOffset 0.1 m와 부합한다. 전체 물체 관측은 없어 당시 다른 후보의 동적 점수까지 재현한 것은 아니다.
근거: [stdout 690–701행](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0.carter-run.stdout.log:690), [선택 규칙 663–680행](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0_carter_run.py:663), [기하 계산과 접촉 근거](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_a0_box06_evidence.json).

## 3. M3: 경유점 전환 때 전진 탈출 후보가 사라짐

접촉 전 74.583–79.583 s의 DBG 10줄이다. 원래/여유 적용 간격을 현재 겹침이나 현재 거리로 해석하면 안 된다.

| DBG 시각 s | 선택 / 이유 | follow / stop t_hit s | 원래 물체 / 여유 적용 간격 m |
|---|---|---|---|
| 75.0, 75.5 | follow / follow_safe | - / - | 1.92, 1.86 / 1.70, 1.69 |
| 76.0, 76.5 | fast / largest_gap | 1.2/1.1, 1.0/0.9 | 1.50, 1.47 / 0.63, 0.48 |
| 77.0, 77.5 | back / largest_gap | 1.0/0.8, 0.8/0.6 | 0.24, -0.10 / -0.79, -1.01 |
| 78.0, 78.5 | back / largest_gap | 0.4/0.3, 0.1/0.1 | 0.06, -0.04 / -0.93, -0.91 |
| 79.0, 79.5 | back / largest_gap | 0.0/0.0, 0.0/0.0 | -0.01, -0.03 / -0.91, -0.77 |

76–76.5 s fast는 안전한 전진이 아닌 all-unsafe 선택이다. 지도 재계산에서 fast의 최소 지도 여유는 0이고 t_hit_map은 2.0→1.6 s였다. 당시 현재 지도 여유는 1 m 이상이므로 미래 경로의 문제다.
기록 자세에 동일 `_next_target`을 재생하면 76.63 s 부근 경유점 인덱스 11→12 전환에서 방향 오차가 커져 `v_follow=0`이 된다. 이때 base의 `v_follow>0` 조건이 꺼져 fast 후보 자체가 사라진다.
77–79.5 s에는 back을 지속 선택한다. 궤적 속도는 76.5 s 약 1.01 m/s에서 77 s 약 0.325 m/s, 77.5 s부터 약 -0.20 m/s로 바뀐다. 79.583 s에 왼쪽 바퀴가 M3와 접촉했다.
따라서 확인되는 실패 흐름은 **위험 중 경유점 전환으로 전진 후보 소실, 이후 위험한 후진 지속**이다. 우회 분기는 아니다. 현재 지도 공간이 있었는데도 그 자리에서 선택 가능한 명령이 제한됐다.
물체 id, points, vx/vy, seen_t는 제출 로그에 없다. 관측 누락, M3 속도 추정오차, 불확실성 여유의 과대 여부는 확정할 수 없다. finite obj_min은 어떤 물체 자료가 있었다는 뜻이지 M3를 정확히 봤다는 증거는 아니다.
사후 장면 기하의 실제 몸체 간격은 77.5 s 약 1.153 m, 79.0 s 0.297 m, 79.5 s 0.039 m다. 이는 관측값이 아닌 기록 자세와 장면 정의로 다시 잰 값이다.
근거: [stdout 650–660행](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0_t2.carter-run.stdout.log:650), [후보 조건 639행](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/isaac_logs/gpu_gpt_oss_120b_pB515_a0_t2_carter_run.py:639), [재생 및 관측 한계](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_a0_m3_evidence.json).

## 4. 시험 0회의 원인과 지시 이행

c02의 content는 0자지만 도구 인자는 **40,682자, UTF-8 42,839바이트**다. 끝의 `"], "sim_seconds": 200}`에 있는 여분 `]` 때문에 40,660번째 문자에서 JSON 파싱이 실패했다. 200 s 인자는 있었고 실제 시험에는 들어가지 않았다.
원시 출력에도 같은 오류가 있으며, 메모리에서 이 `]` 하나만 제거하면 JSON과 Python AST가 정상이다. 원본을 고치거나 시험을 실행하지는 않았다.
`finish=tool_calls`, completion 15,973 / 한도 16,384이므로 길이 제한 절단으로 단정할 수 없다. 보고된 reasoning_tokens 15,953과 달리 raw logprobs 중 15,239개는 도구 인자 본문과 겹쳐, 이를 전부 추론 소비로 해석해서도 안 된다.
[하네스 1086행](/home/jun/SDI-Scenario/accident_scene/arena/agent_loop.py:1086)의 파싱 실패가 시험 카운터와 실행보다 먼저 반환한다. 오류는 c03에 전달됐지만 재호출 없이 코드 답이 왔고, B5는 시험 통과 여부를 강제하지 않은 채 그 답을 채점했다(1247–1256행).
c01의 실제 입력에는 첫 응답 시험, pass 전 제출 금지, JSON 끝의 여분 `]` 주의가 모두 있었다. c01 추론 기록도 시험 필요를 언급한다. 따라서 시험 지시 누락으로 설명할 수 없고, 생략한 내부 원인은 확정할 수 없다.
반면 c03 추론 기록은 긴 전체 파일과 JSON 인자 작성 부담 때문에 시험 지시를 알면서도 도구 없이 답하겠다고 명시한다. 두 최종 답은 전체 파일 지시와 달리 EDIT REGION만 냈으며 하네스가 `spliced=true`로 받아 채점했다.
“답은 언제나 전체 코드”, “도구 없는 답은 제출”과 “먼저 시험”의 단계 구분 및 입력 크기는 정리할 여지가 있다. 다만 이것을 c01 생략의 입증된 원인으로 소급하지 않는다.
근거: [도구 오류](/home/jun/SDI-Scenario/accident_scene/arena/results/turns/prompt_b5_15/gpu_gpt_oss_120b_pB515_a0_c02_tool1.json), [요청 문장, 길이와 원시 출력 증거](/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/b515_a0_tool_evidence.json).

## 5. 변경 제안 5개 — 이번 작업에서는 적용하지 않음

1. **[base]** all-unsafe 선택에서 지도 침범을 먼저 분리한다. 지도상 안전한 후보가 있으면 침범 후보를 제외하고, 음수가 가능한 물체 간격과 내부 0인 지도 거리를 그대로 경쟁시키지 않는다. 후보별 map/obj TTC와 원래/여유 적용 간격을 함께 기록해 선택 이유를 검증한다.
2. **[base]** 추종 입력이 회전으로 바뀌어도 전진 및 곡선 탈출 후보가 통째로 사라지지 않게 후보 구성을 검토한다. 지도와 관측 물체 검사를 통과한 후보만 사용하고, 경유점 전환 및 감속 중 동작까지 CPU와 Isaac에서 점검한다.
3. **[prompt]** 경로 여유 수치뿐 아니라 관문처럼 선분 검증을 거친 직선 연결 단순화와 연속적인 추종 목표를 명시한다. 경유점 통과 반경보다 촘촘한 점을 급히 넘겨 회전 명령으로 바뀌는 경우와 목표 정렬 중 위치 이탈도 점검하게 한다.
4. **[harness, Claude가 사용자 허락을 받아야 함]** 고정 base를 하네스가 보존하고 시험 입력은 수정 영역만 받는 방식으로 긴 JSON 복사를 줄이는 방안을 검토한다. 서버에서 전체 파일을 조립하고 보호 영역과 문법을 검사하며, 프롬프트와 도구 schema의 허용 형식을 일치시킨다.
5. **[harness, Claude가 사용자 허락을 받아야 함]** B5에서 실제 시험이 실행되기 전 무도구 답을 최종 제출로 받지 않는 검증을 넣는다. 첫 호출을 시험 도구로 제한하고 인자 파싱 실패는 시험으로 세지 않은 채 수정 재호출을 요구한다. 시험과 최종 제출 코드의 동일성도 확인한다.
