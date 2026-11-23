# C-2 / C-3 상태와 재개 방법 (2026-11-16 22:45 기준)

## C-2 (코더·검토·분석 모두 gpt-oss-120b, vLLM 127.0.0.1:8420) - 49턴에서 잠정 중단
- 사용자 지시로 49턴을 끝으로 멈춤. 자동 진행(c2_auto.py)은 꺼져 있음.
- 결과: results/turns/prompt_c2_turn0 ~ turn49. 최고 2턴 7/10, 14턴 3/6, 그 뒤 15~49턴은 모두 0 pass.
- 35턴 분석부터 분석 프롬프트가 바뀜(가정 점검 + 구조가 다른 후보). 이전 판: PROMPT_C1_ANALYST_until_turn34.md
- 49턴 진단은 좁은 전방 감지 + 정지·회전(초기 방식)으로 돌아가라는 지시. 50턴은 이 진단과 49턴 코드로 시작함.
- 재개:
  1. vLLM 이 떠 있는지 확인: `curl -s 127.0.0.1:8420/v1/models`
     없으면 `tmux new -d -s vllm120b "GPU_UTIL=0.80 bash fractal_setup/serve_gpt-oss-120b_gpu.sh > ~/logs/vllm120b.log 2>&1"`
     (0.75 이하는 131k 컨텍스트에 KV 캐시가 모자라 뜨지 않음. 로드 직후 GPU 97 GB 까지 잠깐 오름.
      segfault 가 나면 ~/.cache/vllm/torch_compile_cache 를 옮기고 다시 띄움)
  2. C-3 등 다른 Isaac 실행이 없는지 확인 (한 번에 하나만 돎)
  3. `cd ~/SDI-Scenario/accident_scene/arena && PYTHONFAULTHANDLER=1 nohup python3 c2_auto.py --start 50 --max-turn 80 >> ~/logs/c2_auto.log 2>&1 &`
- 턴 로그: ~/logs/c2_turn<N>.log, 자동 진행 로그: ~/logs/c2_auto.log

## C-3 (코더·검토·분석 모두 Claude Opus 5.5) - 2026-11-16 22:44 시작
- 과제, base, 프롬프트, 출발 코드, 하네스 규칙은 C-2 와 같음. LLM 호출만 agent_loop.claude_post:
  `claude -p --tools "" --model claude-opus-5-5 --effort high`, 빈 폴더 ~/c3_agent_cwd, MCP·설정 없음, 호출마다 새 프로세스.
- 결과: results/turns/prompt_c3_turn<N> (태그 claude_opus_5_5_pC3t<N>), 실행 이름 c3t<N>_s<seed>
- 시작: `bash fractal_setup/start_c3.sh` (0턴 뒤 `c2_auto.py --series c3 --start 1 --max-turn 30`)
- 중간에 끊겼으면: `PYTHONFAULTHANDLER=1 nohup python3 c2_auto.py --series c3 --start <다음 턴> --max-turn 30 >> ~/logs/c3_auto.log 2>&1 &`
- 로그: ~/logs/c3_auto.log, ~/logs/c3_turn<N>.log

## 에이전트에게 공개하지 않는 것
- 참조 코드(Claude 작성, 2026-11-16, 같은 조건 10/10 pass, 587 s)와 그 결과: ~/c2_reference_private/ (저장소 밖).
  실행 결과 파일은 results/zrefA_s*. results/turns 아래나 프롬프트에 넣지 않음.

## 알려진 문제
- 분석 단계의 이전 턴 요약 계산(sensor_digest.py static)에서 프로세스가 드물게 죽음(15·40턴 segfault, 29턴 numpy 속성 오류).
  턴별 표를 _prior_table_cache.json 으로 저장해 재계산을 없앰. 호스트도 2026-11-15~16 에 예고 없이 여러 번 재부팅됨 -
  하드웨어 의심, 관리자 확인 필요(커널 로그는 sudo 필요).
- 재부팅되면 crontab @reboot 가 tmux sdi 에 Claude Code(Remote Control)를 다시 띄움. vLLM·실험은 자동으로 켜지지 않음.

## 2026-11-16 23:30 변경 (C-3 3턴 분석부터, C-2 를 재개하면 C-2 에도 적용됨)
- PROMPT_C2.md 에 턴 통과 기준 추가: 한 턴 600초 예산을 실행들이 나눠 쓰고, 10개 모두 합계 600초 안 통과가 기준.
  이전 판: PROMPT_C2_until_c3turn2.md
- 실행 요약 표에 time_limit_s, sim_time_s, ended_by 추가. 분석 입력 제목의 "각 600초 채점 조건"(사실과 다름) 수정.
- 근거 검사기: "stall_abort 가 null" 처럼 없다고 말하는 문장을 거절하던 오류 수정.

## C-4 (2026-11-17 예약): C-3 과 같되 코더만 gpt-oss-120b(vLLM 8420), 검토·분석은 Claude Opus 5.5
- C-3 자동 진행이 "목표 달성"으로 끝나면 fractal_setup/start_c4.sh 가 0턴부터 시작(~/logs/c4_auto.log, c4_turn<N>.log).
- 결과: results/turns/prompt_c4_turn<N> (태그 gptoss_coder_claude_pC4t<N>), 실행 이름 c4t<N>_s<seed>
- 이어서 돌리기: `PYTHONFAULTHANDLER=1 nohup python3 c2_auto.py --series c4 --start <다음 턴> --max-turn 30 >> ~/logs/c4_auto.log 2>&1 &`
- vLLM 이 떠 있어야 함(코더). 분석 입력의 DBG 절이 커도 분석은 Claude 라 무방.
- C-3 성공 시 할 일(사용자 지시): 전체 진행 정리 PPT + 최종 성공 동영상 3개 삽입.

## 결과 (2026-11-17 02:42)
- C-3 (전 역할 Claude): 9턴에 10/10, 578 s. 턴별 0/3, 0/3, 6/7, 6/9, 0/3, 2/5, 7/8, 9/9, 5/8, 10/10.
- C-4 (코더만 gpt-oss-120b): 2턴에 10/10, 484 s. 턴별 1/4, 5/8, 10/10.
- 보고 PPT: ~/c3_report/SDI_arena_C2_C3_report.pptx (build.js 로 재생성, C-4 결과는 아직 안 들어감)
- Isaac 렌더 영상(추종 시점): results/frames_zrec9_follow_s*, ~/c3_report/media/isaac_follow_s*.mp4
- 지금 도는 실험 없음. vLLM 은 8420 에 떠 있음(GPU 78 GB).

## C-5 (2026-11-17 07:33 시작)
- 분석만 gpt-oss-120b(vLLM 8420), 코더·검토는 Claude Opus 5.5. 나머지는 C-3 과 같음. 태그 `claude_gptoss_analyst_pC5t<N>`, 실행 이름 `c5t<N>_s<seed>`.
- 시작: `tmux new -d -s c5 "bash fractal_setup/start_c5.sh > ~/logs/c5_auto.log 2>&1"`; 재개: `python3 c2_auto.py --series c5 --start <N> --max-turn 30 >> ~/logs/c5_auto.log`
- 분석 입력이 gpt-oss 한도(131072 - 답 32768 - 여유 8000 = 90304 토큰)를 넘으면 하네스가 DBG 줄 수를 실행별 10+50 → 5+25 → 3+12 → 2+5 → 0 으로 줄이고, 줄였다는 사실을 DBG 머리말에 적는다(c1_multi_analyze.analyze). Claude 분석에는 적용 안 됨. 이전 파일: ~/logs/c1_multi_analyze_before_c5.py
- C-5 결과(2026-11-17 14:54 종료): 30턴까지 목표 미달성. 최고 17턴 7/10(10회 모두 실행, 601 s), 그다음 2턴 4/5. 최종 코드 2,800줄대. DBG 줄이기 장치는 한 번도 작동하지 않음(분석 입력 최대 78k 토큰). 재개: `python3 c2_auto.py --series c5 --start 31 --max-turn N`.

## C-6 (2026-11-17 16:37 시작)
- 분석만 Qwen3.6-27B(추론 모드, BF16 원본, ~/models/Qwen3.6-27B), 코더·검토는 Claude Opus 5.5. 태그 `claude_qwen36_analyst_pC6t<N>`, 실행 이름 `c6t<N>_s<seed>`.
- 서버: `tmux new -d -s vllmqwen "bash fractal_setup/serve_qwen36-27b_gpu.sh > ~/logs/vllm_qwen36.log 2>&1"` (포트 8420, 0.80, 131072 토큰, `--max-num-seqs 4` 필수 - 기본값 1024 는 기동 실패). gpt-oss 와 동시에 못 띄움.
- 시작: `tmux new -d -s c6 "bash fractal_setup/start_c6.sh > ~/logs/c6_auto.log 2>&1"`; 재개: `python3 c2_auto.py --series c6 --start <N> --max-turn 30 >> ~/logs/c6_auto.log`
- 분석 한 번에 약 4분(C-5 8턴 입력 36.7k 토큰 → 245 s, 추론 5.5k 토큰). 요청 설정은 다른 계열과 같음(temperature 0.7, top_p 0.95, max_tokens 32768).
- C-6 결과(2026-11-17 23:03 종료): 30턴까지 목표 미달성. 최고 5/8(15·16·19턴), 최고 통과 수 5. 분석 한 번에 170~400 s. 재개: `python3 c2_auto.py --series c6 --start 31 --max-turn N` (Qwen 서버를 다시 띄워야 함).

## C-7 (2026-11-17 준비)
- 분석만 Gemma 4 31B(it, BF16, 추론 켬), 코더·검토는 Claude Opus 5.5. 태그 `claude_gemma4_analyst_pC7t<N>`, 실행 이름 `c7t<N>_s<seed>`.
- 서버 `fractal_setup/serve_gemma4-31b_gpu.sh`, 자동 전환 `fractal_setup/switch_to_c7.sh`(tmux c7wait, ~/logs/c7_switch.log), 시작 `fractal_setup/start_c7.sh`(~/logs/c7_auto.log).
- Gemma 는 추론이 기본 꺼짐: `agent_loop.post` 가 모델 이름이 gemma 로 시작하면 `chat_template_kwargs.enable_thinking=true` 를 넣는다.
- C-7 시작 2026-11-18 00:27. Gemma 서버는 `--kv-cache-dtype fp8` 가 있어야 0.80 에서 131072 토큰으로 뜬다(없으면 KV 캐시 23.3 GiB 필요, 남는 것은 15.0 GiB → 기동 실패). 시험 요청: 입력 24.3k 토큰 → 407 s, 추론 8.0k 토큰, 답 형식 정상.
- Isaac 은 GPU 를 약 6~8 GB 쓴다(gpu_watch.csv: vLLM 77~78 GB → 실행 중 84~86 GB).
- C-7 결과(2026-11-18 09:58 종료): 30턴까지 목표 미달성. 최고 8/8(25턴, 600 s 를 8회에 다 씀), 7/9 네 번(18·21·22·24턴), 7/8(23턴). 충돌은 11턴 이후 턴당 0~3건, 통과 실행 평균 72~88 s(필요 60 s). 분석 한 번에 100~400 s. 재개: `python3 c2_auto.py --series c7 --start 31 --max-turn N`.
- C-7 에서 드러난 하네스의 정보 틈(미적용): (1) 예산이 딱 떨어져 남은 실행을 시작하지 못하면 실행 요약에 실패 행이 없어 분석이 "모두 통과"로 읽는다(14·25턴). (2) 검토가 진단에 대해 적은 의문이 다음 분석 입력에 들어가지 않는다. (3) 코더의 답 형식으로는 정의를 지울 수 없다.

## C-7b 와 C-8 (2026-11-18, 사용자 지시: 하네스에 사실 두 가지를 넣고 C-7 을 다시, 그 뒤 C-8)
- 하네스 변경(계열 c71, c8 에만 적용, `c1_cycle.FACT_SERIES`): 분석 입력에 (1) "이번 턴 실행 집계" - 배치 10개 중 몇 개를 실행했고 몇 개를 왜 못 했는지, 통과 실행의 평균·최소·최대 시간과 필요한 실행당 평균 60 s, 통과 기준 충족 여부, (2) 검토 담당이 진단에 대해 적은 의문(diagnosis_concerns) 을 넣는다. 이전 파일: ~/logs/c1_cycle_before_c71.py, ~/logs/c1_multi_analyze_before_c71.py
- C-7b: 계열 이름 `c71`(실행 이름 `c71t<N>_s<seed>`, 기록 `results/turns/prompt_c71_turn<N>`), 분석 Gemma 4 31B, 0턴부터. 2026-11-18 10:23 시작, tmux c71, ~/logs/c71_auto.log. 재개: `python3 c2_auto.py --series c71 --start <N> --max-turn 30`
- C-8: 계열 `c8`, 분석 EXAONE 4.5 33B(`fractal_setup/serve_exaone45-33b_gpu.sh`, 추론 기본 켜짐, 연구·비상업 라이선스). C-7b 가 끝나면 `fractal_setup/switch_to_c8.sh`(tmux c8wait, ~/logs/c8_switch.log)가 서버를 바꾸고 시험한 뒤 `start_c8.sh` 를 띄운다. 문맥은 131072 → 114688 → 98304 순으로 시도하고, 분석 입력 한도는 서버의 max_model_len 에서 읽는다(`c1_multi_analyze.server_context`).
- C-7b 결과(2026-11-18 17:37 종료): 30턴까지 목표 미달성. 9/10 세 번(2턴 573 s, 22턴 560 s, 27턴 523 s), 9/9 한 번(1턴). 3~19턴은 단순 반응형 코드로 대부분 0~3건 통과, 20턴에 코더가 추적·예측 구조를 넣은 뒤 회복. 3턴에서 하네스의 근거 검사 버그(regex 가 "." 를 숫자로 읽음)로 분석이 죽어 고치고 3턴 분석만 다시 받음.
- 2026-11-18 17:38: C-8 전환 중 EXAONE 가중치를 올리던 때(GPU 63 GB 확보 직후, 로딩 20초쯤) 호스트가 예고 없이 죽고 17:39:44 재부팅됨. vLLM 모델 로딩 중 호스트 다운은 세 번째(앞의 두 번은 gpt-oss-120b). C-8 은 시작되지 않음. 재시도: `tmux new -d -s c8wait "bash fractal_setup/switch_to_c8.sh > ~/logs/c8_switch.log 2>&1"` (C-7b 종료와 내려받기 완료는 이미 충족돼 바로 서버를 올림). GPU 감시(`~/logs/gpu_watch.csv`)도 재부팅으로 꺼졌으니 `resume_after_reboot.sh` 의 nohup 줄로 다시 켤 것.
- C-8 시작 2026-11-18 21:01(재시도에서 EXAONE 서버 정상 기동). 21:22 1턴 분석 요청 중 vLLM 엔진이 segfault(flashinfer prefill + fp8 KV 캐시; 호스트는 무사). 서빙 스크립트에 `--attention-backend TRITON_ATTN` 을 넣어 다시 올리고, `fractal_setup/resume_c8_turn1.sh` 로 1턴 분석만 다시 받은 뒤 2턴부터 이어 감. 잘못 시작된 2턴은 `prompt_c8_turn2_aborted_vllmcrash`.

## C-8b (`c81`) paused after turn 18 (2026-11-19, user: token budget; resume the following week)

- Series `c81` = C-8 again (EXAONE 4.5 33B analyst, turn facts) + analyst picks `base_turn`, stage-2 direction on that
  turn's code, prior table with start-code and mean-pass-time columns, parse-error retry text. `c72` = same with Gemma
  (C-7c reached 10/10 at turn 3, 592 s).
- Results so far (turn: pass/run): 0:1/4, 1:5/6, 2:0/3, 3:5/6, 4:6/8, 5:7/8, 6:6/7, 7:7/7, 8:3/5, 9:4/6, 10:6/8,
  11:7/8, 12:4/6, 13:0/3, 14:2/4, 15:0/3, 16:7/7, 17:4/6, 18: see `~/logs/c81_turn18.log`.
- Resume: start the EXAONE server (user approval for a vLLM start; host has crashed during model load before):
  `tmux new -d -s vllmexaone "bash fractal_setup/serve_exaone45-33b_gpu.sh > ~/logs/vllm_exaone45.log 2>&1"`, wait for
  `curl 127.0.0.1:8420/v1/models`, then
  `tmux new -d -s c81 "cd ~/SDI-Scenario/accident_scene/arena && PYTHONFAULTHANDLER=1 python3 c2_auto.py --series c81 --start 19 --max-turn 30 >> ~/logs/c81_auto.log 2>&1"`.
  The autopilot reads turn 18's diagnosis (its `base_turn`) and continues.
- Nothing after commit 6a21786 is committed (C-8, C-7c, C-8b results; revert feature; wall-clock guard).

> 2026-11-22: C-8b was resumed at turn 19 (SIGCONT to the autopilot) and runs to turn 30; C-5b (`c51`, gpt-oss-120b analyst, same method) starts after it via `~/logs/c51_chain.sh`. The pause section above is history.
