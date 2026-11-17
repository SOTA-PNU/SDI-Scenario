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
