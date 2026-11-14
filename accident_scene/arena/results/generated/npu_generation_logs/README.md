# NPU 1회차 생성 로그 (2026-09-21, job scratch에서 2026-10-08 보존)

논문 표 4 주 ᵃ·ᵇ의 근거다. 원래 세션 작업 폴더(`~/.claude/jobs/fdd8b179/tmp/`)에만 있어서 프로젝트로 옮겼다.

- `run_npu.sh` — NPU 1회차 생성 스크립트. 모델마다 예열 요청을 먼저 보낸다(21–26행).
- `npu_generation.log` — 요청별 시간과 출력 토큰.
  - 20행: EXAONE 예열 `http=200 97.4s`.
  - 38행: gpt-oss-120b 예열 `http=503 100.46s`. 41행 a0 `185.6s`는 그 직후 첫 요청이다.
  - 23–27행: EXAONE 트랙 A 5건의 요청 시간 70.8/62.7/73.6/70.7/72.8 s. 원시 JSON에는 `_bench`가 없다.
- `qwen3_remeasure.log` — Qwen3 재측정. 예열 결과를 버렸다(`warmup discarded (14.7s)`).
