# zero2level — 통일 base + 미니멀 프롬프트 벤치마크

`levels/` 의 사다리 방식(레벨 N 의 base = 레벨 N-1 의 solution)과 달리, 여기서는
**네 레벨의 `base_carter_run.py` 가 바이트 단위로 동일한 한 파일**입니다 (md5 일치).
레벨은 폴더 이름(level0~3)에서 읽고, 목표·톨러런스·장애물 같은 givens 는 파일 안의
MISSIONS 테이블에서 뽑습니다. `[EDIT REGION]` 은 빈 컨트롤러(제어 코드 없음)로 시작하고,
프롬프트는 목적 달성에 필요한 요청사항만 담아 접근 지침·구현 힌트를 넣지 않았습니다.

## 구조

```
levels_zero2level/
  level{0..3}/
    base_carter_run.py      # 4개 레벨 모두 동일한 파일 — 폴더 이름으로 레벨 판별,
                            # givens 는 MISSIONS 테이블, [EDIT REGION] 은 빈 컨트롤러
    PROMPT.md               # 전문 = 모델 입력 (필요 요청사항만, 힌트 없음)
    prompted_carter_run.py  # 프롬프트만 보고 작성한 기준 구현 (블라인드 도출, PASS 실측)
    results/                # prompted_result.json 등
```

base 의 공통 하네스는 `levels/common/` 을 그대로 참조합니다 (`carter_env.py` 등).

## 프로토콜

`PROMPT.md` 전문이 곧 모델 입력입니다. 프롬프트는 모델에게 `base_carter_run.py` 를
복사한 `<모델 본인 이름>_carter_run.py` 를 만들어 그 파일의 `[EDIT REGION]` 블록만
수정하라고 지시합니다 — base 는 비교 기준으로 보존됩니다. 검증은 채점자가 별도 수행:

```bash
# 저장소 루트에서 (N = 레벨)
python3 levels/common/check_edit_region.py levels_zero2level/levelN/<모델이름>_carter_run.py
bash levels/run_isaac.sh levels_zero2level/levelN/<모델이름>_carter_run.py
# 판정: levels_zero2level/levelN/results/<모델이름>_result.json 의 verdict
```

## prompted 실측 (2026-08-10, seed 42 — 클로드 블라인드 도출: PROMPT.md+base 만 열람)

| 레벨 | verdict | time_to_goal | final_d | yaw_err | 충돌 | 도출된 접근 |
|---|---|---|---|---|---|---|
| L0 | **PASS** | 1.50 s | 0.13 m | — | 0 | 거리 비례 감속 전진, 정지 반경 0.15 m |
| L1 | **PASS** | 8.37 s | 0.24 m | 0.010 rad | 0 | 3단계 P 제어 (회전→주행→정렬) |
| L2 | **PASS** | 10.08 s | 0.20 m | 0.038 rad | 0 | 스캔으로 박스 모서리를 월드 좌표 추정 → 넓은 쪽 우회 웨이포인트 4개 생성 |
| L3 | **PASS** | 8.10 s (≤12) | 0.12 m | 0.009 rad | 0 | 프롬프트의 박스 좌표 기반 고정 웨이포인트 우회 + 속도 스케줄 |

네 레벨 모두 첫 도출·첫 실측에서 무충돌 통과했습니다.

## 사다리판(`levels/`)과의 비교 메모

- 사다리판 프롬프트는 기존 코드 서사와 접근 지침(회피 여유 0.55 m 등)을 담지만, 이
  트랙은 미션·성공 조건·API 만 줍니다. 그런데도 L2 실측이 10.08 s 로 사다리판
  prompted(17.43 s)보다 빨랐습니다 — 힌트가 반응형(bearing-차단) 접근을 유도한 반면,
  힌트 없는 프롬프트에서는 웨이포인트 우회 접근이 도출된 결과입니다.
- L3 는 이전 레벨 코드를 물려받지 않으므로 "기존 코드 속도 튜닝" 과제가 아니라
  "회피+시간 예산을 처음부터 설계" 과제가 됩니다.
