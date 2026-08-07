# Level 0 — 고정 프롬프트 (로컬 LLM 벤치마크 입력)

## 사용 방법 (채점자용 — 이 섹션은 모델에게 입력하지 않습니다)

모델에게는 아래 **프롬프트 본문**과 `base_carter_run.py` **전문**만 입력하고, 수정된
`[EDIT REGION]` 블록**만** 출력받습니다. 파일 생성·실행·채점을 모델에게 시키지 마십시오
— 절차 실패가 제어 능력 측정을 오염시킵니다. 저장소를 열어 주는 에이전트식 입력
("이 PROMPT.md를 읽고 작업 수행해줘")도 금지합니다 — 이 채점자용 섹션까지 읽고 파일
생성·실행을 시도하다 무너지고, 같은 폴더의 정답 파일(solution/prompted)을 읽어
블라인드가 깨집니다. 에이전트형 CLI 에 붙여넣을 때는 입력 맨 앞에 "파일을 만들거나
실행하지 말고, 수정된 블록만 출력해줘." 한 줄을 덧붙입니다.

저장소 루트에서:

```bash
# 1) 모델 입력 생성 (프롬프트 본문 + base 전문)
{ sed -n '/^## 프롬프트 본문/,$p' levels/level0/PROMPT.md | tail -n +2; \
  echo; echo '--- base_carter_run.py 전문 ---'; cat levels/level0/base_carter_run.py; } > input.txt

# 2) 모델 출력 블록을 block.txt 로 저장해 채점 (이식 → 규칙검사 → Isaac 실측 → 판정, PASS=exit 0)
python3 levels/common/grade_block.py 0 <모델이름> block.txt
```

블록에 마커 줄·코드펜스가 섞여 있어도 grade_block.py 가 정리해 base 사본에 이식하고,
`levels/level0/results/<모델이름>_result.json` 의 `verdict` 로 판정합니다.

이 저장소의 기준 구현: [`prompted_carter_run.py`](prompted_carter_run.py) —
아래 프롬프트만 보고 작성했고 PASS 실측됨 (`results/prompted_result.json`).

## 프롬프트 본문 (여기부터 모델 입력, base_carter_run.py 전문을 뒤에 붙입니다)

당신은 모바일 로봇 제어 코드를 작성하는 엔지니어입니다. 아래에 Isaac Sim에서 Nova Carter
로봇을 구동하는 파이썬 러너 파일 `base_carter_run.py` 전문이 주어집니다. 이 파일의
`[EDIT REGION]` ~ `[END EDIT REGION]` 블록**만** 수정해서 미션을 통과시키십시오.

규칙:
1. `[EDIT REGION]` 블록 밖은 한 글자도 바꾸지 마십시오 (상단 givens, 하단 하네스 포함).
2. `controller(t, pose, env) -> (v, w, done)` 계약을 지키십시오. 매 스텝(1/60 s) 호출되며
   `pose`는 (x, y, yaw) 실측 월드 좌표, `v`는 전진 속도 [m/s], `w`는 yaw 각속도 [rad/s],
   `done=True`면 미션 종료 후 채점됩니다.
3. import 추가 금지 — 이미 import 된 `math` 만 사용하십시오.
4. 난수·벽시계에 의존하지 말고 `pose` 기반으로 결정론적으로 제어하십시오.
5. 최종 출력은 수정된 `[EDIT REGION]` 블록 전체(마커 주석 포함)만 제시하십시오.

미션(레벨 0): 로봇은 (-6.0, -1.0)에서 yaw=π(-x 방향)를 바라보고 스폰됩니다. 목표
(-8.0, -1.0)은 스폰 헤딩 **정면 2.0 m** 앞의 빈 통로에 있습니다. **전진만으로**(v ≥ 0,
w는 항상 0) 목표에 도달해 정지하십시오.

성공 조건: 최종 위치가 목표에서 1.0 m 이내 (yaw는 채점 안 함), 제한 시간 60 s.

접근 지침: 목표까지 남은 거리를 `pose`로 계산해, 허용 반경 안쪽에 충분히 들어왔을 때
정지하고 `done=True`를 반환하십시오. 그 전까지는 일정한 전진 속도로 직진하면 됩니다.
