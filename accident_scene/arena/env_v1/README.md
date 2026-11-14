# env_v1 — 실험 1–9 의 환경 원본 (2026-10-08 보존)

2026-10-08 에 박스 04·05·06·13 의 높이를 0.6 m 로 올리기 직전의 파일을 그대로 복사했다
(`cp -p`, 해시 동일 확인).

| 파일 | 내용 |
|---|---|
| `base_carter_run.py` | v1 MISSION 표 — 박스 04·05·06·13 = 0.25·0.20·0.18·0.20 m |
| `nova_carter_warehouse_arena.yaml` | v1 시나리오 명세 |
| `layout.json` | v1 배치 (상위 폴더의 `layout.json` 과 동일; 분석 스크립트와 논문 그림은 그쪽을 읽는다) |

실험 1–9 에서 만든 실행 파일(`results/generated/`, `results/turns/`, `*_carter_run.py`)은 각자
v1 MISSION 표를 담고 있으므로, 다시 채점하면 v1 환경으로 돈다.

v1 을 다시 쓰려면 이 폴더의 `base_carter_run.py` 와 `nova_carter_warehouse_arena.yaml` 을 상위 폴더로
되돌리면 된다. 변경 내용과 Isaac 실측은 상위 `REPORT.md` 의 '환경 v2' 절에 있다.
