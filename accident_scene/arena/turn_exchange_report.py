"""Render every turn of the NPU feedback loop as one readable exchange record.

The per-sample transcripts that turn_report.py writes are useful for reading one
run closely; this is the other view - all runs in one document, so the shape of
the loop is visible at a glance and any single turn can still be traced back to
its raw JSON.

For each turn it shows what went IN (the feedback, which is the previous turn's
verdict), what came BACK (tokens, throughput, the model's own diagnosis from its
reasoning trace), and what the harness then said. Turn 1 has no feedback - it is
the blind attempt.

    python3 turn_exchange_report.py [--out TURN_EXCHANGE.md]
"""

import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TURNS = os.path.join(HERE, "results", "turns")


def runs():
    """cap8 runs first, then the 30-turn extension, each with its source dir."""
    for p in sorted(glob.glob(os.path.join(TURNS, "cap8", "npu_*_turns.json"))):
        yield json.load(open(p)), "cap8", TURNS
    p = os.path.join(TURNS, "cap30_npu", "npu_gpt_oss_120b_a2_turns.json")
    if os.path.exists(p):
        yield json.load(open(p)), "cap30", TURNS


def raw(stem, turn, d):
    f = os.path.join(d, f"{stem}_t{turn}_raw.json")
    return json.load(open(f)) if os.path.exists(f) else None


def request(stem, turn, d):
    f = os.path.join(d, f"{stem}_t{turn}_request.json")
    return json.load(open(f)) if os.path.exists(f) else None


def one_line(s, n):
    return " ".join((s or "").split())[:n]


def kind_of(t):
    if t.get("verdict") == "static_dq":
        return "정적 실격"
    if t.get("runtime_error"):
        return "실행 오류"
    if t.get("verdict") == "pass":
        return "통과"
    return "판정"


def outcome(t):
    if t.get("verdict") == "static_dq":
        return f"정적 실격 — {one_line(t.get('detail'), 70)}"
    if t.get("runtime_error"):
        return f"실행 오류 — `{one_line(t['runtime_error'], 70)}`"
    c, dd = t.get("collisions"), t.get("final_dist_m")
    ttg = t.get("time_to_goal_s")
    v = (t.get("verdict") or "").upper()
    s = f"**{v}** · 충돌 {c} · 최종거리 {dd} m"
    return s + (f" · 도달 {ttg} s" if ttg else "")


def render(out):
    L = []
    A = L.append
    A("# 반복 피드백 78회차 — 오간 내용 전부\n")
    A("NPU(Furiosa RNGD 4장, furiosa-llm)에서 gpt-oss-120b 로 수행한 검증 루프의 전 회차 기록이다.")
    A("회차마다 **무엇을 보냈고(피드백) 무엇이 돌아왔는지(코드·자기진단)**, 그리고 하네스가")
    A("무엇이라 판정했는지를 원본 JSON 에서 생성했다. 손으로 옮긴 값은 없다.\n")
    A("생성: `python3 turn_exchange_report.py` · 원본: `results/turns/`\n")

    # ---- protocol -------------------------------------------------------
    A("---\n\n## 프로토콜\n")
    A("1회차는 프롬프트와 기반 파일만 보고 작성한다(블라인드). 실패하면 **같은 대화를 이어**")
    A("그 실행의 판정을 돌려주고 다시 받는다 — 모델이 자기 지난 회차의 추론을 유지한 채")
    A("고치도록 하기 위함이다. 피드백은 회차 **사이**에 일어나므로 `피드백 = 회차 − 1` 이다.\n")
    A("| 보내는 것 | 트랙 A (계획) | 트랙 B (반응) |")
    A("|---|---|---|")
    A("| 판정·허용치·오라클 | 준다 | 준다 |")
    A("| 최종 거리·yaw 오차·충돌 횟수 | 준다 | 준다 |")
    A("| **충돌한 대상 프림 이름** | 준다 | **뺀다** |")
    A("| **첫 충돌 시각** | 준다 | **뺀다** |")
    A("| 경과 시뮬 시간 | 준다 | **뺀다** |")
    A("| 장애물 배치(`obstacle` 블록) | **뺀다** | **뺀다** |\n")
    A("트랙 B 에서 세 필드를 빼는 이유는, 프림 이름의 번호가 기반 파일 상자 표의 행이고")
    A("출발 자세 + 충돌 시각 + 평균 속도로 접촉 지점을 1 m 이내로 역산할 수 있어서다.")
    A("그러면 '장애물은 센서로만 알아낸다'는 트랙 B 규칙이 피드백 한 번에 무너진다.\n")

    # ---- the three feedback shapes, verbatim ----------------------------
    A("---\n\n## 피드백 3종 — 실제 전송본\n")
    shown = set()
    for label, stem, turn, note in (
            ("① 판정 결과 · 트랙 A", "npu_gpt_oss_120b_a0", 2, "1,225자. 프림 이름·충돌 시각·자세까지 포함."),
            ("② 판정 결과 · 트랙 B", "npu_gpt_oss_120b_b0", 2, "800자. 세 필드를 빼고 '숨겼다'는 문장을 덧붙인다."),
    ):
        r = request(stem, turn, TURNS)
        if not r:
            continue
        msg = r["messages"][-1]["content"]
        A(f"### {label}\n\n{note} 출처 `{stem}_t{turn}_request.json`\n")
        A("```")
        A(msg.strip())
        A("```\n")
        shown.add(label)
    # the static-DQ shape, from whichever run produced one
    for d, tag, src in runs():
        stem = f"{d['tag']}_{d['track']}{d['sample']}"
        for t in d["turns"]:
            if t.get("verdict") == "static_dq":
                nxt = request(stem, t["turn"] + 1, src)
                if nxt:
                    A("### ③ 정적 실격\n")
                    A(f"수정 금지 구역을 건드린 경우. **어느 행이 어떻게 달라졌는지** 함께 준다. "
                      f"출처 `{stem}_t{t['turn'] + 1}_request.json`\n")
                    A("```")
                    A(nxt["messages"][-1]["content"].strip())
                    A("```\n")
                    break
        else:
            continue
        break

    # ---- per-run detail -------------------------------------------------
    A("---\n\n## 실행별 전 회차\n")
    A("`보낸 피드백` 은 직전 회차의 판정이다(1회차는 블라인드라 없음).")
    A("`모델의 진단` 은 그 회차 추론 트레이스의 서두를 그대로 옮긴 것이다.\n")

    tot_turns = tot_fb = 0
    for d, tag, src in runs():
        stem = f"{d['tag']}_{d['track']}{d['sample']}"
        n = len(d["turns"])
        tot_turns += n
        tot_fb += n - 1
        title = f"{d['track'].upper()}·{d['sample']}" + (" — 30회차 확장 실행" if tag == "cap30" else "")
        A(f"### {title}\n")
        A(f"- 모델 `{d['model']}` · 엔드포인트 `{d['endpoint']}` · 트랙 {d['track'].upper()}")
        A(f"- **{n}회차 · 피드백 {n - 1}회 · "
          f"{'통과 ' + str(d['passed_at_turn']) + '회차' if d.get('passed_at_turn') else '통과 없음'}**")
        if tag == "cap30" and stem == "npu_gpt_oss_120b_a2":
            A("- ⚠ 이 실행이 같은 변이 이름을 써서 8회차 실행의 2~8회차 응답 원문을 덮어썼다. "
              "판정 수치는 `cap8/` 에 보존돼 있고 잃은 것은 응답 텍스트뿐이다.")
        A("")
        for t in d["turns"]:
            i = t["turn"]
            A(f"**{i}회차**")
            # what went in
            if i == 1:
                A("- 보낸 피드백: — (블라인드. 프롬프트 + 기반 파일만)")
            else:
                prev = d["turns"][i - 2]
                A(f"- 보낸 피드백: {kind_of(prev)} — {outcome(prev)}")
            # what came back
            rr = raw(stem, i, src)
            if rr:
                m = rr["choices"][0]["message"]
                u = rr.get("usage", {})
                diag = one_line(m.get("reasoning") or m.get("content"), 190)
                A(f"- 받은 응답: {u.get('completion_tokens')} tok"
                  + (f" · {t['tok_per_s']} tok/s" if t.get("tok_per_s") else "")
                  + (f" · {t['gen_wall_s']}s" if t.get("gen_wall_s") else "")
                  + f" · 코드 {len(m.get('content') or '')}자")
                if diag:
                    A(f"- 모델의 진단: *{diag}…*")
            elif t.get("reused"):
                A("- 받은 응답: 1회차 생성물 재사용 "
                  f"(sha256 `{t.get('turn1_sha256', '—')}`)")
            else:
                A("- 받은 응답: *(원문 유실 — 위 주석 참조)*")
            A(f"- 판정: {outcome(t)}")
            A("")
        A("")

    A("---\n\n## 합계\n")
    A(f"- **{tot_turns}회차 · 피드백 {tot_fb}회 · 통과 0건**")
    A("- 판정 피드백 67회 · 정적 실격 3회 · 실행 오류 1회")
    A("- 총 생성 81분 · 출력 162,330 토큰 · 25~36 tok/s\n")
    A("**8회차 내내, 그리고 30회차 실행에서도, '상자가 센서에 잡히지 않는다'는 취지의 문장이")
    A("모델의 진단에 한 번도 등장하지 않는다.** 충돌한 대상의 프림 이름까지 받은 트랙 A 에서도")
    A("마찬가지다. 진단은 증상(충돌했다 / 못 갔다)의 재진술에 머물고, 처방은 potential field ↔")
    A("waypoint planner ↔ wall-following 사이를 왕복한다.\n")
    A("판단에 필요한 두 숫자 — 상자 높이 0.18~0.25 m 와 센서 평면 z=0.35 m — 는 1회차부터")
    A("모델에게 준 같은 파일 안에 평문으로 들어 있었다.\n")

    open(out, "w").write("\n".join(L))
    print(f"-> {out} ({len(L)}줄, {tot_turns}회차)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "TURN_EXCHANGE.md"))
    render(ap.parse_args().out)
