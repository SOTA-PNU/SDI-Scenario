"""Every track-A turn, with the sentences that actually passed between the two sides.

The feedback wrapper is identical on all 187 turns, so it is printed once in full
and each turn then shows only what changed: the numbers that were substituted in,
the model's own diagnosis in its own words, and the harness' verdict. The initial
6,961-character prompt is likewise shown once, not per turn.

    python3 TRACK_A_EXCHANGE.py [--out TRACK_A_EXCHANGE.md] [--diag-chars 600]
"""

import argparse, glob, json, os, re, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "results", "turns")


def groups():
    tp = [f for f in sorted(glob.glob(os.path.join(T, "cap8", "*_turns.json")))
          if json.load(open(f))["track"] == "a"]
    lg = [os.path.join(T, "cap30_npu", "npu_gpt_oss_120b_a2_turns.json")]
    ab = sorted(glob.glob(os.path.join(T, "ablation_feedback", "*_turns.json")))
    return [("턴 프로토콜 (상한 8회차)", tp, T),
            ("장기 실행 (상한 30회차)", lg, T),
            ("피드백 절제 실험 (상한 8회차)", ab, os.path.join(T, "ablation_feedback"))]


def raw(stem, n, d):
    p = os.path.join(d, f"{stem}_t{n}_raw.json")
    return json.load(open(p)) if os.path.exists(p) else None


def req(stem, n, d):
    p = os.path.join(d, f"{stem}_t{n}_request.json")
    return json.load(open(p)) if os.path.exists(p) else None


def sent_values(t):
    """The numbers the feedback substituted in — i.e. the previous turn's verdict."""
    if t.get("verdict") == "static_dq":
        return "정적 실격 — " + " ".join((t.get("detail") or "").split())[:120]
    if t.get("runtime_error"):
        return f"실행 오류 — `{t['runtime_error']}`"
    c, d_, y = t.get("collisions"), t.get("final_dist_m"), t.get("partners")
    s = f"verdict=fail · final_dist={d_} m (tol 0.75) · collision_count={c}"
    if y:
        s += f" · partners={y}"
    return s


def diagnosis(rr, n):
    if not rr:
        return None
    m = rr["choices"][0]["message"]
    txt = m.get("reasoning") or m.get("content") or ""
    txt = " ".join(txt.split())
    return txt[:n] + ("…" if len(txt) > n else "")


def main(out, dc):
    L = []; A = L.append
    A("# 트랙 A — 반복 회차별로 오간 내용\n")
    A("프롬프트 A(`PROMPT.md`)로 수행한 모든 반복의 기록이다. 회차마다 **보낸 값**(직전 판정),")
    A("**모델이 돌려준 진단**(자기 추론 트레이스의 말 그대로), **하네스의 판정**을 싣는다.")
    A("반복되는 문구와 6,961자 초기 프롬프트는 아래에 한 번만 전문으로 싣고, 회차별로는")
    A("달라지는 부분만 보인다. 원본은 `results/turns/*_request.json` · `*_raw.json`.\n")
    A("생성: `python3 TRACK_A_EXCHANGE.py`\n")

    # ---------- 반복 횟수 표 ----------
    A("---\n\n## 1. 반복 횟수\n")
    A("| 구분 | 실행 | 가속기 | 회차 | 피드백 | 통과 |")
    A("|---|---|---|---|---|---|")
    tot_r = tot_t = tot_f = 0
    for label, files, _ in groups():
        for f in files:
            d = json.load(open(f))
            n = len(d["turns"]); acc = "NPU" if d["tag"].startswith("npu_") else "GPU"
            cond = d.get("condition", "default")
            name = f"{d['track']}{d['sample']}" + ("" if cond == "default" else f"·{cond}")
            A(f"| {label} | `{name}` | {acc} | {n} | {n - 1} | "
              f"{d['passed_at_turn'] if d.get('passed_at_turn') else '—'} |")
            tot_r += 1; tot_t += n; tot_f += n - 1
    A(f"| **합계** | **{tot_r}실행** | | **{tot_t}** | **{tot_f}** | **1** |\n")
    A("피드백은 회차 **사이**에 일어나므로 실행마다 `회차 − 1` 이다. 1회차는 블라인드라")
    A("앞에 피드백이 없다. 통과 1건은 `gpu a4` 가 5회차에 달성한 것이다.\n")

    # ---------- 오가는 문구의 틀 ----------
    A("---\n\n## 2. 오가는 문구 — 전문 (모든 회차 공통)\n")
    A("### 2.1 1회차에 보낸 것\n")
    A("`PROMPT.md` 전문(1,934자)과 `base_carter_run.py` 전문(5,027자)을 아래 형태로 이어붙인")
    A("**단일 user 메시지 6,961자**. 시스템 프롬프트·도구·추가 지시 없음.\n")
    A("```")
    A("<PROMPT.md 전문>")
    A("")
    A("---")
    A("")
    A("`base_carter_run.py`:")
    A("")
    A("```python")
    A("<base_carter_run.py 전문>")
    A("```")
    A("```\n")

    A("### 2.2 2회차 이후 보낸 것 — 피드백\n")
    A("직전 회차의 판정을 **같은 대화에 이어서** 보낸다. 문구는 모든 회차가 동일하고,")
    A("```json``` 안의 값만 바뀐다.\n")
    A("```")
    A("제출한 코드를 고정 하네스에서 실행했습니다. 판정 결과입니다.")
    A("")
    A("```json")
    A("{ verdict, criteria(허용치 4종), oracles(reached_goal / no_collision 과 각 detail),")
    A("  metrics(time_to_goal_s, min_goal_dist_m, final_dist_m, final_yaw_err_rad,")
    A("          final_pose, collision_count, collision_partners, first_collision_t,")
    A("          sim_time_s, wall_time_s, frames), aborted_after_collision, seed }")
    A("```")
    A("")
    A("같은 규칙 그대로, `[EDIT REGION]` 블록을 고쳐 다시 제출하십시오.")
    A("```\n")
    A("**장애물 배치(`obstacle` 블록)는 뺀다** — 박스 14개의 좌표·크기·높이가 그대로")
    A("들어 있어, 주면 과제가 무너진다.\n")
    r = req("npu_gpt_oss_120b_a0", 2, T)
    if r:
        A("실제 전송본 한 건 (`npu_gpt_oss_120b_a0_t2_request.json`):\n")
        A("```")
        A(r["messages"][-1]["content"].strip())
        A("```\n")

    A("### 2.3 정적 실격일 때 보낸 것\n")
    for label, files, src in groups():
        done = False
        for f in files:
            d = json.load(open(f)); stem = f"{d['tag']}_{d['track']}{d['sample']}" + \
                ("" if d.get("condition", "default") == "default" else f"_{d['condition']}")
            for t in d["turns"]:
                if t.get("verdict") == "static_dq":
                    nx = req(stem, t["turn"] + 1, src)
                    if nx:
                        A("```")
                        A(nx["messages"][-1]["content"].strip())
                        A("```\n"); done = True; break
            if done: break
        if done: break

    # ---------- 회차별 ----------
    A("---\n\n## 3. 회차별 기록\n")
    A("`보낸 값` 은 위 2.2 틀의 `json` 안에 채워진 값이다. `모델의 진단` 은 그 회차 추론")
    A("트레이스의 첫 부분을 말 그대로 옮긴 것이다.\n")
    for label, files, src in groups():
        A(f"\n## — {label} —\n")
        for f in files:
            d = json.load(open(f))
            stem = f"{d['tag']}_{d['track']}{d['sample']}" + \
                ("" if d.get("condition", "default") == "default" else f"_{d['condition']}")
            acc = "NPU" if d["tag"].startswith("npu_") else "GPU"
            cond = d.get("condition", "default")
            A(f"### {acc} · a{d['sample']}" + ("" if cond == "default" else f" · {cond} 조건"))
            A(f"\n{len(d['turns'])}회차 · 피드백 {len(d['turns']) - 1}회 · "
              f"{'**' + str(d['passed_at_turn']) + '회차 통과**' if d.get('passed_at_turn') else '통과 없음'}"
              + ("  ·  ⚠ 이 실행이 8회차 실행의 2~8회차 응답 원문을 덮어썼다"
                 if (src == T and cond == "default" and "cap30" in f) else "") + "\n")
            for t in d["turns"]:
                i = t["turn"]
                A(f"**{i}회차**\n")
                if i == 1:
                    A("- 보낸 것: 2.1 (블라인드)")
                else:
                    A(f"- 보낸 값: `{sent_values(d['turns'][i - 2])}`")
                rr = raw(stem, i, src)
                dg = diagnosis(rr, dc)
                if t.get("reused"):
                    A("- 받은 것: 1회차 생성물 재사용")
                elif dg:
                    u = rr.get("usage", {})
                    A(f"- 받은 것: {u.get('completion_tokens')} tok"
                      + (f" · {t['tok_per_s']} tok/s" if t.get("tok_per_s") else ""))
                    A(f"- 모델의 진단:\n\n  > {dg}\n")
                else:
                    A("- 받은 것: *(원문 유실)*")
                A(f"- 판정: {sent_values(t)}\n")
    A("\n---\n\n## 4. 합계\n")
    A(f"- 트랙 A **{tot_r}실행 · {tot_t}회차 · 피드백 {tot_f}회 · 통과 1건**")
    A("- 통과는 `gpu a4` 5회차 (거리 0.697 m, 허용 0.75 m, 충돌 0, 56.4 s)")
    A("- 187회차 어디에서도 모델의 진단에 '상자가 센서 평면 아래라 검출되지 않는다' 취지의")
    A("  문장이 나오지 않는다. 판단에 필요한 두 값(상자 높이 0.18~0.25 m, 센서 평면 0.35 m)은")
    A("  1회차부터 같은 파일에 평문으로 주어져 있었다.\n")
    open(out, "w").write("\n".join(L))
    print(f"-> {out} ({len(L)}줄 · {sum(len(x) for x in L):,}자 · {tot_t}회차)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "TRACK_A_EXCHANGE.md"))
    ap.add_argument("--diag-chars", type=int, default=600)
    a = ap.parse_args()
    main(a.out, a.diag_chars)
