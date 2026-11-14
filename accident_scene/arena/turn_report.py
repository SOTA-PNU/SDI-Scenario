"""Render one sample's turn protocol as a readable transcript.

The turn log is machine-shaped; this is the human record the paper cites:
for every turn, what was asked, what came back, how fast, and what the
harness said about it.  The full request bodies stay in
results/turns/*_request.json - this quotes only the part that changed,
because the first message is the fixed prompt and repeating it per turn
would bury the feedback.

    python3 turn_report.py results/turns/npu_gpt_oss_120b_b0_turns.json
"""

import json
import sys
from pathlib import Path


def brief(code, n=40):
    lines = [l for l in code.splitlines()]
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[:n]) + f"\n... ({len(lines) - n}줄 생략)"


def render(log_path):
    log = json.loads(Path(log_path).read_text())
    turns_dir = Path(log_path).parent
    arena = turns_dir.parent.parent
    out = []
    stem = f"{log['tag']}_{log['track']}{log['sample']}"
    out.append(f"# 턴 프로토콜 기록 — {stem}\n")
    out.append(f"- 모델: `{log['model']}`  ·  엔드포인트: `{log['endpoint']}`")
    out.append(f"- 트랙: {log['track'].upper()}  ·  샘플: {log['sample']}")
    passed = log.get("passed_at_turn")
    n_turns = len(log["turns"])
    status = f"{passed}회차에 통과" if passed else f"{n_turns}회차까지 실패"
    out.append(f"- 결과: **{status}**\n")

    for t in log["turns"]:
        n = t["turn"]
        out.append(f"\n---\n\n## {n}회차\n")
        req = turns_dir / f"{t['variant']}_request.json"
        if n == 1 and not req.exists():
            out.append("**보낸 것**: `PROMPT.md`/`PROMPT_B.md` + `base_carter_run.py` (블라인드, 1회차)\n")
        elif req.exists():
            msgs = json.loads(req.read_text())["messages"]
            last_user = [m for m in msgs if m["role"] == "user"][-1]
            out.append(f"**보낸 것** (대화 {len(msgs)}턴, 마지막 사용자 메시지):\n")
            out.append("```\n" + last_user["content"].strip() + "\n```\n")
        if t.get("trimmed_exchanges"):
            out.append(f"> 컨텍스트 한도로 오래된 교환 {t['trimmed_exchanges']}건을 잘랐습니다.\n")

        if "completion_tokens" in t:
            out.append(f"**생성**: {t['gen_wall_s']}s · 출력 {t['completion_tokens']} tok "
                       f"({t.get('tok_per_s')} tok/s) · 추론 {t.get('reasoning_tokens', 0)} tok "
                       f"· 프롬프트 {t.get('prompt_tokens')} tok · finish={t.get('finish')}"
                       + (" · splice 복원" if t.get("spliced") else "") + "\n")
        sub = arena / f"{t['variant']}_carter_run.py"
        if sub.exists():
            src = sub.read_text()
            begin = "# [EDIT REGION]"
            end = "# ========================= [END EDIT REGION]"
            if begin in src and end in src:
                block = src[src.index(begin):src.index(end)]
            else:
                block = src
            out.append(f"**나온 코드** (`{sub.name}`, EDIT REGION {len(block.splitlines())}줄):\n")
            out.append("```python\n" + brief(block) + "\n```\n")

        v = t.get("verdict")
        if v == "static_dq":
            out.append(f"**판정**: 정적 실격 — {t.get('detail')}\n")
        elif t.get("runtime_error"):
            out.append(f"**판정**: 런타임 오류 — `{t['runtime_error']}`\n")
        elif v:
            out.append(f"**판정**: **{v.upper()}** · 최종거리 {t.get('final_dist_m')} m · "
                       f"충돌 {t.get('collisions')} · 도달 {t.get('time_to_goal_s')} s\n")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        md = Path(p).with_suffix("").with_suffix("")
        dest = Path(p).parent / (Path(p).stem.replace("_turns", "") + "_transcript.md")
        dest.write_text(render(p))
        print("wrote", dest)
