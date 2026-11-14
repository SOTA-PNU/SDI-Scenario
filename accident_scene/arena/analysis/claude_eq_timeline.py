"""Render a sandboxed Claude run (REPORT.md 실험 7) as a per-turn event timeline.

Reads the session + subagent transcripts that claude_agent_loop.py copied out
of the sandbox and lays every event on one clock: tool calls (with a one-line
input and result), thinking summaries, the agent's own narration, and the
messages the harness sent. Subagent events are tagged with the subagent's type
so "who did what" stays readable when the main agent delegates.

    python3 analysis/claude_eq_timeline.py claude_eq_a0
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "turns" / "claude_eq"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def one_line(s, n=110):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def tool_input(name, i):
    if name == "Bash":
        cmd = (i.get("command") or "").strip().splitlines()
        return one_line((i.get("description") + " :: " if i.get("description") else "")
                        + (cmd[0] if cmd else ""), 140)
    for k in ("file_path", "pattern", "description", "path", "url"):
        if i.get(k):
            extra = f" ({len(i['content'])}자)" if name == "Write" and i.get("content") else ""
            return one_line(f"{i[k]}{extra}", 140)
    return one_line(json.dumps(i, ensure_ascii=False), 140)


def result_text(b):
    c = b.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    lines = [l for l in str(c or "").splitlines() if l.strip()]
    if not lines:
        return "(출력 없음)"
    return one_line(lines[-1], 120) + (f"  [{len(lines)}줄]" if len(lines) > 1 else "")


def events(path, who):
    out = []
    pending = {}
    for ln in path.read_text().splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        t = ts(d.get("timestamp"))
        m = d.get("message") or {}
        c = m.get("content") if isinstance(m, dict) else None
        if d.get("type") == "user" and isinstance(c, str) and who == "main":
            out.append((t, who, "PROMPT", f"하네스 → 에이전트 메시지 {len(c)}자: {one_line(c, 90)}"))
        for b in c if isinstance(c, list) else []:
            k = b.get("type")
            if k == "tool_use":
                pending[b["id"]] = len(out)
                out.append([t, who, b["name"], tool_input(b["name"], b.get("input") or {}), None])
            elif k == "tool_result" and b.get("tool_use_id") in pending:
                row = out[pending.pop(b["tool_use_id"])]
                row[4] = ("⚠ " if b.get("is_error") else "") + result_text(b)
            elif k == "thinking" and (b.get("thinking") or "").strip():
                out.append((t, who, "THINK", one_line(b["thinking"], 400)))
            elif k == "text" and b.get("text", "").strip() and d.get("type") == "assistant":
                out.append((t, who, "SAY", one_line(b["text"], 400)))
    return out


def main(stem):
    tdir = OUT / f"{stem}_transcripts" / "projects"
    mains = [p for p in tdir.glob("*/*.jsonl")]
    if not mains:
        sys.exit(f"no transcripts under {tdir}")
    evs = []
    for p in mains:
        evs += events(p, "main")
        for sj in sorted((p.parent / p.stem / "subagents").glob("*.jsonl")):
            meta = sj.with_suffix(".meta.json")
            kind = json.loads(meta.read_text()).get("agentType", "?") if meta.exists() else "?"
            evs += events(sj, f"sub:{kind}:{sj.stem[-6:]}")
    evs = [list(e) + [None] * (5 - len(e)) for e in evs if e[0] is not None]
    evs.sort(key=lambda e: e[0])
    lines = [f"# {stem} — 이벤트 타임라인", "",
             "세션·서브에이전트 전사에서 자동 생성 (`analysis/claude_eq_timeline.py`).",
             "시각은 각 회차 메시지 수신 기준 경과 시간. `THINK` = 사고 요약, `SAY` = 에이전트가 쓴 문장.", ""]
    turn, t0 = 0, None
    for t, who, kind, what, res in evs:
        if kind == "PROMPT":
            turn += 1
            t0 = t
            lines += ["", f"## {turn}회차", "", f"- `+00:00` {what}"]
            continue
        dt = (t - t0).total_seconds() if t0 else 0
        stamp = f"+{int(dt // 60):02d}:{int(dt % 60):02d}"
        tag = "" if who == "main" else f" **[{who}]**"
        if kind in ("THINK", "SAY"):
            lines.append(f"- `{stamp}`{tag} *{kind}* {what}")
        else:
            lines.append(f"- `{stamp}`{tag} **{kind}** `{what.replace('`', '´')}`"
                         + (f" → {res.replace('`', '´')}" if res else ""))
    out = OUT / f"{stem}_timeline.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(evs)} events, {turn} turns")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "claude_eq_a0")
