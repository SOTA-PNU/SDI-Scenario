"""C-1: one analyst call over many runs of the same code on different random mover layouts (2026-11-15).

Input: PROMPT_C1_ANALYST.md (with a header saying what this call gets instead of one run), the coder's task prompt, the
base, the code, the coder's stated reason, multi_run_digest.py's compact digest of every run, and the attempt history.

    python3 c1_multi_analyze.py --code results/turns/prompt_c1_turn1/gpu_gpt_oss_120b_pC1t1_block.py \\
        --turn results/turns/prompt_c1_turn1/gpu_gpt_oss_120b_pC1t1_turn.json --seeds results/c1t1_seeds.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as AL            # noqa: E402
import c1_once as C1               # noqa: E402
import multi_run_digest as MR      # noqa: E402
import urllib.request
import c1_turn as CT               # noqa: E402  REASONING_EFFORT, MAX_TOKENS

HEADER = """
## 이번 분석에서 받는 자료 (위 목록 대신)

이번에는 같은 코드를 물체 배치만 바꿔 여러 번 돌린 결과를 한꺼번에 받습니다. 실행 하나의 원본 기록 대신, 하네스가 실행마다 계산한
압축 요약(`여러 실행 요약`)을 받습니다: 결과, 충돌 순간 로봇 상태와 상대 물체의 로봇 기준 위치·속도·접근 속도, 충돌 전 5초를 0.25초
간격으로 본 로봇 상태와 라이다로 본 지도에 없는 가장 가까운 물체, 처음 보인 시점, 정지·제자리 회전 구간. 맨 앞 `table` 은 실행별 한 줄
비교입니다. 물체는 번호(mover_00 ...)로만 부르며 구간·속도는 주지 않습니다(배치는 매번 바뀝니다).
코드가 'DBG' 로 시작하는 줄을 print 하면 `DBG 출력` 에 실행별로 (처음 10줄과 마지막 50줄, 줄당 300자까지) 들어옵니다.
잘린 양은 실행마다 적혀 있고(출력한 줄 수, 전달한 줄 수, 뒤가 잘린 줄 수), 잘린 줄 끝에는 `…(잘림)` 표시가 붙습니다.
한 실행만 보고 결론 내지 말고, 여러 실행에 공통된 실패 원인을 찾아 근거에 여러 실행을 인용하십시오. 실패 유형이 여러 가지면 가장 많은
실패를 일으킨 원인부터 다룹니다.
"""


PRIOR_HEADER = """
## 이전 턴 요약 (하네스가 계산한 표)

이전 턴마다 시험한 코드의 설정값, 그 설정을 만든 지시의 요지, 결과(통과 수)와 실패 유형 개수입니다. 숫자는 하네스가 각 턴의 시험
결과에서 센 값입니다. 이 표로 어떤 설정과 지시가 결과를 좋게 했고 나쁘게 했는지 비교해, 결과가 나빴던 설정이나 방향을 되풀이하지
마십시오. **이 표는 이전 턴 것이며 이번 진단의 evidence 가 아닙니다.** evidence 는 이번 턴 `여러 실행 요약` 의 값만 인용합니다.
"""
PRIOR_DIRECTION_CHARS = 160

# C-8b on (user 2026-11-19): the analyst chooses which turn's code the next coder starts from. Until now the next turn
# always built on the latest code, so an analyst that saw results get worse could only pile instructions on top
# (C-8: 5/8 at turn 17, 1,055 lines -> 0/3 at turn 30, 1,759 lines, with no way back).
BASE_TURN_SECTION = """
## 다음 턴의 시작 코드 고르기 (`base_turn`)

다음 턴의 수정 담당은 **당신이 고른 턴의 코드**에서 시작해 당신의 `direction` 을 반영합니다. 고를 수 있는 턴: {allowed}.
이번 턴은 {turn}턴이고, 이번에 시험한 코드는 {started} 만든 것입니다.

결과가 이전보다 나빠졌다면 먼저 **직전 지시가 옳았는지** 판단하십시오(`이전 턴 요약` 표의 통과 수와 통과 실행 평균 시간을 비교).
그런 다음 둘 중 하나를 고릅니다.

- 직전 지시가 옳았고 모자란 것만 채우면 된다: `base_turn` 을 이번 턴({turn})으로 두고, 이번 코드에 더할 지시를 씁니다.
- 직전 지시(또는 그 앞 몇 턴의 지시)가 틀렸다: 결과가 더 좋았던 이전 턴을 `base_turn` 으로 고르고, 그 턴의 코드에 적용할
  **새 지시**를 씁니다. 그 턴 뒤에 더해진 변경은 모두 버려집니다. 이미 실패한 지시를 그 코드에 다시 내리지 마십시오.

이전 턴의 코드는 이 입력에 없습니다(이번 코드만 있습니다). 이전 턴으로 돌아갈 때는 그 턴 뒤에 생긴 이름(상수, 함수)에 기대지 말고,
`direction` 을 동작으로 적으십시오(무엇을 언제 하게 할지). 그 턴의 설정값은 `이전 턴 요약` 표에 있습니다.

JSON 에 다음 두 항목을 더합니다.

 "last_direction_ok": "yes|no|unclear 와 그렇게 본 근거 한 문장 (직전 지시가 결과를 좋게 했는지)",
 "base_turn": 다음 턴이 시작할 코드의 턴 번호(정수)
"""


def _constants(block):
    """Top-level constant assignments of a code block, as 'NAME=value' (value source text, shortened)."""
    import ast
    out = []
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return out
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name.isupper() or (name.startswith("_") and name[1:].replace("_", "").isupper()):
                if name in ("_WAYPOINTS",):
                    continue
                src = ast.get_source_segment(block, n.value) or ""
                if len(src) <= 40 and "\n" not in src:
                    out.append(f"{name}={src}")
    return out


def _failure_counts(table):
    c = {"stall_abort": 0, "progress_abort": 0, "collision_driving": 0, "collision_stopped_or_turning": 0,
         "collision_static": 0, "time_limit": 0, "other_fail": 0}
    for r in table:
        if r.get("verdict") == "pass":
            continue
        if r.get("stall_abort_t") is not None:
            c["stall_abort"] += 1
        elif r.get("progress_abort_t") is not None:
            c["progress_abort"] += 1
        elif r.get("collision_t") is not None:
            if not str(r.get("partner") or "").startswith("mover"):
                c["collision_static"] += 1
            elif str(r.get("robot_at_collision") or "").startswith("주행"):
                c["collision_driving"] += 1
            else:
                c["collision_stopped_or_turning"] += 1
        elif r.get("ended_by") == "time_limit":
            c["time_limit"] += 1
        else:
            c["other_fail"] += 1
    return {k: v for k, v in c.items() if v}


def prior_analyses(turn_dirs, extra=False):
    """Harness-made table of earlier turns: settings -> result -> failure-type counts (no run names, no reasoning).
    Raw earlier diagnoses + reasoning made the analyst copy earlier turns' runs as its own evidence (C-2 turns 5-9)."""
    rows = []
    for n, d in turn_dirs:
        d = Path(d)
        tj = [p for p in d.glob("*_turn.json") if not p.name.endswith("coder_review.json")]
        blk = sorted(d.glob("*_block.py"))
        diags = [p for p in sorted(d.glob("*_multi*_diagnosis.json")) if "BAD" not in p.name and "ARCH" not in p.name]
        if not tj or not diags:
            continue
        try:
            tdata = json.loads(tj[0].read_text())
            table = json.loads(diags[0].read_text()).get("table") or []
        except ValueError:
            continue
        # recompute from the runs' files where possible (saved tables predate the static-partner fix)
        series = d.name.split("_turn")[0].replace("prompt_", "")
        variants = [f"{series}t{n}_s{s}" for s in (tdata.get("seeds") or [])
                    if (HERE / "results" / f"{series}t{n}_s{s}_result.json").exists()]
        # cached per turn (2026-11-16): recomputing every earlier turn's lidar digest on every turn took longer each
        # turn, and the process died there three times (turn 15 and 40 segfault, turn 29 a broken numpy lookup)
        cache = d / "_prior_table_cache.json"
        try:
            if cache.exists():
                c = json.loads(cache.read_text())
                if c.get("variants") == variants:
                    table, variants = c["table"], []
            if variants:
                table = MR.table([MR.run_summary(v, HERE / "results") for v in variants])
                cache.write_text(json.dumps({"variants": variants, "table": table}, ensure_ascii=False))
        except (OSError, ValueError, KeyError):
            pass
        if not table:
            continue
        direction = str((tdata.get("diagnosis_in") or {}).get("direction") or "출발점 코드 그대로")
        if len(direction) > PRIOR_DIRECTION_CHARS:
            direction = direction[:PRIOR_DIRECTION_CHARS] + "…"
        passes = sum(1 for r in table if r.get("verdict") == "pass")
        if extra:      # C-8b: which code the turn started from, and how long its passing runs took
            tp = [r.get("sim_time_s") or 0.0 for r in table if r.get("verdict") == "pass"]
            bt = tdata.get("base_turn")
            rows.append(f"| {n} | {'출발점' if n == 0 else (f'{bt}턴 코드' if bt is not None else '-')} | "
                        f"{', '.join(_constants(blk[0].read_text())) if blk else '-'} | {direction} | "
                        f"{passes}/{len(table)} | {f'{sum(tp) / len(tp):.0f}초' if tp else '-'} | "
                        f"{json.dumps(_failure_counts(table), ensure_ascii=False)} |")
            continue
        rows.append(f"| {n} | {', '.join(_constants(blk[0].read_text())) if blk else '-'} | {direction} | "
                    f"{passes}/{len(table)} | {json.dumps(_failure_counts(table), ensure_ascii=False)} |")
    if not rows:
        return ""
    if extra:
        return ("| 턴 | 시작한 코드 | 시험한 코드의 설정값 | 그 코드를 만든 지시(요지) | 통과 | 통과 실행 평균 시간 | 실패 유형 개수 |\n"
                "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    return ("| 턴 | 시험한 코드의 설정값 | 그 코드를 만든 지시(요지) | 통과 | 실패 유형 개수 |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")


DBG_HEAD, DBG_TAIL, DBG_CHARS = 10, 50, 300


def dbg_section(variants, log_dir, DBG_HEAD=DBG_HEAD, DBG_TAIL=DBG_TAIL, why=""):
    """The code's own DBG prints per run (C-3 turn 2: the analyst asked the coder for DBG lines, the coder printed them,
    and this multi-run input never carried them). First DBG_HEAD and last DBG_TAIL lines of each run.
    What was cut is stated per run and marked on the line (user 2026-11-17: in C-3 turn 6 over half of the lines were
    dropped and 48-92 lines a run lost their last 20-28 characters without a sign)."""
    parts = []
    for v in variants:
        f = Path(log_dir) / f"{v}.isaac.log"
        full = C1.RD.dbg_lines(f.read_text(errors="replace")) if f.exists() else []
        if not full:
            continue
        n_all, n_long, longest = len(full), sum(len(ln) > DBG_CHARS for ln in full), max(len(ln) for ln in full)
        lines = [ln if len(ln) <= DBG_CHARS else ln[:DBG_CHARS] + " …(잘림)" for ln in full]
        if len(lines) > DBG_HEAD + DBG_TAIL:
            lines = lines[:DBG_HEAD] + [f"... ({n_all - DBG_HEAD - DBG_TAIL}줄 생략)"] + lines[len(lines) - DBG_TAIL:]
        sent = min(n_all, DBG_HEAD + DBG_TAIL)
        stat = f"코드가 출력한 {n_all}줄 중 {sent}줄 전달"
        if n_long:
            stat += f", {DBG_CHARS}자를 넘어 뒤가 잘린 줄 {n_long}개(전체 기준, 가장 긴 줄 {longest}자)"
        parts.append(f"### {v}: {stat}\n```\n" + "\n".join(lines) + "\n```")
    if not parts:
        return "\n## DBG 출력\n\n(코드가 출력한 DBG 줄 없음)\n"
    return ("\n## DBG 출력 (코드가 print 한 'DBG' 로 시작하는 줄. 실행별 처음 %d줄과 마지막 %d줄, 줄당 %d자까지만 전달하며 "
            "넘는 부분은 버려집니다%s)\n\n" % (DBG_HEAD, DBG_TAIL, DBG_CHARS, why) + "\n\n".join(parts) + "\n")


STAGE2_PROMPT = """
당신은 다음 턴을 **{bt}턴의 코드**에서 시작하기로 했습니다. 방금 낸 `direction` 은 그 코드를 보지 않고 쓴 것입니다.
아래가 {bt}턴에 시험한 코드입니다({bt}턴 뒤의 변경은 들어 있지 않습니다). 이 코드를 읽고, 수정 담당이 **이 코드에** 적용할 지시를
다시 쓰십시오.

- 이 코드에 실제로 있는 이름(상수, 함수)만 가리키십시오. 없는 것을 새로 만들게 하려면 무엇을 하는 것인지 적습니다.
- 이번 턴 실행에서 본 실패를 이 코드가 그대로 겪을지, 이 코드에는 없는 문제인지 구분하십시오. 이번 턴의 근거는 이번 코드의
  실행에서 나온 것이고, 이 코드의 결과는 `이전 턴 요약` 표의 {bt}턴 줄입니다.
- 이 코드 다음에 내렸던 지시(`시도 기록` 의 #{nx})는 이미 시험했습니다. 그 뒤 결과가 좋아지지 않았다면 같은 지시를 다시 내리지 마십시오.
- 통과 기준은 배치 10개 모두 통과, 시뮬 합계 600초 이내입니다.

## {bt}턴의 코드

```python
{code}```

JSON 객체 하나만 냅니다. 코드 펜스나 설명을 붙이지 않습니다.

{{"chosen": "이 코드에 무엇을 할지와 이유 한 문장",
 "target": "고칠 곳(이 코드에 있는 이름)",
 "direction": "어떻게 구현할지 6문장 이내 (수정 에이전트와 검토 담당은 이 지시만 따릅니다)"}}
"""


def base_turn_errors(diag, revert):
    """C-8b: the answer must name the turn the next coder starts from (EXAONE left the new keys out when first tried)."""
    if not revert or not diag:
        return []
    try:
        ok = int(diag.get("base_turn")) in revert["allowed_set"]
    except (TypeError, ValueError):
        ok = False
    errs = [] if ok else [f"`base_turn` 이 없거나 고를 수 없는 값입니다({diag.get('base_turn')!r}). 고를 수 있는 턴: {revert['allowed']}. "
                          f"이번 코드에서 이어 가려면 {revert['turn']} 을 적습니다."]
    if not str(diag.get("last_direction_ok") or "").strip():
        errs.append("`last_direction_ok` 가 없습니다. 직전 지시가 결과를 좋게 했는지(yes|no|unclear)와 근거 한 문장을 적으십시오.")
    return errs


EVIDENCE_RETRIES = 2
RUN_RE = re.compile(r"(c\d+t\d+_s\d+)")


def check_evidence(diag, rows):
    """Evidence must match this turn's runs (C-2 turn 5 copied turn 4's stall numbers onto its own run names)."""
    if not diag or not isinstance(diag.get("evidence"), list):
        return ["진단 JSON 이나 evidence 목록이 없습니다."]
    errs = []
    for ev in diag["evidence"]:
        ev = str(ev)
        for run in set(RUN_RE.findall(ev)):
            row = rows.get(run)
            if row is None:
                errs.append(f"{run}: 이번 턴에 없는 실행입니다.")
                continue
            actual = (f"실제 {run}: verdict={row['verdict']}, stall_abort_t={row.get('stall_abort_t')}, "
                      f"progress_abort_t={row.get('progress_abort_t')}, collision_t={row.get('collision_t')}, "
                      f"partner={row.get('partner')}, robot_at_collision={row.get('robot_at_collision')}, "
                      f"stop_turns={row.get('stop_turns')}")
            m = re.search(r"stall_abort(?:_t)?\s*[=:]?\s*([0-9]+\.?[0-9]*)", ev)   # a digit is required: "stall_abort. 로봇" gave "." (C-7b turn 3 crash)
            # an evidence line that says the stall abort is absent ("stall_abort ... null/None/없") is not a claim of one
            # (C-3 turn 2: a correct "충돌·stall_abort·progress_abort 가 모두 null" was rejected three times)
            denies = re.search(r"stall[^.。]{0,60}(null|none|없)", ev, re.I)
            if (m or re.search(r"stall|정체 중단", ev, re.I)) and row.get("stall_abort_t") is None and "stall" in ev.lower() \
                    and not denies:
                errs.append(f"{run} 는 정체 중단이 없습니다. {actual}")
            elif m and row.get("stall_abort_t") is not None and abs(float(m.group(1).rstrip(".")) - row["stall_abort_t"]) > 0.2:
                errs.append(f"{run} 의 stall_abort_t 가 다릅니다. {actual}")
            m = re.search(r"collision(?:_t| at t| at)?\s*[=:]?\s*t?\s*=?\s*([0-9]+\.?[0-9]*)", ev)
            if m and row.get("collision_t") is None:
                errs.append(f"{run} 는 충돌이 없습니다. {actual}")
            elif m and abs(float(m.group(1).rstrip(".")) - row["collision_t"]) > 0.2:
                errs.append(f"{run} 의 충돌 시각이 다릅니다. {actual}")
            m = re.search(r"stop_turn(?:_episodes|s)?\s*[=:]?\s*([0-9]+)", ev)
            if m and row.get("stop_turns") is not None and int(m.group(1)) != row["stop_turns"]:
                errs.append(f"{run} 의 stop_turns 가 다릅니다. {actual}")
    return errs


CONTEXT_TOKENS = 131072


def server_context(base_url):
    """max_model_len of the model the vLLM server holds (C-8: EXAONE may be served with less than 131072)."""
    try:
        d = json.loads(urllib.request.urlopen(base_url + "/models", timeout=30).read())
        return int(d["data"][0].get("max_model_len") or CONTEXT_TOKENS)
    except Exception:
        return CONTEXT_TOKENS


def count_tokens(base_url, model, text):
    """Prompt tokens as the vLLM server counts them (its /tokenize); a rough guess if that fails."""
    try:
        req = urllib.request.Request(base_url.rsplit("/v1", 1)[0] + "/tokenize", method="POST",
                                     data=json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        return int(json.loads(urllib.request.urlopen(req, timeout=120).read())["count"])
    except Exception:
        return len(text) // 2


def analyze(code, reason, history, variants, log_dir, out, model="gpt-oss-120b", base_url="http://127.0.0.1:8420/v1",
            prior="", facts="", revert=None):
    """One analyst call over `variants` (runs of `code`); writes <out>_request/_raw/_diagnosis.json, returns (diag, table)."""
    base = (HERE / C1.BASE_FILE).read_text()
    dg = MR.digest(variants, log_dir)
    def make_body(dbg):
        return ((HERE / C1.ANALYST_PROMPT).read_text() + "\n" + HEADER
                + "\n---\n\n## 과제 프롬프트 (수정 에이전트가 받은 것)\n\n" + C1.task_prompt()
                + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + base + "```\n"
                + "\n---\n\n## 이번 코드\n\n```python\n" + code + "```\n"
                + "\n## 수정 에이전트가 밝힌 변경 이유\n\n" + (reason or "") + "\n"
                + facts
                + f"\n## 여러 실행 요약 ({len(variants)}개 배치. 턴 예산은 모든 실행 합계 600초이고, 실행별 시간 한도는 표의 time_limit_s)\n\n```json\n" + json.dumps(dg, ensure_ascii=False) + "\n```\n"
                + dbg
                + "\n## 시도 기록\n\n" + history + "\n"
                + (BASE_TURN_SECTION.format(**revert) if revert else "")
                + ((PRIOR_HEADER + "\n" + prior + "\n") if prior else "")
                + ("\n답 JSON 에는 `last_direction_ok` 와 `base_turn` 을 반드시 넣으십시오.\n" if revert and prior else ""))
    body = make_body(dbg_section(variants, log_dir))
    if not str(model).startswith("claude"):
        # C-5 (2026-11-17): a vLLM analyst has a context limit (C-3's inputs reached 130-167k tokens with DBG lines).
        # Fewer DBG lines per run until the input fits; the DBG header tells the analyst what was delivered and why.
        limit = server_context(base_url) - CT.MAX_TOKENS - 8000   # room for the answer and the evidence-check retries
        for head, tail in ((DBG_HEAD, DBG_TAIL), (5, 25), (3, 12), (2, 5), (0, 0)):
            if (head, tail) != (DBG_HEAD, DBG_TAIL):
                body = make_body(dbg_section(variants, log_dir, head, tail,
                                             ". 분석 모델의 입력 길이 한도 때문에 이번에는 줄 수를 줄였습니다"))
            n_tok = count_tokens(base_url, model, body)
            print(f"[multi] 입력 토큰 {n_tok} (한도 {limit}), DBG 실행별 {head}+{tail}줄", flush=True)
            if n_tok <= limit:
                break
    payload = {"model": model, "messages": [{"role": "user", "content": body}], "max_tokens": CT.MAX_TOKENS,
               "temperature": 0.7, "top_p": 0.95, "seed": 1000, "reasoning_effort": CT.effort_for("analyst")}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    t0 = time.time()
    rows = {r["run"]: r for r in dg["table"]}
    checks = []
    for attempt in range(EVIDENCE_RETRIES + 1):
        r = AL.post(base_url, payload, 7200)
        Path(f"{out}_raw.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        Path(f"{out}_raw_try{attempt}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))  # keep rejected answers
        u = r.get("usage", {})
        content = r["choices"][0]["message"].get("content") or ""
        print(f"[multi] {time.time() - t0:.0f}s 입력={u.get('prompt_tokens')} 출력={u.get('completion_tokens')} 답={len(content)}자", flush=True)
        s, e = content.find("{"), content.rfind("}")
        perr = None
        try:
            diag = json.loads(content[s:e + 1])
        except (json.JSONDecodeError, ValueError) as ex:
            diag = None
            if revert:      # C-8b: say what was wrong. EXAONE's rejected answers were bracket slips, and the retry text
                pos = getattr(ex, "pos", 0)     # (stale evidence) never told it so - it repeated them three times
                perr = [f"답이 올바른 JSON 이 아니어서 읽지 못했습니다({ex}). 오류 근처: …{content[s:e + 1][max(0, pos - 80):pos + 40]}… "
                        "괄호와 쉼표를 확인해, 같은 내용을 올바른 JSON 객체 하나로 다시 내십시오."]
        errs = perr or (check_evidence(diag, rows) + base_turn_errors(diag, revert))
        checks.append({"attempt": attempt, "errors": errs})
        if not errs:
            break
        print(f"[multi] 근거 검사 실패 {len(errs)}건 - 다시 요청: {errs[0][:160]}", flush=True)
        payload["messages"] = payload["messages"][:1] + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": ("답에 고칠 것이 있습니다.\n\n" if revert else
                                          "근거가 이번 실행 요약과 맞지 않습니다. 이전 턴 기록의 숫자를 이번 근거로 쓰면 안 됩니다.\n\n")
             + "\n".join(f"- {x}" for x in errs)
             + ("\n\n같은 JSON 형식 하나만 냅니다." if perr else
                "\n\n이번 턴 `여러 실행 요약` 의 실제 값만 근거로 진단을 다시 내십시오. 같은 JSON 형식 하나만 냅니다.")}]
        Path(f"{out}_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    if checks[-1]["errors"] and prior:
        # every attempt kept citing earlier turns' runs (C-2 turn 8, 39k-token input): ask once more without the record
        print("[multi] 근거 검사 계속 실패 - 이전 턴 분석 기록을 빼고 한 번 더 요청", flush=True)
        payload["messages"] = [{"role": "user", "content": body.split(PRIOR_HEADER)[0]}]
        Path(f"{out}_request_noprior.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        r = AL.post(base_url, payload, 7200)
        Path(f"{out}_raw.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        content = r["choices"][0]["message"].get("content") or ""
        s, e = content.find("{"), content.rfind("}")
        try:
            diag = json.loads(content[s:e + 1])
        except (json.JSONDecodeError, ValueError):
            diag = None
        checks.append({"attempt": "no_prior", "errors": check_evidence(diag, rows) + base_turn_errors(diag, revert)})
        print(f"[multi] 기록 없이: 근거 검사 {len(checks[-1]['errors'])}건", flush=True)
    if revert and diag:
        # an unreadable or not-allowed base_turn means "this turn's code" (the behaviour of every earlier series)
        bt = diag.get("base_turn")
        try:
            bt = int(bt)
        except (TypeError, ValueError):
            bt = None
        if bt not in revert["allowed_set"]:
            print(f"[multi] base_turn {diag.get('base_turn')!r} 은 고를 수 없는 값 - 이번 턴({revert['turn']}) 코드로 둠", flush=True)
            diag["base_turn_given"], bt = diag.get("base_turn"), revert["turn"]
        diag["base_turn"] = bt
        print(f"[multi] 다음 턴 시작 코드: {bt}턴 | 직전 지시 판단: {str(diag.get('last_direction_ok'))[:160]}", flush=True)
        code_k = (revert.get("code_of") or {}).get(bt)
        if bt != revert["turn"] and code_k and not checks[-1]["errors"]:
            # stage 2 (user 2026-11-19): the analyst wrote its direction without the older turn's code in view (in the
            # first trial it went back to turn 16 and named constants that exist only in turn 20). Show it that code
            # and have it write the direction again; the rest of the diagnosis stays as it is.
            msgs = payload["messages"][:1] + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": STAGE2_PROMPT.format(bt=bt, nx=bt + 1, code=Path(code_k).read_text())}]
            p2 = {**payload, "messages": msgs}
            s2 = None
            for attempt in range(EVIDENCE_RETRIES + 1):
                Path(f"{out}_request_stage2.json").write_text(json.dumps(p2, ensure_ascii=False, indent=1))
                r2 = AL.post(base_url, p2, 7200)
                Path(f"{out}_raw_stage2_try{attempt}.json").write_text(json.dumps(r2, ensure_ascii=False, indent=1))
                c2 = r2["choices"][0]["message"].get("content") or ""
                u2 = r2.get("usage", {})
                print(f"[multi] 2단계 {time.time() - t0:.0f}s 입력={u2.get('prompt_tokens')} 출력={u2.get('completion_tokens')} 답={len(c2)}자", flush=True)
                a_, b_ = c2.find("{"), c2.rfind("}")
                try:
                    s2 = json.loads(c2[a_:b_ + 1])
                    why = None if str(s2.get("direction") or "").strip() else "`direction` 이 비어 있습니다."
                except (json.JSONDecodeError, ValueError) as ex:
                    pos = getattr(ex, "pos", 0)
                    s2, why = None, (f"답이 올바른 JSON 이 아니어서 읽지 못했습니다({ex}). 오류 근처: …{c2[a_:b_ + 1][max(0, pos - 80):pos + 40]}… "
                                     "괄호와 쉼표를 확인하십시오.")
                if not why:
                    break
                s2 = None
                print(f"[multi] 2단계 답을 쓰지 못함 - 다시 요청: {why[:160]}", flush=True)
                p2 = {**payload, "messages": msgs + [{"role": "assistant", "content": c2},
                                                     {"role": "user", "content": why + " 같은 JSON 형식 하나만 다시 내십시오."}]}
            if s2:
                diag["stage1"] = {k: diag.get(k) for k in ("chosen", "target", "direction")}
                for k in ("chosen", "target", "direction"):
                    if str(s2.get(k) or "").strip():
                        diag[k] = s2[k]
                print(f"[multi] 2단계: {bt}턴 코드를 보고 지시를 다시 씀", flush=True)
            else:
                diag["stage2_failed"] = True
                print("[multi] 2단계 실패 - 1단계 지시를 그대로 씀", flush=True)
    failed = bool(checks[-1]["errors"])
    Path(f"{out}_diagnosis.json").write_text(json.dumps({"diagnosis": diag, "raw": content, "table": dg["table"],
                                                         "evidence_checks": checks, "evidence_failed": failed},
                                                        ensure_ascii=False, indent=1))
    return diag, dg["table"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--turn", required=True, help="c1_turn.py's *_turn.json (coder reason, diagnosis in)")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--prefix", default="c1t1_s")
    ap.add_argument("--log-dir", default="results")
    ap.add_argument("--out", default="results/turns/prompt_c1_turn1/gpu_gpt_oss_120b_pC1t1_multi10")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base", default="http://127.0.0.1:8420/v1")
    a = ap.parse_args()
    base = (HERE / C1.BASE_FILE).read_text()
    code = (HERE / a.code).read_text()
    turn = json.loads((HERE / a.turn).read_text())
    reason = next((c.get("reason") for c in reversed(turn["coder"]) if c.get("review", {}) and c["review"].get("verdict") == "pass"), "")
    seeds = json.loads((HERE / a.seeds).read_text())
    dg = MR.digest([f"{a.prefix}{s}" for s in seeds], HERE / a.log_dir)
    n_fail = sum(1 for r in dg["table"] if r["verdict"] != "pass")
    history = ("#0 현재 코드(A-4 경유점, 속도 5배, 센서 미사용): 출발 직후 mover_00 과 충돌\n"
               f"#1 이번 코드(진단 {json.dumps(turn['diagnosis_in'].get('direction'), ensure_ascii=False)} 반영, 검토 통과): "
               f"배치 1개에서 pass(53.1 s) 뒤, 새 배치 {len(seeds)}개 중 {len(seeds) - n_fail}개 pass, {n_fail}개 fail")
    body = ((HERE / C1.ANALYST_PROMPT).read_text() + "\n" + HEADER
            + "\n---\n\n## 과제 프롬프트 (수정 에이전트가 받은 것)\n\n" + C1.task_prompt()
            + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + base + "```\n"
            + "\n---\n\n## 이번 코드\n\n```python\n" + code + "```\n"
            + "\n## 수정 에이전트가 밝힌 변경 이유\n\n" + (reason or "") + "\n"
            + f"\n## 여러 실행 요약 ({len(seeds)}개 배치. 턴 예산은 모든 실행 합계 600초이고, 실행별 시간 한도는 표의 time_limit_s)\n\n```json\n" + json.dumps(dg, ensure_ascii=False) + "\n```\n"
            + "\n## 시도 기록\n\n" + history + "\n")
    payload = {"model": a.model, "messages": [{"role": "user", "content": body}], "max_tokens": 16384,
               "temperature": 0.7, "top_p": 0.95, "seed": 1000}
    out = HERE / a.out
    (out.parent).mkdir(parents=True, exist_ok=True)
    Path(f"{out}_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    t0 = time.time()
    r = AL.post(a.base, payload, 7200)
    Path(f"{out}_raw.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    u = r.get("usage", {})
    content = r["choices"][0]["message"].get("content") or ""
    print(f"[multi] {time.time() - t0:.0f}s 입력={u.get('prompt_tokens')} 출력={u.get('completion_tokens')} 답={len(content)}자", flush=True)
    s, e = content.find("{"), content.rfind("}")
    try:
        diag = json.loads(content[s:e + 1])
    except (json.JSONDecodeError, ValueError):
        diag = None
    Path(f"{out}_diagnosis.json").write_text(json.dumps({"diagnosis": diag, "raw": content, "table": dg["table"]},
                                                        ensure_ascii=False, indent=1))
    print(json.dumps(diag, ensure_ascii=False, indent=1) if diag else content, flush=True)


if __name__ == "__main__":
    main()
