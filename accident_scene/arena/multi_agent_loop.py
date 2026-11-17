"""M-1 (DESIGN_MULTI_AGENT.md): the B-5-29 task split between a harness and two model roles.

The harness (this file, no model) owns the rules: it runs every code a coder answer contains as a
200 s test drive on the grading scene, sends a test-passing code unchanged to grading, and stops on
a grading pass, --max-grades failed gradings, --max-iters iterations or --max-wall-h hours.
Each iteration asks two fresh conversations of the same model:
  coder    PROMPT_M1_CODER.md + base + the previous iteration's code + the latest diagnosis + the attempt
           log -> one EDIT REGION block
  analyst  PROMPT_M1_ANALYST.md + the coder's prompt + base + this code + run_digest + raw DBG windows
           + the attempt log -> one diagnosis JSON
Test drives reuse agent_loop.Run.test (same runner files, isaac_logs, cleanup); grading reuses
turn_loop.grade. Nothing of agent_loop's turn loop runs here.

Kept for every run (--out-dir):
  <stem>.log is the caller's redirect; <stem>_events.jsonl, <stem>_i<NN>_<role>[_r<k>]_request.json /
  _raw.json / _raw_text.txt, <stem>_i<NN>_digest.json, <stem>_i<NN>_diagnosis.json, <stem>_attempts.jsonl,
  <stem>_turns.json (one record per grading, same keys run_*_sequential.sh reads), isaac_logs/.

    python3 multi_agent_loop.py --model gpt-oss-120b --tag gpu_gpt_oss_120b_pM1 --sample 0 \\
        --base http://127.0.0.1:8420/v1 --out-dir results/turns/prompt_m1
"""

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as AL            # noqa: E402  Run.test, build, static_check, region_of, code_key, post, raw_text
import run_digest as RD            # noqa: E402
import turn_loop as TL             # noqa: E402

CODER_PROMPT = "PROMPT_M1_CODER.md"
ANALYST_PROMPT = "PROMPT_M1_ANALYST.md"
FORMAT_RETRIES = 2                 # extra coder/analyst asks inside one iteration when the answer has the wrong form
ATTEMPT_LOG_LINES = 10
NONE_STREAK_STOP = 2               # analyst said function=none this many times in a row -> stop for a human
MODES = {"no_path", "stall", "collision", "runtime_error", "static", "slow_progress", "other"}
FUNCS = {"plan_path", "lookahead_target", "follow_cmd", "none"}
CHANGE_RE = re.compile(r"#\s*\[시험\s*\d+\][^\n]*")
FUNC_RE = re.compile(r"^def (\w+)\(", re.M)
DIFF_CAP_CHARS = 8000
EDITABLE = ("plan_path", "lookahead_target", "follow_cmd")
# directions the task prompt forbids (PROMPT_B5_29.md 경로와 지도 계약 / 구현 규칙); seen in M-1 v1 and M-1c diagnoses
FORBIDDEN = [
    (r"직선.{0,20}(대체|fallback|반환|시도)|(straight|direct)[- ]?(line|segment).{0,30}(fallback|return)|fallback.{0,30}(직선|straight)",
     "계획 실패 때 검사하지 않은 시작-목표 직선으로 대체하는 것은 금지입니다."),
    (r"segment_free.{0,30}(제거|생략|빼|없애|건너|skip|remove|drop)|(제거|생략|skip|remove).{0,20}segment_free",
     "이웃·지름길·끝점 연결의 segment_free 검사를 빼는 것은 금지입니다."),
    (r"observed_objects|choose_command\s*\(|센서.{0,10}(직접|호출)",
     "세 함수 안에서 센서·관측 함수나 choose_command 를 직접 부르는 것은 금지입니다."),
    (r"(GOAL|goal_xy|목표\s*(점|좌표)).{0,30}(옮|이동|shift|offset|변환)",
     "GOAL 좌표를 옮기는 것은 금지입니다(성공 판정은 실제 GOAL 기준)."),
    (r"(EDIT|base).{0,15}(밖|외부).{0,15}(수정|변경)|base\s*(코드|루프).{0,10}(수정|변경)",
     "EDIT REGION 밖이나 base 를 바꾸는 것은 금지입니다."),
    (r"(clearance_m|여유).{0,30}(무시|낮추|줄이).{0,10}(요청|임의)?",
     "요청된 clearance_m 를 임의로 낮추거나 무시하는 것은 금지입니다."),
]


def base_block():
    return "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + AL.BASE + "```\n"


def region_text(code):
    """The EDIT REGION of a coder answer's code, or None when the markers are missing.
    splice() would wrap any text (even a refusal) into a region, so both markers must be in the answer."""
    if TL.BEGIN_MARK not in code or TL.END_MARK not in code:
        return None
    text = AL.build(code)
    b, e = text.find(TL.BEGIN_MARK), text.find(TL.END_MARK)
    if not 0 <= b < e:
        return None
    eol = text.find("\n", e)
    return text[b: eol + 1 if eol >= 0 else len(text)]     # the whole block, both marker lines included


def key_of(block):
    """agent_loop.code_key of the block, the same key B-gate compares (comments, blank lines, docstrings ignored)."""
    return AL.code_key(AL.region_of(AL.build(block)).rstrip())


def change_note(region):
    m = CHANGE_RE.search(region or "")
    return m.group(0)[:200] if m else "(변경 주석 없음)"


def functions_of(block):
    """Top-level functions of an EDIT block: name -> its source (nested helpers belong to their function)."""
    starts = [(m.start(), m.group(1)) for m in FUNC_RE.finditer(block)]
    end_mark = block.find(TL.END_MARK)
    out = {}
    for i, (s, n) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else (end_mark if end_mark > s else len(block))
        out[n] = block[s:e]
    return out


def diff_summary(prev, cur, cap=DIFF_CAP_CHARS):
    """(one-line per-function change summary, unified diff) of this code against the previous iteration's.
    The coder never wrote the prompt's `# [시험 N] ... 변경:` line in M-1 v1, so the harness records the change."""
    if not prev:
        return "첫 코드", ""
    a, b = functions_of(prev), functions_of(cur)
    parts = []
    for n in dict.fromkeys(list(a) + list(b)):
        if n not in a:
            parts.append(f"{n} 추가")
        elif n not in b:
            parts.append(f"{n} 삭제")
        elif a[n] != b[n]:
            d = [ln for ln in difflib.unified_diff(a[n].splitlines(), b[n].splitlines(), lineterm="", n=0)
                 if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
            add = sum(1 for ln in d if ln.startswith("+"))
            parts.append(f"{n} +{add}/-{len(d) - add}")
    ud = "\n".join(difflib.unified_diff(prev.splitlines(), cur.splitlines(), "직전 코드", "이번 코드",
                                        lineterm="", n=2))
    if len(ud) > cap:
        ud = ud[:cap] + f"\n... (diff {len(ud)}자 중 앞 {cap}자만)"
    return ", ".join(parts) or "함수 변화 없음(주석·공백만)", ud


def parse_coder(content, current):
    """M-1d coder answer -> (block, reason, changed names, error).

    The answer is a short reason followed by one python fence holding only the functions it changes, each a
    complete top-level def (a whole EDIT block is accepted too). The harness puts them into `current` (the
    previous code, or the base's template on the first iteration), so a function the coder leaves out stays as
    it was - M-1c i02 replaced plan_path with the template while its reasoning said it would touch only
    lookahead_target. Any of the three may change; plan_path must end up implemented (no NotImplementedError)."""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if not fences:
        return None, None, None, "python 코드 펜스가 없습니다."
    code = max(fences, key=len)
    reason = re.sub(r"```(?:python)?\s*\n.*?```", "", content, flags=re.S).strip()
    if len(reason) < 10:
        return None, None, None, "코드 펜스 앞에 무엇을 왜 바꿨는지(변경 이유)를 쓰지 않았습니다."
    if TL.BEGIN_MARK in code:
        block = region_text(code)
        if block is None:
            return None, None, None, "EDIT REGION 시작·끝 주석이 깨졌습니다."
        new = {n: s for n, s in functions_of(block).items() if n in EDITABLE}
    else:
        try:
            tree = ast.parse(code)
        except SyntaxError as ex:
            return None, None, None, f"코드 펜스 안 구문 오류: line {ex.lineno}: {ex.msg}"
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        other = [type(n).__name__ for n in tree.body
                 if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                 and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
        bad = [d.name for d in defs if d.name not in EDITABLE]
        if other or bad or not defs:
            return None, None, None, ("코드 펜스에는 plan_path, lookahead_target, follow_cmd 중 바꾼 함수의 완전한 def 만 넣습니다"
                                      + (f" (허용되지 않은 최상위 함수: {', '.join(bad)})" if bad else "")
                                      + (" (함수 밖 문장이 있습니다)" if other else "") + ".")
        lines = code.splitlines()
        new = {}
        for i, d in enumerate(defs):
            s = d.lineno - 1 - len(d.decorator_list)
            e = defs[i + 1].lineno - 1 - len(defs[i + 1].decorator_list) if i + 1 < len(defs) else len(lines)
            new[d.name] = "\n".join(lines[s:e]).rstrip() + "\n\n\n"
    block = current
    have = functions_of(block)
    for n, src in new.items():
        block = block.replace(have[n], src, 1)
    final = functions_of(block)
    missing = [n for n in EDITABLE if n not in final]
    if missing:
        return None, None, None, f"함수가 없습니다: {', '.join(missing)}"
    stub = [n for n in EDITABLE if "NotImplementedError" in final[n]]
    if stub:
        return None, None, None, (f"{', '.join(stub)} 가 아직 빈 틀(NotImplementedError)입니다. "
                                  "plan_path 를 포함해 세 함수 모두 실제로 구현해야 합니다.")
    return block, reason, sorted(new), None


def rule_violation(d):
    """A diagnosis that steers the coder to something the task prompt forbids (M-1c: 3 of 18)."""
    for pat, why in FORBIDDEN:
        if re.search(pat, str(d.get("direction", "")), re.I):
            return why
    return None


def parse_diagnosis(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None, "JSON 객체가 없습니다."
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError as ex:
        return None, f"JSON 구문 오류: {ex}"
    if not isinstance(d, dict):
        return None, "JSON 객체가 아닙니다."
    bad = [k for k in ("failure_mode", "evidence", "function", "direction", "confidence")
           if k not in d]
    if bad:
        return None, f"빠진 필드: {', '.join(bad)}"
    if d["failure_mode"] not in MODES:
        return None, f"failure_mode 값이 허용 목록에 없습니다: {d['failure_mode']!r}"
    if d["function"] not in FUNCS:
        return None, f"function 값이 허용 목록에 없습니다: {d['function']!r}"
    if not isinstance(d["evidence"], list):
        return None, "evidence 는 문자열 목록이어야 합니다."
    why = rule_violation(d)
    if why:
        return None, f"과제 규칙 위반 제안입니다: {why} 규칙 안에서 고치는 방향으로 다시 진단하십시오."
    return d, None


class M1:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # agent_loop.Run gives Run.test: runner files, isaac_logs, carter_run_locked.sh, cleanup
        self.runner = AL.Run(Namespace(out_dir=str(self.out), tag=a.tag, sample=a.sample))
        self.stem = self.runner.stem
        self.ev_path = self.out / f"{self.stem}_events.jsonl"
        self.att_path = self.out / f"{self.stem}_attempts.jsonl"
        self.log_path = self.out / f"{self.stem}_turns.json"
        self.calls = 0
        self.coder_prompt = (HERE / CODER_PROMPT).read_text()
        self.analyst_prompt = (HERE / ANALYST_PROMPT).read_text()
        self.attempts = []

    # ------------------------------------------------------------ bookkeeping
    def event(self, kind, **kw):
        rec = {"ts": round(time.time(), 3), "time": time.strftime("%F %T"), "type": kind, **kw}
        with open(self.ev_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def say(self, text):
        print(f"{time.strftime('%F %T')} {text}", flush=True)

    def save(self, name, obj):
        (self.out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------ model
    def ask(self, it, role, msgs, retry):
        a = self.a
        self.calls += 1
        name = f"{self.stem}_i{it:02d}_{role}" + (f"_r{retry}" if retry else "")
        payload = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens,
                   "temperature": a.temperature, "top_p": a.top_p,
                   "seed": a.seed_base + a.sample * 100 + self.calls}
        self.save(f"{name}_request.json", payload)
        chars = sum(len(m["content"]) for m in msgs)
        self.event("request", iter=it, role=role, retry=retry, call=self.calls, file=f"{name}_request.json",
                   chars=chars)
        waits = 0
        while True:
            t0 = time.time()
            try:
                r = AL.post(a.base, payload, a.request_timeout)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:2000]
                self.event("http_error", iter=it, role=role, code=e.code, body=body)
                if e.code in (502, 503, 504) and waits < 30:
                    waits += 1
                    self.say(f"[i{it:02d} {role}] 서버 {e.code} - 60초 뒤 다시 요청")
                    time.sleep(60)
                    continue
                raise
            except (TimeoutError, OSError) as e:
                if waits >= TL.REQUEST_RETRIES:
                    raise
                waits += 1
                self.say(f"[i{it:02d} {role}] 요청 실패({type(e).__name__}) - 서버를 기다린 뒤 재요청")
                TL.wait_for_server(a.base)
        dt = time.time() - t0
        r["_bench"] = {"t_start_epoch": t0, "wall_s": round(dt, 3), "endpoint": a.base, "role": role,
                       "iter": it, "call": self.calls}
        self.save(f"{name}_raw.json", r)
        txt = AL.raw_text(r)
        if txt:
            (self.out / f"{name}_raw_text.txt").write_text(txt)
        u = r.get("usage", {})
        content = (r["choices"][0]["message"].get("content") or "")
        info = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "finish": r["choices"][0].get("finish_reason"), "content_chars": len(content),
                "wall_s": round(dt, 1)}
        self.event("response", iter=it, role=role, retry=retry, call=self.calls, **info)
        self.say(f"[i{it:02d} {role}] 생성 {dt:.0f}s 입력={info['prompt_tokens']} 출력={info['completion_tokens']} "
                 f"답={info['content_chars']}자 finish={info['finish']}")
        return content

    def attempt_lines(self):
        return "\n".join(x["line"] for x in self.attempts[-ATTEMPT_LOG_LINES:]) or "(없음)"

    def coder(self, it, code, diagnosis):
        """-> (block, key, retries, reason, changed functions). M-1d: the answer holds only the changed functions."""
        parts = [self.coder_prompt, base_block()]
        if code:
            parts.append("\n---\n\n## 현재 코드 (직전 시도의 EDIT 블록)\n\n```python\n" + code + "```\n")
        if diagnosis:
            parts.append("\n## 진단\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=1) + "\n```\n")
        parts.append("\n## 시도 기록\n\n" + self.attempt_lines() + "\n")
        parts.append(f"\n이번은 시도 {it}입니다. "
                     + ("세 함수를 모두 구현해 " if not code else "바꿀 함수만 ")
                     + "변경 이유와 python 코드 펜스 하나로 내십시오.\n")
        msgs = [{"role": "user", "content": "".join(parts)}]
        seen = {x["code_key"]: x["iter"] for x in self.attempts if x.get("code_key")}
        current = code or region_text(AL.BASE)          # first iteration: the base's template block
        for retry in range(FORMAT_RETRIES + 1):
            content = self.ask(it, "coder", msgs, retry)
            block, reason, changed, why = parse_coder(content, current)
            if block is not None:
                key = key_of(block)
                if key in seen:
                    why = (f"바꿔 끼운 결과가 시도 #{seen[key]}의 코드와 같습니다(주석·빈 줄·docstring 차이 무시). "
                           "진단에 따라 실제로 바뀐 코드를 내십시오.")
                else:
                    return block, key, retry, reason, changed
            self.event("coder_rejected", iter=it, retry=retry, reason=why)
            self.say(f"[i{it:02d} coder] 형식 거부: {why}")
            msgs = msgs[:1] + [{"role": "assistant", "content": content},
                               {"role": "user", "content": why + " 변경 이유와 바꾼 함수의 완전한 def 를 담은 python 코드 펜스 "
                                                                 "하나로 다시 내십시오."}]
        return None, None, FORMAT_RETRIES, None, None

    def analyst(self, it, region, kind, dg, raw, change, udiff, reason=None):
        body = (self.analyst_prompt
                + "\n\n---\n\n## 과제 프롬프트 (수정 에이전트가 받은 것)\n\n" + self.coder_prompt
                + base_block()
                + "\n---\n\n## 이번 코드\n\n```python\n" + region + "```\n"
                + "\n## 수정 에이전트가 밝힌 변경 이유\n\n" + (reason or "(없음)") + "\n"
                + f"\n## 이번 수정 (직전 코드와의 차이, 하네스 계산)\n\n함수별: {change}\n"
                + ("\n```diff\n" + udiff + "\n```\n" if udiff else "")
                + f"\n## 실행 종류\n\n{kind}\n"
                + "\n## 실행 요약\n\n```json\n" + json.dumps(dg, ensure_ascii=False, indent=1) + "\n```\n"
                + "\n## DBG 원본 구간\n\n```\n" + ("\n".join(raw) if raw else "(Isaac 에서 실행하지 않아 DBG 가 없습니다)")
                + "\n```\n"
                + "\n## 시도 기록\n\n" + self.attempt_lines() + "\n")
        msgs = [{"role": "user", "content": body}]
        for retry in range(FORMAT_RETRIES):
            content = self.ask(it, "analyst", msgs, retry)
            d, err = parse_diagnosis(content)
            if d:
                return d
            self.event("analyst_rejected", iter=it, retry=retry, reason=err)
            self.say(f"[i{it:02d} analyst] 형식 거부: {err}")
            msgs = msgs[:1] + [{"role": "assistant", "content": content},
                               {"role": "user", "content": f"답 형식 오류: {err} 지정한 JSON 객체 하나만 다시 내십시오."}]
        # no usable diagnosis: hand the coder the harness facts instead of nothing
        r = dg.get("result", {})
        if r.get("static_check"):
            return {"failure_mode": "static", "function": "unknown", "confidence": "low",
                    "evidence": [str(r["static_check"])[:600]],
                    "direction": "분석 담당이 진단을 내지 못했습니다. 정적 검사 오류를 고치십시오.",
                    "_fallback": True}
        st = (dg.get("stalls") or [{}])[0]
        return {"failure_mode": "other", "function": "unknown", "confidence": "low",
                "evidence": [f"verdict={r.get('verdict')} final_dist={r.get('final_dist_m')} "
                             f"collisions={r.get('collision_count')} runtime_error={r.get('runtime_error')}",
                             f"첫 정체 {st.get('t_start')}~{st.get('t_end')} s ({st.get('x')}, {st.get('y')}) "
                             f"pick={st.get('pick')}" if st else "정체 없음"],
                "direction": "분석 담당이 진단을 내지 못했습니다. 위 근거로 판단하십시오.",
                "_fallback": True}

    # ------------------------------------------------------------ runs
    def grade(self, g, region):
        variant = self.stem if g == 1 else f"{self.stem}_t{g}"
        (HERE / f"{variant}_carter_run.py").write_text(AL.build(region))
        t0 = time.time()
        result = TL.grade(variant)
        mt = result.get("metrics", {})
        rec = {"turn": g, "variant": variant, "verdict": result.get("verdict"),
               "final_dist_m": mt.get("final_dist_m"), "collisions": mt.get("collision_count"),
               "partners": mt.get("collision_partners"), "first_collision_t": mt.get("first_collision_t"),
               "time_to_goal_s": mt.get("time_to_goal_s"), "sim_time_s": mt.get("sim_time_s"),
               "runtime_error": result.get("_runtime_error"), "wall_s": round(time.time() - t0, 1)}
        return variant, result, rec

    def digest_of(self, variant, runtime_error=None):
        dg = RD.digest(variant, TL.ISAAC_LOG_DIR)
        if runtime_error:
            dg["result"]["runtime_error"] = runtime_error[:1500]
        raw = RD.dbg_window(variant, TL.ISAAC_LOG_DIR, RD.analyst_windows(dg), self.a.dbg_window_chars)
        return dg, raw

    def main(self):
        a = self.a
        os.environ["ARENA_MOVERS"] = a.movers          # tests and grading on the same scene (B-5-29 --b-test-movers)
        log = {"mode": "m1", "model": a.model, "tag": a.tag, "sample": a.sample, "endpoint": a.base,
               "coder_prompt": CODER_PROMPT, "analyst_prompt": ANALYST_PROMPT, "base_file": AL.BASE_FILE.name,
               "coder_prompt_sha16": hashlib.sha256(self.coder_prompt.encode()).hexdigest()[:16],
               "analyst_prompt_sha16": hashlib.sha256(self.analyst_prompt.encode()).hexdigest()[:16],
               "movers": a.movers, "test_sim_s": a.test_sim_s, "max_iters": a.max_iters,
               "max_wall_h": a.max_wall_h, "max_grades": a.max_grades, "dbg_window_chars": a.dbg_window_chars,
               "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
               "seed_base": a.seed_base, "started": time.strftime("%F %T"), "turns": []}
        self.event("start", **{k: v for k, v in log.items() if k != "turns"})
        self.say(f"[start] {self.stem} M-1 coder={CODER_PROMPT} analyst={ANALYST_PROMPT} base={AL.BASE_FILE.name}")
        t_start = time.time()
        code, diagnosis, g, none_streak, stop = None, None, 1, 0, None
        for it in range(1, a.max_iters + 1):
            if time.time() - t_start > a.max_wall_h * 3600:
                stop = "wall"
                break
            region, key, retries, reason, changed = self.coder(it, code, diagnosis)
            target = (diagnosis or {}).get("function")
            att = {"iter": it, "coder_retries": retries, "code_key": key, "reason": reason, "changed": changed,
                   "diagnosis_target": target}
            if region is None:
                att.update(kind="no_code", line=f"#{it} 코드 없음(수정 에이전트가 형식에 맞는 EDIT 블록을 내지 못함)")
                self.attempts.append(att)
                with open(self.att_path, "a") as f:
                    f.write(json.dumps(att, ensure_ascii=False) + "\n")
                continue
            change, udiff = diff_summary(code, region)
            off = sorted(set(changed or []) - {target}) if target in EDITABLE else []
            if off:
                change += f" (진단 대상 {target} 밖: {', '.join(off)})"
            self.say(f"[i{it:02d} change] {change}")
            self.say(f"[i{it:02d} reason] {(reason or '').replace(chr(10), ' ')[:300]}")
            self.say(f"[i{it:02d} test] 시험 주행 {a.test_sim_s:.0f} s")
            out, trec = self.runner.test(g, it, {"code": region, "sim_seconds": a.test_sim_s})
            att.update(kind="test", variant=trec["variant"] if trec else None, change=change,
                       comment=change_note(region))
            if trec is None or not trec.get("static_ok"):
                # every change is judged by the analyst, a failed compile/static check included (user 2026-11-15)
                detail = out.get("static_check") or out.get("error") or "정적 검사 실패"
                self.say(f"[i{it:02d} test] 정적 실격: {detail.splitlines()[0][:120]}")
                dg = {"result": {"verdict": "static_dq", "static_check": detail[:3000]}}
                self.save(f"{self.stem}_i{it:02d}_digest.json", {"variant": None, "digest": dg, "dbg_window": []})
                diagnosis = self.analyst(it, region, "정적 검사(컴파일) 실패 - Isaac 에서 실행하지 않음",
                                         dg, [], change, udiff, reason)
                att.update(verdict="static_dq",
                           line=(f"#{it} 정적 실격: {detail.splitlines()[0][:100]} | 변경 {change[:120]} | 이유 {(reason or '')[:120]}"
                                 f" | 진단 {diagnosis.get('function')}/{diagnosis.get('failure_mode')}: "
                                 f"{str(diagnosis.get('direction'))[:140]}"))
                self.say(f"[i{it:02d} analyst] {diagnosis.get('failure_mode')} fn={diagnosis.get('function')} "
                         f"conf={diagnosis.get('confidence')}")
            else:
                self.say(f"[i{it:02d} test] {str(trec['verdict']).upper()} dist={trec['final_dist_m']} "
                         f"col={trec['collisions']} ttg={trec['time_to_goal_s']} wall={trec['wall_s']}s")
                att.update(verdict=trec["verdict"], final_dist_m=trec["final_dist_m"], collisions=trec["collisions"])
                variant, kind = trec["variant"], f"시험 주행 {a.test_sim_s:.0f}초"
                runtime_error = out.get("runtime_error")
                if trec["verdict"] == "pass":
                    variant, result, grec = self.grade(g, region)
                    grec["from_test"] = trec["variant"]
                    log["turns"].append(grec)
                    self.save(self.log_path.name, log)
                    self.event("grading", **grec)
                    self.say(f"[g{g}] 채점 {str(grec['verdict']).upper()} dist={grec['final_dist_m']} "
                             f"col={grec['collisions']} ttg={grec['time_to_goal_s']} (시도 {it})")
                    att.update(graded=g, graded_verdict=grec["verdict"])
                    if grec["verdict"] == "pass":
                        att["line"] = f"#{it} 시험 pass -> 채점 {g} pass"
                        self.attempts.append(att)
                        with open(self.att_path, "a") as f:
                            f.write(json.dumps(att, ensure_ascii=False) + "\n")
                        stop = "passed"
                        break
                    kind = "최종 채점 600초 (시험 주행은 pass, 채점은 실패)"
                    runtime_error = None
                    g += 1
                dg, raw = self.digest_of(variant, runtime_error)
                self.save(f"{self.stem}_i{it:02d}_digest.json", {"variant": variant, "digest": dg, "dbg_window": raw})
                if g > a.max_grades:
                    att["line"] = f"#{it} 시험 pass -> 채점 실패 (채점 한도 {a.max_grades}회)"
                    self.attempts.append(att)
                    with open(self.att_path, "a") as f:
                        f.write(json.dumps(att, ensure_ascii=False) + "\n")
                    stop = "max_grades"
                    break
                diagnosis = self.analyst(it, region, kind, dg, raw, change, udiff, reason)
                st = (dg.get("stalls") or [None])[0]
                where = f" 정체@{st['t_start']}s({st['x']},{st['y']})" if st else ""
                res = (f"채점{g - 1} {att.get('graded_verdict')}" if att.get("graded") else f"시험 {trec['verdict']}")
                att["line"] = (f"#{it} {res} dist={dg['result'].get('final_dist_m')}{where} | 변경 {change[:120]}"
                               f" | 이유 {(reason or '').replace(chr(10), ' ')[:120]}"
                               f" | 진단 {diagnosis.get('function')}/{diagnosis.get('failure_mode')}: "
                               f"{str(diagnosis.get('direction'))[:140]}")
                self.say(f"[i{it:02d} analyst] {diagnosis.get('failure_mode')} fn={diagnosis.get('function')} "
                         f"conf={diagnosis.get('confidence')}")
                none_streak = none_streak + 1 if diagnosis.get("function") == "none" else 0
            self.save(f"{self.stem}_i{it:02d}_diagnosis.json", diagnosis)
            self.attempts.append(att)
            with open(self.att_path, "a") as f:
                f.write(json.dumps({**att, "diagnosis": diagnosis}, ensure_ascii=False) + "\n")
            code = region
            if none_streak >= NONE_STREAK_STOP:
                stop = "base_limited"
                break
        else:
            stop = "max_iters"
        log.update(stop=stop, iterations=len(self.attempts), model_calls=self.calls,
                   passed_at_turn=next((t["turn"] for t in log["turns"] if t["verdict"] == "pass"), None),
                   ended=time.strftime("%F %T"))
        self.save(self.log_path.name, log)
        self.event("end", stop=stop, iterations=len(self.attempts), gradings=len(log["turns"]))
        self.say(f"[done] {self.stem}: stop={stop} 시도 {len(self.attempts)}회 채점 {len(log['turns'])}회 "
                 f"-> {self.log_path}")
        return 0 if stop == "passed" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8420/v1")
    ap.add_argument("--out-dir", default="results/turns/prompt_m1")
    ap.add_argument("--base-file", default="base_carter_run_ros_map_lidartime_objects_check_v17.py")
    ap.add_argument("--movers", default="v1")
    ap.add_argument("--test-sim-s", type=float, default=200.0)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--max-wall-h", type=float, default=3.0)
    ap.add_argument("--max-grades", type=int, default=2)
    ap.add_argument("--dbg-window-chars", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--request-timeout", type=int, default=7200)
    a = ap.parse_args()
    # the same base swap agent_loop.main does for --base-file
    AL.BASE_FILE = HERE / a.base_file
    AL.BASE = AL.BASE_FILE.read_text()
    AL.SP.BASE = AL.BASE
    if TL._outside(AL.BASE) is None or eval(AL.BASE.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0],
                                            {"dict": dict}) != AL._MISSION:
        raise SystemExit(f"--base-file {a.base_file}: EDIT REGION 표시나 MISSION 이 A-2 base 와 다릅니다.")
    import arena_movers
    arena_movers.scenario(a.movers)
    if not Path(a.out_dir).is_absolute():
        a.out_dir = str(HERE / a.out_dir)
    return M1(a).main()


if __name__ == "__main__":
    sys.exit(main())
