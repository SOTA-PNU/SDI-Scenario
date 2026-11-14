"""Run the benchmark's turn protocol on one sample until it passes.

Turn 1 is blind: prompt + base, nothing else.  After a failing run the model
gets that run's own result - redacted for track B, which may not learn the
layout - and writes a new submission *inside the same conversation*, so it
keeps its own reasoning from the previous turn.  That is the same shape the
Claude track-B entry went through (one feedback, second turn passed); this
just automates it and records how many turns each model needs.

What the model is told between turns is only outcome, never geometry:

  track A  the whole result minus the `obstacle` block.  A is allowed to read
           MISSION, so the box table is not new information and naming the
           prim it struck is legitimate debugging signal.
  track B  mask_feedback.mask(): verdict, criteria, the five outcome metrics,
           and which oracle failed.  The prim name, the contact time and the
           obstacle block are all withheld - each of them reconstructs the
           layout that track B is supposed to discover with the scanner.

    python3 turn_loop.py --model openai/gpt-oss-120b --tag gpu_gpt_oss_120b \
        --track b --sample 0 --max-turns 6 --base http://127.0.0.1:8000/v1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
TURNS = RESULTS / "turns"
CARTER_RUN = Path.home() / "carter_ws" / "tools" / "bin" / "carter-run"
# Kit writes one session log per Isaac launch here and keeps only the newest few,
# so a run that wants its logs has to copy them out right after it finishes.
KIT_LOG_DIR = (Path.home() / "carter_ws" / "mamba" / "envs" / "isaacsim" / "lib" / "python3.10"
               / "site-packages" / "omni" / "logs" / "Kit" / "Isaac-Sim Python" / "4.5")
ISAAC_LOG_DIR = None  # --isaac-log-dir: keep every run's console output, Kit log and result files

sys.path.insert(0, str(HERE))
from generate import api, extract, model_input          # noqa: E402
from mask_feedback import mask                          # noqa: E402
from splice import splice                               # noqa: E402

FEEDBACK = """\
제출한 코드를 고정 하네스에서 실행했습니다. 판정 결과입니다.

```json
{result}
```

같은 규칙 그대로, `[EDIT REGION]` 블록을 고쳐 다시 제출하십시오.
"""

RUNTIME_FEEDBACK = """\
제출한 코드를 고정 하네스에서 실행했더니 `controller()` 가 예외를 던져 실행이 중단되었습니다.

```
{err}
```

같은 규칙 그대로, `[EDIT REGION]` 블록을 고쳐 다시 제출하십시오.
"""

DQ_FEEDBACK = """\
제출한 코드가 채점 전 정적 검사에서 실격되었습니다.

{detail}

같은 규칙 그대로, `[EDIT REGION]` 블록을 고쳐 다시 제출하십시오.
"""


BEGIN_MARK = "# [EDIT REGION]"
END_MARK = "# ========================= [END EDIT REGION]"
# punctuation a model substitutes for ASCII while "just retyping" a docstring
LOOKALIKE = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
             "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"',
             "\u201d": '"', "\u2026": "...", "\u00a0": " ", "\u2212": "-"}


def _outside(text):
    lines = text.splitlines(keepends=True)
    b = e = None
    for i, ln in enumerate(lines):
        if BEGIN_MARK in ln and b is None:
            b = i
        if ln.startswith(END_MARK):
            e = i
    if b is None or e is None or e <= b:
        return None
    return lines[: b - 1] + lines[e + 1:]


def _fold(s):
    for k, v in LOOKALIKE.items():
        s = s.replace(k, v)
    return s


def edit_region_diff(cand_path):
    """Say WHICH line outside the region changed, and whether it is only glyphs.

    check_edit_region.py answers yes/no, which is all a grader needs but not
    enough for the model to act on: told only "header differs" it re-emitted
    the same file four turns running. Naming the line - and separating a
    retyped en-dash from a real edit - turns a dead end into a fixable note.
    """
    base = _outside((HERE / "base_carter_run.py").read_text())
    cand = _outside(Path(cand_path).read_text())
    if base is None or cand is None:
        return "EDIT REGION 표시가 없거나 깨졌습니다.", False
    glyph_only = True
    notes = []
    for i, (b, c) in enumerate(zip(base, cand), 1):
        if b == c:
            continue
        if _fold(b) != _fold(c):
            glyph_only = False
        notes.append(f"{i}행: base=<<{b.rstrip()[:90]}>> 제출=<<{c.rstrip()[:90]}>>")
        if len(notes) >= 3:
            break
    if len(base) != len(cand):
        glyph_only = False
        notes.append(f"줄 수가 다릅니다: base {len(base)}줄, 제출 {len(cand)}줄")
    if not notes:
        return "차이를 찾지 못했습니다(공백/개행 차이일 수 있습니다).", False
    head = ("EDIT REGION 밖에서 바뀐 것은 문장부호 글리프뿐입니다"
            " (ASCII 하이픈/따옴표가 유니코드로 바뀌었습니다). "
            "base 의 글자를 그대로 두십시오.\n" if glyph_only
            else "EDIT REGION 밖이 base 와 다릅니다.\n")
    return head + "\n".join(notes), glyph_only


def trim(msgs, budget_chars, rec):
    """Keep the original prompt and drop the oldest exchanges if the thread grows.

    The conversation carries every previous answer, so a model with a 32k
    window runs out around turn five.  Dropping the oldest exchange keeps the
    task statement (msgs[0], which is the whole contract) and the most recent
    feedback, and the drop is recorded so a trimmed run is never read as an
    untrimmed one.
    """
    dropped = 0
    while sum(len(m["content"]) for m in msgs) > budget_chars and len(msgs) > 3:
        del msgs[1:3]          # oldest assistant answer + the feedback after it
        dropped += 1
    if dropped:
        rec["trimmed_exchanges"] = dropped
        print(f"       컨텍스트 초과 - 오래된 교환 {dropped}건 제거", flush=True)
    return msgs


def feedback_json(result, track, fb_profile=None):
    """What the model is shown after a failing run.

    Default behaviour is unchanged, so every run recorded before the ablation
    existed stays reproducible: track B gets the layout-redacted view, track A
    gets the whole result minus the obstacle block. Passing a profile name
    switches to the ablation's fixed field sets instead.
    """
    if fb_profile:
        from feedback_profile import profile
        return json.dumps(profile(result, fb_profile), indent=2, ensure_ascii=False)
    if track == "b":
        return json.dumps(mask(result), indent=2, ensure_ascii=False)
    shown = {k: v for k, v in result.items() if k != "obstacle"}
    return json.dumps(shown, indent=2, ensure_ascii=False)


def static_check(path, track):
    """Return (ok, detail). Cheap gates before Isaac gets the file."""
    src = path.read_text()
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        return False, f"구문 오류: line {e.lineno}: {e.msg}"
    p = subprocess.run(["python3", str(REPO / "levels/common/check_edit_region.py"),
                        str(path)], capture_output=True, text=True)
    if p.returncode != 0:
        detail, _ = edit_region_diff(path)
        return False, detail
    if track == "b":
        p = subprocess.run(["python3", str(HERE / "check_b_track.py"), str(path)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            hard = [l.strip() for l in (p.stdout + p.stderr).splitlines()
                    if l.strip().startswith("HARD")]
            return False, "규칙 6 위반(배치 하드코딩): " + (hard[0][:200] if hard else "")
    if "def controller" not in src:
        return False, "controller() 가 없습니다."
    return True, ""


REQUEST_RETRIES = 6


def wait_for_server(base, every=30, give_up_s=12 * 3600):
    """Block until the endpoint answers again; return the seconds spent waiting.

    A dropped SSH tunnel or a router restart is an outage, not an answer: the
    call that was in flight is asked again once the server is back, so an
    outage costs time but never a sample.
    """
    t0 = time.time()
    while True:
        try:
            api(base, "/models", None, timeout=15, retries=0)
            return round(time.time() - t0, 1)
        except (TimeoutError, OSError):
            if time.time() - t0 > give_up_s:
                raise
            time.sleep(every)


def keep_isaac_logs(variant, p, t0, t1, waits):
    """--isaac-log-dir: keep everything one Isaac run left behind.

    The console output of carter-run (stdout and stderr, whole), the Kit session
    log(s) written during the run, the run's result/trajectory files and the
    graded submission. Grading is serial (carter-run guard), so any Kit log
    touched after t0 belongs to this run.
    """
    if ISAAC_LOG_DIR is None:
        return
    import shutil
    d = ISAAC_LOG_DIR
    head = (f"# carter-run accident_scene/arena/{variant}_carter_run.py\n"
            f"# start_epoch={t0:.3f} end_epoch={t1:.3f} wall_s={t1 - t0:.1f} "
            f"returncode={p.returncode} serial_guard_waits={waits}\n")
    (d / f"{variant}.carter-run.stdout.log").write_text(head + (p.stdout or ""))
    (d / f"{variant}.carter-run.stderr.log").write_text(head + (p.stderr or ""))
    kept = {"kit_logs": [], "result_files": [], "returncode": p.returncode,
            "start_epoch": t0, "end_epoch": t1, "serial_guard_waits": waits}
    if KIT_LOG_DIR.is_dir():
        for k in sorted(KIT_LOG_DIR.glob("kit_*.log")):
            if k.stat().st_mtime >= t0 - 1:
                shutil.copy2(k, d / f"{variant}.{k.name}")
                kept["kit_logs"].append(k.name)
    for f in sorted(RESULTS.glob(f"{variant}_*")):
        if f.is_file() and f.stat().st_mtime >= t0 - 1:
            shutil.copy2(f, d / f.name)
            kept["result_files"].append(f.name)
    sub = HERE / f"{variant}_carter_run.py"
    if sub.exists():
        shutil.copy2(sub, d / sub.name)
    (d / f"{variant}.kept.json").write_text(json.dumps(kept, indent=2, ensure_ascii=False))


def grade(variant):
    """Run one submission in Isaac and return its result dict."""
    out = RESULTS / f"{variant}_result.json"
    if out.exists():
        out.unlink()
    env = {**os.environ, "CARTER_REPO": str(REPO)}
    # drop anything that would make this run non-comparable
    env.pop("CARTER_WHEEL_TOE_DEG", None)
    waits = 0
    for _ in range(480):
        # several loops may share the one Isaac; the guard keeps grading serial,
        # so wait for it instead of failing (results are unaffected)
        t0 = time.time()
        p = subprocess.run([str(CARTER_RUN), f"accident_scene/arena/{variant}_carter_run.py"],
                           env=env, capture_output=True, text=True, timeout=1800)
        if p.returncode != 75:
            break
        waits += 1
        time.sleep(30)
    else:
        raise RuntimeError("다른 Isaac 인스턴스가 4시간 넘게 실행 중입니다(직렬 실행 가드).")
    keep_isaac_logs(variant, p, t0, time.time(), waits)
    if not out.exists():
        # The controller raised, so the harness never scored the run. That is a
        # real failure mode, not missing data: report it as one, and keep the
        # exception line - it is an outcome, not geometry, so track B may see it.
        log = RESULTS / f"{variant}.isaac.log"
        blob = (p.stdout + p.stderr)
        if log.exists():
            log.write_text(blob)
        return {"verdict": "fail", "_runtime_error": exception_line(blob),
                "criteria": {}, "oracles": [], "metrics": {},
                "_wall_s": round(time.time() - t0, 1)}
    r = json.loads(out.read_text())
    r["_wall_s"] = round(time.time() - t0, 1)
    return r


TRACE_RE = re.compile(r"^\s*(?:\[py stderr\]:\s*)?([A-Za-z_.]*(?:Error|Exception)\b.*)$", re.M)


# a Kit extension that fails to import at boot (isaacsim.asset.browser wants `requests`) leaves a full traceback
# in every run's log; it is not the controller's error and must never be reported as one (실험 31 a1 test 5)
BOOT_TB_RE = re.compile(r"Traceback \(most recent call last\):\n(?:[ \t].*\n)*?[A-Za-z_.]*(?:Error|Exception)\b.*", re.M)
FATAL_RE = re.compile(r"^Fatal Python error: (?P<kind>[^\n]+)$", re.M)
FRAME_RE = re.compile(r'\s*File "(?P<f>[^"]+)", line (?P<n>\d+) in (?P<fn>\S+)')


def drop_boot_tracebacks(clean):
    return BOOT_TB_RE.sub(lambda m: "" if "custom_importer.py" in m.group(0) else m.group(0), clean)


def fatal_text(clean):
    """A process that died without a Python traceback (segfault, abort): the faulthandler's current-thread
    frames that lie in the run file, most recent first; None if the log has no 'Fatal Python error'."""
    m = FATAL_RE.search(clean)
    if not m:
        return None
    tail = clean[m.end():]
    j = tail.find("Current thread")
    frames = []
    for ln in (tail[j:].splitlines()[1:60] if j >= 0 else []):
        fm = FRAME_RE.match(ln)
        if not fm:
            if frames:
                break
            continue
        if fm["f"].endswith("_carter_run.py"):
            frames.append(f"{fm['fn']} (line {fm['n']})")
    return (f"Fatal Python error: {m['kind']}. controller() 실행 중에 프로세스가 죽었습니다. Python 예외가 아니라서 "
            f"traceback 이 없습니다." + (f" 그때 실행 중이던 호출(최근 것부터): {' <- '.join(frames[:8])}" if frames else ""))


def exception_line(blob):
    """Pull the controller's own exception out of a very noisy Kit log."""
    # strip Kit's timestamp/prefix so the message reads like a plain traceback
    clean = drop_boot_tracebacks(re.sub(r"^.*?\[py stderr\]:\s?", "", blob, flags=re.M))
    hits = [m for m in TRACE_RE.findall(clean)
            if "Fatal Python error" not in m and "omni" not in m]
    if hits:
        return hits[-1].strip()[:300]
    return (fatal_text(clean) or "컨트롤러가 예외를 던져 실행이 중단되었습니다.")[:300]


def write_submission(variant, content):
    """Save the model's answer as a runnable submission; splice if it is a block."""
    code = extract(content)
    spliced = splice(code)
    path = HERE / f"{variant}_carter_run.py"
    text = spliced if spliced is not None else (code if code.endswith("\n") else code + "\n")
    path.write_text(text)
    return path, spliced is not None


def resume_state(a, log_path, log):
    """Rebuild the conversation exactly as it stood after the last recorded turn.

    That turn's request (post-trim), its answer, and the feedback the loop would
    have sent - all recomputed from files, nothing re-generated. Returns
    (msgs, first turn to run), or None when the run had already finished.
    """
    stem = log_path.name[:-len("_turns.json")]
    old = json.loads(log_path.read_text())
    if old.get("passed_at_turn") or len(old["turns"]) >= a.max_turns:
        print(f"[resume] {stem}: 이미 종료된 실행", flush=True)
        return None
    log["turns"] = old["turns"]
    log["resumed_after_turn"] = len(old["turns"])
    msgs = [{"role": "user", "content": model_input(a.track, a.prompt_file)}]
    if old["turns"]:
        last = old["turns"][-1]
        msgs = json.loads((TURNS / last["request_file"]).read_text())["messages"]
        raw = json.loads((TURNS / last["raw_file"]).read_text())
        msgs.append({"role": "assistant",
                     "content": raw["choices"][0]["message"].get("content") or ""})
        if last["verdict"] == "static_dq":
            msgs.append({"role": "user", "content": DQ_FEEDBACK.format(detail=last["detail"])})
        elif last.get("runtime_error"):
            msgs.append({"role": "user", "content": RUNTIME_FEEDBACK.format(err=last["runtime_error"])})
        else:
            res = json.loads((RESULTS / f"{last['variant']}_result.json").read_text())
            res["_wall_s"] = last.get("sim_wall_s")
            msgs.append({"role": "user", "content": FEEDBACK.format(
                result=feedback_json(res, a.track, a.feedback_profile))})
    start = len(old["turns"]) + 1
    print(f"[resume] {stem}: {start}회차부터 이어 갑니다 (메시지 {len(msgs)}개)", flush=True)
    return msgs, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--track", choices=["a", "b"], required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--base", default=os.environ.get("LLM_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--feedback-profile", choices=["lean", "rich"],
                    help="ablation only: fixed feedback field set instead of the default")
    ap.add_argument("--out-suffix", default="",
                    help="appended to the stem so a condition never overwrites another")
    ap.add_argument("--turns-dir", help="write turn records here instead of results/turns")
    ap.add_argument("--budget-chars", type=int, default=60000,
                    help="trim the oldest exchanges once the thread exceeds this")
    ap.add_argument("--request-timeout", type=int, default=7200,
                    help="seconds to wait for one generation (slow shared NPU: 2400 was too short)")
    ap.add_argument("--resume", action="store_true",
                    help="continue a crashed run from its saved turn records")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="request field reasoning_effort (실험 8 T3); omitted = server default")
    ap.add_argument("--prompt-file",
                    help="prompt file in arena/ to use instead of the track default (e.g. PROMPT_A1.md)")
    ap.add_argument("--isaac-log-dir",
                    help="keep every Isaac run's console output, Kit session log and result files here")
    ap.add_argument("--reuse-turn1", action="store_true",
                    help="grade the existing turn-1 submission instead of regenerating it")
    a = ap.parse_args()

    global TURNS, ISAAC_LOG_DIR
    if a.isaac_log_dir:
        ISAAC_LOG_DIR = Path(a.isaac_log_dir)
        ISAAC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if a.turns_dir:
        TURNS = Path(a.turns_dir)
    TURNS.mkdir(parents=True, exist_ok=True)
    stem = f"{a.tag}_{a.track}{a.sample}{a.out_suffix}"
    seed_stem = f"{a.tag}_{a.track}{a.sample}"
    log = {"model": a.model, "tag": a.tag, "track": a.track, "sample": a.sample,
           "endpoint": a.base, "feedback_profile": a.feedback_profile or "default",
           "condition": a.out_suffix.lstrip("_") or "default",
           "reasoning_effort": a.reasoning_effort, "turns": []}
    log_path = TURNS / f"{stem}_turns.json"

    msgs = [{"role": "user", "content": model_input(a.track, a.prompt_file)}]
    log["prompt_file"] = a.prompt_file or ("PROMPT.md" if a.track == "a" else "PROMPT_B.md")
    log["input_sha16"] = __import__("hashlib").sha256(
        msgs[0]["content"].encode()).hexdigest()[:16]
    log["input_chars"] = len(msgs[0]["content"])
    passed_at = None
    start = 1
    if a.resume and log_path.exists():
        state = resume_state(a, log_path, log)
        if state is None:
            return 0
        msgs, start = state

    for turn in range(start, a.max_turns + 1):
        variant = stem if turn == 1 else f"{stem}_t{turn}"
        rec = {"turn": turn, "variant": variant}

        if turn == 1 and a.reuse_turn1:
            raw = RESULTS / "generated" / f"{seed_stem}_raw.json"
            sub = HERE / f"{seed_stem}_carter_run.py"
            if not (raw.exists() and sub.exists()):
                print(f"[!] --reuse-turn1 인데 {stem} 의 1회차가 없습니다.", flush=True)
                return 2
            content = json.load(raw.open())["choices"][0]["message"].get("content") or ""
            msgs.append({"role": "assistant", "content": content})
            rec["reused"] = True
            rec["turn1_source"] = seed_stem
            rec["turn1_sha256"] = __import__("hashlib").sha256(
                content.encode()).hexdigest()[:16]
            print(f"[t1] 기존 제출 재사용: {sub.name}", flush=True)
        else:
            trim(msgs, a.budget_chars, rec)
            payload = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens,
                       "temperature": a.temperature, "top_p": a.top_p,
                       "seed": a.seed_base + a.sample * 100 + turn}
            if a.reasoning_effort:
                payload["reasoning_effort"] = a.reasoning_effort
            n_retry = 0
            while True:
                t0 = time.time()
                try:
                    r = api(a.base, "/chat/completions", payload, timeout=a.request_timeout)
                    break
                except (TimeoutError, OSError) as e:
                    # no answer came back, so nothing was learned: once the server
                    # answers again, ask the same turn again
                    n_retry += 1
                    rec["request_retries"] = n_retry
                    if n_retry > REQUEST_RETRIES:
                        raise
                    print(f"[t{turn}] 요청 실패({type(e).__name__}) - 서버 응답을 기다린 뒤 재요청 "
                          f"{n_retry}/{REQUEST_RETRIES}", flush=True)
                    rec.setdefault("server_wait_s", []).append(wait_for_server(a.base))
            dt = time.time() - t0
            # the exact request is part of the record: a turn is only
            # reproducible if what was asked is kept next to what came back
            (TURNS / f"{variant}_request.json").write_text(json.dumps(
                {"model": a.model, "endpoint": a.base, "temperature": a.temperature,
                 "top_p": a.top_p, "seed": a.seed_base + a.sample * 100 + turn,
                 "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort,
                 "messages": msgs},
                indent=2, ensure_ascii=False))
            r["_bench"] = {"t_start_epoch": t0, "t_end_epoch": t0 + dt,
                           "wall_s": round(dt, 3), "endpoint": a.base,
                           "requested_model": a.model, "track": a.track,
                           "sample": a.sample, "turn": turn}
            (TURNS / f"{variant}_raw.json").write_text(
                json.dumps(r, indent=2, ensure_ascii=False))
            reply = r["choices"][0]["message"]
            content = reply.get("content") or ""
            u = r.get("usage", {})
            ct = u.get("completion_tokens") or 0
            rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            rec.update(gen_wall_s=round(dt, 1), prompt_tokens=u.get("prompt_tokens"),
                       completion_tokens=ct, reasoning_tokens=rt,
                       tok_per_s=round(ct / dt, 1) if dt > 0 else None,
                       finish=r["choices"][0].get("finish_reason"), chars=len(content),
                       request_file=f"{variant}_request.json",
                       raw_file=f"{variant}_raw.json")
            print(f"[t{turn}] 생성 {dt:.0f}s out={ct} ({round(ct / dt, 1) if dt else 0} tok/s) "
                  f"추론tok={rt} chars={len(content)}", flush=True)
            if not content.strip():
                rec["verdict"] = "empty_response"
                log["turns"].append(rec)
                log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
                print(f"[t{turn}] 빈 응답 - 중단", flush=True)
                break
            msgs.append({"role": "assistant", "content": content})
            (TURNS / f"{variant}_reply.md").write_text(content)
            sub, was_spliced = write_submission(variant, content)
            rec["spliced"] = was_spliced
            rec["submission"] = sub.name

        path = HERE / f"{seed_stem if (turn == 1 and a.reuse_turn1) else variant}_carter_run.py"
        ok, detail = static_check(path, a.track)
        if not ok:
            rec.update(verdict="static_dq", detail=detail)
            print(f"[t{turn}] 정적 실격: {detail}", flush=True)
            msgs.append({"role": "user", "content": DQ_FEEDBACK.format(detail=detail)})
            log["turns"].append(rec)
            log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
            continue

        cached = RESULTS / f"{seed_stem if turn == 1 else variant}_result.json"
        if turn == 1 and a.reuse_turn1 and cached.exists():
            # turn 1 was graded in the batch run; Isaac is deterministic here,
            # so re-running it would only cost two minutes per sample
            result = json.loads(cached.read_text())
            print(f"[t{turn}] 기존 채점 재사용", flush=True)
        else:
            result = grade(variant)
        m = result.get("metrics", {})
        if result.get("_runtime_error"):
            rec["runtime_error"] = result["_runtime_error"]
        rec.update(verdict=result["verdict"], final_dist_m=m.get("final_dist_m"),
                   collisions=m.get("collision_count"),
                   partners=m.get("collision_partners"),
                   time_to_goal_s=m.get("time_to_goal_s"),
                   sim_wall_s=result.get("_wall_s"))
        print(f"[t{turn}] {result['verdict'].upper()}  dist={m.get('final_dist_m')} "
              f"col={m.get('collision_count')} ttg={m.get('time_to_goal_s')}", flush=True)
        log["turns"].append(rec)
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

        if result["verdict"] == "pass":
            passed_at = turn
            break
        if result.get("_runtime_error"):
            msgs.append({"role": "user", "content": RUNTIME_FEEDBACK.format(
                err=result["_runtime_error"])})
        else:
            msgs.append({"role": "user", "content": FEEDBACK.format(
                result=feedback_json(result, a.track, a.feedback_profile))})

    log["passed_at_turn"] = passed_at
    log["turns_used"] = len(log["turns"])
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[done] {stem}: " + (f"{passed_at}회차에 통과" if passed_at
                                else f"{len(log['turns'])}회차까지 실패") +
          f" -> {log_path}", flush=True)
    return 0 if passed_at else 1


if __name__ == "__main__":
    sys.exit(main())
