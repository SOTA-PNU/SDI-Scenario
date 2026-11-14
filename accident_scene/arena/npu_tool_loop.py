"""Turn protocol with a Python tool for NPU gpt-oss-120b (REPORT.md 실험 8, T2).

T3's setting (request field reasoning_effort=high) plus one tool, so the model
can do what the equal-input Claude agent did between receiving the task and
handing in a file: run its own code - checks, a simplified simulation it writes
itself - before answering. The code runs in a network-less bubblewrap sandbox
whose only visible folder is the run's own working directory (kept for the
record). Everything else is turn_loop.py: the same first message, static check,
Isaac grading, feedback wording and 8-turn cap. The added system message and
tool description are quoted, with their leak audit, in REPORT.md 실험 8.

    python3 npu_tool_loop.py --sample 0 [--max-turns 8]

T2P (실험 9) reuses this loop with options that all default to T2's behaviour:
--system-file (the procedure text), --tool-timeout, --provide (files placed in
the working folder at a fresh start), and --require-file / --min-tool-calls with
--reminder-file / --max-reminders (a final answer that comes before the required
work exists is answered once with a fixed, pre-registered reminder).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import turn_loop                                                # noqa: E402
from generate import api, extract, model_input                  # noqa: E402
from splice import splice                                       # noqa: E402
from turn_loop import (DQ_FEEDBACK, FEEDBACK, REQUEST_RETRIES,  # noqa: E402
                       RUNTIME_FEEDBACK, feedback_json, static_check, wait_for_server)

OUT = HERE / "results" / "turns" / "transfer_t2"
SYSTEM = ("이 작업에는 python 도구가 있습니다. 답을 내기 전에 직접 작성한 Python 코드를 실행해 볼 수 있습니다"
          "(예: 작성한 컨트롤러 점검, 직접 만든 간단한 시뮬레이션). 실행 환경은 네트워크가 없는 빈 작업 폴더이고 "
          "python3와 numpy를 쓸 수 있습니다. 실제 Isaac 시뮬레이터는 없으며, 프롬프트 규칙 7의 '시뮬레이터'는 그것을 "
          "뜻합니다. 작업이 끝나면 수정한 파일을 작업 폴더에 `<모델 이름>_carter_run.py` 로 저장하거나(그 경우 그 "
          "파일을 채점합니다) 파일 전문을 답으로 출력하십시오.")
def tool_spec(timeout=300):
    return {"type": "function", "function": {
        "name": "python",
        "description": ("Run Python 3 code in an isolated sandbox: no network; the working directory persists "
                        f"across calls in this task; python3 with numpy is available; time limit {timeout} s; "
                        "stdout and stderr are returned (truncated)."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python 3 source to execute"}},
            "required": ["code"]}}}


TOOL = tool_spec()
FORCE_MSG = "도구 사용 한도에 도달했습니다. 지금까지의 작업을 바탕으로 최종 답을 내십시오."


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_code(work, code, n, timeout=300, limit=8000):
    """Execute one cell in a network-less sandbox that sees only `work`."""
    cells = work / ".cells"
    cells.mkdir(parents=True, exist_ok=True)
    cell = cells / f"cell_{n:03d}.py"
    cell.write_text(code)
    cmd = ["bwrap", "--die-with-parent", "--unshare-all", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--bind", str(work), "/work", "--chdir", "/work",
           "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/bin:/bin",
           "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "MPLBACKEND", "Agg",
           "timeout", "-k", "10", str(timeout), "python3", f"/work/.cells/{cell.name}"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    secs = time.time() - t0
    out = p.stdout or ""
    if p.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + p.stderr
    if len(out) > limit:
        out = out[:limit // 2] + f"\n...[{len(out) - limit}자 생략]...\n" + out[-limit // 2:]
    note = f"[exit {p.returncode}{' (시간 초과)' if p.returncode in (124, 137) else ''}, {secs:.1f} s]"
    return (out + ("\n" if out else "") + note), p.returncode, secs


def size(msgs):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in msgs)


def compress(msgs, budget, keep_last=6):
    """Shrink the oldest tool outputs (then their code) until the thread fits."""
    tools = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    old = tools[:-keep_last] if len(tools) > keep_last else []
    n = 0
    for i in old:
        if size(msgs) <= budget:
            break
        c = msgs[i].get("content") or ""
        if len(c) > 300:
            msgs[i]["content"] = f"[이전 실행 출력 생략: {len(c)}자]"
            n += 1
    for m in msgs:
        if size(msgs) <= budget:
            break
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments") or ""
            if len(a) > 300 and m is not msgs[-1]:
                tc["function"]["arguments"] = json.dumps({"code": f"# [이전 코드 생략: {len(a)}자]"})
                n += 1
    return n


def chat(a, msgs, force, rec=None):
    payload = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "top_p": a.top_p,
               "reasoning_effort": a.reasoning_effort, "tools": [tool_spec(a.tool_timeout)],
               "tool_choice": "none" if force else "auto"}
    n_retry = 0
    while True:
        t0 = time.time()
        try:
            r = api(a.base, "/chat/completions", payload, timeout=a.request_timeout)
            break
        except urllib.error.HTTPError as e:
            # 4xx is an answer (the tool_choice fallback depends on it); 5xx is an outage
            if e.code < 500:
                raise
            err = e
        except (TimeoutError, OSError) as e:
            err = e
        # no answer came back, so nothing was learned: once the server answers
        # again, ask the same call again
        n_retry += 1
        if rec is not None:
            rec["request_retries"] = rec.get("request_retries", 0) + 1
        if n_retry > REQUEST_RETRIES:
            raise err
        print(f"       요청 실패({type(err).__name__}) - 서버 응답을 기다린 뒤 재요청 "
              f"{n_retry}/{REQUEST_RETRIES}", flush=True)
        waited = wait_for_server(a.base)
        if rec is not None:
            rec.setdefault("server_wait_s", []).append(waited)
    return payload, r, time.time() - t0


def unmet(a, work, t0, rec):
    """What the turn still lacks before a final answer counts (T2P); [] = nothing."""
    miss = [f for f in a.require_file
            if not ((work / f).is_file() and (work / f).stat().st_mtime >= t0 - 1)]
    if len(rec["tools"]) < a.min_tool_calls:
        miss.append(f"도구 호출 {len(rec['tools'])}/{a.min_tool_calls}")
    return miss


def find_submission(work, t0):
    cands = [p for p in work.rglob("*_carter_run.py")
             if p.is_file() and p.name != "base_carter_run.py" and ".cells" not in p.parts]
    fresh = [p for p in cands if p.stat().st_mtime >= t0 - 1]
    return (max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None), sorted(p.name for p in cands)


def main():
    # '@file' reads further arguments from a file, one per line. T2P passes its options that way so that
    # no command line names '*_carter_run.py' - carter-run's serial guard counts any python process
    # whose command line does, and would wait on these loops forever.
    ap = argparse.ArgumentParser(fromfile_prefix_chars="@")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--tag", default="npu_gpt_oss_120b_t2")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-tool-calls", type=int, default=12)
    ap.add_argument("--turn-wall-min", type=float, default=60.0)
    ap.add_argument("--budget-chars", type=int, default=300000)
    ap.add_argument("--work-root", default=str(Path.home() / ".cache" / "npu_t2"))
    ap.add_argument("--request-timeout", type=int, default=7200)
    ap.add_argument("--resume", action="store_true",
                    help="restart the interrupted turn from its saved start state")
    ap.add_argument("--turns-dir", help="write records here instead of results/turns/transfer_t2")
    ap.add_argument("--system-file", help="system message text (default: the T2 text)")
    ap.add_argument("--tool-timeout", type=int, default=300, help="seconds per tool call (also shown to the model)")
    ap.add_argument("--provide", action="append", default=[],
                    help="file copied into the working folder at a fresh start (path relative to this folder)")
    ap.add_argument("--require-file", action="append", default=[],
                    help="working-folder file that must be written this turn before a final answer counts")
    ap.add_argument("--min-tool-calls", type=int, default=0,
                    help="tool calls this turn before a final answer counts")
    ap.add_argument("--reminder-file", help="text sent when a final answer comes before the requirements are met")
    ap.add_argument("--max-reminders", type=int, default=0, help="reminders per turn (0 = never)")
    a = ap.parse_args()
    global OUT
    if a.turns_dir:
        OUT = Path(a.turns_dir)
    system = Path(a.system_file).read_text().strip() if a.system_file else SYSTEM
    reminder = Path(a.reminder_file).read_text().strip() if a.reminder_file else None

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.tag}_a{a.sample}"
    log_path = OUT / f"{stem}_turns.json"
    state_path = OUT / f"{stem}_state.json"
    work = Path(a.work_root) / stem / "work"
    work0 = Path(a.work_root) / stem / "work_turnstart"
    if a.resume:
        if not state_path.exists():
            print(f"[!] {state_path.name} 가 없습니다 - 이어 갈 수 없습니다", flush=True)
            return 2
        if log_path.exists() and json.loads(log_path.read_text()).get("status"):
            print(f"[resume] {stem}: 이미 종료된 실행", flush=True)
            return 0
    elif log_path.exists() or state_path.exists():
        print(f"[!] {log_path.name} 가 이미 있습니다 - 새 이름(--tag)으로 실행하십시오", flush=True)
        return 2
    if not a.resume:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for f in a.provide:
            shutil.copy2(HERE / f, work / Path(f).name)

    # the sandbox must work before any model call is spent on it
    out, rc, _ = run_code(work, "import numpy, sys; print('sandbox ok', sys.version.split()[0], numpy.__version__)", 0)
    if rc != 0 or "sandbox ok" not in out:
        print(f"[!] 샌드박스 자가 점검 실패: {out[-300:]}", flush=True)
        return 3
    print(f"[{stem}] {out.splitlines()[0]}", flush=True)
    cell = work / ".cells" / "cell_000.py"
    if cell.exists():
        cell.unlink()

    first = model_input("a")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    log = {"tag": a.tag, "sample": a.sample, "track": "a", "model": a.model, "endpoint": a.base,
           "reasoning_effort": a.reasoning_effort, "max_tokens": a.max_tokens,
           "temperature": a.temperature, "top_p": a.top_p, "max_tool_calls": a.max_tool_calls,
           "turn_wall_min": a.turn_wall_min, "budget_chars": a.budget_chars,
           "system_sha16": sha16(system), "first_message_sha16": sha16(first), "turns": []}
    if system != SYSTEM or a.tool_timeout != 300 or a.provide or a.require_file or a.min_tool_calls:
        log.update(system_file=a.system_file, tool_timeout=a.tool_timeout,
                   tool_sha16=sha16(json.dumps(tool_spec(a.tool_timeout), ensure_ascii=False)),
                   provided={Path(f).name: sha16((HERE / f).read_text()) for f in a.provide},
                   require_file=a.require_file, min_tool_calls=a.min_tool_calls,
                   reminder_sha16=sha16(reminder) if reminder else None, max_reminders=a.max_reminders)

    def save():
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    cell_no, start = 0, 1
    if a.resume:
        # restart the interrupted turn from exactly where it began: same thread,
        # same working folder (files the lost half-turn wrote are rolled back)
        st = json.loads(state_path.read_text())
        msgs, cell_no, log, start = st["msgs"], st["cell_no"], st["log"], st["turn"]
        log["resumed_at_turn"] = log.get("resumed_at_turn", []) + [start]
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(work0, work)
        print(f"[resume] {stem}: {start}회차를 처음부터 다시 (메시지 {len(msgs)}개)", flush=True)

    for turn in range(start, a.max_turns + 1):
        variant = stem if turn == 1 else f"{stem}_t{turn}"
        rec = {"turn": turn, "variant": variant, "calls": [], "tools": []}
        state_path.write_text(json.dumps({"turn": turn, "msgs": msgs, "cell_no": cell_no,
                                          "log": log}, ensure_ascii=False))
        if work0.exists():
            shutil.rmtree(work0)
        shutil.copytree(work, work0)
        t_turn = time.time()
        final, k, forced, fallback = None, 0, False, False
        while True:
            k += 1
            force = (len(rec["tools"]) >= a.max_tool_calls
                     or time.time() - t_turn > a.turn_wall_min * 60)
            forced = forced or force
            rec.setdefault("compressed", 0)
            rec["compressed"] += compress(msgs, a.budget_chars)
            try:
                payload, r, dt = chat(a, msgs, force, rec)
            except urllib.error.HTTPError as e:
                if not force:
                    raise
                # the server refused tool_choice "none": ask in words instead (logged)
                msgs.append({"role": "user", "content": FORCE_MSG})
                fallback = True
                payload, r, dt = chat(a, msgs, False, rec)
            (OUT / f"{variant}_call{k:02d}_raw.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
            ch = r["choices"][0]
            msg, u = ch["message"], r.get("usage", {})
            calls = msg.get("tool_calls") or []
            rec["calls"].append({"k": k, "gen_s": round(dt, 1), "finish": ch.get("finish_reason"),
                                 "completion_tokens": u.get("completion_tokens"),
                                 "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                 "prompt_tokens": u.get("prompt_tokens"), "tool_calls": len(calls),
                                 "forced": force, "thread_chars": size(msgs)})
            print(f"[t{turn}.{k}] {dt:.0f}s out={u.get('completion_tokens')} "
                  f"reason={rec['calls'][-1]['reasoning_tokens']} tools={len(calls)}"
                  f"{' (최종 요구)' if force else ''}", flush=True)
            if calls and not (force and not fallback):
                msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
                for tc in calls:
                    cell_no += 1
                    try:
                        code = json.loads(tc["function"].get("arguments") or "{}").get("code", "")
                    except json.JSONDecodeError as e:
                        code, res, rc, secs = "", f"[도구 인자 JSON 오류: {e}]", -1, 0.0
                    else:
                        res, rc, secs = run_code(work, code, cell_no, timeout=a.tool_timeout)
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
                    rec["tools"].append({"cell": cell_no, "call": k, "code_chars": len(code), "exit": rc,
                                         "secs": round(secs, 1), "out_chars": len(res)})
                    print(f"       cell {cell_no}: {len(code)}자 → exit {rc} ({secs:.0f}s)", flush=True)
                if fallback:
                    final = ""
                    break
                continue
            final = msg.get("content") or ""
            msgs.append({"role": "assistant", "content": final})
            missing = [] if force or not reminder else unmet(a, work, t_turn, rec)
            if missing and rec.get("reminders_sent", 0) < a.max_reminders:
                # T2P: the answer came before the required work exists - one fixed reminder
                rec["reminders_sent"] = rec.get("reminders_sent", 0) + 1
                rec.setdefault("reminders", []).append({"call": k, "missing": missing})
                msgs.append({"role": "user", "content": reminder})
                print(f"       최종 답 보류 ({', '.join(missing)}) → 절차 안내 "
                      f"{rec['reminders_sent']}/{a.max_reminders}", flush=True)
                continue
            break
        (OUT / f"{variant}_reply.md").write_text(final or "")
        rec.update(forced_final=forced, force_by_message=fallback, model_calls=k,
                   tool_calls=len(rec["tools"]),
                   gen_wall_s=round(sum(c["gen_s"] for c in rec["calls"]), 1),
                   tool_wall_s=round(sum(t["secs"] for t in rec["tools"]), 1),
                   turn_wall_s=round(time.time() - t_turn, 1),
                   completion_tokens=sum(c["completion_tokens"] or 0 for c in rec["calls"]),
                   reasoning_tokens=sum(c["reasoning_tokens"] or 0 for c in rec["calls"]))
        snap = OUT / f"{stem}_files" / variant
        if snap.exists():
            shutil.rmtree(snap)
        shutil.copytree(work, snap)

        sub, cands = find_submission(work, t_turn)
        if sub is not None:
            src, channel = sub.read_text(), "file"
        else:
            src, channel = extract(final or ""), "text"
        spliced = splice(src)
        text = spliced if spliced is not None else (src if src.endswith("\n") else src + "\n")
        path = HERE / f"{variant}_carter_run.py"
        path.write_text(text)
        rec["submission"] = {"channel": channel, "agent_file": sub.name if sub else None,
                             "candidates": cands, "spliced": spliced is not None,
                             "file": path.name, "sha16": sha16(text)}
        ok, detail = static_check(path, "a")
        rec["static_ok"] = ok
        if not ok:
            rec.update(verdict="static_dq", detail=detail, feedback_kind="static_dq")
            msgs.append({"role": "user", "content": DQ_FEEDBACK.format(detail=detail)})
            print(f"[t{turn}] 정적 실격: {detail}", flush=True)
            log["turns"].append(rec)
            save()
            continue
        result = turn_loop.grade(variant)
        m = result.get("metrics", {})
        if result.get("_runtime_error"):
            rec["runtime_error"] = result["_runtime_error"]
        rec.update(verdict=result["verdict"], final_dist_m=m.get("final_dist_m"),
                   final_yaw_err_rad=m.get("final_yaw_err_rad"), collisions=m.get("collision_count"),
                   partners=m.get("collision_partners"), time_to_goal_s=m.get("time_to_goal_s"),
                   sim_wall_s=result.get("_wall_s"))
        print(f"[t{turn}] {result['verdict'].upper()} dist={m.get('final_dist_m')} "
              f"col={m.get('collision_count')} {m.get('collision_partners')} "
              f"tools={rec['tool_calls']} gen={rec['gen_wall_s']}s", flush=True)
        if result["verdict"] == "pass":
            rec["feedback_kind"] = None
            log["turns"].append(rec)
            log.update(passed_at_turn=turn, status="pass")
            save()
            print(f"[done] {stem}: {turn}회차에 통과 -> {log_path}", flush=True)
            return 0
        if result.get("_runtime_error"):
            msgs.append({"role": "user", "content": RUNTIME_FEEDBACK.format(err=result["_runtime_error"])})
            rec["feedback_kind"] = "runtime_error"
        else:
            msgs.append({"role": "user", "content": FEEDBACK.format(result=feedback_json(result, "a"))})
            rec["feedback_kind"] = "result"
        log["turns"].append(rec)
        save()

    log.update(passed_at_turn=None, status=f"fail_{a.max_turns}")
    save()
    print(f"[done] {stem}: {a.max_turns}회차까지 실패 -> {log_path}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
