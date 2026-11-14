"""Prompt A-4 through furio (openclaude on the NPU router) - REPORT.md 실험 14.

One sample is up to --turns sessions of the furio coding agent in a clean workspace outside
the repository, holding only base_carter_run.py (the A-2 base) and ./run_isaac_sim. Turn 1
gets PROMPT_A4F.md plus the base, in the form agent_loop gives A-4; a later turn resumes
the same session with turn_loop's feedback. After each session the file the agent named
(else the newest *_carter_run.py it wrote) is graded once, the way agent_loop grades.

The agent: furio_direct.sh (~/.local/bin/furio, but through llm_log_proxy.py) with furio's own
system prompt (no --bare: --bare cut it to 182 characters and dropped the Write tool), a fresh
config dir, tools Bash/Read/Write/Edit, --permission-mode dontAsk with only Read/Edit/Write
inside the workspace and Bash(./run_isaac_sim:*) allowed. No CLAUDE.md exists on the
workspace's path, so none is picked up.

Kept under --out-dir:
  <stem>_t<N>_stream.jsonl   openclaude stream-json: every message, tool use, tool result
  <stem>_t<N>_stderr.log
  <stem>_llm/                every request and response between openclaude and the router
  <stem>_tool_calls.jsonl    every ./run_isaac_sim call: file, sim_seconds, what it printed
  isaac_logs/                every Isaac run, tests and gradings
  <stem>_session/            openclaude's own session transcript
  <stem>_workspace_t<N>/     the workspace after turn N
  <stem>_turns.json          per-turn summary

    python3 furio_a4_run.py --sample 0 --port 8420
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as A   # noqa: E402  A-2 base, static gate, lock wrapper
import turn_loop as TL   # noqa: E402

FEEDBACK_NOTE = "\n이번 회차에도 `./run_isaac_sim` 으로 최대 {budget}번 시험 주행할 수 있습니다.\n"


def say(text):
    print(f"{time.strftime('%F %T')} {text}", flush=True)


def write_wrapper(ws, out, tag, sample, turn, budget):
    w = ws / "run_isaac_sim"
    w.write_text("#!/usr/bin/env bash\n"
                 "# Isaac Sim test drive on this server: ./run_isaac_sim <file> [sim_seconds]\n"
                 'RUN_ISAAC_FILE="$(realpath "${1:-}" 2>/dev/null || echo "${1:-}")" RUN_ISAAC_SIM_S="${2:-}" '
                 f"exec python3 {HERE}/furio_isaac_tool.py --out-dir {out} --tag {tag} --sample {sample} "
                 f"--turn {turn} --budget {budget}\n")
    os.chmod(w, 0o755)


def parse_stream(path):
    """What the session did, from openclaude's stream-json."""
    info = {"tool_uses": 0, "bash": 0, "denied": 0, "writes": 0, "result_text": None, "num_turns": None,
            "subtype": None, "is_error": None}
    for line in open(path, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type")
        if t == "assistant":
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    info["tool_uses"] += 1
                    info["bash"] += c.get("name") == "Bash"
                    info["writes"] += c.get("name") in ("Write", "Edit")
        elif t == "user":
            for c in e.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    s = c.get("content") if isinstance(c.get("content"), str) else json.dumps(c.get("content"))
                    info["denied"] += "has been denied" in (s or "")
        elif t == "result":
            info.update(result_text=e.get("result"), num_turns=e.get("num_turns"), subtype=e.get("subtype"),
                        is_error=e.get("is_error"))
    return info


def find_submission(ws, text):
    names = re.findall(r"([\w.\-]+_carter_run\.py)", text or "")
    for n in reversed(names):
        p = ws / n
        if n != "base_carter_run.py" and p.exists():
            return p, "named"
    cands = [p for p in ws.glob("*_carter_run.py") if p.name != "base_carter_run.py"]
    if cands:
        return max(cands, key=lambda p: p.stat().st_mtime), "newest"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", default="npu_furio_pA4")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--prompt-file", default="PROMPT_A4F.md")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "turns" / "prompt_a4_furio"))
    ap.add_argument("--ws-root", required=True, help="scratch directory outside the repository")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--test-budget", type=int, default=10)
    ap.add_argument("--agent-max-turns", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--session-timeout", type=int, default=4 * 3600)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    TL.ISAAC_LOG_DIR = out / "isaac_logs"
    TL.ISAAC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{a.tag}_a{a.sample}"
    root = Path(a.ws_root) / stem
    if root.exists():
        raise SystemExit(f"{root} exists - refusing to reuse a workspace")
    ws, cfg = root / "ws", root / "config"
    ws.mkdir(parents=True)
    cfg.mkdir(parents=True)
    (cfg / "settings.json").write_text(json.dumps({"model": a.model}))
    (ws / "base_carter_run.py").write_text(A.BASE)
    first = A.model_input(a.prompt_file)
    log = {"model": a.model, "tag": a.tag, "sample": a.sample, "agent": "furio (openclaude) via furio_direct.sh",
           "prompt_file": a.prompt_file, "input_chars": len(first),
           "input_sha16": hashlib.sha256(first.encode()).hexdigest()[:16],
           "workspace": str(ws), "test_budget": a.test_budget, "turns_cap": a.turns,
           "agent_max_turns": a.agent_max_turns, "max_output_tokens": a.max_tokens,
           "flags": "--tools Bash,Read,Write,Edit --permission-mode dontAsk "
                    "--allowed-tools Read(./**) Edit(./**) Write(./**) Bash(./run_isaac_sim:*)",
           "started": time.strftime("%F %T"), "turns": []}
    say(f"[start] {stem} 입력 {log['input_chars']}자 sha16 {log['input_sha16']} 작업 폴더 {ws}")
    proxy = subprocess.Popen([sys.executable, str(HERE / "llm_log_proxy.py"), "--port", str(a.port),
                              "--log-dir", str(out / f"{stem}_llm"), "--inject-logprobs"],
                             stdout=subprocess.DEVNULL, stderr=open(out / f"{stem}_proxy.err", "w"))
    time.sleep(1)
    session = str(uuid.uuid4())
    prompt, passed_at = first, None
    try:
        for turn in range(1, a.turns + 1):
            write_wrapper(ws, out, a.tag, a.sample, turn, a.test_budget)
            env = dict(os.environ, OPENAI_BASE_URL=f"http://127.0.0.1:{a.port}/v1", OPENCLAUDE_CONFIG_DIR=str(cfg),
                       OPENAI_MODEL=a.model, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(a.max_tokens),
                       BASH_DEFAULT_TIMEOUT_MS="3600000", BASH_MAX_TIMEOUT_MS="3600000")
            cmd = ["bash", str(HERE / "furio_direct.sh"), "--tools", "Bash,Read,Write,Edit",
                   "--permission-mode", "dontAsk",
                   "--allowed-tools", "Read(./**)", "Edit(./**)", "Write(./**)", "Bash(./run_isaac_sim:*)",
                   "--max-turns", str(a.agent_max_turns), "--output-format", "stream-json", "--verbose"]
            cmd += ["--resume", session] if turn > 1 else ["--session-id", session]
            cmd += ["-p", prompt]
            stream, err = out / f"{stem}_t{turn}_stream.jsonl", out / f"{stem}_t{turn}_stderr.log"
            say(f"[t{turn}] furio 세션 시작 ({'이어 가기' if turn > 1 else '새 세션'} {session})")
            t0 = time.time()
            with open(stream, "w") as so, open(err, "w") as se:
                try:
                    rc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                        timeout=a.session_timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            wall = round(time.time() - t0, 1)
            info = parse_stream(stream)
            shutil.copytree(ws, out / f"{stem}_workspace_t{turn}")
            rec = {"turn": turn, "session_id": session, "returncode": rc, "wall_s": wall, **info}
            say(f"[t{turn}] 세션 끝 rc={rc} {wall:.0f}s 도구 {info['tool_uses']}번(Bash {info['bash']}, "
                f"쓰기 {info['writes']}, 거부 {info['denied']}) 에이전트 턴 {info['num_turns']}")
            sub, how = find_submission(ws, info["result_text"])
            variant = stem if turn == 1 else f"{stem}_t{turn}"
            if sub is None:
                rec.update(verdict="no_submission")
                log["turns"].append(rec)
                (out / f"{stem}_turns.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
                say(f"[t{turn}] 제출 파일 없음")
                prompt = ("작업 폴더에 제출 파일이 없습니다. `<모델 본인 이름>_carter_run.py` 로 저장하고 "
                          "마지막 답에 그 파일 이름을 적으십시오." + FEEDBACK_NOTE.format(budget=a.test_budget))
                continue
            text = A.build(sub.read_text())
            (HERE / f"{variant}_carter_run.py").write_text(text)
            rec.update(submission=sub.name, submission_found_by=how)
            ok, detail = A.static_check(text)
            if not ok:
                rec.update(verdict="static_dq", detail=detail)
                log["turns"].append(rec)
                (out / f"{stem}_turns.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
                say(f"[t{turn}] 제출 정적 실격: {detail.splitlines()[0]}")
                prompt = TL.DQ_FEEDBACK.format(detail=detail) + FEEDBACK_NOTE.format(budget=a.test_budget)
                continue
            result = TL.grade(variant)
            mt = result.get("metrics", {})
            rec.update(verdict=result["verdict"], final_dist_m=mt.get("final_dist_m"),
                       collisions=mt.get("collision_count"), partners=mt.get("collision_partners"),
                       first_collision_t=mt.get("first_collision_t"), time_to_goal_s=mt.get("time_to_goal_s"),
                       runtime_error=result.get("_runtime_error"))
            log["turns"].append(rec)
            (out / f"{stem}_turns.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
            say(f"[t{turn}] 채점 {result['verdict'].upper()} dist={mt.get('final_dist_m')} "
                f"col={mt.get('collision_count')} {mt.get('collision_partners') or ''} ttg={mt.get('time_to_goal_s')}")
            if result["verdict"] == "pass":
                passed_at = turn
                break
            if result.get("_runtime_error"):
                prompt = TL.RUNTIME_FEEDBACK.format(err=result["_runtime_error"]) + FEEDBACK_NOTE.format(budget=a.test_budget)
            else:
                prompt = TL.FEEDBACK.format(result=TL.feedback_json(result, "a")) + FEEDBACK_NOTE.format(budget=a.test_budget)
    finally:
        proxy.terminate()
        proj = cfg / "projects"
        if proj.exists():
            shutil.copytree(proj, out / f"{stem}_session", dirs_exist_ok=True)
        log["passed_at_turn"] = passed_at
        log["turns_used"] = len(log["turns"])
        log["ended"] = time.strftime("%F %T")
        (out / f"{stem}_turns.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
        say(f"[done] {stem}: " + (f"{passed_at}회차에 통과" if passed_at else f"{len(log['turns'])}회차까지 실패"))
    return 0 if passed_at else 1


if __name__ == "__main__":
    sys.exit(main())
