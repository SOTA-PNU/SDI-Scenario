"""Turn protocol with a sandboxed Claude Code agent as the entrant (REPORT.md 실험 7).

Same contract as turn_loop.py - the same first message (generate.model_input,
6,961 chars for track A), the same static check, the same Isaac grading and
the same feedback wording - but the entrant is a Claude Code agent with its
default tools, run inside claude_sandbox.sh so that the prompt is the only
task information it can reach. What the agent *did* between receiving a
message and handing in a file is the point here, so every turn keeps:

  <variant>_request.md      the exact message sent
  <variant>_stream.jsonl    Claude Code's full stream-json event log
  <variant>_reply.md        the agent's final text for the turn
  <stem>_files/<variant>/   its workspace and /tmp after the turn
  <stem>_transcripts/       session + subagent transcripts (cumulative)
  <stem>_init.json          the harness it ran in: model, tools, agents, MCP
  <stem>_turns.json         per turn: tools per agent, subagents, tokens,
                            wall time, submission, static check, verdict

    python3 claude_agent_loop.py --sample 0 --max-turns 8 [--dry-run]
    python3 claude_agent_loop.py --sample 0 --resume-run     # continue a stopped run
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
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import turn_loop                                                # noqa: E402
from generate import extract, model_input                       # noqa: E402
from splice import splice                                       # noqa: E402
from turn_loop import (DQ_FEEDBACK, FEEDBACK, RUNTIME_FEEDBACK,  # noqa: E402
                       feedback_json, static_check)

SANDBOX = HERE / "claude_sandbox.sh"
OUT = HERE / "results" / "turns" / "claude_eq"
CRED = Path.home() / ".claude" / ".credentials.json"
TOKEN_FILE = Path(os.environ.get("CLAUDE_EQ_TOKEN_FILE",
                                 Path.home() / ".config" / "claude_eq" / "oauth_token"))
REDACTED = "<REDACTED_OAUTH_TOKEN>"
# Information parity with the NPU entry (no web), plus tools that act outside
# the sandbox on the user's account. None of them bears on writing a
# controller; everything else in Claude Code's default set stays available.
DISALLOWED = ["WebSearch", "WebFetch", "Artifact", "ArtifactComments", "ArtifactData",
              "SendUserFile", "PushNotification", "RemoteTrigger", "CronCreate",
              "CronDelete", "ScheduleWakeup", "ShareOnboardingGuide", "DesignSync",
              "SendFeedback"]
NET_RE = re.compile(r"\b(curl|wget|pip3?\s+install|git\s+clone|apt(-get)?\s|https?://|ssh\s)")
SPAWN_TOOLS = ("Agent", "Task")
LIMIT_RE = re.compile(r"(session|usage|rate|weekly)\s+limit", re.I)
AUTH_RE = re.compile(r"(oauth token has expired|authentication_error|invalid.{0,20}token|"
                     r"please run /login|\b401\b)", re.I)
RESET_RE = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*(?:\(([^)]+)\))?", re.I)
# sent only after a usage-limit stop, to let the agent finish the same turn;
# it carries no task information and is logged as an interruption, not a turn
CONTINUE_MSG = "(외부 사정으로 작업이 중단되었습니다. 중단된 지점부터 이어서 진행하십시오.)"


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def token():
    """(secret, expiry epoch) of whatever claude_sandbox.sh will hand the agent."""
    if TOKEN_FILE.exists() and TOKEN_FILE.read_text().strip():
        return "".join(TOKEN_FILE.read_text().split()), time.time() + 365 * 86400
    d = json.loads(CRED.read_text())["claudeAiOauth"]
    return d["accessToken"], d.get("expiresAt", 0) / 1000.0


def redact_tree(root, secret):
    """The token is in the agent's environment; make sure no copy of it leaves."""
    hits = 0
    paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for p in paths:
        try:
            s = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if secret in s:
            p.write_text(s.replace(secret, REDACTED))
            hits += 1
    return hits


def copy_tree(src, dst, skip=(), max_bytes=5_000_000):
    if dst.exists():
        shutil.rmtree(dst)
    if not src.exists():
        return 0
    n = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(rel.parts[:len(s.split("/"))] == tuple(s.split("/")) for s in skip):
            continue
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= max_bytes:
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            n += 1
    return n


def parse_stream(lines):
    """Split Claude Code's event log by agent: main loop vs each subagent."""
    init = result = None
    agents = {}
    spawns = {}
    bash = []

    def slot(key):
        return agents.setdefault(key, {"tools": Counter(), "text_blocks": 0, "text_chars": 0,
                                       "thinking_blocks": 0, "thinking_chars": 0})
    for ln in lines:
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            init = ev
        elif kind == "result":
            result = ev
        elif kind == "assistant":
            key = ev.get("parent_tool_use_id") or "main"
            a = slot(key)
            for b in (ev.get("message") or {}).get("content") or []:
                t = b.get("type")
                if t == "tool_use":
                    a["tools"][b["name"]] += 1
                    inp = b.get("input") or {}
                    if b["name"] in SPAWN_TOOLS:
                        spawns[b["id"]] = {
                            "by": key, "subagent_type": inp.get("subagent_type"),
                            "description": inp.get("description"), "model": inp.get("model"),
                            "run_in_background": inp.get("run_in_background"),
                            "prompt_chars": len(inp.get("prompt") or "")}
                    if b["name"] == "Bash":
                        bash.append((key, inp.get("command") or ""))
                elif t == "thinking":
                    a["thinking_blocks"] += 1
                    a["thinking_chars"] += len(b.get("thinking") or "")
                elif t == "text":
                    a["text_blocks"] += 1
                    a["text_chars"] += len(b.get("text") or "")
    for a in agents.values():
        a["tools"] = dict(a["tools"])
    return init, result, agents, spawns, bash


def transcript_agents(projects):
    """Per-subagent facts from Claude Code's own transcripts (meta + tool counts)."""
    out = []
    for jl in sorted(projects.rglob("subagents/*.jsonl")):
        meta_p = jl.with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        tools, models, first, last = Counter(), set(), None, None
        for ln in jl.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            first = first or ts
            last = ts or last
            m = d.get("message") or {}
            if isinstance(m, dict) and m.get("model"):
                models.add(m["model"])
            for b in (m.get("content") if isinstance(m, dict) else None) or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools[b["name"]] += 1
        out.append({"agent": jl.stem, "meta": meta, "models": sorted(models),
                    "tools": dict(tools), "first": first, "last": last})
    return out


def find_submission(run_dir, t0):
    """Newest *_carter_run.py the agent wrote; workspace first, then /tmp."""
    for root in (run_dir / "home" / "workspace", run_dir / "tmp"):
        cands = [p for p in root.rglob("*_carter_run.py")
                 if p.is_file() and p.name != "base_carter_run.py"]
        if cands:
            fresh = [p for p in cands if p.stat().st_mtime >= t0 - 1]
            pick = max(fresh or cands, key=lambda p: p.stat().st_mtime)
            return pick, bool(fresh), sorted(str(p.relative_to(run_dir)) for p in cands)
    return None, False, []


def grade_serial(variant):
    """turn_loop.grade, waiting out the carter-run guard instead of failing."""
    for _ in range(60):
        try:
            return turn_loop.grade(variant)
        except RuntimeError as e:
            print(f"       Isaac 사용 중 ({e}) - 60 s 뒤 재시도", flush=True)
            time.sleep(60)
    raise RuntimeError("carter-run guard busy for 60 min")


def wait_for_token(min_left_min, turn):
    """Block until the access token has min_left_min minutes left.

    The parent Claude Code session refreshes the shared credentials file as a
    side effect of its own requests; this loop only ever reads it, so waiting
    is the whole job. One line per 10 min keeps the monitor informed.
    """
    last = 0.0
    while True:
        secret, exp = token()
        left = (exp - time.time()) / 60
        if left >= min_left_min:
            return secret, exp
        if time.time() - last >= 600:
            print(f"[t{turn}] 접근 토큰 {left:.0f}분 남음 - 갱신될 때까지 대기", flush=True)
            last = time.time()
        time.sleep(60)


def usage_limit_reset(result):
    """When the turn ended on the account's usage limit, the time to resume."""
    if not result or not result.get("is_error"):
        return None
    text = str(result.get("result") or "")
    if AUTH_RE.search(text):
        # an expired access token: resume as soon as a fresh one is on disk
        # (the caller blocks in wait_for_token before continuing)
        return datetime.now().astimezone()
    if not LIMIT_RE.search(text):
        return None
    m = RESET_RE.search(text)
    tz = None
    if m and m.group(4):
        try:
            tz = ZoneInfo(m.group(4).strip())
        except Exception:
            tz = None
    now = datetime.now(tz) if tz else datetime.now().astimezone()
    if not m:
        return now + timedelta(minutes=30)
    hour = int(m.group(1)) % 12 + (12 if (m.group(3) or "").lower() == "pm" else 0) \
        if m.group(3) else int(m.group(1))
    wake = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if wake <= now:
        wake = now + timedelta(minutes=5) if now - wake < timedelta(hours=1) else wake + timedelta(days=1)
    return wake + timedelta(minutes=2)


def run_agent(cmd, message, stream_p, err_p, timeout):
    """One claude -p process; its event stream goes straight to disk."""
    t0 = time.time()
    with stream_p.open("w") as so, err_p.open("w") as se:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=so, stderr=se, text=True)
        try:
            p.communicate(input=message, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = "timeout"
    return rc, stream_p.read_text(), err_p.read_text(), time.time() - t0


def merge_segments(parsed):
    """Combine the per-agent counts of a turn that was split by interruptions."""
    agents, spawns, bash = {}, {}, []
    for _, _, ag, sp, bs in parsed:
        for key, v in ag.items():
            a = agents.setdefault(key, {"tools": Counter(), "text_blocks": 0, "text_chars": 0,
                                        "thinking_blocks": 0, "thinking_chars": 0})
            a["tools"].update(v["tools"])
            for f in ("text_blocks", "text_chars", "thinking_blocks", "thinking_chars"):
                a[f] += v[f]
        spawns.update(sp)
        bash += bs
    for a in agents.values():
        a["tools"] = dict(a["tools"])
    return agents, spawns, bash


def claude_cmd(a, run_dir, sid, first):
    cmd = [str(SANDBOX), str(run_dir), "-p", "--output-format", "stream-json", "--verbose",
           "--model", a.model, "--effort", a.effort, "--permission-mode", "bypassPermissions",
           "--disallowedTools", *DISALLOWED, "--strict-mcp-config",
           "--mcp-config", '{"mcpServers":{}}',
           "--settings", json.dumps({"showThinkingSummaries": True})]
    cmd += ["--session-id", sid] if first else ["--resume", sid]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--model", default="claude-opus-5-5[1m]")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--tag", default="claude_eq")
    ap.add_argument("--sandbox-root", default=str(Path.home() / ".cache" / "claude_eq"),
                    help="where the agent's fresh home lives; kept out of the project tree")
    ap.add_argument("--turn-timeout", type=int, default=4 * 3600)
    ap.add_argument("--min-token-min", type=int, default=90,
                    help="wait for a refreshed access token instead of starting a turn "
                         "the current one cannot finish")
    ap.add_argument("--resume-run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.tag}_a{a.sample}"
    run_dir = Path(a.sandbox_root) / stem
    log_path = OUT / f"{stem}_turns.json"

    if a.resume_run:
        log = json.loads(log_path.read_text())
        if log.get("finished"):
            print(f"[done] {stem} 이미 종료: {log.get('status')}")
            return 0
    else:
        if log_path.exists() and not a.dry_run:
            print(f"[!] {log_path.name} 가 이미 있습니다. 이어 하려면 --resume-run", flush=True)
            return 2
        first_msg = model_input("a")
        log = {"tag": a.tag, "sample": a.sample, "track": "a", "model": a.model,
               "effort": a.effort, "max_turns": a.max_turns, "session_id": str(uuid.uuid4()),
               "sandbox": str(SANDBOX.name), "disallowed_tools": DISALLOWED,
               "settings": {"showThinkingSummaries": True},
               "first_message_chars": len(first_msg), "first_message_sha16": sha16(first_msg),
               "next_prompt": first_msg, "turns": [], "subagents": []}

    if a.dry_run:
        cmd = claude_cmd(a, run_dir, log["session_id"], not log["turns"])
        print("[dry-run] stem", stem, "| run_dir", run_dir)
        print("[dry-run] first message", log["first_message_chars"], "chars sha16",
              log["first_message_sha16"])
        print("[dry-run] next turn", len(log["turns"]) + 1, "prompt chars", len(log["next_prompt"]))
        print("[dry-run] command:", " ".join(cmd))
        _, exp = token()
        print(f"[dry-run] access token valid for {(exp - time.time()) / 60:.0f} min")
        return 0

    def save():
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    run_dir.mkdir(parents=True, exist_ok=True)
    prev_sha = log["turns"][-1].get("submission", {}).get("sha16") if log["turns"] else None
    prev_result = log.get("_last_result")

    for turn in range(len(log["turns"]) + 1, a.max_turns + 1):
        variant = stem if turn == 1 else f"{stem}_t{turn}"
        secret, exp = wait_for_token(a.min_token_min, turn)

        prompt = log["next_prompt"]
        (OUT / f"{variant}_request.md").write_text(prompt)
        rec = {"turn": turn, "variant": variant, "prompt_chars": len(prompt),
               "prompt_sha16": sha16(prompt)}
        print(f"[t{turn}] 에이전트 시작 ({len(prompt)}자)", flush=True)

        # The event stream goes straight to disk, so a run killed mid-turn still
        # leaves everything the agent did up to that moment. A stop on the
        # account's usage limit is waited out and the SAME turn continued with
        # CONTINUE_MSG; every segment keeps its own stream file and is logged.
        t0 = time.time()
        segs, message, wall = [], prompt, 0.0
        while True:
            k = len(segs)
            sfx = "" if k == 0 else f"_r{k}"
            sp_, ep_ = OUT / f"{variant}_stream{sfx}.jsonl", OUT / f"{variant}_stderr{sfx}.txt"
            rc, out, err, w = run_agent(
                claude_cmd(a, run_dir, log["session_id"], turn == 1 and k == 0),
                message, sp_, ep_, a.turn_timeout)
            wall += w
            segs.append((sp_, ep_, parse_stream(out.splitlines())))
            wake = usage_limit_reset(segs[-1][2][1])
            if wake is None or k >= 3:
                break
            rec.setdefault("interruptions", []).append(
                {"segment": k, "message": segs[-1][2][1].get("result"),
                 "stopped_at": datetime.now().strftime("%H:%M:%S"),
                 "resume_after": wake.strftime("%H:%M %Z")})
            print(f"[t{turn}] 사용 한도로 중단 - {wake:%H:%M} 이후 같은 회차를 이어서 진행",
                  flush=True)
            time.sleep(max(0.0, (wake - datetime.now(wake.tzinfo)).total_seconds()))
            secret, exp = wait_for_token(a.min_token_min, turn)
            message = CONTINUE_MSG
        stream_p, err_p = segs[0][0], segs[-1][1]
        init = segs[0][2][0]
        result = segs[-1][2][1]
        agents, spawns, bash = merge_segments([s[2] for s in segs])
        if init and turn == 1:
            (OUT / f"{stem}_init.json").write_text(json.dumps(init, indent=2, ensure_ascii=False))
        reply = (result or {}).get("result") or ""
        (OUT / f"{variant}_reply.md").write_text(reply)
        rec.update(
            agent_exit=rc, agent_wall_s=round(wall, 1),
            duration_ms=(result or {}).get("duration_ms"),
            duration_api_ms=(result or {}).get("duration_api_ms"),
            api_turns=(result or {}).get("num_turns"),
            is_error=(result or {}).get("is_error"),
            cost_usd_reported=(result or {}).get("total_cost_usd"),
            usage=(result or {}).get("usage"), model_usage=(result or {}).get("modelUsage"),
            permission_denials=(result or {}).get("permission_denials"),
            segments=len(segs), agents=agents, spawned=spawns, reply_chars=len(reply),
            bash_calls=len(bash),
            network_like_cmds=[c[:200] for _, c in bash if NET_RE.search(c)])

        # keep what the agent made and how it got there, outside its sandbox
        files_dir = OUT / f"{stem}_files" / variant
        rec["files_copied"] = (
            copy_tree(run_dir / "home" / "workspace", files_dir / "workspace")
            + copy_tree(run_dir / "tmp", files_dir / "tmp"))
        tdir = OUT / f"{stem}_transcripts"
        copy_tree(run_dir / "home" / ".claude", tdir, skip=("statsig", "cache", "shell-snapshots"))
        rec["redacted_files"] = sum(redact_tree(x, secret) for x in
                                    [q for s_ in segs for q in s_[:2]]
                                    + [OUT / f"{variant}_reply.md", files_dir, tdir])
        seen = {s["agent"] for s in log["subagents"]}
        for s in transcript_agents(tdir / "projects"):
            if s["agent"] not in seen:
                s["first_turn"] = turn
                log["subagents"].append(s)
            else:
                for old in log["subagents"]:
                    if old["agent"] == s["agent"]:
                        old.update({k: s[k] for k in ("tools", "models", "last")})
        tool_line = {k if k == "main" else spawns.get(k, {}).get("subagent_type", k[:12]):
                     sum(v["tools"].values()) for k, v in agents.items()}
        u = (result or {}).get("usage") or {}
        print(f"[t{turn}] 에이전트 종료 rc={rc} {wall:.0f}s api_turns={rec['api_turns']} "
              f"tools={tool_line} spawned={len(spawns)} out_tok={u.get('output_tokens')} "
              f"cost=${rec['cost_usd_reported']}", flush=True)

        if result is None or result.get("is_error"):
            # an error text is not a submission: stop instead of grading it
            rec["verdict"] = "agent_error"
            rec["agent_error"] = (result or {}).get("result") or err_p.read_text()[-400:]
            log["turns"].append(rec)
            log.update(status=f"agent_error rc={rc}", finished=True)
            save()
            print(f"[t{turn}] 에이전트 오류로 중단: {str(rec['agent_error'])[:160]}", flush=True)
            return 4

        sub, fresh, cands = find_submission(run_dir, t0)
        if sub is not None:
            src, channel = sub.read_text(), "file"
            sub_rel = str(sub.relative_to(run_dir))
        else:
            src, channel, sub_rel = extract(reply), "text", None
        spliced = splice(src)
        text = spliced if spliced is not None else (src if src.endswith("\n") else src + "\n")
        path = HERE / f"{variant}_carter_run.py"
        path.write_text(text)
        rec["submission"] = {"channel": channel, "agent_path": sub_rel, "fresh_this_turn": fresh,
                             "candidates": cands, "spliced": spliced is not None,
                             "file": path.name, "sha16": sha16(text)}

        ok, detail = static_check(path, "a")
        rec["static_ok"] = ok
        if not ok:
            rec.update(verdict="static_dq", detail=detail)
            print(f"[t{turn}] 정적 실격: {detail}", flush=True)
            log["next_prompt"] = DQ_FEEDBACK.format(detail=detail)
            rec["feedback_kind"] = "static_dq"
            log["turns"].append(rec)
            save()
            continue

        if rec["submission"]["sha16"] == prev_sha and prev_result is not None:
            # same bytes as the last graded file; Isaac is deterministic here
            result_d = prev_result
            rec["identical_resubmission"] = True
            print(f"[t{turn}] 직전 제출과 동일 - 이전 채점 재사용", flush=True)
        else:
            result_d = grade_serial(variant)
        m = result_d.get("metrics", {})
        if result_d.get("_runtime_error"):
            rec["runtime_error"] = result_d["_runtime_error"]
        rec.update(verdict=result_d["verdict"], final_dist_m=m.get("final_dist_m"),
                   final_yaw_err_rad=m.get("final_yaw_err_rad"),
                   collisions=m.get("collision_count"), partners=m.get("collision_partners"),
                   first_collision_t=m.get("first_collision_t"),
                   time_to_goal_s=m.get("time_to_goal_s"), sim_wall_s=result_d.get("_wall_s"))
        print(f"[t{turn}] {result_d['verdict'].upper()} dist={m.get('final_dist_m')} "
              f"col={m.get('collision_count')} {m.get('collision_partners')} "
              f"ttg={m.get('time_to_goal_s')}", flush=True)
        prev_sha, prev_result = rec["submission"]["sha16"], result_d
        log["_last_result"] = result_d

        if result_d["verdict"] == "pass":
            rec["feedback_kind"] = None
            log["turns"].append(rec)
            log.update(passed_at_turn=turn, status="pass", finished=True)
            save()
            print(f"[done] {stem}: {turn}회차에 통과 -> {log_path}", flush=True)
            return 0
        if result_d.get("_runtime_error"):
            log["next_prompt"] = RUNTIME_FEEDBACK.format(err=result_d["_runtime_error"])
            rec["feedback_kind"] = "runtime_error"
        else:
            log["next_prompt"] = FEEDBACK.format(result=feedback_json(result_d, "a"))
            rec["feedback_kind"] = "result"
        log["turns"].append(rec)
        save()

    log.update(passed_at_turn=None, status=f"fail_{a.max_turns}", finished=True)
    save()
    print(f"[done] {stem}: {a.max_turns}회차까지 실패 -> {log_path}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
