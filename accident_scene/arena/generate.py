"""Collect k samples per model from an OpenAI-compatible endpoint.

The model input is exactly two files - the track's PROMPT*.md and
base_carter_run.py - concatenated, and nothing else. No system prompt, no
tool loop, no orchestrator instructions: a transcript audit of an earlier
round found that sentences added around the prompt ("pay attention to ...")
pointed at the scenario's trap and made those entries non-comparable.

Every response is saved raw, so the model id, token counts and finish reason
stay recoverable - the earlier NPU round lost all three.

    python3 generate.py MODEL --track b --k 5 [--tag name] [--base URL]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "generated"


def api(base, path, payload=None, timeout=2400, retries=2):
    """POST and parse, retrying a 400 on a fresh connection.

    furiosa-llm's gpt-oss path intermittently rejects a well-formed request
    with a pydantic error about message content, always on a connection that
    has already served a long generation, and never on the first request of a
    fresh connection. The cause is unknown; reconnecting recovers it. Connection:
    close forces a new socket per attempt so a poisoned one is never reused.
    """
    body = None if payload is None else json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base}{path}", data=body, headers={
            "Content-Type": "application/json", "Connection": "close"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 400 or attempt == retries:
                raise
            print(f"       HTTP 400 - 새 연결로 재시도 {attempt + 1}/{retries}", flush=True)
            time.sleep(3)


def model_input(track, prompt_file=None):
    """The model's whole input: one prompt file plus base, nothing else.

    prompt_file overrides the track default. It exists for the prescription
    arm, which needs PROMPT.md plus one sentence; the fixed prompts themselves
    stay byte-identical so every earlier entry remains comparable.
    """
    prompt = (HERE / prompt_file) if prompt_file else (
        HERE / ("PROMPT.md" if track == "a" else "PROMPT_B.md"))
    return (prompt.read_text()
            + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n"
            + (HERE / "base_carter_run.py").read_text() + "```\n")


def extract(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--track", choices=["a", "b"], default="b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag")
    ap.add_argument("--base", default=os.environ.get("LLM_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--warmup", action="store_true",
                    help="send one discarded request first (MoE backends ramp slowly)")
    ap.add_argument("--start", type=int, default=0, help="first sample index")
    ap.add_argument("--prompt-file", help="use this prompt instead of the track default")
    ap.add_argument("--out-dir", help="write submissions and raw responses here")
    a = ap.parse_args()

    tag = a.tag or re.sub(r"[^a-z0-9]+", "_", a.model.split("/")[-1].lower()).strip("_")
    msg = model_input(a.track, a.prompt_file)
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] {a.model} track={a.track} k={a.k} prompt_chars={len(msg)}", flush=True)

    fails = 0
    if a.warmup:
        # gpt-oss ramps 4-6 -> 31 tok/s over ~130 s AFTER the backend reports
        # ready; measuring without this makes the first sample 6x slow and
        # penalises the model unfairly. The response is discarded.
        w0 = time.time()
        try:
            api(a.base, "/chat/completions", {
                "model": a.model, "messages": [{"role": "user", "content": msg}],
                "max_tokens": 512, "temperature": a.temperature, "top_p": a.top_p})
            print(f"[gen] warmup discarded ({time.time() - w0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[gen] warmup failed: {str(e)[:120]}", flush=True)

    for i in range(a.start, a.start + a.k):
        name = f"{tag}_{a.track}{i}"
        t0 = time.time()
        try:
            r = api(a.base, "/chat/completions", {
                "model": a.model, "messages": [{"role": "user", "content": msg}],
                "max_tokens": a.max_tokens, "temperature": a.temperature,
                "top_p": a.top_p, "seed": 1000 + i})
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            (out_dir / f"{name}_error.txt").write_text(f"HTTP {e.code}\n{body}")
            print(f"  [{i}] HTTP {e.code}: {body[:200]}", flush=True)
            fails += 1
            if fails >= 2:
                print("  [!] 연속 2회 실패 - 중단합니다. 서버 상태를 확인하세요.", flush=True)
                return 2
            continue
        except Exception as e:
            print(f"  [{i}] REQUEST FAILED: {str(e)[:160]}", flush=True)
            fails += 1
            if fails >= 2:
                print("  [!] 연속 2회 실패 - 중단합니다.", flush=True)
                return 2
            continue
        fails = 0
        dt = time.time() - t0
        # stamp the request window so the run can be aligned against the
        # serving box's own power / throughput logs after the fact
        r["_bench"] = {"t_start_epoch": t0, "t_end_epoch": t0 + dt,
                       "wall_s": round(dt, 3), "endpoint": a.base,
                       "requested_model": a.model, "track": a.track,
                       "prompt_file": a.prompt_file or "",
                       "sample": i, "seed": 1000 + i,
                       "temperature": a.temperature, "top_p": a.top_p,
                       "max_tokens": a.max_tokens}
        (out_dir / f"{name}_raw.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))

        ch = r["choices"][0]
        reply = ch["message"]          # NOT `msg` - that holds the prompt
        content = reply.get("content") or ""
        # Servers disagree about where reasoning goes: vLLM uses
        # reasoning_content, furiosa-llm uses reasoning, and gpt-oss handles
        # harmony inside the server. If content is empty but some other field
        # holds text, say which - silently writing an empty file would look
        # like a model failure when it is a field-name mismatch.
        if not content.strip():
            alt = {k: len(v) for k, v in reply.items()
                   if isinstance(v, str) and v.strip() and k != "content"}
            if alt:
                print(f"       content empty; other text fields: {alt}", flush=True)
        u = r.get("usage", {})
        fin = ch.get("finish_reason")
        print(f"  [{i}] {dt:6.1f}s finish={fin} out={u.get('completion_tokens')} "
              f"chars={len(content)}", flush=True)
        if not content.strip():
            print(f"       EMPTY (reasoning-only) -> no submission file", flush=True)
            continue
        code = extract(content)
        p = out_dir / f"{name}_carter_run.py"
        p.write_text(code if code.endswith("\n") else code + "\n")

    print(f"[gen] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
