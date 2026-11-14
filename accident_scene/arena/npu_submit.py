"""Send the fixed track-B benchmark to a model on the NPU router and save its file.

The model gets EXACTLY two things, concatenated, and nothing else:
    1. accident_scene/arena/PROMPT_B.md   (verbatim)
    2. accident_scene/arena/base_carter_run.py   (verbatim, inside a fence)
No extra instructions, no hints, no restatement - a transcript audit of the
Claude entries showed that orchestrator-added "pay attention to ..." sentences
are what made those entries non-comparable. Keep this wrapper empty.

    python3 accident_scene/arena/npu_submit.py MODEL_ID [--tag name] [--max-tokens N]
    python3 accident_scene/arena/npu_submit.py --list

Writes (under accident_scene/arena/):
    <tag>_b_carter_run.py                 the submission, ready to grade
    results/submissions/<tag>_b_raw.json  the full API response (audit trail)

The router is reached through the SSH tunnel on 127.0.0.1:8400 (override with
NPU_BASE). Sampling is pinned to temperature 0 so a rerun reproduces the entry.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = os.environ.get("NPU_BASE", "http://127.0.0.1:8400/v1")
PROMPT = HERE / "PROMPT_B.md"
BASE_FILE = HERE / "base_carter_run.py"


def api(path, payload=None, timeout=1800):
    url = f"{BASE_URL}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_message():
    """The whole model input: prompt file + base file. Nothing else."""
    return (PROMPT.read_text()
            + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n"
            + BASE_FILE.read_text()
            + "```\n")


def extract_python(text):
    """Pull the file out of the reply: fenced block if present, else raw."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="model id from --list")
    ap.add_argument("--tag", help="short name for the files (default: sanitised model id)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--list", action="store_true", help="list router models and exit")
    a = ap.parse_args()

    if a.list:
        for m in api("/models")["data"]:
            print(m["id"])
        return 0
    if not a.model:
        ap.error("model id required (see --list)")

    tag = a.tag or re.sub(r"[^a-z0-9]+", "_", a.model.lower()).strip("_")
    msg = build_message()
    print(f"[npu] model={a.model} tag={tag} prompt_chars={len(msg)}", flush=True)

    resp = api("/chat/completions", {
        "model": a.model,
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    })

    out_dir = HERE / "results" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_b_raw.json").write_text(json.dumps(resp, indent=2, ensure_ascii=False))

    choice = resp["choices"][0]["message"]
    content = choice.get("content") or ""
    reasoning = choice.get("reasoning") or ""
    usage = resp.get("usage", {})
    print(f"[npu] finish={resp['choices'][0].get('finish_reason')} "
          f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
          f"reasoning={usage.get('completion_tokens_details', {}).get('reasoning_tokens')}", flush=True)

    if not content.strip():
        print("[npu] EMPTY content - the model spent the budget on reasoning. "
              "Raise --max-tokens or pick a non-thinking variant.", file=sys.stderr)
        return 2

    code = extract_python(content)
    target = HERE / f"{tag}_b_carter_run.py"
    target.write_text(code if code.endswith("\n") else code + "\n")
    print(f"[npu] wrote {target} ({len(code)} chars)")
    print("[npu] next:")
    print(f"  python3 levels/common/check_edit_region.py accident_scene/arena/{target.name}")
    print(f"  python3 accident_scene/arena/check_b_track.py accident_scene/arena/{target.name}")
    print(f"  CARTER_REPO=<repo> $CARTER_WS/tools/bin/carter-run accident_scene/arena/{target.name}")
    return 0


sys.exit(main())
