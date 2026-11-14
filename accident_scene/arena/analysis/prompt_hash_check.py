"""Recompute the first message and tool hashes of every A-2 family prompt with the current
agent_loop.py and compare them with what the real runs logged (results/turns/<dir>/*_turns.json).
A later prompt (A-2-3, 실험 17; A-4T, 실험 18; B-2 revision, 실험 21) must leave every earlier one byte-identical.
No Isaac, no LLM.

    python3 analysis/prompt_hash_check.py
"""
import hashlib
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop as A  # noqa: E402

CASES = [  # name, flags (A21, A22, A23[, A4T]), prompt, run dir of the logged runs
    ("A-3", (False, False, False), "PROMPT_A3.md", "prompt_a3"),
    ("A-4", (False, False, False), "PROMPT_A4.md", "prompt_a4"),
    ("A-4T", (False, False, False, True), "PROMPT_A4T.md", "prompt_a4t"),
    ("B-1", (False, False, False, False, "b1"), "PROMPT_B1.md", "prompt_b1"),
    # 실험 20 ran B-2 as PROMPT_B2.md (front 2D lidar code only); that file is kept byte-identical as
    # PROMPT_B2_front2d.md and PROMPT_B2.md now holds the all-sensor B-2 (실험 21, results/turns/prompt_b2_all)
    ("B-2f", (False, False, False, False, "b2"), "PROMPT_B2_front2d.md", "prompt_b2", "PROMPT_B2.md"),
    ("B-2", (False, False, False, False, "b2"), "PROMPT_B2.md", "prompt_b2_all"),
    # B-2-1 (실험 22): the B-2 sensor base code moved from the prompt into the base the model sees
    ("B-2-1", (False, False, False, False, "b2"), "PROMPT_B2_1.md", "prompt_b2_1", None, "base_carter_run_ros_withenv.py"),
    # B-3 (실험 23): B-2-1 + the prompt says the final grading scene has objects the test drives lack
    ("B-3", (False, False, False, False, "b3"), "PROMPT_B3.md", "prompt_b3", None, "base_carter_run_ros_withenv.py"),
    # B-4 (실험 24): B-3 without the simulator-only env.raycast_scan (prompt text gone, env=None in the base)
    ("B-4", (False, False, False, False, "b4"), "PROMPT_B4.md", "prompt_b4", None, "base_carter_run_ros_noenv.py"),
    ("A-2", (False, False, False), "PROMPT_A2.md", "prompt_a2"),
    ("A-2-1", (True, False, False), "PROMPT_A2_1.md", "prompt_a2_1"),
    ("A-2-2", (True, True, False), "PROMPT_A2_2.md", "prompt_a2_2"),
    ("A-2-3", (True, True, True), "PROMPT_A2_3.md", "prompt_a2_3"),
]


A2_BASE = A.BASE


def hashes(flags, prompt, base_file=None):
    A.BASE = (ARENA / base_file).read_text() if base_file else A2_BASE
    A.A21, A.A22, A.A23 = flags[:3]
    A.A4T = len(flags) > 3 and flags[3]
    A.B_MODE = flags[4] if len(flags) > 4 else None
    A.A4T_SEED = (A.load_a4t_seed("npu_gpt_oss_120b_pA4_a0_carter_run.py", "results/npu_gpt_oss_120b_pA4_a0_result.json",
                                  "results/npu_gpt_oss_120b_pA4_a0_trajectory.csv") if A.A4T else None)
    text = A.model_input(prompt)
    tools = A.tools() if (A.A21 or A.B_MODE) else A.TOOL
    return {"input_sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "input_chars": len(text),
            "tool_sha16": hashlib.sha256(json.dumps(tools, ensure_ascii=False).encode()).hexdigest()[:16],
            "tools": [t["function"]["name"] for t in (tools if isinstance(tools, list) else [tools])]}


bad = 0
out = {}
for name, flags, prompt, d, *extra in CASES:
    logged_as = [extra[0]] if extra and extra[0] else []
    now = hashes(flags, prompt, extra[1] if len(extra) > 1 else None)
    logged = []
    for f in sorted((ARENA / "results" / "turns" / d).glob("*_turns.json")) if d else []:
        t = json.loads(f.read_text())
        if t.get("prompt_file") == (logged_as[0] if logged_as else prompt):
            logged.append({k: t.get(k) for k in ("input_sha16", "input_chars", "tool_sha16")} | {"file": f.name})
    same = all(l["input_sha16"] == now["input_sha16"] and l["tool_sha16"] == now["tool_sha16"]
               and l["input_chars"] == now["input_chars"] for l in logged)
    bad += not same
    out[name] = {"now": now, "logged": logged, "same_as_logged": same}
    print(f"{name:6} input {now['input_sha16']} ({now['input_chars']}자) tools {now['tool_sha16']} "
          f"{now['tools']}  logged runs {len(logged)}: {'SAME' if same else 'DIFFERENT'}")
(ARENA / "results" / "analysis" / "prompt_hash_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
sys.exit(1 if bad else 0)
