"""`run_isaac_sim` for a furio (openclaude) agent: one Isaac test drive, same as agent_loop's tool.

The agent works in a clean workspace and may only run `./run_isaac_sim <file> [sim_seconds]`.
That script calls this file, which reads the agent's runner (whole file or EDIT REGION
block), applies the same static gate against base_carter_run_a2.py, runs it in Isaac Sim
on this server through the same path that grades (turn_loop.grade behind
carter_run_locked.sh), keeps every log under --out-dir/isaac_logs, and prints the same JSON
agent_loop.py returns to the model. A per-run counter enforces the test budget per turn.

    RUN_ISAAC_FILE=<file> [RUN_ISAAC_SIM_S=<s>] python3 furio_isaac_tool.py --out-dir <dir> \
        --tag npu_furio_pA4 --sample 0 --turn 1 --budget 10

The file name comes in through the environment, never argv: carter-run's serial guard
counts any python process whose command line names a "*carter_run.py" as a running
simulator, so `python3 ... base_carter_run.py` on argv made it refuse forever (exit 75).
"""
import os
import argparse
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as A   # noqa: E402  (sets the A-2 base and the lock wrapper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--turn", type=int, default=1)
    ap.add_argument("--budget", type=int, default=10)
    a = ap.parse_args()
    a.file = os.environ.get("RUN_ISAAC_FILE", "")
    a.sim_seconds = os.environ.get("RUN_ISAAC_SIM_S") or None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / f"{a.tag}_a{a.sample}_tool_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    key = f"t{a.turn}"
    used = state.get(key, 0)
    ev = out / f"{a.tag}_a{a.sample}_tool_calls.jsonl"
    rec_base = {"ts": round(time.time(), 3), "time": time.strftime("%F %T"), "turn": a.turn,
                "file": a.file, "sim_seconds_arg": a.sim_seconds}
    if used >= a.budget:
        res = {"error": f"이번 회차의 시험 주행 한도({a.budget}번)를 다 썼습니다. "
                        "더 실행하지 말고 최종 파일을 완성해 마치십시오."}
        print(json.dumps(res, ensure_ascii=False, indent=1))
        with open(ev, "a") as f:
            f.write(json.dumps({**rec_base, "result": res}, ensure_ascii=False) + "\n")
        return 0
    try:
        code = Path(a.file).read_text()
    except OSError as e:
        res = {"error": f"파일을 읽지 못했습니다: {e}"}
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    used += 1
    state[key] = used
    state_p.write_text(json.dumps(state))
    args = types.SimpleNamespace(model="furio", tag=a.tag, sample=a.sample, base="", out_dir=str(out),
                                 max_turns=2, test_budget=a.budget, max_tokens=0, temperature=0, top_p=0,
                                 seed_base=0, budget_chars=0, request_timeout=0, resume=False)
    run = A.Run(args)
    tool_args = {"code": code}
    if a.sim_seconds is not None:
        tool_args["sim_seconds"] = a.sim_seconds
    res, trec = run.test(a.turn, used, tool_args)
    res = {"test": used, "tests_left": a.budget - used, **res}
    if trec:
        trec["test"] = used
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(ev, "a") as f:
        f.write(json.dumps({**rec_base, "test": used, "record": trec, "result": res}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
