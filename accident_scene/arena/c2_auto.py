"""C-2 autopilot (2026-11-15, user): keep running turns without asking until one turn runs all 10 tests inside the
600 s simulation budget and every one of them passes.

Each turn is c1_cycle.py --series c2 --turn N, fed by the last good turn (its reviewed code, its analyst diagnosis and
its history). A turn that ends without reviewed code or without a readable diagnosis is skipped as a source: the next
turn reuses the last good turn's inputs. Stops at the goal or after --max-turn.

    python3 c2_auto.py --start 4 --wait-pid <pid of the running turn 3>
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path.home() / "logs"


def say(s):
    print(f"{time.strftime('%F %T')} [auto] {s}", flush=True)


SERIES = "c2"                 # --series c3: the all-Claude series (2026-11-16); c4: gpt-oss coder, Claude others
MODEL_TAGS = {"c3": "claude_opus_5_5", "c4": "gptoss_coder_claude", "c5": "claude_gptoss_analyst", "c6": "claude_qwen36_analyst", "c7": "claude_gemma4_analyst", "c71": "claude_gemma4_analyst",
              "c8": "claude_exaone45_analyst", "c81": "claude_exaone45_analyst", "c72": "claude_gemma4_analyst", "c51": "claude_gptoss_analyst"}
REVERT_SERIES = ("c81", "c72", "c51")      # the analyst's diagnosis names the turn whose code the next coder starts from (base_turn)


def turn_files(n):
    d = HERE / f"results/turns/prompt_{SERIES}_turn{n}"
    tag = f"{MODEL_TAGS.get(SERIES, 'gpu_gpt_oss_120b')}_p{SERIES.upper()}t{n}"
    diag = sorted(glob.glob(str(d / f"{tag}_multi*_diagnosis.json")))
    diag = [p for p in diag if "rerun" not in p and "BAD" not in p and "ARCH" not in p]   # archived, never an input
    return {"dir": d, "block": d / f"{tag}_block.py", "history": d / f"{tag}_history.txt",
            "turn": d / f"{tag}_turn.json", "diag": Path(diag[0]) if diag else None}


def outcome(n):
    """(usable as a source?, runs, passes)"""
    f = turn_files(n)
    if not f["turn"].exists():
        return False, 0, 0
    t = json.loads(f["turn"].read_text())
    runs, passes = len(t.get("seeds") or []), t.get("passes") or 0
    diag_ok = f["diag"] is not None and json.loads(f["diag"].read_text()).get("diagnosis")
    usable = f["block"].exists() and f["history"].exists() and bool(diag_ok)
    return usable, runs, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="first turn this autopilot runs")
    ap.add_argument("--wait-pid", type=int, default=None, help="a turn still running: wait for it first")
    ap.add_argument("--max-turn", type=int, default=30)
    ap.add_argument("--goal-runs", type=int, default=10)
    ap.add_argument("--series", default="c2", choices=["c2", "c3", "c4", "c5", "c6", "c7", "c71", "c8", "c81", "c72", "c51"])
    a = ap.parse_args()
    global SERIES
    SERIES = a.series
    if a.wait_pid:
        say(f"turn {a.start - 1} 진행 중(pid {a.wait_pid}) - 끝나기를 기다림")
        while os.path.exists(f"/proc/{a.wait_pid}"):
            time.sleep(20)
    last_good = None
    for n in range(a.start - 1, -1, -1):
        if outcome(n)[0]:
            last_good = n
            break
    if last_good is None:
        say("시작할 수 있는 이전 턴이 없음")
        return 1
    usable, runs, passes = outcome(a.start - 1)
    say(f"turn {a.start - 1}: {passes}/{runs} pass, 다음 턴 입력은 turn {last_good}")
    if runs >= a.goal_runs and passes >= a.goal_runs:
        say(f"목표 달성(turn {a.start - 1}: {passes}/{runs}) - 종료")
        return 0
    for n in range(a.start, a.max_turn + 1):
        src = turn_files(last_good)
        code_turn = last_good
        if SERIES in REVERT_SERIES:
            bt = (json.loads(src["diag"].read_text()).get("diagnosis") or {}).get("base_turn")
            if isinstance(bt, int) and 0 <= bt <= last_good and turn_files(bt)["block"].exists():
                code_turn = bt
            elif bt is not None:
                say(f"base_turn {bt!r} 의 코드가 없음 - turn {last_good} 코드로 진행")
        say(f"turn {n} 시작 (입력: turn {code_turn} 코드, turn {last_good} 진단·기록)" if code_turn != last_good
            else f"turn {n} 시작 (입력: turn {last_good} 코드·진단·기록)")
        cmd = ["python3", "c1_cycle.py", "--series", SERIES, "--turn", str(n), "--code", str(turn_files(code_turn)["block"]),
               "--diagnosis", str(src["diag"]), "--history-file", str(src["history"])]
        if SERIES in REVERT_SERIES:
            cmd += ["--base-turn", str(code_turn), "--diag-turn", str(last_good)]
        with open(LOG / f"{SERIES}_turn{n}.log", "w") as fh:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "37")}).returncode
        usable, runs, passes = outcome(n)
        say(f"turn {n} 끝 (rc={rc}): {passes}/{runs} pass, 다음 입력으로 {'사용' if usable else '사용 안 함'}")
        if usable:
            last_good = n
        if runs >= a.goal_runs and passes >= a.goal_runs:
            say(f"목표 달성(turn {n}: {passes}/{runs}, 600 s 예산 안) - 종료")
            return 0
    say(f"turn {a.max_turn} 까지 목표 미달성 - 종료")
    return 2


if __name__ == "__main__":
    sys.exit(main())
