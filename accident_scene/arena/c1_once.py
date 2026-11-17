"""C-1 one shot (2026-11-15): one Isaac run of the current code on a random mover layout, then one analyst diagnosis.

The current code is the EDIT block of a4_speedx5_carter_run.py (A-4 a0 waypoints, B-5-29 speeds) on the C-1 base
base_carter_run_ros_noenv.py (sensor helpers, env=None). The run uses ARENA_MOVERS=v1r with a fresh seed (or --seed).
The analyst gets PROMPT_C1_ANALYST.md, the coder's task prompt (PROMPT_C1_DRAFT.md up to its setup memo), the base,
the code, run_digest's summary plus mover facts (closest approach per mover, the collision moment), the drawn mover
values, and the code's own DBG lines.

    python3 c1_once.py [--seed N] [--out-dir results/turns/prompt_c1_once]
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as AL            # noqa: E402
import arena_movers as AM          # noqa: E402
import run_digest as RD            # noqa: E402
import sensor_digest as SDG        # noqa: E402
import turn_loop as TL             # noqa: E402

BASE_FILE = "base_carter_run_ros_noenv.py"
CODE_FILE = "a4_speedx5_carter_run.py"
TASK_PROMPT = "PROMPT_C1_DRAFT.md"
ANALYST_PROMPT = "PROMPT_C1_ANALYST.md"
KEYS = ("failure_mode", "evidence", "target", "direction", "confidence")


def task_prompt():
    t = (HERE / TASK_PROMPT).read_text()
    return t.split("\n---\n\n## 실험 설정 메모")[0].replace("> 초안(2026-11-15). 수정 에이전트가 받는 과제 프롬프트다. "
                                                          "하네스·분석 에이전트 쪽 설정은 맨 끝 `실험 설정 메모`에 적었다.\n", "")


def mover_facts(variant, log_dir, result):
    """Closest approach of every mover to the robot, and the collision moment, from the trajectory and the drawn movers."""
    d = Path(log_dir)
    movers = json.load(open(d / f"{variant}_movers.json"))["movers"]
    traj = [(float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"])) for r in csv.DictReader(open(d / f"{variant}_trajectory.csv"))]
    out = {"note": "물체 중심과 로봇 중심 사이 거리(m). 물체는 0.6 m 정사각, 로봇 몸체는 약 0.95 x 0.68 m.", "closest": []}
    for i, m in enumerate(movers):
        best = min(((math.hypot(x - AM.position(m, t)[0], y - AM.position(m, t)[1]), t, x, y) for t, x, y, _ in traj),
                   key=lambda r: r[0])
        mx, my = AM.position(m, best[1])
        out["closest"].append({"mover": f"arena_mover_{i:02d} ({m['name']})", "min_center_dist_m": round(best[0], 2),
                               "t": round(best[1], 2), "robot_xy": [round(best[2], 2), round(best[3], 2)],
                               "mover_xy": [round(mx, 2), round(my, 2)]})
    m = (result or {}).get("metrics", {})
    fc = m.get("first_collision_t")
    if fc is not None:
        k = min(range(len(traj)), key=lambda j: abs(traj[j][0] - fc))
        j = max(0, k - 6)
        dt = traj[k][0] - traj[j][0]
        sp = math.hypot(traj[k][1] - traj[j][1], traj[k][2] - traj[j][2]) / dt if dt > 0 else None
        out["collision"] = {"t": fc, "partners": m.get("collision_partners"),
                            "robot_pose": [round(v, 2) for v in traj[k][1:]], "robot_speed_m_s": round(sp, 2) if sp else sp,
                            "movers_at_t": [{"mover": f"arena_mover_{i:02d} ({mm['name']})",
                                             "xy": [round(c, 2) for c in AM.position(mm, fc)]} for i, mm in enumerate(movers)]}
    return out, movers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sim-s", type=float, default=200.0)
    ap.add_argument("--out-dir", default="results/turns/prompt_c1_once")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base", default="http://127.0.0.1:8420/v1")
    ap.add_argument("--tag", default="gpu_gpt_oss_120b_pC1once")
    a = ap.parse_args()
    AL.BASE_FILE = HERE / BASE_FILE
    AL.BASE = AL.BASE_FILE.read_text()
    AL.SP.BASE = AL.BASE
    out_dir = HERE / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(1, 2 ** 31)
    os.environ["ARENA_MOVERS"] = "v1r"
    os.environ["ARENA_MOVERS_SEED"] = str(seed)
    os.environ["CARTER_SENSOR_LOG"] = "1"        # both 2D lidars every 0.1 s, for the analyst (sensor_digest.py)
    src = (HERE / CODE_FILE).read_text()
    b, e = src.find(TL.BEGIN_MARK), src.find(TL.END_MARK)
    block = src[b: src.find("\n", e) + 1]
    run = AL.Run(Namespace(out_dir=str(out_dir), tag=a.tag, sample=0))
    print(f"[c1] seed {seed}, Isaac {a.sim_s:.0f} s ...", flush=True)
    out, rec = run.test(1, 1, {"code": block, "sim_seconds": a.sim_s})
    if not rec or not rec.get("static_ok"):
        print("[c1] static check failed:", out, flush=True)
        return 1
    variant = rec["variant"]
    print(f"[c1] {variant}: {rec['verdict']} dist={rec['final_dist_m']} col={rec['collisions']} {rec['partners']} "
          f"t_col={rec['first_collision_t']} wall={rec['wall_s']}s", flush=True)
    result = json.loads((TL.ISAAC_LOG_DIR / f"{variant}_result.json").read_text())
    dg = RD.digest(variant, TL.ISAAC_LOG_DIR)
    facts, movers = mover_facts(variant, TL.ISAAC_LOG_DIR, result)
    t_end = (result.get("metrics") or {}).get("first_collision_t")
    lidar = SDG.digest(variant, TL.ISAAC_LOG_DIR, t_end=t_end, window=5.0)
    stdout = (TL.ISAAC_LOG_DIR / f"{variant}.carter-run.stdout.log").read_text(errors="replace")
    dbg = RD.dbg_lines(stdout)[:200]
    body = ((HERE / ANALYST_PROMPT).read_text()
            + "\n\n---\n\n## 과제 프롬프트 (수정 에이전트가 받은 것)\n\n" + task_prompt()
            + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + AL.BASE + "```\n"
            + "\n---\n\n## 이번 코드\n\n```python\n" + block + "```\n"
            + f"\n## 실행 종류\n\n시험 주행 {a.sim_s:.0f}초, 움직이는 물체 무작위 배치\n"
            + "\n## 실행 요약\n\n```json\n" + json.dumps({**dg, "movers_vs_robot": facts}, ensure_ascii=False, indent=1) + "\n```\n"
            + "\n## 2D 라이다 기록 (하네스가 0.1 s 마다 저장, "
            + ("첫 충돌 전 5 s ~ 후 0.5 s" if t_end is not None else "마지막 5 s") + ")\n\n```json\n"
            + json.dumps(lidar, ensure_ascii=False) + "\n```\n"
            + "\n## 움직이는 물체 실제 값 (이번 실행만, 분석용)\n\n```json\n"
            + json.dumps([{k: m[k] for k in ("name", "p0", "p1", "speed", "phase")} for m in movers], ensure_ascii=False) + "\n```\n"
            + "\n## DBG 출력\n\n```\n" + ("\n".join(dbg) if dbg else "(코드가 출력한 DBG 없음)") + "\n```\n"
            + "\n## 시도 기록\n\n(첫 실행. 이번 코드는 수정 전 현재 코드)\n")
    payload = {"model": a.model, "messages": [{"role": "user", "content": body}], "max_tokens": 16384,
               "temperature": 0.7, "top_p": 0.95, "seed": 1000}
    (out_dir / f"{a.tag}_analyst_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    t0 = time.time()
    r = AL.post(a.base, payload, 7200)
    (out_dir / f"{a.tag}_analyst_raw.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    u = r.get("usage", {})
    print(f"[c1] analyst {time.time() - t0:.0f}s 입력={u.get('prompt_tokens')} 출력={u.get('completion_tokens')}", flush=True)
    s, e2 = content.find("{"), content.rfind("}")
    try:
        diag = json.loads(content[s:e2 + 1])
        missing = [k for k in KEYS if k not in diag]
    except (json.JSONDecodeError, ValueError):
        diag, missing = None, list(KEYS)
    (out_dir / f"{a.tag}_diagnosis.json").write_text(json.dumps({"seed": seed, "variant": variant, "run": rec,
                                                                 "movers_vs_robot": facts, "lidar_first_not_on_map": lidar.get("first_not_on_map"), "diagnosis": diag,
                                                                 "raw_content": content, "missing": missing},
                                                                ensure_ascii=False, indent=1))
    print(json.dumps(diag, ensure_ascii=False, indent=1) if diag else content, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
