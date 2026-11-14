"""Summary of prompt A-4T (REPORT.md 실험 18): per turn the graded verdict, completion time,
time_to_goal, tests and their outcomes, the waypoints and speed constants of the submitted code,
and the speed the robot actually reached (from the graded trajectory). No Isaac, no LLM.

    python3 analysis/prompt_a4t_summary.py [--turns-dir results/turns/prompt_a4t] [--tag npu_gpt_oss_120b_pA4T]
"""
import argparse
import ast
import csv
import glob
import json
import math
import os
import sys

ARENA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ARENA)
import agent_loop as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--turns-dir", default="results/turns/prompt_a4t")
ap.add_argument("--tag", default="npu_gpt_oss_120b_pA4T")
ap.add_argument("--out", default="results/analysis/prompt_a4t_summary.json")
a = ap.parse_args()
os.chdir(ARENA)
ISO = os.path.join(a.turns_dir, "isaac_logs")


def waypoints(region):
    """The first list/tuple of >= 3 numeric (x, y) pairs assigned at top level or in a function."""
    try:
        tree = ast.parse(region)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 3:
            pts = []
            for e in node.elts:
                try:
                    v = ast.literal_eval(e)
                except Exception:
                    try:  # (GOAL[0], GOAL[1])
                        v = (A.BASE_GOAL[0], A.BASE_GOAL[1]) if "GOAL" in ast.unparse(e) else None
                    except Exception:
                        v = None
                if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
                    pts.append((float(v[0]), float(v[1])))
                else:
                    pts = None
                    break
            if pts:
                return pts
    return None


def path_len(pts):
    p = [A.BASE_START[:2]] + pts
    if math.dist(p[-1], A.BASE_GOAL[:2]) > 0.01:
        p.append(A.BASE_GOAL[:2])
    return round(sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1)), 1)


def numbers(region):
    """Numeric constants that look like speed limits: arguments of min()/clamp() and assignments."""
    try:
        tree = ast.parse(region)
    except SyntaxError:
        return None
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def traj_speed(path):
    if not os.path.exists(path):
        return None
    rows = [(float(r["t"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(open(path))]
    sp = [math.dist(rows[i][1:], rows[i + 6][1:]) / (rows[i + 6][0] - rows[i][0])
          for i in range(0, len(rows) - 6, 6) if rows[i + 6][0] > rows[i][0]]
    dist = sum(math.dist(rows[i][1:], rows[i + 1][1:]) for i in range(len(rows) - 1))
    return {"peak_mps": round(max(sp), 2) if sp else None, "driven_m": round(dist, 1),
            "mean_mps": round(dist / rows[-1][0], 2) if rows[-1][0] else None}


seed = A.load_a4t_seed("npu_gpt_oss_120b_pA4_a0_carter_run.py", "results/npu_gpt_oss_120b_pA4_a0_result.json",
                       "results/npu_gpt_oss_120b_pA4_a0_trajectory.csv")
out = {"seed": {"completion_s": seed["completion_s"], "time_to_goal_s": seed["time_to_goal_s"],
                "waypoints": waypoints(seed["region"]), "path_m": path_len(waypoints(seed["region"])),
                "constants": numbers(seed["region"]),
                "trajectory": traj_speed("results/npu_gpt_oss_120b_pA4_a0_trajectory.csv")}, "runs": {}}
print("seed", out["seed"]["completion_s"], out["seed"]["path_m"], out["seed"]["trajectory"])
for f in sorted(glob.glob(os.path.join(a.turns_dir, f"{a.tag}_a*_turns.json"))):
    log = json.load(open(f))
    run = os.path.basename(f)[: -len("_turns.json")]
    turns = []
    for t in log["turns"]:
        sub = t.get("submission")
        region = A.region_of(open(os.path.join(ISO, sub)).read()) if sub and os.path.exists(os.path.join(ISO, sub)) else \
            (A.region_of(open(sub).read()) if sub and os.path.exists(sub) else None)
        wp = waypoints(region) if region else None
        tj = traj_speed(os.path.join("results", f"{t['variant']}_trajectory.csv"))
        tests = [(x.get("test"), x.get("verdict") if x.get("static_ok", True) else "static_dq",
                  x.get("partners"), x.get("final_dist_m"), x.get("time_to_goal_s")) for x in t.get("tests", [])]
        row = {"turn": t["turn"], "verdict": t.get("verdict"), "completion_s": t.get("completion_s"),
               "time_to_goal_s": t.get("time_to_goal_s"), "partners": t.get("partners"),
               "final_dist_m": t.get("final_dist_m"), "tests": tests,
               "tests_passed": sum(1 for x in tests if x[1] == "pass"),
               "waypoints": wp, "path_m": path_len(wp) if wp else None,
               "same_waypoints_as_seed": wp == out["seed"]["waypoints"] if wp else None,
               "constants": numbers(region) if region else None, "trajectory": tj}
        turns.append(row)
        print(run, f"t{t['turn']}", row["verdict"], "comp", row["completion_s"], "ttg", row["time_to_goal_s"],
              "tests", len(tests), "pass", row["tests_passed"], "path", row["path_m"],
              "same_wp", row["same_waypoints_as_seed"], "traj", tj)
    out["runs"][run] = {"best_completion_s": log.get("best_completion_s"), "completions": log.get("completions"),
                        "passed_at_turn": log.get("passed_at_turn"), "model_calls": log.get("model_calls"),
                        "turns": turns}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print("->", a.out)
