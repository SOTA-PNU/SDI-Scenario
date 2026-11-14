"""Experiment 10 (prompt A-1): one row per turn - what the controller did, what Isaac said.

Reads results/turns/prompt_a1/<tag>_a<k>_turns.json and each turn's submitted file
and reports, per turn:
  outcome    verdict, final distance, collisions and partners, time to goal
             (from the turn log, which copied them from the Isaac result JSON)
  waypoints  indicators that the EDIT REGION follows a sequence of intermediate
             targets: a waypoint/route/path name, numeric (x, y) pairs, a list of
             pairs, an index that advances. The REPORT.md definition (a sequence of
             at least two intermediate targets followed in order) is applied by
             reading the code; these columns say where to look.
  scan       every env.raycast_scan(...) call, parsed; a call 'departs' when any
             argument differs from the docstring defaults (n_beams=61,
             fov_deg=180.0, z=0.35, max_range=6.0) or is not a literal
  logs       every LLM and Isaac file the run promised to keep is on disk, and
             each kept Kit log names this turn's submission (a Kit log from a
             neighbouring serial run would name another file)
Definitions are the ones fixed in REPORT.md (실험 10) before the run.

Run:  python3 analysis/prompt_a1_summary.py [--turns-dir results/turns/prompt_a1]
      (writes results/analysis/prompt_a1_summary.json)
"""
import argparse
import ast
import glob
import io
import json
import os
import re
import tokenize
from collections import Counter
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
os.chdir(ARENA)
DEFAULTS = {"n_beams": 61, "fov_deg": 180.0, "z": 0.35, "max_range": 6.0}
POSITIONAL = ["n_beams", "fov_deg", "z", "max_range"]


def code_only(src):
    """Source with comments dropped and long strings blanked (docstrings say nothing)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING:
                out.append(tok.string if len(tok.string) < 12 else '""')
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return " ".join(out)


def region(src):
    m = re.search(r"#\s*\[EDIT REGION\][^\n]*\n(.*?)\n#\s*=+\s*\[END EDIT REGION\]", src, re.S)
    return m.group(1) if m else None


def scan_calls(reg):
    """Every raycast_scan(...) call in the region, arguments resolved against the defaults."""
    calls = []
    try:
        tree = ast.parse(reg)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "raycast_scan"):
            continue
        args = {}
        for name, val in zip(POSITIONAL, node.args):
            args[name] = val
        for kw in node.keywords:
            args[kw.arg or "**"] = kw.value
        resolved, departs = {}, False
        for name, val in args.items():
            if isinstance(val, ast.Constant) and isinstance(val.value, (int, float)):
                resolved[name] = val.value
                if name in DEFAULTS and float(val.value) != float(DEFAULTS[name]):
                    departs = True
            else:
                resolved[name] = "expr:" + ast.unparse(val)[:60]
                departs = True
        calls.append({"line": node.lineno, "args": resolved, "departs": departs})
    return calls


def waypoint_indicators(reg):
    code = code_only(reg)
    pairs = re.findall(r"[\(\[]\s*-?\s*\d+(?:\.\d+)?\s*,\s*-?\s*\d+(?:\.\d+)?\s*[\)\]]", code)
    names = sorted(set(re.findall(r"\b\w*(?:waypoint|WAYPOINT|Waypoint|route|ROUTE|_wp|wp_|WP_|_WP|"
                                  r"\bwps\b|\bWPS\b|\bwp\b|\bWP\b|path|PATH|Path)\w*\b", code)))
    names = [n for n in names if n not in ("Path", "pathlib")]
    pair_list = bool(re.search(r"\[\s*[\(\[]\s*-?\s*\d+(?:\.\d+)?\s*,\s*-?\s*\d+(?:\.\d+)?\s*[\)\]]\s*,"
                               r"\s*[\(\[]\s*-?\s*\d+", code))
    advance = bool(re.search(r"(\w*(?:idx|index|i_wp|wp_i|cur|current|target_i|seg)\w*\s*\+=\s*1)"
                             r"|\.pop\s*\(\s*0\s*\)|\bnext\s*\(", code, re.I))
    return {"wp_names": names[:8], "numeric_xy_pairs": len(pairs), "pair_list": pair_list,
            "index_advances": advance,
            "uses_box_table": bool(re.search(r"""MISSION\s*\[\s*["']boxes["']\s*\]|MISSION\s*\.\s*get\s*\(\s*["']boxes""", code)),
            "uses_GOAL": bool(re.search(r"\bGOAL\b", code)),
            "uses_START": bool(re.search(r"\bSTART\b", code))}


def kit_logs_ok(variant, d):
    """Kept Kit logs for this turn, and whether each one names this turn's submission."""
    out = []
    for k in sorted(d.glob(f"{variant}.kit_*.log")):
        txt = k.read_text(errors="replace")
        names_self = f"{variant}_carter_run.py" in txt
        others = sorted(set(re.findall(r"(npu_gpt_oss_120b_pA1_a\d(?:_t\d+)?)_carter_run\.py", txt)) - {variant})
        out.append({"file": k.name, "bytes": k.stat().st_size, "names_self": names_self, "names_other": others})
    return out


def logs_present(variant, turn_rec, tdir):
    iso = tdir / "isaac_logs"
    llm = {s: (tdir / f"{variant}_{s}").exists() for s in ("request.json", "raw.json", "reply.md")}
    graded = turn_rec.get("verdict") not in ("static_dq", "empty_response", None)
    isaac = None
    if graded:
        isaac = {
            "stdout": (iso / f"{variant}.carter-run.stdout.log").exists(),
            "stderr": (iso / f"{variant}.carter-run.stderr.log").exists(),
            "kept_json": (iso / f"{variant}.kept.json").exists(),
            "submission": (iso / f"{variant}_carter_run.py").exists(),
            "result_json": (iso / f"{variant}_result.json").exists() or bool(turn_rec.get("runtime_error")),
            "trajectory": (iso / f"{variant}_trajectory.csv").exists() or bool(turn_rec.get("runtime_error")),
            "kit_logs": kit_logs_ok(variant, iso),
        }
    return {"llm": llm, "graded": graded, "isaac": isaac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-dir", default="results/turns/prompt_a1")
    ap.add_argument("--tag", default="npu_gpt_oss_120b_pA1")
    ap.add_argument("--out", default="results/analysis/prompt_a1_summary.json")
    ap.add_argument("--max-turn", type=int,
                    help="ignore turns after this one (the A-1 runs were stopped after turn 2)")
    a = ap.parse_args()
    tdir = Path(a.turns_dir)
    runs = []
    for f in sorted(glob.glob(str(tdir / f"{a.tag}_a[0-9]_turns.json"))):
        d = json.load(open(f))
        stem = Path(f).name[:-len("_turns.json")]
        rows = []
        for t in d.get("turns", []):
            n = t.get("turn")
            if a.max_turn and n > a.max_turn:
                continue
            variant = stem if n == 1 else f"{stem}_t{n}"
            sub = Path(f"{variant}_carter_run.py")
            kept = tdir / "isaac_logs" / f"{variant}_carter_run.py"
            src = sub.read_text() if sub.exists() else (kept.read_text() if kept.exists() else None)
            reg = region(src) if src else None
            row = {"turn": n, "variant": variant, "verdict": t.get("verdict"),
                   "final_dist_m": t.get("final_dist_m"), "collisions": t.get("collisions"),
                   "partners": t.get("partners"), "time_to_goal_s": t.get("time_to_goal_s"),
                   "runtime_error": t.get("runtime_error"), "detail": t.get("detail"),
                   "gen_wall_s": t.get("gen_wall_s"), "completion_tokens": t.get("completion_tokens"),
                   "reasoning_tokens": t.get("reasoning_tokens"), "finish": t.get("finish"),
                   "spliced": t.get("spliced"), "sim_wall_s": t.get("sim_wall_s"),
                   "request_retries": t.get("request_retries")}
            if reg is not None:
                calls = scan_calls(reg)
                row["scan_calls"] = calls
                row["scan_departs"] = None if calls is None else any(c["departs"] for c in calls)
                row["scan_alias"] = bool(re.search(r"raycast_scan\b(?!\s*\()", code_only(reg)))
                row["env_other"] = sorted(set(re.findall(r"\benv\s*\.\s*(\w+)", code_only(reg))) - {"raycast_scan"})
                row["prints"] = bool(re.search(r"\bprint\s*\(", code_only(reg)))
                row.update(waypoint_indicators(reg))
                row["region_lines"] = reg.count("\n") + 1
            else:
                row["code"] = "no submission file" if src is None else "no edit region"
            row["logs"] = logs_present(variant, t, tdir)
            rows.append(row)
        runs.append({"run": stem, "sample": d.get("sample"), "passed_at_turn": d.get("passed_at_turn"),
                     "turns_used": d.get("turns_used", len(rows)), "finished": "turns_used" in d,
                     "prompt_file": d.get("prompt_file"), "input_sha16": d.get("input_sha16"),
                     "turns": rows})

    summary = {"runs": len(runs),
               "finished": sum(r["finished"] for r in runs),
               "passed_within_cap": [(r["run"], r["passed_at_turn"]) for r in runs if r["passed_at_turn"]],
               "scan_departs_any_turn": [r["run"] for r in runs
                                         if any(t.get("scan_departs") for t in r["turns"])],
               "scan_departs_turn1": [r["run"] for r in runs
                                      if r["turns"] and r["turns"][0].get("scan_departs")],
               "partners": Counter(p.replace("/World/", "") for r in runs for t in r["turns"]
                                   for p in (t.get("partners") or [])),
               "verdicts": Counter(t.get("verdict") for r in runs for t in r["turns"]),
               "missing_logs": [(t["variant"], k) for r in runs for t in r["turns"]
                                for k, v in list(t["logs"]["llm"].items()) +
                                list(((t["logs"]["isaac"] or {}).items()))
                                if v is False or (k == "kit_logs" and v == [])],
               "kit_logs_naming_other_runs": [(t["variant"], k["file"], k["names_other"])
                                              for r in runs for t in r["turns"]
                                              for k in ((t["logs"]["isaac"] or {}).get("kit_logs") or [])
                                              if k["names_other"] or not k["names_self"]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1))

    for r in runs:
        print(f"== {r['run']}  passed_at={r['passed_at_turn']}  turns={r['turns_used']}"
              f"  finished={r['finished']}")
        for t in r["turns"]:
            sc = t.get("scan_calls")
            scan = "-" if sc is None else f"{len(sc)}call" + ("/DEPART" if t.get("scan_departs") else "")
            parts = ",".join(p.replace("/World/", "") for p in (t.get("partners") or []))
            print(f"  t{t['turn']}: {t['verdict']:<14} dist={t.get('final_dist_m')} col={t.get('collisions')}"
                  f" ttg={t.get('time_to_goal_s')} [{parts}] scan={scan}"
                  f" wp_names={t.get('wp_names')} xy={t.get('numeric_xy_pairs')}"
                  f" list={t.get('pair_list')} adv={t.get('index_advances')}"
                  f" boxes={t.get('uses_box_table')} env+={t.get('env_other')}"
                  + (f" ERR={t['runtime_error'][:80]}" if t.get("runtime_error") else "")
                  + (f" DQ={str(t.get('detail'))[:80]}" if t.get("verdict") == "static_dq" else ""))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
