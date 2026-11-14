"""Experiment 11 (prompt A-2): one row per model call, test drive and graded submission.

Reads results/turns/prompt_a2/<tag>_a<k>_turns.json and _events.jsonl and reports,
per turn: the model calls (lost and recovered ones marked), every Isaac test drive
(verdict, partners, sim_seconds, DBG lines) with the code features of the tested
file (waypoints, scan arguments - same indicators as prompt_a1_summary.py), the
graded submission, and whether every file the run promised to keep is on disk:
  per call   <stem>_c<NN>_request.json, _raw.json (and _raw_text.txt once logprobs were on)
  per tool   <stem>_c<NN>_tool<k>.json
  per Isaac  isaac_logs/<variant>.carter-run.stdout/.stderr.log, .kept.json, the runner,
             result JSON, trajectory CSV (none after a runtime error), Kit log(s) naming
             that runner; a statically rejected test keeps the runner and the reason

Run:  python3 analysis/prompt_a2_summary.py   (writes results/analysis/prompt_a2_summary.json)
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompt_a1_summary as A1   # noqa: E402  (chdir to the arena, code feature helpers)


def features(src):
    reg = A1.region(src) if src else None
    if reg is None:
        return {"code": "no edit region" if src else "no file"}
    calls = A1.scan_calls(reg)
    w = A1.waypoint_indicators(reg)
    return {"scan_calls": None if calls is None else len(calls),
            "scan_departs": None if calls is None else any(c["departs"] for c in calls),
            "scan_args": None if calls is None else [c["args"] for c in calls][:6],
            "prints_dbg": bool(re.search(r"print\s*\(\s*[\"']DBG", reg)),
            "numeric_xy_pairs": w["numeric_xy_pairs"], "pair_list": w["pair_list"],
            "index_advances": w["index_advances"], "wp_names": w["wp_names"][:5],
            "uses_box_table": w["uses_box_table"]}


def isaac_files(variant, iso, runtime_error=False):
    have = {"stdout": (iso / f"{variant}.carter-run.stdout.log").exists(),
            "stderr": (iso / f"{variant}.carter-run.stderr.log").exists(),
            "kept_json": (iso / f"{variant}.kept.json").exists(),
            "runner": (iso / f"{variant}_carter_run.py").exists(),
            "result_json": (iso / f"{variant}_result.json").exists() or runtime_error,
            "trajectory": (iso / f"{variant}_trajectory.csv").exists() or runtime_error}
    kits = []
    for k in sorted(iso.glob(f"{variant}.kit_*.log")):
        txt = k.read_text(errors="replace")
        kits.append({"file": k.name, "names_self": f"{variant}_carter_run.py" in txt})
    have["kit_logs"] = kits
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-dir", default="results/turns/prompt_a2")
    ap.add_argument("--tag", default="npu_gpt_oss_120b_pA2")
    ap.add_argument("--out", default="results/analysis/prompt_a2_summary.json")
    a = ap.parse_args()
    tdir = Path(a.turns_dir)
    iso = tdir / "isaac_logs"
    runs = []
    for f in sorted(glob.glob(str(tdir / f"{a.tag}_a[0-9]_turns.json"))):
        d = json.load(open(f))
        stem = Path(f).name[:-len("_turns.json")]
        events = [json.loads(l) for l in open(tdir / f"{stem}_events.jsonl")]
        recovered = {e["call"] for e in events if e["type"] == "recovered_tool_call"}
        lost = {e["call"] for e in events if e["type"] == "lost_response"}
        turns = []
        for t in d["turns"]:
            calls = []
            for c in t["calls"]:
                n = c["call"]
                files = {"request": (tdir / f"{stem}_c{n:02d}_request.json").exists(),
                         "raw": (tdir / f"{stem}_c{n:02d}_raw.json").exists(),
                         "raw_text": (tdir / f"{stem}_c{n:02d}_raw_text.txt").exists(),
                         "tool_results": len(list(tdir.glob(f"{stem}_c{n:02d}_tool*.json")))}
                calls.append({"call": n, "tool_calls": c["tool_calls"], "content_chars": c["content_chars"],
                              "completion_tokens": c["completion_tokens"], "reasoning_tokens": c["reasoning_tokens"],
                              "gen_wall_s": c["gen_wall_s"], "lost": n in lost or bool(c.get("lost")),
                              "recovered": n in recovered or bool(c.get("recovered_tool_call") is not None
                                                                  and "recovered_tool_call" in c),
                              "files": files})
            tests = []
            for x in t["tests"]:
                v = x["variant"]
                src = (iso / f"{v}_carter_run.py")
                tests.append({**{k: x.get(k) for k in ("test", "variant", "sim_seconds", "static_ok", "verdict",
                                                       "final_dist_m", "collisions", "partners", "first_collision_t",
                                                       "time_to_goal_s", "sim_time_s", "dbg_lines", "wall_s",
                                                       "runtime_error", "detail")},
                              "code": features(src.read_text() if src.exists() else None),
                              "isaac": (isaac_files(v, iso, bool(x.get("runtime_error"))) if x.get("static_ok")
                                        else {"runner": src.exists(),
                                              "reason": (iso / f"{v}.static_check.txt").exists()})})
            sub = t.get("submission")
            sub_src = Path(sub).read_text() if sub and Path(sub).exists() else None
            row = {"turn": t["turn"], "verdict": t.get("verdict"), "final_dist_m": t.get("final_dist_m"),
                   "collisions": t.get("collisions"), "partners": t.get("partners"),
                   "first_collision_t": t.get("first_collision_t"), "time_to_goal_s": t.get("time_to_goal_s"),
                   "sim_time_s": t.get("sim_time_s"), "tests_used": t.get("tests_used"),
                   "resumed": t.get("resumed"), "calls": calls, "tests": tests,
                   "submission": sub, "submission_code": features(sub_src),
                   "submission_same_as_last_test": (bool(tests) and sub_src is not None and
                                                    A1.region(sub_src) is not None and
                                                    (iso / f"{tests[-1]['variant']}_carter_run.py").exists() and
                                                    A1.region(sub_src) == A1.region((iso / f"{tests[-1]['variant']}_carter_run.py").read_text())),
                   "reply_file": (tdir / t["reply_file"]).exists() if t.get("reply_file") else None}
            if t.get("verdict") not in ("static_dq", "empty_response", "lost_response", "no_submission", None):
                row["isaac"] = isaac_files(t["variant"], iso, bool(t.get("runtime_error")))
            turns.append(row)
        runs.append({"run": stem, "passed_at_turn": d.get("passed_at_turn"), "turns_used": d.get("turns_used"),
                     "model_calls": d.get("model_calls"), "input_sha16": d.get("input_sha16"),
                     "tool_sha16": d.get("tool_sha16"), "started": d.get("started"), "ended": d.get("ended"),
                     "turns": turns})

    def missing(r):
        out = []
        for t in r["turns"]:
            for c in t["calls"]:
                if not (c["files"]["request"] and c["files"]["raw"]):
                    out.append((f"t{t['turn']} c{c['call']:02d}", "request/raw"))
                if c["tool_calls"] and c["files"]["tool_results"] < c["tool_calls"]:
                    out.append((f"t{t['turn']} c{c['call']:02d}", "tool result"))
            for x in t["tests"]:
                # a Kit log of the run just before can be copied too when it ended within a
                # second of this one's start; what must be there is this run's own log
                for k, v in x["isaac"].items():
                    if v is False or (k == "kit_logs" and not any(kk["names_self"] for kk in v)):
                        out.append((x["variant"], k))
            for k, v in (t.get("isaac") or {}).items():
                if v is False or (k == "kit_logs" and not any(kk["names_self"] for kk in v)):
                    out.append((f"t{t['turn']} graded", k))
        return out

    summary = {"runs": len(runs),
               "passed": [(r["run"], r["passed_at_turn"]) for r in runs if r["passed_at_turn"]],
               "tests_per_turn": {r["run"]: [t["tests_used"] for t in r["turns"]] for r in runs},
               "test_verdicts": Counter(x.get("verdict") or ("static_dq" if not x.get("static_ok") else None)
                                        for r in runs for t in r["turns"] for x in t["tests"]),
               "partners_tests": Counter(p.split("/")[-1] for r in runs for t in r["turns"] for x in t["tests"]
                                         for p in (x.get("partners") or [])),
               "partners_graded": Counter(p.split("/")[-1] for r in runs for t in r["turns"]
                                          for p in (t.get("partners") or [])),
               "waypoints_turn1_submission": [r["run"] for r in runs if r["turns"] and
                                              r["turns"][0]["submission_code"].get("pair_list")],
               "scan_departs_any": [r["run"] for r in runs if any(
                   (x["code"].get("scan_departs") for t in r["turns"] for x in t["tests"]))
                   or any(t["submission_code"].get("scan_departs") for t in r["turns"])],
               "dbg_prints_any": [r["run"] for r in runs if any(
                   x["code"].get("prints_dbg") for t in r["turns"] for x in t["tests"])],
               "lost_calls": {r["run"]: [c["call"] for t in r["turns"] for c in t["calls"] if c["lost"]] for r in runs},
               "recovered_calls": {r["run"]: [c["call"] for t in r["turns"] for c in t["calls"] if c["recovered"]]
                                   for r in runs},
               "missing_files": {r["run"]: missing(r) for r in runs}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1))
    for r in runs:
        print(f"== {r['run']}  passed_at={r['passed_at_turn']}  calls={r['model_calls']}  {r['started']} -> {r['ended']}")
        for t in r["turns"]:
            print(f"  t{t['turn']}: {t['verdict']} dist={t['final_dist_m']} col={t['collisions']} "
                  f"{[p.split('/')[-1] for p in (t['partners'] or [])]} tests={t['tests_used']} "
                  f"same_as_last_test={t['submission_same_as_last_test']} sub_wp_list={t['submission_code'].get('pair_list')} "
                  f"sub_xy={t['submission_code'].get('numeric_xy_pairs')} sub_scan_departs={t['submission_code'].get('scan_departs')}")
            for x in t["tests"]:
                print(f"     test{x['test']:02d} sim={x['sim_seconds']} {x.get('verdict') or 'static_dq'} "
                      f"dist={x.get('final_dist_m')} col={x.get('collisions')} "
                      f"{[p.split('/')[-1] for p in (x.get('partners') or [])]} dbg={x.get('dbg_lines')} "
                      f"wp_list={x['code'].get('pair_list')} xy={x['code'].get('numeric_xy_pairs')} "
                      f"scan_departs={x['code'].get('scan_departs')}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
