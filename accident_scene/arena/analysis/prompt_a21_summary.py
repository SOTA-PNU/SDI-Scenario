"""Experiment 15 (prompt A-2-1): what the runs did with the added Isaac information.

Reads results/turns/prompt_a2_1/<tag>_a<k>_turns.json, _events.jsonl, the per-call tool
files and isaac_logs, and reports per turn:
  - model calls (lost / recovered), tests (verdict, partners, telemetry digest: stall modes,
    contacts scored/unscored, closest objects), scene tool use (scene_obstacles region-less /
    with region, check_route calls and their min_gap / scored overlaps);
  - whether the scene listing reached the model before the turn's submission (an explicit
    region-less scene_obstacles call or the listing attached to a test result);
  - the graded submission: verdict, partners, whether its EDIT REGION equals a tested one
    (and that test's verdict), the first literal waypoint list in it and check_route on it
    (scored overlaps, min gap);
  - every file the run promised to keep (A-2's list plus the telemetry file per test).

    python3 analysis/prompt_a21_summary.py     (writes results/analysis/prompt_a21_summary.json)
"""
import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ARENA))
import prompt_a2_summary as P2      # noqa: E402  (features(), isaac_files(); chdir to the arena via prompt_a1_summary)
import route_clearance_check as RC  # noqa: E402  (first_pair_list)
import a21_scene as S               # noqa: E402
import agent_loop as AL             # noqa: E402  (region_of, norm_region)

START, GOAL = (-9.0, -10.8, 1.0138), (8.0, 16.5, 1.0138)


def sha_region(text):
    return hashlib.sha256(AL.norm_region(AL.region_of(text).rstrip()).encode()).hexdigest()[:8]


def route_check(sc, src):
    reg = AL.region_of(src) if src else ""
    pts = RC.first_pair_list(reg) if reg else None
    if not pts:
        return None
    if abs(pts[0][0] - START[0]) < 0.05 and abs(pts[0][1] - START[1]) < 0.05:
        pts = pts[1:]
    try:
        r = S.check_route(sc, [list(p) for p in pts], START, GOAL)
    except ValueError as e:
        return {"waypoints": pts, "error": str(e)}
    scored_ov = sorted({o[0] for ov in r["overlaps"] for o in ov["objects"] if o[1]}
                       | {o[0] for ov in r["turn_overlaps"] for o in ov["objects"] if o[1]})
    unscored_ov = sorted({o[0] for ov in r["overlaps"] for o in ov["objects"] if not o[1]}
                         | {o[0] for ov in r["turn_overlaps"] for o in ov["objects"] if not o[1]})
    near_scored = min([sg[4] for sg in r["segments"] if sg[6]] + [t[3] for t in r["turns"] if t[5]] or [None],
                      key=lambda v: 9e9 if v is None else v)
    return {"waypoints": pts, "min_gap": r["min_gap"], "min_gap_scored": near_scored,
            "scored_overlaps": scored_ov, "unscored_overlaps": unscored_ov, "end_to_goal_m": r.get("end_to_goal_m")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-dir", default="results/turns/prompt_a2_1")
    ap.add_argument("--tag", default="npu_gpt_oss_120b_pA21")
    ap.add_argument("--out", default="results/analysis/prompt_a21_summary.json")
    a = ap.parse_args()
    tdir = ARENA / a.turns_dir
    iso = tdir / "isaac_logs"
    sc = S.Scene()
    runs = []
    for f in sorted(glob.glob(str(tdir / f"{a.tag}_a[0-9]_turns.json"))):
        d = json.load(open(f))
        stem = Path(f).name[:-len("_turns.json")]
        events = [json.loads(l) for l in open(tdir / f"{stem}_events.jsonl")]
        recovered = {e["call"] for e in events if e["type"] == "recovered_tool_call"}
        lost = {e["call"] for e in events if e["type"] == "lost_response"}
        tool_files = {}
        for tf in sorted(tdir.glob(f"{stem}_c*_tool*.json")):
            tool_files[tf.name] = json.loads(tf.read_text())
        turns = []
        tested = {}                                  # sha8 -> (turn, test, verdict)
        for t in d["turns"]:
            calls = [c["call"] for c in t["calls"]]
            ctools = []
            for c in calls:
                for name, tf in sorted(tool_files.items()):
                    if name.startswith(f"{stem}_c{c:02d}_tool"):
                        tc = tf["tool_call"]["function"]["name"]
                        res = tf["result"] or {}
                        ctools.append({"call": c, "file": name, "tool": tc, "error": res.get("error"),
                                       "region": (tf.get("arguments") or {}).get("region"),
                                       "scene_attached": "scene" in res,
                                       "min_gap": res.get("min_gap") if tc == "check_route" else None,
                                       "scored_overlaps": sorted({o[0] for ov in res.get("overlaps", []) + res.get("turn_overlaps", [])
                                                                  for o in ov["objects"] if o[1]}) if tc == "check_route" else None})
            seen_scene = any((x["tool"] == "scene_obstacles" and not x["region"] and not x["error"]) or x["scene_attached"]
                             for x in ctools)
            tests = []
            for x in t["tests"]:
                v = x["variant"]
                src = iso / f"{v}_carter_run.py"
                text = src.read_text() if src.exists() else None
                dg = iso / f"{v}_a21diag.json"
                a21 = x.get("a21") or {}
                row = {**{k: x.get(k) for k in ("test", "variant", "sim_seconds", "static_ok", "verdict", "final_dist_m",
                                                 "collisions", "partners", "first_collision_t", "time_to_goal_s",
                                                 "sim_time_s", "wall_s", "runtime_error", "detail")},
                       "code_ran": x.get("code_ran"), "a21": a21, "code": P2.features(text),
                       "telemetry_file": dg.exists()}
                if text and x.get("static_ok"):
                    tested.setdefault(sha_region(text), (t["turn"], x["test"], x.get("verdict")))
                    row["route"] = route_check(sc, text)
                    row["isaac"] = P2.isaac_files(v, iso, bool(x.get("runtime_error")))
                tests.append(row)
            sub = t.get("submission")
            sub_src = Path(sub).read_text() if sub and Path(sub).exists() else None
            same = tested.get(sha_region(sub_src)) if sub_src and AL.region_of(sub_src) else None
            row = {"turn": t["turn"], "verdict": t.get("verdict"), "final_dist_m": t.get("final_dist_m"),
                   "collisions": t.get("collisions"), "partners": t.get("partners"),
                   "first_collision_t": t.get("first_collision_t"), "time_to_goal_s": t.get("time_to_goal_s"),
                   "tests_used": t.get("tests_used"), "queries": t.get("queries", []),
                   "query_counts": Counter(q["tool"] + ("(region)" if (q.get("args") or {}).get("region") else "")
                                           for q in t.get("queries", []) if not q.get("error")),
                   "tool_calls": ctools, "scene_seen_before_submission": seen_scene,
                   "calls": [{"call": c["call"], "tool_calls": c["tool_calls"], "content_chars": c["content_chars"],
                              "completion_tokens": c["completion_tokens"], "gen_wall_s": c["gen_wall_s"],
                              "lost": c["call"] in lost or bool(c.get("lost")),
                              "recovered": c["call"] in recovered} for c in t["calls"]],
                   "tests": tests, "submission": sub, "submission_code": P2.features(sub_src),
                   "submission_equals_test": same, "submission_route": route_check(sc, sub_src)}
            if t.get("verdict") not in ("static_dq", "empty_response", "lost_response", "no_submission", None):
                row["isaac"] = P2.isaac_files(t["variant"], iso, bool(t.get("runtime_error")))
            turns.append(row)
        runs.append({"run": stem, "passed_at_turn": d.get("passed_at_turn"), "turns_used": d.get("turns_used"),
                     "model_calls": d.get("model_calls"),
                     **{k: d.get(k) for k in ("input_sha16", "tool_sha16", "scene_sha16", "listing_sha16",
                                              "inject_sha16", "a21_code_sha16", "started", "ended")},
                     "turns": turns})
    summary = {"runs": len(runs),
               "passed": [(r["run"], r["passed_at_turn"]) for r in runs if r["passed_at_turn"]],
               "tests_per_turn": {r["run"]: [t["tests_used"] for t in r["turns"]] for r in runs},
               "queries_per_turn": {r["run"]: [dict(t["query_counts"]) for t in r["turns"]] for r in runs},
               "scene_seen_before_submission": {r["run"]: [t["scene_seen_before_submission"] for t in r["turns"]] for r in runs},
               "graded": {r["run"]: [(t["turn"], t["verdict"], [p.split("/")[-1] for p in (t["partners"] or [])],
                                      t["final_dist_m"]) for t in r["turns"]] for r in runs},
               "submission_equals_test": {r["run"]: [t["submission_equals_test"] for t in r["turns"]] for r in runs},
               "submission_route_scored_overlaps": {r["run"]: [(t["submission_route"] or {}).get("scored_overlaps")
                                                               for t in r["turns"]] for r in runs},
               "test_verdicts": Counter(x.get("verdict") or ("static_dq" if not x.get("static_ok") else None)
                                        for r in runs for t in r["turns"] for x in t["tests"]),
               "stall_modes": Counter(m for r in runs for t in r["turns"] for x in t["tests"]
                                      for m, _ in ((x.get("a21") or {}).get("stalls") or [])),
               "lost_calls": {r["run"]: [c["call"] for t in r["turns"] for c in t["calls"] if c["lost"]] for r in runs},
               "recovered_calls": {r["run"]: [c["call"] for t in r["turns"] for c in t["calls"] if c["recovered"]] for r in runs},
               "missing_telemetry": [x["variant"] for r in runs for t in r["turns"] for x in t["tests"]
                                     if x.get("static_ok") and not x["telemetry_file"]]}
    out = ARENA / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1, default=str))
    for r in runs:
        print(f"== {r['run']}  passed_at={r['passed_at_turn']}  calls={r['model_calls']}  {r['started']} -> {r['ended']}")
        for t in r["turns"]:
            sr = t["submission_route"] or {}
            print(f"  t{t['turn']}: {t['verdict']} dist={t['final_dist_m']} col={t['collisions']} "
                  f"{[p.split('/')[-1] for p in (t['partners'] or [])]} tests={t['tests_used']} queries={dict(t['query_counts'])} "
                  f"scene_seen={t['scene_seen_before_submission']} sub==test={t['submission_equals_test']} "
                  f"route: wp={len(sr.get('waypoints') or [])} min_gap={sr.get('min_gap')} scored_ov={sr.get('scored_overlaps')}")
            for q in t["tool_calls"]:
                if q["tool"] == "check_route":
                    print(f"     c{q['call']:02d} check_route min_gap={q['min_gap']} scored_overlaps={q['scored_overlaps']} err={q['error']}")
                elif q["tool"] == "scene_obstacles":
                    print(f"     c{q['call']:02d} scene_obstacles region={q['region']} err={q['error']}")
            for x in t["tests"]:
                a21 = x.get("a21") or {}
                print(f"     test{x['test']:02d} sim={x['sim_seconds']} {x.get('verdict') or 'static_dq'} dist={x.get('final_dist_m')} "
                      f"col={x.get('collisions')} {[p.split('/')[-1] for p in (x.get('partners') or [])]} "
                      f"stalls={a21.get('stalls')} contacts={a21.get('contacts')} code_ran={x.get('code_ran')} "
                      f"route_scored_ov={(x.get('route') or {}).get('scored_overlaps')}")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
