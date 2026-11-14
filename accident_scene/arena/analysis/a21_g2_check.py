"""Gate G2 analysis (REPORT.md 실험 15): instrumented test runs vs plain runs of the same code.

Cases (runners g2_<case>_{plain,inst}_carter_run.py, made from the archived A-2 tests with a
shorter timeout; plus the A-3 reference controller): verdict, partners, first collision time,
the largest position gap at common timestamps between instrumented and plain today, and between
today's plain run and the archived A-2 test (yesterday, plain) as the natural spread. Then the
digest each instrumented run would return, against the expected diagnosis. Writes
results/analysis/a21_g2_check.json.
"""
import csv
import json
import math
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene as S  # noqa: E402
import agent_loop as A  # noqa: E402

R = ARENA / "results/analysis/a21_gates"      # gate runners, results, telemetry (moved here after the gate)
ISO2 = ARENA / "results/turns/prompt_a2/isaac_logs"
CASES = {"a0t1t01": "npu_gpt_oss_120b_pA2_a0_t1_test01", "a0t2t01": "npu_gpt_oss_120b_pA2_a0_t2_test01",
         "a0t2t02": "npu_gpt_oss_120b_pA2_a0_t2_test02", "a1t2t01": "npu_gpt_oss_120b_pA2_a1_t2_test01",
         "a1t2t03": "npu_gpt_oss_120b_pA2_a1_t2_test03"}
# instrumented runs of the final telemetry block are g2f_*; a1t2t01 was run once, with the first block
INST = {"a0t1t01": "g2f", "a0t2t01": "g2f", "a0t2t02": "g2f", "a1t2t01": "g2", "a1t2t03": "g2f", "a3ref": "g2f"}


def traj(f):
    if not Path(f).exists():
        return {}
    return {round(float(q["t"]), 3): (float(q["x"]), float(q["y"])) for q in csv.DictReader(open(f))}


def gap(a, b):
    common = sorted(set(a) & set(b))
    if not common:
        return None, 0
    return round(max(math.hypot(a[t][0] - b[t][0], a[t][1] - b[t][1]) for t in common), 4), len(common)


def res(f):
    if not Path(f).exists():
        return None
    m = json.loads(Path(f).read_text())["metrics"]
    return {"verdict": json.loads(Path(f).read_text())["verdict"], "partners": m["collision_partners"],
            "first_collision_t": m["first_collision_t"], "final_pose": m["final_pose"], "metrics": m}


def main():
    sc = S.Scene()
    out = {}
    for case, orig in list(CASES.items()) + [("a3ref", None)]:
        inst = R / f"{INST[case]}_{case}_inst"
        plain = R / f"g2_{case}_plain" if case != "a3ref" else R / "a21val_plain"
        o = {"inst": res(f"{inst}_result.json"), "plain": res(f"{plain}_result.json")}
        o["max_gap_inst_vs_plain"], o["common"] = gap(traj(f"{inst}_trajectory.csv"), traj(f"{plain}_trajectory.csv"))
        fct = [r["first_collision_t"] for r in (o["inst"], o["plain"]) if r and r["first_collision_t"] is not None]
        if fct:
            cut = min(fct)
            a_ = {t: v for t, v in traj(f"{inst}_trajectory.csv").items() if t < cut}
            b_ = {t: v for t, v in traj(f"{plain}_trajectory.csv").items() if t < cut}
            o["max_gap_before_first_collision"], _ = gap(a_, b_)
        if orig:
            o["max_gap_plain_vs_archived"], _ = gap(traj(f"{plain}_trajectory.csv"), traj(ISO2 / f"{orig}_trajectory.csv"))
        dg = Path(f"{inst}_a21diag.json")
        if dg.exists():
            diag = json.loads(dg.read_text())
            tele, keys = S.digest(sc, diag, (8.0, 16.5, 1.0138), (o["inst"] or {}).get("metrics"))
            runner = R / f"{INST[case]}_{case}_inst_carter_run.py"
            if runner.exists():
                lines = runner.read_text().splitlines()
                start = next((i for i, ln in enumerate(lines, 1) if ln.startswith("# [EDIT REGION]")), None)
                end = next((i for i, ln in enumerate(lines, 1) if ln.startswith("# ========================= [END EDIT")), None)
                o["runtime_error"] = S.runtime_error_out(diag, start, end, runner.name)
                region = A.region_of(runner.read_text())
                o["same_as_base"] = A.norm_region(region) == A.norm_region(A.region_of(A.BASE))
            o["digest"] = tele
            o["chars"] = len(json.dumps({**tele, "trajectory": S.trajectory(diag["rows"], keys)}, ensure_ascii=False)) if diag.get("rows") else 0
        out[case] = o
        brief = lambda r: None if not r else {k: v for k, v in r.items() if k != "metrics"}
        print(f"{case}: inst={brief(o['inst'])} plain={brief(o['plain'])}")
        print(f"   max gap inst-plain={o['max_gap_inst_vs_plain']} (n={o['common']}), before first collision="
              f"{o.get('max_gap_before_first_collision')}, plain-archived={o.get('max_gap_plain_vs_archived')}")
        t = o.get("digest") or {}
        print(f"   stalls={[(s['mode'], s['t'], s['nearest'], s.get('scan_min')) for s in t.get('stalls', [])]}")
        print(f"   contacts={[(c['obj'], c['scored'], c['t'], c.get('part'), c.get('sep')) for c in t.get('contacts', [])]}")
        print(f"   before={t.get('before_contact', {}).get('event')} closest={[(c['obj'], c['gap'], c['part']) for c in t.get('closest_approach', [])]}")
        print(f"   runtime_error={o.get('runtime_error')} same_as_base={o.get('same_as_base')} chars={o.get('chars')}")
    exp = {
        "a1t2t03 blocked + Rackshield_43 unscored": any(s["mode"] == "blocked" for s in out["a1t2t03"]["digest"]["stalls"])
        and any(c["obj"] == "SM_Rackshield_43" and c["scored"] == 0 for c in out["a1t2t03"]["digest"]["contacts"]),
        "a0t2t01 oscillating near Forklift_01": any(s["mode"] == "oscillating" and s["nearest"][0] == "Forklift_01"
                                                    for s in out["a0t2t01"]["digest"]["stalls"]),
        "a1t2t01 same_as_base + idle": out["a1t2t01"].get("same_as_base") and any(
            s["mode"] == "idle" for s in out["a1t2t01"]["digest"]["stalls"]),
        "a0t2t02 NameError with region line": bool(out["a0t2t02"].get("runtime_error"))
        and out["a0t2t02"]["runtime_error"].get("type") == "NameError" and out["a0t2t02"]["runtime_error"].get("region_line"),
        "a0t1t01 arena_box_00 front-right + before_contact": any(
            c["obj"] == "arena_box_00" and str(c.get("part", "")).startswith("front") for c in out["a0t1t01"]["digest"]["contacts"])
        and out["a0t1t01"]["digest"].get("before_contact", {}).get("event") == "scored_contact",
        "verdicts and partners equal": all(o["inst"] and o["plain"] and o["inst"]["verdict"] == o["plain"]["verdict"]
                                           and o["inst"]["partners"] == o["plain"]["partners"] for o in out.values()
                                           if o["inst"] and o["plain"]),
    }
    for k, v in exp.items():
        print(("PASS " if v else "FAIL ") + k)
    out["checks"] = exp
    (ARENA / "results/analysis/a21_g2_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
