"""Audit saved v12/v13 CPU results and the exact a1 fault fixture; no rollout."""
import ast
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import b515_chooser_sim as SIM
import b525_build_base as BUILD
from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
P = ARENA / "results/analysis"
OUT = P / "b525_comparison.json"
KEYS = ("t", "pose", "nominal", "target", "command", "actual_velocity", "info", "objects", "done")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = {}
    def read(name):
        evidence[str((P/name).relative_to(ARENA))] = sha(P/name)
        return json.loads((P/name).read_text())
    old = read("b524_reference_sim.json")
    new = read("b525_reference_sim.json")
    old_bad = read("b524_bad_functions.json")
    bad = read("b525_bad_functions.json")
    offline = read("b525_check_offline.json")
    fixture = read("b525_a1_fixture.json")
    probe = read("b524_a1_planner_evidence.json")
    checks = []
    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
    base_hash = sha(BUILD.OUT)
    ref_hash = sha(ARENA / "analysis/b525_reference.py")
    check("Current v13 reproduces from unchanged v12", sha(BUILD.SRC) == BUILD.SOURCE_SHA256 and BUILD.build(BUILD.SRC.read_text()) == BUILD.OUT.read_text())
    check("All reference sources and snapshots remain byte-identical", all(d["reference_sha256"] == sha(d["reference_file"]) == sha(d["reference_snapshot"]) == ref_hash for d in (old,new,old_bad,bad)))
    check("Old and current base hashes match their source files", old["base_sha256"] == old_bad["base_sha256"] == sha(BUILD.SRC) and new["base_sha256"] == bad["base_sha256"] == offline["base_sha256"] == base_hash)
    check("Calibrated CPU engine is unchanged", all(d["calibrated_engine_sha256"] == sha(SIM.__file__) == "8d0308dbf55c288297ade1c92bcd4c71570b888562cf58b03e94e3518e9a8358" for d in (old,new,old_bad,bad)))
    expected = [(label,[m["phase"] for m in movers]) for label,movers in SIM.phase_sets("standard61")]
    check("Both normal batches use exact same61 phases and seeds", all([(r["label"],r["phases"]) for r in d["results"]] == expected and all(r["sensor_noise_seed"] == 515+i for i,r in enumerate(d["results"])) for d in (old,new)))
    check("All68 current individual run records match aggregates", len(new["results"]) == 61 and len(bad["results"]) == 7 and all(json.loads((P/stem/"runs"/(r["label"]+".json")).read_text()) == r for stem,d in (("b525_reference_sim",new),("b525_bad_functions",bad)) for r in d["results"]))
    check("Current CPU batches complete with actual-loop audits", all(d["complete"] and d["audit_pass"] and d["runtime_errors"] == 0 for d in (new,bad)) and all(r["base_sha256"] == base_hash and r["reference_sha256"] == ref_hash and r["audit"]["ticks"] == r["audit"]["choose_calls"] and r["audit"]["returned_command_changes"] == r["audit"]["controller_stub_calls"] == 0 for d in (new,bad) for r in d["results"]))
    check("Reference60+v1 pass without errors", new["random_passed"] == new["random_total"] == 60 and new["v1_pass"] and all(r["verdict"] == "pass" and r["hit"] is None for r in new["results"]) and new["hook_error_ticks"] == new["plan_error_attempts"] == 0)
    pairs=[]
    for a,b in zip(old["results"],new["results"]):
        digests=[hashlib.sha256(),hashlib.sha256()]
        count=0; equal=True; last=None
        with gzip.open(a["trace"],"rb") as fa,gzip.open(b["trace"],"rb") as fb:
            for x,y in itertools.zip_longest(fa,fb):
                count+=1
                if x is not None:digests[0].update(x)
                if y is not None:digests[1].update(y)
                if x is None or y is None:equal=False;continue
                left,last=json.loads(x),json.loads(y)
                equal &= all(left.get(k)==last.get(k) for k in KEYS)
        pairs.append({"label":b["label"],"rows":count,"semantic_equal":equal,
            "old_sha_matches":digests[0].hexdigest()==a["trace_sha256"],"new_sha_matches":digests[1].hexdigest()==b["trace_sha256"],
            "terminal_done_zero":bool(last and last["done"] and last["command"]==[0.,0.] and last["t"]==b["t"])})
    check("Normal61 traces are identical across44842 observations", len(pairs)==61 and sum(p["rows"] for p in pairs)==44842 and all(p["semantic_equal"] for p in pairs))
    check("All122 normal raw trace hashes and terminal states match", all(p["old_sha_matches"] and p["new_sha_matches"] and p["terminal_done_zero"] for p in pairs))
    bad_traces=[]
    for r in bad["results"]:
        h=hashlib.sha256(); last=None; n=0
        with gzip.open(r["trace"],"rb") as f:
            for line in f:h.update(line);last=json.loads(line);n+=1
        bad_traces.append({"label":r["label"],"rows":n,"sha_matches":h.hexdigest()==r["trace_sha256"],"pass_done_zero":r["verdict"]!="pass" or bool(last and last["done"] and last["command"]==[0.,0.])})
    check("All7 fault trace hashes match",all(r["sha_matches"] and r["pass_done_zero"] for r in bad_traces))
    variants={"goal_target","empty_path","plan_path_exception","lookahead_target_exception","follow_cmd_exception","grid_endpoints","a1_unchecked_edges"}
    by_variant={r["variant"]:r for r in bad["results"]}
    check("Seven faults use same v1 with sensor seed515",set(by_variant)==variants and all(r["sensor_noise_seed"]==515 and r["phases"]==[m["phase"] for m in SIM.AM.scenario("v1")] for r in bad["results"]))
    check("Fault outcomes are7 contact-free,2 mission passes,5timeouts",all(r["hit"] is None for r in bad["results"]) and {r["variant"] for r in bad["results"] if r["verdict"]=="pass"}=={"goal_target","grid_endpoints"} and sum(r["verdict"]=="timeout" and r["t"]==200 for r in bad["results"])==5)
    check("Previous6 fault outcomes unchanged",all(all(r[k]==by_variant[r["variant"]][k] for k in ("verdict","t","pose","hit")) for r in old_bad["results"]))
    a1=by_variant["a1_unchecked_edges"]
    events=a1["audit"]["base_plan_events"]
    check("A1 rejected geometry retries21batches84attempts without adoption",len(events)==21 and [e["t"] for e in events]==list(range(0,201,10)) and a1["audit"]["function_calls"]["plan_path"]==84 and a1["audit"]["function_calls"]["lookahead_target"]==0 and all(e["accepted_path"] is None and len(e["attempts"])==4 and all(not a["accepted"] and a["reason"]=="check_path" for a in e["attempts"]) for e in events))
    check("A1 batch summary is recorded with explicit no_path",any("plan_summary" in s and "status=no_path" in s for s in a1["audit"]["dbg_examples"]))
    original=ast.parse((ARENA/fixture["source"]).read_text())
    copied=ast.parse((ARENA/fixture["fixture"]).read_text())
    check("A1 fixture is exact original saved plan_path AST",sha(ARENA/fixture["source"])==fixture["source_sha256"] and sha(ARENA/fixture["fixture"])==fixture["fixture_sha256"]==bad["a1_fixture_sha256"] and ast.dump(next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="plan_path"))==ast.dump(next(n for n in copied.body if isinstance(n,ast.FunctionDef) and n.name=="plan_path")))
    check("Original planner cause and single-edge-predicate correction verified",probe["passed"]==probe["total"]==10 and all(r["validation"]["accepted"] for r in probe["neighbor_edge_ablation"]["results"]))
    check("Offline v13 checks all pass",offline["passed"]==offline["total"] and offline["total"]>=62)
    check("Read-only input evidence hashes remain unchanged",all(sha(ARENA/p)==h for p,h in evidence.items()))
    result={"verdict":"pass" if all(c["pass"] for c in checks) else "fail","passed":sum(c["pass"] for c in checks),"total":len(checks),"checks":checks,
        "base_sha256":base_hash,"reference_sha256":ref_hash,"input_sha256":evidence,
        "normal":{"total":61,"random_passed":new["random_passed"],"v1_pass":new["v1_pass"],"ticks":sum(r["audit"]["ticks"] for r in new["results"]),"trace_rows":sum(p["rows"] for p in pairs),"pairs":pairs},
        "bad":{"total":7,"contact_free":sum(r["hit"] is None for r in bad["results"]),"mission_passed":sum(r["verdict"]=="pass" for r in bad["results"]),"traces":bad_traces,"a1_first_batch":events[0]},
        "limits":["choose_ms measures runtime and is excluded from semantic trace equality.","CPU approximation does not establish Isaac or model4/4 success.","The a1 defect is retained and rejected; the corrected-edge ablation checks path geometry, not driving."]}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("COMPARISON",result["verdict"],f"{result['passed']}/{result['total']}")
    for c in checks:
        if not c["pass"]:print("FAIL",c["name"])
    return 0 if result["verdict"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
