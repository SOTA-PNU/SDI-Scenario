"""Integrate task27 completed evidence without running models or simulations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(A), str(A/"analysis")]
import agent_loop as AL
import b525_build_base as BB
import b525_build_prompt as BP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    P = A/"results/analysis"
    out = P/"b525_final_validation.json"
    if out.exists():
        raise SystemExit("Archive previous final validation before rerun")
    inputs={}
    def read(name):
        path=P/name;inputs[str(path.relative_to(A))]=sha(path)
        return json.loads(path.read_text())
    checks=[]
    def check(name,value):
        checks.append({"name":name,"pass":bool(value)})

    before=read("b525_protected_before.json")
    protected={p:(A/p).is_file() and sha(A/p)==h for p,h in before.items() if p!="REPORT.md"}
    check("Every protected source and original result remains unchanged",len(protected)==len(before)-1 and all(protected.values()))
    harness=read("b525_harness_before.json")
    check("All harness files and calibrated engine unchanged",all(sha(p)==h for p,h in harness.items()))
    update=read("b525_report_update.json")
    check("REPORT exact pre-edit backup exists",sha(A/update["archive"])==before["REPORT.md"]==update["before_sha256"])
    check("REPORT contains only the reviewed addition",(A/"REPORT.md").read_bytes()==(A/update["archive"]).read_bytes()+(A/update["addition_file"]).read_bytes())
    addition=(A/update["addition_file"]).read_text()
    check("REPORT records correct experiments and avoids middle dots",all(s in addition for s in ("실험 50 결과","실험 51 준비","145.233","31.6702","4/4")) and "·" not in addition)
    analysis=P/"b524_a1_analysis.md"
    check("Requested a1 analysis is at most60 lines",0<len(analysis.read_text().splitlines())<=60)
    proof=read("b524_a1_planner_evidence.json")
    revisions=read("b524_a1_model_revision_evidence.json")
    run=read("b524_a1_run_evidence.json")
    check("Original planner and corrected-edge evidence all pass",proof["passed"]==proof["total"]==10)
    check("Actual model revision and delivered feedback evidence all pass",revisions["passed"]==revisions["total"]==9 and all(run["checks"].values()))
    check("Original-run evidence hashes match current files",all(sha(A/p)==h for p,h in run["source_sha256"].items()))
    check("Planner evidence hashes match original files",all(sha(p)==h for p,h in proof["input_sha256"].items()))
    check("Model revision evidence hashes match original files",all(sha(A/p)==h for p,h in revisions["source_sha256"].items()))

    base=BP.BASE.read_text();prompt=BP.OUT.read_text();bh=sha(BP.BASE)
    ref=A/"analysis/b525_reference.py";rh=sha(ref)
    check("v13 rebuild reproduces saved base",BB.build(BB.SRC.read_text())==base)
    check("B525 prompt rebuild reproduces saved prompt",BP.build(BP.SRC.read_text(),base)==prompt)
    check("Reference byte-identical to previous unchanged implementation",ref.read_bytes()==(A/"analysis/b524_reference.py").read_bytes())
    selftest=read("b525_selftest.json")
    AL.BASE_FILE,AL.BASE,AL.SP.BASE=BP.BASE,base,base
    AL.B_MODE,AL.A21,AL.A22,AL.A23,AL.A4T="b5",False,False,False,False
    model_input=AL.model_input(BP.OUT.name)
    check("Actual final model input hash and character count match",hashlib.sha256(model_input.encode()).hexdigest()[:16]==selftest["prompt_sha16"] and len(model_input)==selftest["prompt_chars"])
    check("Prompt file metadata match",sha(BP.OUT)[:16]==selftest["prompt_file_sha16"] and len(prompt)==selftest["prompt_file_chars"])
    check("Selftest pins final base and reference",selftest["base_sha256"]==bh and selftest["reference_sha256"]==rh)
    validation={}
    for name in ("b525_selftest.json","b525_prompt_guard.json","b525_check_offline.json","b525_gate_selftest.json","b525_runner_check.json","b525_comparison.json"):
        d=read(name);validation[name]={"passed":d["passed"],"total":d["total"]}
        check(name+" passes all checks",d["passed"]==d["total"] and d["total"]>0)
    sim=read("b525_reference_sim.json");bad=read("b525_bad_functions.json")
    for name,d,n in (("normal",sim,61),("bad7",bad,7)):
        check(name+" uses final base and reference",d["base_sha256"]==bh and d["reference_sha256"]==rh)
        check(name+" completes required cases with exact-loop audit",d["complete"] and d["audit_pass"] and d["runtime_errors"]==0 and len(d["results"])==n)
    check("Random60+v1 pass without hook errors",sim["random_passed"]==sim["random_total"]==60 and sim["v1_pass"] and sim["hook_error_ticks"]==sim["plan_error_attempts"]==0)
    check("Seven faults are contact-free with exactly two mission successes",all(r["hit"] is None for r in bad["results"]) and sum(r["verdict"]=="pass" for r in bad["results"])==2)
    check("A1 fault is present and its unsafe paths remain rejected",any(r["variant"]=="a1_unchecked_edges" and r["verdict"]=="timeout" and all(e["accepted_path"] is None for e in r["audit"]["base_plan_events"]) for r in bad["results"]))
    prep=read("b525_launch_preparation.json")
    check("Launch manifest pins current operational files",all(sha(A/p)==h for p,h in prep["source_sha256"].items()))
    check("Launch manifest pins gate and runner validation",all(sha(A/v["artifact"])==v["sha256"] for v in prep["validation"].values()))
    check("Actual new GPU output remains absent",all(not (A/p).exists() for p in ("results/analysis/b525_gate_isaac","results/turns/prompt_b5_25","b525_gate_probe_carter_run.py")))
    for p in ("run_b525_gpu_steps.sh","run_b525_sequential.sh"):
        result=subprocess.run(["bash","-n",str(A/p)],capture_output=True,text=True)
        check("Shell syntax "+p,result.returncode==0)
    check("Evidence inputs unchanged during integration",all(sha(A/p)==h for p,h in inputs.items()))
    sources=sorted(set(prep["source_sha256"])|{"analysis/b525_final_check.py","analysis/b525_build_base.py","analysis/b525_build_prompt.py","analysis/b525_check_offline.py","analysis/b525_selftest.py","analysis/b525_prompt_guard.py","analysis/b525_reference_sim.py","analysis/b525_compare.py","analysis/b525_a1_unchecked_edges.py","analysis/b524_a1_planner_probe.py","analysis/b524_a1_run_evidence.py","analysis/b524_a1_model_revision_evidence.py","results/analysis/b524_a1_analysis.md","REPORT.md"})
    result={"task":27,"verdict":"pass" if all(c["pass"] for c in checks) else "fail","passed":sum(c["pass"] for c in checks),"total":len(checks),"checks":checks,"validation":validation,
        "base_file":BP.BASE.name,"base_sha256":bh,"base_sha16":bh[:16],"prompt_sha16":selftest["prompt_sha16"],"prompt_chars":selftest["prompt_chars"],"prompt_file_sha16":selftest["prompt_file_sha16"],"prompt_file_chars":selftest["prompt_file_chars"],
        "protected_unchanged":protected,"report_update":update,"source_sha256":{p:sha(A/p) for p in sources},"evidence_sha256":inputs,
        "launch_cmd":"bash "+str(A/"run_b525_gpu_steps.sh"),"execution":"Preparation and CPU checks only; no actual GPU, Isaac, HTTP, host process inspection, or model run.","limitations":"A1 geometry remains rejected; improved diagnostics and instructions require actual model validation. CPU60+v1 is not model4/4 stability."}
    with out.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print("FINAL",result["verdict"],f"{result['passed']}/{result['total']}")
    for c in checks:
        if not c["pass"]:print("FAIL",c["name"])
    return 0 if result["verdict"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
