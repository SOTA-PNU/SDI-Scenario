"""A-2-2 submission-gate audit: for every gated answer and every test run, the EDIT REGION
key the gate used (ast.dump, docstrings count) and a key with docstrings stripped, so a
hold caused only by docstring edits shows up. Reads results/turns/prompt_a2_2/*; no Isaac.

  python3 analysis/a22_gate_audit.py [--turns-dir results/turns/prompt_a2_2] [--tag npu_gpt_oss_120b_pA22]
  python3 analysis/a22_gate_audit.py --a23 --turns-dir results/turns/prompt_a2_3 --tag npu_gpt_oss_120b_pA23 \
      --out results/analysis/prompt_a23_gate_audit.json   # A-2-3 (실험 17): the gate's keys drop docstrings
"""
import argparse, ast, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import agent_loop  # noqa: E402
from agent_loop import region_of, code_key  # noqa: E402


def key_nodoc(region):
    try:
        tree = ast.parse(region)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:12]


def answer_text(path):
    r = json.load(open(path))
    m = r["choices"][0]["message"]
    return m.get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-dir", default="results/turns/prompt_a2_2")
    ap.add_argument("--tag", default="npu_gpt_oss_120b_pA22")
    ap.add_argument("--out", default="results/analysis/prompt_a22_gate_audit.json")
    ap.add_argument("--a23", action="store_true", help="compute keys as the --a23 gate does (docstrings dropped)")
    a = ap.parse_args()
    agent_loop.A23 = a.a23
    out = {}
    for evf in sorted(glob.glob(os.path.join(a.turns_dir, f"{a.tag}_a*_events.jsonl"))):
        run = os.path.basename(evf)[: -len("_events.jsonl")]
        ev = [json.loads(l) for l in open(evf)]
        tests = []
        for e in ev:
            if e["type"] != "tool_call":
                continue
            tj = os.path.join(a.turns_dir, f"{run}_c{e['call']:02d}_tool1.json")
            code = json.load(open(tj))["arguments"].get("code", "")
            reg = region_of(code) or code
            res = next((x for x in ev if x["type"] == "tool_result" and x.get("call") == e["call"]), {})
            tests.append(dict(turn=e["turn"], call=e["call"], test=e["test"],
                              key=code_key(reg), key_nodoc=key_nodoc(reg),
                              verdict=res.get("verdict")))
        gates = []
        for e in ev:
            if e["type"] != "submission_gate":
                continue
            reg = region_of(answer_text(os.path.join(a.turns_dir, f"{run}_c{e['call']:02d}_raw.json"))) or ""
            kn = key_nodoc(reg)
            same = [t["test"] for t in tests if t["turn"] == e["turn"] and t["key"] == e["code_key"]]
            same_nodoc = [t["test"] for t in tests if t["turn"] == e["turn"] and kn and t["key_nodoc"] == kn]
            gates.append(dict(turn=e["turn"], call=e["call"], decision=e["decision"],
                              notes_sent=e["notes_sent"], tests_left=e["tests_left"],
                              code_key=e["code_key"], recomputed_key=code_key(reg) if reg else None,
                              key_nodoc=kn, same_as_test=same, same_as_test_nodoc=same_nodoc))
        out[run] = dict(tests=tests, gates=gates)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for run, d in out.items():
        print("==", run)
        for t in d["tests"]:
            print(f"  t{t['turn']} c{t['call']:02d} test{t['test']:02d} {t['verdict']} key={t['key']} nodoc={t['key_nodoc']}")
        for g in d["gates"]:
            print(f"  t{g['turn']} c{g['call']:02d} gate {g['decision']:<12} key={g['code_key']} (re {g['recomputed_key']}) "
                  f"nodoc={g['key_nodoc']} same={g['same_as_test']} same_nodoc={g['same_as_test_nodoc']}")
    print("->", a.out)


if __name__ == "__main__":
    main()
