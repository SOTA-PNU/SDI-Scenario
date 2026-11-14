"""Did the controllers that hit a low box have ANY information about that box?

For every low-box collision in the paper's scope (first turn 60, cap-8 loop
turns 2-8, transfer conditions T3/T2/T2P) this reads the submitted EDIT REGION
and checks, with comments and long strings removed:
  * every env.raycast_scan(...) call - are the arguments empty (default z=0.35)?
  * does the code read MISSION["boxes"] (the box table)?
  * does a literal (x, y) pair of the collided low box appear?
The low boxes' tops are 0.18-0.25 m, below the 0.35 m scan plane (Isaac check:
analysis/box_visibility_v1.log), so a default-height scan cannot return them.
NPU a2 cap-8 turns 2-8 were overwritten by the cap-30 run (same variant names),
so their code is reported as lost instead of being read from the wrong file.

Run:  python3 analysis/lowbox_controller_check.py   (writes analysis/lowbox_controller_check.json)
"""
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
LOW = {"arena_box_04": (2.5, 3.0), "arena_box_05": (3.9, 4.2),
       "arena_box_06": (7.625, 13.4), "arena_box_13": (0.2, -9.6)}
LOST = {f"npu_gpt_oss_120b_a2_t{k}" for k in range(2, 9)}   # cap-8 code overwritten by cap-30


def code_only(src):
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            out.append(tok.string if len(tok.string) < 12 else '""')
            continue
        out.append(tok.string)
    return " ".join(out)


def region(src):
    m = re.search(r"#\s*\[EDIT REGION\][^\n]*\n(.*?)\n#\s*=+\s*\[END EDIT REGION\]", src, re.S)
    return m.group(1) if m else None


def scan_args(code):
    """Argument text of every raycast_scan(...) call, with balanced parentheses."""
    out = []
    for m in re.finditer(r"raycast_scan \(", code):
        i, depth, j = m.end(), 1, m.end()
        while j < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[j], 0)
            j += 1
        out.append(code[i:j - 1])
    return out


def near_pair(flat, x, y):
    """A box centre written as a coordinate: (x, y) / [x, y] / {'x': x, 'y': y} / x=x, y=y.

    `flat` is the code with whitespace removed. Two unrelated constants that happen
    to equal x and y (e.g. gains 2.5 and 3.0 on adjacent lines) do not count.
    """
    X, Y = re.escape(str(x)), re.escape(str(y))
    num = r"(?![\d])"
    pats = [rf"[\(\[]{X}{num},{Y}{num}[\)\],]",
            rf"""['"]x['"]:{X}{num},['"]y['"]:{Y}{num}""",
            rf"\bx={X}{num},y={Y}{num}"]
    return any(re.search(q, flat) for q in pats)


def analyse(v, partners, source):
    if source == "cap8" and v in LOST:
        return {"status": "code lost (overwritten by the cap-30 run)"}
    f = Path(f"{v}_carter_run.py")
    if not f.exists():
        return {"status": "no code file"}
    reg = region(f.read_text())
    if reg is None:
        return {"status": "no edit region"}
    code = code_only(reg)
    calls = scan_args(code)
    flat = re.sub(r"\s+", "", code)
    pairs = re.findall(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)", code)
    return {
        "status": "checked",
        "scan_calls": len(calls),
        "scan_all_default": all(not c.strip() for c in calls),
        "uses_box_table": bool(re.search(r"""MISSION \[ ["']boxes["'] \]|MISSION \. get \( ["']boxes["']""", code)),
        "uses_env_other_than_scan": sorted(set(re.findall(r"\benv \. (\w+)", code)) - {"raycast_scan"}),
        "has_lowbox_xy_literal": any(near_pair(flat, LOW[p][0], LOW[p][1]) for p in partners),
        "numeric_xy_pairs": len(pairs),   # waypoint-like constants (not necessarily from the table)
    }


scope = []
for f in sorted(glob.glob("results/*_result.json")):
    v = os.path.basename(f)[:-12]
    if re.fullmatch(r"(gpu|npu)_(gpt_oss_120b|qwen3_32b_fp8|exaone_4_0_32b_fp8)_[ab][0-4]", v):
        parts = json.load(open(f))["metrics"]["collision_partners"]
        low = sorted({k for p in parts for k in LOW if k in p})
        if low:
            scope.append(("first turn", v, low))
for f in sorted(glob.glob("results/turns/cap8/*_turns.json")):
    d = json.load(open(f))
    turns = d["turns"] if isinstance(d, dict) else d
    base = os.path.basename(f)[:-11]
    for t in turns:
        low = sorted({k for p in (t.get("partners") or []) for k in LOW if k in p})
        if low and t.get("turn") != 1:            # turn 1 is the reused first-turn run
            scope.append(("cap8", f"{base}_t{t['turn']}", low))
ts = json.load(open("results/analysis/transfer_summary.json"))
for cond, tag in (("T3", "t3"), ("T2", "t2"), ("T2P", "t2p")):
    for run in ts[cond]["runs"]:
        for i, (k, parts) in enumerate(zip(run["kinds"], run["partners"])):
            if k == "low_box":
                low = sorted({kk for p in parts for kk in LOW if kk in p})
                scope.append((cond, f"npu_gpt_oss_120b_{tag}_a{run['sample']}" + ("" if i == 0 else f"_t{i + 1}"), low))

rows = [{"source": s, "variant": v, "low_partners": low, **analyse(v, low, s)} for s, v, low in scope]
checked = [r for r in rows if r["status"] == "checked"]
summary = {
    "in_scope": len(rows),
    "by_source": Counter(r["source"] for r in rows),
    "checked": len(checked),
    "checked_scan_default_only_and_no_table_no_xy": sum(
        r["scan_all_default"] and not r["uses_box_table"] and not r["has_lowbox_xy_literal"]
        and not r["uses_env_other_than_scan"] for r in checked),
    "checked_with_numeric_xy_pairs": [(r["variant"], r["numeric_xy_pairs"]) for r in checked if r["numeric_xy_pairs"]],
    "not_checked": [(r["variant"], r["status"]) for r in rows if r["status"] != "checked"],
}
Path("analysis/lowbox_controller_check.json").write_text(
    json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
print(json.dumps(summary, ensure_ascii=False, indent=1))
