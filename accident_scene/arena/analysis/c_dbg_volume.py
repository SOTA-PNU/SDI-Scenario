"""How much DBG text would the model get if it saw every DBG line its controller printed? (B-5-10 runs)

The base drops all but the first DBG line per 0.5 s of sim time (10 in the first 0.5 s) and never reports how many it
dropped, and agent_loop.dbg_lines keeps the first 40 and last 40 of what is left. So the full count is estimated:
a run whose kept lines fill (almost) every 0.5 s window is taken to print one DBG line per control call (60 per sim
second), which is what the final codes' unconditional end-of-controller prints do; a sparse run is taken at its kept
count. Tokens per line come from the 120b server's own tokenizer (/tokenize on :8420, read only).

    python3 analysis/c_dbg_volume.py
"""
import csv
import json
import math
import statistics as st
import sys
import urllib.request
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
LOGS = ARENA / "results/turns/prompt_b5_10/isaac_logs"
sys.path.insert(0, str(ARENA))
from agent_loop import dbg_lines  # noqa: E402


def ntok(text):
    req = urllib.request.Request("http://localhost:8420/tokenize", data=json.dumps({"model": "gpt-oss-120b", "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["count"]


rows = []
for so in sorted(LOGS.glob("*.carter-run.stdout.log")):
    v = so.name[:-len(".carter-run.stdout.log")]
    tr = next((p for p in (LOGS / f"{v}_trajectory.csv", ARENA / "results" / f"{v}_trajectory.csv") if p.exists()), None)
    if tr is None:
        continue
    traj = list(csv.DictReader(open(tr)))
    if not traj:
        continue
    T = float(traj[-1]["t"])
    ticks = round(T * 60) + 1                                     # controller() runs every 1/60 s step; long trajectories are thinned
    text = so.read_text(errors="replace")
    kept = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("DBG")]
    shown, omitted = dbg_lines(text)
    windows = math.ceil(T / 0.5)
    fill = len(kept) / max(1, 9 + windows)
    dense = fill > 0.9
    full = ticks if dense else len(kept)
    sample = kept[:400]
    tpl = ntok("\n".join(sample)) / len(sample) if sample else 0.0
    rows.append(dict(v=v.replace("gpu_gpt_oss_120b_pB510_", ""), T=T, ticks=ticks, kept=len(kept), shown=len(shown),
                     shown_tok=ntok("\n".join(shown)) if shown else 0, dense=dense, full=full, tpl=tpl, full_tok=full * tpl,
                     chars=st.mean(len(s) for s in kept) if kept else 0))

print(f"{'run':<14}{'sim s':>8}{'calls':>7}{'printed':>9}{'shown':>7}{'shown tok':>10}{'dense':>7}{'full lines':>11}{'tok/line':>9}{'full tok':>10}")
for r in rows:
    print(f"{r['v']:<14}{r['T']:>8.1f}{r['ticks']:>7}{r['kept']:>9}{r['shown']:>7}{r['shown_tok']:>10}{str(r['dense']):>7}{r['full']:>11}{r['tpl']:>9.1f}{r['full_tok']:>10.0f}")
d = [r for r in rows if r["dense"]]
print(f"\nruns {len(rows)}, dense {len(d)}")
turns = {}
for r in rows:
    if "_test" in r["v"]:
        key = r["v"].split("_test")[0]
        turns.setdefault(key, []).append(r)
print("\nper turn (test runs the model saw inside one turn):")
for key, rs in turns.items():
    print(f"  {key:<8} tests {len(rs):>2}  shown tok {sum(r['shown_tok'] for r in rs):>7}  full tok {sum(r['full_tok'] for r in rs):>11,.0f}  max one run {max(r['full_tok'] for r in rs):>10,.0f}")
for k in ("T", "full", "full_tok", "tpl", "chars", "shown_tok"):
    xs = [r[k] for r in rows]
    print(f"{k:>9}: median {st.median(xs):.1f}  min {min(xs):.1f}  max {max(xs):.1f}  sum {sum(xs):.0f}")
