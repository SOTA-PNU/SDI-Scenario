"""Tabulate every turn-protocol run: does feedback converge, and toward what?

The per-turn numbers matter more than the verdict here, because none of the
open models pass: what the table shows is which oracle each model moves. A
run whose collision count falls to zero while its final distance grows has
not got closer to solving the task - it has stopped driving.

    python3 turn_summary.py [results/turns]
"""

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(d):
    for p in sorted(glob.glob(os.path.join(d, "*_turns.json"))):
        yield json.loads(open(p).read())


def main(d):
    rows = []
    print(f"{'실행':30} {'턴':>3} {'통과':>5}  {'회차별 충돌':38} {'회차별 최종거리 m'}")
    print("-" * 132)
    for log in load(d):
        stem = f"{log['tag']}_{log['track']}{log['sample']}"
        cols, dists = [], []
        for t in log["turns"]:
            if t.get("verdict") == "static_dq":
                cols.append("DQ"); dists.append("-")
            elif t.get("runtime_error"):
                cols.append("ERR"); dists.append("-")
            else:
                c = t.get("collisions")
                cols.append("-" if c is None else str(c))
                dd = t.get("final_dist_m")
                dists.append("-" if dd is None else f"{dd:.1f}")
        passed = log.get("passed_at_turn")
        print(f"{stem:30} {len(log['turns']):3d} {(str(passed) if passed else 'X'):>5}  "
              f"{' '.join(f'{c:>4}' for c in cols):38} {' '.join(f'{x:>5}' for x in dists)}")
        gen = [t for t in log["turns"] if t.get("tok_per_s")]
        rows.append(dict(
            run=stem, model=log["model"], endpoint=log["endpoint"],
            track=log["track"], turns=len(log["turns"]),
            passed_at_turn=passed or "",
            collisions_by_turn="|".join(cols), final_dist_by_turn="|".join(dists),
            mean_tok_per_s=round(sum(t["tok_per_s"] for t in gen) / len(gen), 1) if gen else "",
            total_completion_tokens=sum(t.get("completion_tokens") or 0 for t in log["turns"]),
            total_gen_s=round(sum(t.get("gen_wall_s") or 0 for t in log["turns"]), 1)))
    if not rows:
        print("턴 기록이 없습니다."); return
    out = os.path.join(d, "turns.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)}개 실행 -> {out}")
    print("\n생성 비용")
    print(f"{'실행':30} {'평균 tok/s':>10} {'총 출력 tok':>12} {'총 생성 s':>10}")
    for r in rows:
        print(f"{r['run']:30} {r['mean_tok_per_s']:>10} "
              f"{r['total_completion_tokens']:>12} {r['total_gen_s']:>10}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "turns"))
