"""실험 8 (Claude 절차의 NPU 이식) — T0/T3/T2 결과를 회차 기록에서 그대로 집계한다.

Every number comes from the saved *_turns.json; nothing is typed in by hand.
Failure types are fixed before reading the outcome: a turn is pass, static_dq,
empty_response, runtime_error, a collision (low box = 04/05/06/13, the four below
the scan plane; tall box = any other arena box; other = any other prim), or a
timeout without contact.

    python3 analysis/transfer_summary.py [--json results/analysis/transfer_summary.json]
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LOW = {"arena_box_04", "arena_box_05", "arena_box_06", "arena_box_13"}
CONDS = [
    ("T0", "results/turns/cap8/npu_gpt_oss_120b_a{}_turns.json", range(5)),
    ("T3", "results/turns/transfer_t3/npu_gpt_oss_120b_t3_a{}_turns.json", range(5)),
    ("T2", "results/turns/transfer_t2/npu_gpt_oss_120b_t2_a{}_turns.json", range(3)),
    ("T2P", "results/turns/transfer_t2p/npu_gpt_oss_120b_t2p_a{}_turns.json", range(3)),
]
TYPES = ["pass", "low_box", "tall_box", "other_prim", "timeout", "static_dq", "empty_response", "runtime_error"]


def kind(t):
    v = t.get("verdict")
    if v in ("pass", "static_dq", "empty_response"):
        return v
    if t.get("runtime_error"):
        return "runtime_error"
    parts = [p.rsplit("/", 1)[-1] for p in (t.get("partners") or [])]
    if parts or (t.get("collisions") or 0) > 0:
        if any(p in LOW for p in parts):
            return "low_box"
        if any(p.startswith("arena_box_") for p in parts):
            return "tall_box"
        return "other_prim"
    return "timeout"


def med(xs):
    xs = [x for x in xs if x is not None]
    return (round(st.median(xs), 1), round(min(xs), 1), round(max(xs), 1)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "results/analysis/transfer_summary.json"))
    a = ap.parse_args()
    out = {}
    for cond, pat, samples in CONDS:
        runs = []
        for i in samples:
            d = json.loads((HERE / pat.format(i)).read_text())
            turns = d["turns"]
            ks = [kind(t) for t in turns]
            last = turns[-1]
            end = ("pass" if d.get("passed_at_turn") else
                   "empty_response" if last.get("verdict") == "empty_response" else
                   f"fail_{len(turns)}")
            runs.append({
                "sample": i, "turns": len(turns), "passed_at_turn": d.get("passed_at_turn"), "end": end,
                "kinds": ks,
                "dist": [t.get("final_dist_m") for t in turns],
                "partners": [[p.rsplit("/", 1)[-1] for p in (t.get("partners") or [])] for t in turns],
                "best_dist": min((t["final_dist_m"] for t in turns if t.get("final_dist_m") is not None), default=None),
                "gen_s": [t.get("gen_wall_s") for t in turns],
                "reasoning_tokens": [t.get("reasoning_tokens") for t in turns],
                "completion_tokens": [t.get("completion_tokens") for t in turns],
                "finish_length": sum(1 for t in turns if t.get("finish") == "length"
                                     or any(c.get("finish") == "length" for c in t.get("calls") or [])),
                "tool_calls": [t.get("tool_calls") for t in turns] if cond in ("T2", "T2P") else None,
            })
        allk = [k for r in runs for k in r["kinds"]]
        judged = [k for k in allk if k not in ("empty_response",)]
        out[cond] = {
            "runs": runs,
            "passed_runs": sum(1 for r in runs if r["passed_at_turn"]),
            "n_runs": len(runs),
            "turns_total": len(allk),
            "turns_graded": sum(1 for k in allk if k not in ("static_dq", "empty_response")),
            "kinds": {k: allk.count(k) for k in TYPES},
            "gen_s_per_turn": med([g for r in runs for g in r["gen_s"]]),
            "reasoning_tokens_per_turn": med([x for r in runs for x in r["reasoning_tokens"]]),
            "finish_length_turns": sum(r["finish_length"] for r in runs),
        }
        if cond in ("T2", "T2P"):
            tc = [x for r in runs for x in r["tool_calls"] if x is not None]
            out[cond]["tool_calls_per_turn_mean"] = round(sum(tc) / len(tc), 2)
            out[cond]["turns_with_tools"] = sum(1 for x in tc if x)
            out[cond]["tool_calls_total"] = sum(tc)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("| 조건 | 통과 실행 | 회차 수 | " + " | ".join(TYPES) + " | 회차당 생성 s 중앙값(범위) | 추론 tok 중앙값 |")
    print("|---|---|---|" + "---|" * len(TYPES) + "---|---|")
    for cond, s in out.items():
        print(f"| {cond} | {s['passed_runs']}/{s['n_runs']} | {s['turns_total']} | "
              + " | ".join(str(s["kinds"][k]) for k in TYPES)
              + f" | {s['gen_s_per_turn']} | {s['reasoning_tokens_per_turn']} |")
    print()
    for cond, s in out.items():
        for r in s["runs"]:
            seq = " ".join(f"{k[:4]}{'' if not p else '(' + ','.join(x.replace('arena_box_', 'b') for x in p) + ')'}"
                           for k, p in zip(r["kinds"], r["partners"]))
            print(f"{cond} a{r['sample']}: end={r['end']:<14} best={r['best_dist']}  {seq}")
    for c in ("T2", "T2P"):
        if c in out:
            s = out[c]
            print(f"{c} 도구: 총 {s['tool_calls_total']}회, 도구를 쓴 회차 {s['turns_with_tools']}/{s['turns_total']}, 회차당 평균 {s['tool_calls_per_turn_mean']}")
    print(f"출력 한도 도달(finish=length) 회차: " + ", ".join(f"{c} {s['finish_length_turns']}" for c, s in out.items()))
    print(f"-> {a.json}")


if __name__ == "__main__":
    main()
