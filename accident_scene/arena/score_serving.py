"""Summarise the serving-side numbers of every generation, per model and track.

The raw API responses carry more than the code they produced: token counts,
how much of the output was reasoning rather than answer, and how long the
request took. Those are what the paper reports about the serving side - not
power, which on the NPU box moves *inversely* with load and cannot be read as
consumption (see the system table's footnote).

Throughput here is end-to-end (tokens / request wall time), which folds in
queueing and the network hop. The backends' own prefill/decode rates are the
cleaner numbers and come from their logs; this is the client's view.

    python3 score_serving.py [results/generated]   -> table + serving.csv
"""

import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(d):
    for f in sorted(glob.glob(os.path.join(d, "*_raw.json"))):
        r = json.load(open(f))
        u = r.get("usage") or {}
        b = r.get("_bench") or {}
        name = os.path.basename(f)[: -len("_raw.json")]
        track = name[-2]                      # ..._a0 / ..._b3
        tag = name[: -3]
        ct = u.get("completion_tokens") or 0
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        content = (r["choices"][0]["message"].get("content") or "")
        yield dict(
            tag=tag, track=track, sample=name[-1],
            accel="npu" if tag.startswith("npu_") else "gpu",
            prompt_tokens=u.get("prompt_tokens") or 0,
            cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            completion_tokens=ct, reasoning_tokens=rt,
            reasoning_frac=round(rt / ct, 3) if ct else None,
            code_chars=len(content),
            # chars per token separates prose from code: English prose runs
            # 3.5-5 chars/token, code 2.5-3.5. A model whose output is nearly
            # all code lands at the bottom of that range.
            chars_per_tok=round(len(content) / (ct - rt), 2) if ct - rt > 0 else None,
            finish=r["choices"][0].get("finish_reason"),
            wall_s=b.get("wall_s"),
            tok_per_s=round(ct / b["wall_s"], 1) if b.get("wall_s") else None,
            t_start=b.get("t_start_epoch"), t_end=b.get("t_end_epoch"),
            empty=ct == 0 or not content.strip())


def main(d):
    data = list(rows(d))
    if not data:
        print("no responses yet"); return
    g = defaultdict(list)
    for r in data:
        g[(r["accel"], r["tag"], r["track"])].append(r)

    def med(v):
        v = [x for x in v if x is not None]
        return round(st.median(v), 1) if v else None

    print(f"{'가속기':6} {'모델':26} {'트랙':4} {'n':>2} {'출력tok':>8} {'추론%':>6} "
          f"{'코드자':>7} {'자/tok':>7} {'초':>6} {'tok/s':>6} {'빈':>3}")
    print("-" * 100)
    for (accel, tag, track), rs in sorted(g.items()):
        name = tag[4:] if tag.startswith("npu_") else tag
        rf = [r["reasoning_frac"] for r in rs if r["reasoning_frac"] is not None]
        print(f"{accel:6} {name:26} {track:4} {len(rs):2d} "
              f"{med([r['completion_tokens'] for r in rs]):8} "
              f"{round(100*st.median(rf)) if rf else '-':>6} "
              f"{med([r['code_chars'] for r in rs]):7} "
              f"{med([r['chars_per_tok'] for r in rs]) or '-':>7} "
              f"{med([r['wall_s'] for r in rs]) or '-':>6} "
              f"{med([r['tok_per_s'] for r in rs]) or '-':>6} "
              f"{sum(r['empty'] for r in rs):3d}")

    out = os.path.join(d, "serving.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)
    print(f"\n{len(data)}건 -> {out}   (빈 응답 {sum(r['empty'] for r in data)}건)")


main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "generated"))
