"""Draw the turn traces: collisions falling while the robot leaves the traps.

One panel per phenotype, because the claim is that the three shapes are
different shapes and not three samples of one. Collision count is drawn
against the left axis, distance to the nearest sub-plane box against the
right; when the two cross, the loop is trading the oracle it can read for the
region where the other oracle lives.

Hand-authored SVG so the figure prints at one column (88 mm) without a
rasteriser in the way.

    python3 analysis/fig3_exposure_collision.py [--out figs/fig3_exposure_collision.svg]
"""

import argparse
import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA = os.path.dirname(HERE)

W, PH = 300, 104                    # panel width, panel height
L, R, T = 38, 34, 16                # margins inside a panel
MAXC, MAXE = 360.0, 9.0             # collision cap (harness aborts at 360), metres
COL_C, COL_E, INK, MUTED = "#1F4B73", "#C1462B", "#101820", "#8892A0"
ORDER = [("sequential_closure", "오라클 순차 폐쇄"),
         ("stall_converge", "정지로 수렴"),
         ("oscillate", "진동")]


def panels(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["phenotype"]][r["run_id"]].append(r)
    return by


def xy(i, n, w):
    return L + (w - L - R) * (0 if n < 2 else i / (n - 1))


def series(rs, key, cap):
    out = []
    for r in rs:
        v = r[key]
        out.append(None if v in ("", "None") else min(float(v), cap))
    return out


def polyline(vals, top, cap, n):
    pts, seg, segs = [], [], []
    for i, v in enumerate(vals):
        if v is None:
            if len(seg) > 1:
                segs.append(seg)
            seg = []
            continue
        seg.append((xy(i, n, W), top + (PH - T) * (1 - v / cap)))
    if len(seg) > 1:
        segs.append(seg)
    for i, v in enumerate(vals):
        if v is not None:
            pts.append((xy(i, n, W), top + (PH - T) * (1 - v / cap)))
    return segs, pts


def draw(by, out):
    H = 30 + len(ORDER) * (PH + 26)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="\'Malgun Gothic\',\'Noto Sans KR\',sans-serif" '
         f'role="img" aria-label="phenotype 별 턴 궤적: 충돌 횟수와 저평면 박스까지의 '
         f'거리">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    s.append(f'<text x="{L}" y="13" font-size="9" fill="{COL_C}">■ 충돌 횟수 (좌)</text>'
             f'<text x="{L + 108}" y="13" font-size="9" fill="{COL_E}">■ 저평면 박스까지 '
             f'거리 m (우)</text>')
    y0 = 30
    for key, label in ORDER:
        runs = by.get(key, {})
        s.append(f'<text x="{L}" y="{y0 - 4}" font-size="9.5" fill="{INK}" '
                 f'font-weight="600">{label} (n={len(runs)})</text>')
        # frame + axes
        s.append(f'<line x1="{L}" y1="{y0}" x2="{L}" y2="{y0 + PH - T}" stroke="{INK}" '
                 f'stroke-width="0.8"/>')
        s.append(f'<line x1="{W - R}" y1="{y0}" x2="{W - R}" y2="{y0 + PH - T}" '
                 f'stroke="{INK}" stroke-width="0.8"/>')
        s.append(f'<line x1="{L}" y1="{y0 + PH - T}" x2="{W - R}" y2="{y0 + PH - T}" '
                 f'stroke="{INK}" stroke-width="0.8"/>')
        for frac, lab in ((0, "0"), (0.5, "180"), (1.0, "360")):
            yy = y0 + (PH - T) * (1 - frac)
            s.append(f'<text x="{L - 4}" y="{yy + 3}" font-size="8" fill="{COL_C}" '
                     f'text-anchor="end">{lab}</text>')
        for frac, lab in ((0, "0"), (0.667, "6.0"), (1.0, "9")):
            yy = y0 + (PH - T) * (1 - frac)
            s.append(f'<text x="{W - R + 4}" y="{yy + 3}" font-size="8" fill="{COL_E}">'
                     f'{lab}</text>')
        # max_range line: beyond it the sub-plane boxes are outside the sensor
        ymr = y0 + (PH - T) * (1 - 6.0 / MAXE)
        s.append(f'<line x1="{L}" y1="{ymr}" x2="{W - R}" y2="{ymr}" stroke="{COL_E}" '
                 f'stroke-width="0.7" stroke-dasharray="2 3" opacity="0.6"/>')
        s.append(f'<text x="{W - R - 2}" y="{ymr - 3}" font-size="7.6" fill="{COL_E}" '
                 f'text-anchor="end" opacity="0.8">max_range 6.0 m</text>')

        for rid, rs in sorted(runs.items()):
            rs.sort(key=lambda r: int(r["turn_idx"]))
            n = len(rs)
            hero = rid == "gpu_gpt_oss_120b_a4"
            wid = 1.9 if hero else 0.9
            op = 1.0 if hero else 0.5
            for vals, colr, cap in ((series(rs, "collision_count", MAXC), COL_C, MAXC),
                                    (series(rs, "trap_exposure_m", MAXE), COL_E, MAXE)):
                segs, pts = polyline(vals, y0, cap, n)
                for seg in segs:
                    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in seg)
                    s.append(f'<polyline points="{d}" fill="none" stroke="{colr}" '
                             f'stroke-width="{wid}" opacity="{op}"/>')
                for x, y in pts:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.8 if hero else 1.1}" '
                             f'fill="{colr}" opacity="{op}"/>')
            if hero:
                lx = xy(n - 1, n, W)
                s.append(f'<text x="{lx - 4}" y="{y0 + 10}" font-size="8" fill="{INK}" '
                         f'text-anchor="end" font-weight="600">통과 (5회차)</text>')
        s.append(f'<text x="{(L + W - R) / 2:.0f}" y="{y0 + PH - T + 13}" font-size="8" '
                 f'fill="{MUTED}" text-anchor="middle">회차 →</text>')
        y0 += PH + 26
    s.append("</svg>")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(s))
    # 88 mm column / W units = mm per unit; 1 pt = 0.3528 mm
    mm_per_unit = 88.0 / W
    smallest = 7.6
    print(f"-> {out}  ({W}x{H} 단위)")
    print(f"   88 mm 단 폭 기준 최소 글자 {smallest * mm_per_unit / 0.3528:.1f} pt "
          f"(권장 하한 6 pt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.path.join(ARENA, "results", "analysis",
                                                    "turn_trace.csv"))
    ap.add_argument("--out", default=os.path.join(ARENA, "figs",
                                                  "fig3_exposure_collision.svg"))
    a = ap.parse_args()
    draw(panels(list(csv.DictReader(open(a.trace)))), a.out)
