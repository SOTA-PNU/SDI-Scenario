"""Slide figure: the equal-input Claude controller's real Isaac trajectory (REPORT.md 실험 7).

Same map conventions as analysis/fig_paper_figs.py (north to the right; blue =
boxes the default scan sees, orange hatched = sub-plane obstacles, grey = fixed
structures), drawn large for a 16:9 slide.

    <python with matplotlib> analysis/fig_claude_traj.py [variant]
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import font_manager                 # noqa: E402
from matplotlib.patches import Rectangle, Circle    # noqa: E402
import numpy as np                                  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, INK2, STRUCT, PATH = "#2a78d6", "#eb6834", "#14213d", "#52514e", "#cfcdc6", "#14213d"
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name})
plt.rcParams.update({"font.family": cjk[0] if cjk else "sans-serif", "font.size": 12,
                     "axes.unicode_minus": False, "hatch.linewidth": 1.0})


def main(variant="claude_eq_a0"):
    L = json.loads((HERE / "layout.json").read_text())
    M = json.loads((HERE / "arena_map.json").read_text())
    R = json.loads((HERE / "results" / f"{variant}_result.json").read_text())
    rows = list(csv.DictReader(open(HERE / "results" / f"{variant}_trajectory.csv")))
    xs = np.array([float(r["x"]) for r in rows])
    ys = np.array([float(r["y"]) for r in rows])
    low = {i for i, b in enumerate(L["boxes"]) if b["height"] < 0.35}

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.10)
    g = np.array(M["grid"], dtype=float)
    img = np.ma.masked_where(g.T == 0, g.T)
    x0, y0, s = M["x0"], M["y0"], M["step"]
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap([STRUCT]), origin="upper",
              extent=[y0, y0 + g.shape[0] * s, x0 + g.shape[1] * s, x0], interpolation="nearest")
    for i, b in enumerate(L["boxes"]):
        ax.add_patch(Rectangle((b["y"] - b["depth"] / 2, b["x"] - b["width"] / 2), b["depth"], b["width"],
                               facecolor=ORANGE if i in low else BLUE, edgecolor="white", lw=0.8,
                               hatch="////" if i in low else None, zorder=3))
        if i in low:
            off = {6: (-1.15, 0.0)}.get(i, (0.0, 1.25))
            ax.text(b["y"] + off[0], b["x"] + off[1], f"{i:02d}", ha="center", va="center",
                    fontsize=11, color=ORANGE, fontweight="bold", zorder=5)
    ax.plot(ys, xs, color=PATH, lw=2.2, zorder=6, solid_capstyle="round")
    st, go = L["start"], L["goal"]
    ax.add_patch(Circle((go["y"], go["x"]), L["criteria"]["position_tolerance_m"], fill=False,
                        ls=(0, (2, 1.5)), lw=1.2, ec=INK, zorder=7))
    ax.plot(st["y"], st["x"], "o", ms=8, mfc="white", mec=INK, mew=1.6, zorder=8)
    ax.plot(ys[-1], xs[-1], "s", ms=7, mfc=INK, mec=INK, zorder=8)
    ax.text(st["y"] + 0.4, st["x"] - 0.9, "출발", fontsize=13, ha="left", va="center", color=INK)
    m = R["metrics"]
    ax.annotate(f"정지: 목표에서 {m['final_dist_m']:.2f} m\n도달 {m['time_to_goal_s']} s · 접촉 {m['collision_count']}",
                xy=(ys[-1] - 0.3, xs[-1]), xytext=(5.0, 6.6), fontsize=12, ha="left", va="top",
                color=INK, linespacing=1.3, zorder=9)
    ax.set_xlim(-13.2, 19.2)
    ax.set_ylim(10.4, -11.4)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m] (북 →)")
    ax.set_ylabel("x [m] (동 ↓)")
    ax.tick_params(labelsize=11)
    out = HERE / "figs" / f"{variant}_isaac_traj.png"
    fig.savefig(out)
    print("wrote", out, "| points", len(xs), "| final", m["final_dist_m"], m["time_to_goal_s"])


if __name__ == "__main__":
    main(*sys.argv[1:])
