"""Slide figure: the one NPU pass (실험 8 T3 a4) — turns 6, 7, 8 over the arena.

Same frame as paper Fig. 1 (north to the right: horizontal = y, vertical = x
increasing downward), drawn from layout.json / arena_map.json and the three
Isaac trajectory files. Low boxes (below the z = 0.35 m scan plane) are the
orange hatched ones; tall boxes are neutral so the three runs keep their own
colours (validated categorical trio, plus line style and an end label each).

    <python with matplotlib> analysis/fig_transfer_pass.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.patches import Circle, Rectangle     # noqa: E402
import numpy as np                                   # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "figs"
BG, INK, INK2, MUTED, STRUCT, TALL, LOW = "#F7F6F2", "#14213D", "#3B4556", "#6B7280", "#D9D6CC", "#B8C2D3", "#EB6834"
RUNS = [("t6", "#E87BA4", (0, (1.2, 1.6)), 2.4, "6회차: box_04 충돌"),
        ("t7", "#4A3AA7", (0, (5, 3)), 2.4, "7회차: 시간 초과 (6.83 m 남음)"),
        ("t8", "#2A78D6", "-", 3.4, "8회차: 통과 0.45 m · 33.7 s")]

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name})
plt.rcParams.update({"font.family": cjk[0] if cjk else "sans-serif", "font.size": 17,
                     "axes.unicode_minus": False, "hatch.linewidth": 1.2, "svg.fonttype": "path"})


def traj(variant):
    rows = list(csv.DictReader(open(HERE / "results" / f"{variant}_trajectory.csv")))
    return np.array([[float(r["x"]), float(r["y"])] for r in rows])


def main():
    L = json.loads((HERE / "layout.json").read_text())
    M = json.loads((HERE / "arena_map.json").read_text())
    low = {i for i, b in enumerate(L["boxes"]) if b["height"] < 0.35}
    fig = plt.figure(figsize=(10.6, 7.6), facecolor=BG)
    ax = fig.add_axes([0.07, 0.13, 0.91, 0.85], facecolor=BG)
    g = np.array(M["grid"], dtype=float)
    img = np.ma.masked_where(g.T == 0, g.T)
    x0, y0, s = M["x0"], M["y0"], M["step"]
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap([STRUCT]), origin="upper",
              extent=[y0, y0 + g.shape[0] * s, x0 + g.shape[1] * s, x0], interpolation="nearest", zorder=1)
    for i, b in enumerate(L["boxes"]):
        lo = i in low
        ax.add_patch(Rectangle((b["y"] - b["depth"] / 2, b["x"] - b["width"] / 2), b["depth"], b["width"],
                               facecolor=LOW if lo else TALL, edgecolor=BG, lw=1.0,
                               hatch="///" if lo else None, zorder=3))
        if lo:
            off = {6: (-1.25, 0.0)}.get(i, (0.0, 1.3))
            ax.text(b["y"] + off[0], b["x"] + off[1], f"{i:02d}", ha="center", va="center",
                    fontsize=15, color="#B4461F", fontweight="bold", zorder=5)
    st, go = L["start"], L["goal"]
    ax.add_patch(Circle((go["y"], go["x"]), L["criteria"]["position_tolerance_m"], fill=False,
                        ls=(0, (2, 1.5)), lw=1.4, ec=INK, zorder=6))
    for key, color, ls, lw, label in RUNS:
        p = traj(f"npu_gpt_oss_120b_t3_a4_{key}")
        ax.plot(p[:, 1], p[:, 0], ls=ls, lw=lw, color=color, zorder=8 if key == "t8" else 7,
                solid_capstyle="round", dash_capstyle="round")
        ex, ey = p[-1]
        # end marker carries the outcome: x = contact, open = timeout, filled = pass (run names are in the slide legend)
        mk = {"t6": dict(marker="X", ms=15, mfc=color, mec=BG, mew=1.5),
              "t7": dict(marker="o", ms=13, mfc=BG, mec=color, mew=3),
              "t8": dict(marker="o", ms=13, mfc=color, mec=BG, mew=1.5)}[key]
        ax.plot(ey, ex, zorder=9, **mk)
    ax.plot(st["y"], st["x"], "o", ms=11, mfc=BG, mec=INK, mew=2, zorder=9)
    ax.text(st["y"] + 0.4, st["x"] - 1.0, "출발", fontsize=16, color=INK, ha="left", va="center")
    ax.text(go["y"] + 0.9, go["x"] - 1.4, "목표", fontsize=16, color=INK, ha="left", va="center")
    ax.set_xlim(-13.2, 19.2)
    ax.set_ylim(10.4, -11.4)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m] (북 →)", fontsize=15, color=INK2, labelpad=2)
    ax.set_ylabel("x [m] (동 ↓)", fontsize=15, color=INK2, labelpad=2)
    ax.tick_params(labelsize=14, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    h = [Rectangle((0, 0), 1, 1, fc=TALL, ec=BG), Rectangle((0, 0), 1, 1, fc=LOW, ec=BG, hatch="///"),
         Rectangle((0, 0), 1, 1, fc=STRUCT, ec="none")]
    ax.legend(h, ["상자 (1.0 m, 스캔에 잡힘)", "낮은 상자 (0.18–0.25 m, 스캔 평면 아래)", "고정 구조물"],
              loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=3, fontsize=14, frameon=False,
              handlelength=1.4, columnspacing=1.2)
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"transfer_t3a4_traj.{ext}", dpi=200, facecolor=BG)
    print("wrote", OUT / "transfer_t3a4_traj.png")


if __name__ == "__main__":
    main()
