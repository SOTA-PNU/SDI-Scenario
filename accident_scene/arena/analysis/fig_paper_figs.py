"""Figures 1 and 2 of the IEMEK paper draft (PAPER_IEMEK.md).

Fig. 1 (one column): (a) top view of the arena drawn from layout.json and the
fixed-structure occupancy grid in arena_map.json, north to the right;
(b) schematic cross-section of the scan plane against a low and a tall box.
Fig. 2 (two columns): the turn loop and the generator conditions.

Colors are the dataviz reference categorical slots 1-2 (blue = boxes the
default scan sees, orange = sub-plane obstacles; validated: CVD dE 24.7), and
the sub-plane boxes also carry a hatch so the figure survives grey printing.

    <python with matplotlib> analysis/fig_paper_figs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np                                   # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "figs"
MM = 1 / 25.4
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, STRUCT = "#0b0b0b", "#52514e", "#8a8984", "#cfcdc6"

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name})
plt.rcParams.update({"font.family": cjk[0] if cjk else "sans-serif", "font.size": 6,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "svg.fonttype": "path",
                     "hatch.linewidth": 0.6, "axes.unicode_minus": False})


def fig1(width_mm=85, height_mm=92, fs=1.0, legend_ncol=3, panel_b=True):
    """fs scales every font size; at width_mm=67.4 (the IEMEK template column, 191 pt)
    the figure is placed unscaled, so the sizes below are the printed sizes."""
    L = json.loads((HERE / "layout.json").read_text())
    M = json.loads((HERE / "arena_map.json").read_text())
    low = {i for i, b in enumerate(L["boxes"]) if b["height"] < 0.35}

    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    narrow = width_mm < 80      # IEMEK column: wider margins, shorter labels, legend in one row
    if narrow and not panel_b:  # top view only (3-page version): map above ticks and a two-row legend
        ga = fig.add_axes([0.185, 13.0 / height_mm, 0.75, 34.0 / height_mm])  # 50.6 mm wide map
        gb = None
    elif narrow:
        ga = fig.add_axes([0.135, 0.441, 0.845, 0.525])
        gb = fig.add_axes([0.135, 0.062, 0.845, 0.233])
    else:
        ga = fig.add_axes([0.10, 0.415, 0.88, 0.57])
        gb = fig.add_axes([0.10, 0.075, 0.88, 0.235])

    # (a) north to the right: horizontal = y, vertical = x increasing downward
    g = np.array(M["grid"], dtype=float)           # rows = y, cols = x
    img = np.ma.masked_where(g.T == 0, g.T)        # rows = x, cols = y
    x0, y0, s = M["x0"], M["y0"], M["step"]
    ga.imshow(img, cmap=matplotlib.colors.ListedColormap([STRUCT]), origin="upper",
              extent=[y0, y0 + g.shape[0] * s, x0 + g.shape[1] * s, x0],
              interpolation="nearest", zorder=1)
    for i, b in enumerate(L["boxes"]):
        r = Rectangle((b["y"] - b["depth"] / 2, b["x"] - b["width"] / 2), b["depth"], b["width"],
                      facecolor=ORANGE if i in low else BLUE, edgecolor="white", lw=0.4,
                      hatch="//////" if i in low else None, zorder=3)
        ga.add_patch(r)
        dy, dx = (0.0, -1.05) if i not in low else (0.0, 1.15)
        off = {3: (0.95, 0.0), 6: (-1.05, 0.0)}   # long barrier 03 and wall-flush 06: label beside
        dy, dx = off.get(i, (dy, dx))
        ga.text(b["y"] + dy, b["x"] + dx, f"{i:02d}", ha="center", va="center", fontsize=4.6 * fs,
                color=ORANGE if i in low else INK2,
                fontweight="bold" if i in low else "normal", zorder=5)
    st, go = L["start"], L["goal"]
    ga.plot([st["y"], go["y"]], [st["x"], go["x"]], ls=(0, (3, 2)), lw=0.5, color=MUTED, zorder=2)
    for p, name in ((st, "출발"), (go, "목표")):
        ga.annotate("", xy=(p["y"] + 2.2 * np.sin(p["yaw"]), p["x"] + 2.2 * np.cos(p["yaw"])),
                    xytext=(p["y"], p["x"]), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK,
                                                             mutation_scale=5), zorder=6)
        ga.plot(p["y"], p["x"], "o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=7)
    ga.add_patch(Circle((go["y"], go["x"]), L["criteria"]["position_tolerance_m"], fill=False,
                        ls=(0, (1.5, 1)), lw=0.5, ec=INK, zorder=6))
    ga.text(st["y"] - 0.6, st["x"] + 1.5, "출발", fontsize=5.5 * fs, ha="center", va="center", color=INK)
    if narrow:                  # the radius is in the text; "목표" alone clears box 06
        ga.text(go["y"] + 0.1, go["x"] - 1.1, "목표", fontsize=5.5 * fs, ha="center", va="bottom", color=INK)
    else:
        ga.text(go["y"] + 0.2, go["x"] - 1.2, "목표\n(반경 0.75 m)", fontsize=5.2 * fs, ha="center",
                va="bottom", color=INK, linespacing=1.15)
    ga.set_xlim(-13.2, 19.2)
    ga.set_ylim(10.4, -11.4)
    ga.set_aspect("equal")
    ga.set_xlabel("y [m] (북 →)", labelpad=1, fontsize=6 * fs)
    ga.set_ylabel("x [m] (동 ↓)", labelpad=1, fontsize=6 * fs)
    ga.set_xticks(range(-10, 20, 5))
    ga.set_yticks(range(-10, 11, 5))
    ga.tick_params(labelsize=5.5 * fs, pad=1)
    h = [Rectangle((0, 0), 1, 1, fc=BLUE, ec="white"),
         Rectangle((0, 0), 1, 1, fc=ORANGE, ec="white", hatch="//////"),
         Rectangle((0, 0), 1, 1, fc=STRUCT, ec="none")]
    names = (["상자 (1.0 m)", "저평면 장애물 (0.18–0.25 m)", "고정 구조물"] if narrow
             else ["상자 10개 (1.0 m)", "저평면 장애물 4개 (0.18–0.25 m)", "고정 구조물"])
    if narrow and not panel_b:  # explain the dashed start-goal line and the goal circle too
        h += [Line2D([0], [0], ls=(0, (3, 2)), lw=0.5, color=MUTED),
              Line2D([0], [0], marker="o", ls="none", ms=3.6, mfc="none", mec=INK, mew=0.5)]
        names = names + ["직선 경로", "목표 반경 0.75 m"]
        order = [0, 3, 1, 4, 2]  # legend fills column-major: row 1 keeps the old three, row 2 the new two
        h, names = [h[i] for i in order], [names[i] for i in order]
    ga.legend(h, names, loc="upper center",
              bbox_to_anchor=(0.5, (-0.175 if not panel_b else -0.155) if narrow else -0.125),
              ncol=legend_ncol, fontsize=5.0 * fs,
              frameon=False, handlelength=1.2, columnspacing=0.9, handletextpad=0.35,
              borderaxespad=0)
    if gb is None:
        return fig
    ga.text(-0.155 if narrow else -0.10, 1.04 if narrow else 1.02, "(a)", transform=ga.transAxes,
            fontsize=6.5 * fs, fontweight="bold", va="top")

    # (b) cross-section, schematic in x (not to scale), true heights in z
    gb.axhline(0, color=INK, lw=0.6)
    gb.add_patch(FancyBboxPatch((0.15, 0.07), 0.62, 0.30, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=INK, lw=0.6))
    for wx in (0.27, 0.65):
        gb.add_patch(Circle((wx, 0.06), 0.055, fc=INK2, ec="none"))
    gb.plot(0.77, 0.35, "o", ms=2.6, color=INK)
    gb.annotate("", xy=(4.08, 0.35), xytext=(0.80, 0.35),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, mutation_scale=5))
    gb.plot(4.08, 0.35, marker="x", ms=4, mew=0.9, color=INK)
    gb.add_patch(Rectangle((1.75, 0), 0.9, 0.20, fc=ORANGE, ec="white", lw=0.4, hatch="//////"))
    gb.add_patch(Rectangle((4.1, 0), 0.75, 1.0, fc=BLUE, ec="white", lw=0.4))
    gb.axhline(0.35, color=MUTED, lw=0.5, ls=(0, (1.5, 1.5)), zorder=0)
    gb.text(4.98, 0.39, "스캔 평면\nz = 0.35 m", fontsize=5.1 * fs, ha="left", va="bottom",
            color=INK2, linespacing=1.15)
    gb.text(2.45 if narrow else 2.2, 0.42 if narrow else 0.43, "저평면 장애물 0.18–0.25 m\n(레이가 위로 지나감)",
            fontsize=5.1 * fs, ha="center", va="bottom", color=INK, linespacing=1.15)
    gb.text(4.475, 1.04, "상자 1.0 m", fontsize=5.2 * fs, ha="center", va="bottom", color=INK)
    gb.text(4.0, 0.29, "검출", fontsize=5.1 * fs, ha="right", va="top", color=INK)
    gb.text(0.15, 0.80 if narrow else 0.72, "로봇 · 레이 팬\n(61빔, 180°, 최대 6 m)", fontsize=5.1 * fs,
            ha="left", va="bottom", color=INK, linespacing=1.15)
    gb.set_xlim(0, 6.0)
    gb.set_ylim(0, 1.25)
    gb.set_yticks([0, 0.35, 1.0])
    gb.set_yticklabels(["0", "0.35", "1.0"])
    gb.set_xticks([])
    gb.set_ylabel("높이 z [m]", labelpad=1, fontsize=6 * fs)
    gb.tick_params(labelsize=5.5 * fs, pad=1)
    for side in ("top", "right"):
        gb.spines[side].set_visible(False)
    gb.text(-0.155 if narrow else -0.09, 1.10 if narrow else 1.05, "(b)", transform=gb.transAxes,
            fontsize=6.5 * fs, fontweight="bold", va="top")
    gb.text(1.0, -0.06, "개념도 (수평 축척 무시)", transform=gb.transAxes, fontsize=4.8 * fs,
            ha="right", va="top", color=MUTED)
    return fig


def box(ax, x, y, w, h, text, fc="white", ec=INK2, lw=0.6, fs=5.6, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight=weight, zorder=4, linespacing=1.25)


def arrow(ax, p, q, ls="-", lw=0.7, color=INK2):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=lw, ls=ls, color=color, mutation_scale=5,
                                shrinkA=0, shrinkB=0))


def fig2():
    fig = plt.figure(figsize=(175 * MM, 66 * MM))
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    yt, ht = 21.5, 9.5
    # the fixed harness the model cannot touch
    ax.add_patch(FancyBboxPatch((49.3, yt - 2.4), 50.2, ht + 4.6, boxstyle="round,pad=0,rounding_size=1",
                                fc="#f0efec", ec="none", zorder=1))
    ax.text(74.4, yt + ht + 1.25, "고정 하네스 (제출 파일 밖)", ha="center", va="center",
            fontsize=5.4, color=INK2, zorder=2)
    box(ax, 0.8, yt, 14.2, ht, "입력 메시지 1개\n트랙 프롬프트 + 기반 파일\n(트랙 A 6,961자)")
    box(ax, 18.0, yt, 13.0, ht, "생성기\n(아래 조건 중 하나)", weight="bold")
    box(ax, 34.0, yt, 12.5, ht, "제출 코드\n[EDIT REGION]\n수정본")
    box(ax, 51.0, yt, 14.8, ht, "정적 검증\n① 구문 ② 고정부 동일\n③ 트랙 B 규칙 ④ controller()")
    box(ax, 68.3, yt, 14.5, ht, "물리 실행\nIsaac Sim 4.5 · 60 Hz\n직렬 채점", ec=INK, lw=0.8)
    box(ax, 85.3, yt, 13.8, ht, "판정\n위치 ≤ 0.75 m\nyaw ≤ 0.26 rad · 접촉 0")
    mid = yt + ht / 2
    for a, b in ((15.0, 18.0), (31.0, 34.0), (46.5, 51.0), (65.8, 68.3), (82.8, 85.3)):
        arrow(ax, (a, mid), (b, mid))
    # feedback: result JSON (solid) and static disqualification (dashed), both back to the generator
    top = yt + ht
    ax.plot([92.2, 92.2, 26.0, 26.0], [top, 37.6, 37.6, top + 0.15], color=INK2, lw=0.7, zorder=2)
    arrow(ax, (26.0, top + 1.0), (26.0, top))
    ax.text(59.0, 38.4, "실패: 판정 JSON 피드백 (트랙 A는 장애물 배치 블록만 제외) · 같은 대화 · 상한 8회차",
            ha="center", va="bottom", fontsize=5.4, color=INK)
    ax.plot([58.4, 58.4, 23.0, 23.0], [top + 2.3, 34.6, 34.6, top + 0.15], color=INK2, lw=0.7,
            ls=(0, (2.5, 1.5)), zorder=2)
    arrow(ax, (23.0, top + 1.0), (23.0, top), ls=(0, (2.5, 1.5)))
    ax.plot([58.4, 58.4], [top, top + 2.3], color=INK2, lw=0.7, ls=(0, (2.5, 1.5)), zorder=2)
    ax.text(40.7, 35.3, "정적 실격: 달라진 행 안내", ha="center", va="bottom", fontsize=5.2,
            color=INK2)
    ax.text(92.2, yt - 3.3, "통과 → 종료", ha="center", va="center", fontsize=5.4, color=INK,
            fontweight="bold")
    arrow(ax, (92.2, yt), (92.2, yt - 2.1))
    # generator conditions, joined to the generator by a bus
    yb, hb = 1.2, 12.0
    gens = [(0.8, "GPU 서빙\nvLLM 0.29.0 · RTX PRO 6000 1장\ngpt-oss-120b · Qwen3-32B-FP8\nEXAONE-4.0-32B-FP8"),
            (25.6, "NPU 서빙\nfuriosa-llm 2026.3.0 · RNGD 4장\n같은 모델명\n(입력·피드백 동일)"),
            (50.4, "Claude Code 에이전트 (Opus 5.5)\nbubblewrap 격리 · 빈 작업 폴더\n기본 도구 · 웹·MCP 차단\n작업 절차 기록 (이벤트·파일)"),
            (75.2, "절차 요인 이식 (NPU gpt-oss-120b)\nT3 숙고 (reasoning_effort high)\nT2 = T3 + Python 실행 도구\nT2P = T2의 안내 대신 11단계 작업 순서 지시")]
    for x, t in gens:
        box(ax, x, yb, 24.0, hb, t, fs=5.3, ls="-" if "계획" not in t else (0, (3, 1.5)))
    bus = yb + hb + 2.4
    ax.plot([12.8, 87.2], [bus, bus], color=INK2, lw=0.7, zorder=2)
    for x, _ in gens:
        ax.plot([x + 12.0, x + 12.0], [yb + hb, bus], color=INK2, lw=0.7, zorder=2)
    arrow(ax, (24.5, bus), (24.5, yt))
    return fig



def fig2_compact():
    """One-column version of Fig. 2 for the 4-page paper: the turn loop and the generators."""
    fig = plt.figure(figsize=(85 * MM, 58 * MM))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    fs = 5.4
    box(ax, 2, 47, 33, 18, "입력 메시지 1개\n트랙 프롬프트\n+ 기반 파일", fs=fs)
    box(ax, 50, 41, 48, 27, "생성기 (조건 하나)\n공개 모델 3종 (GPU·NPU 서빙)\nClaude Code 에이전트\n이식 조건 T3·T2·T2P (NPU)",
        fs=fs, weight="bold")
    box(ax, 66, 6, 32, 22, "정적 검증\n구문 · 구역 밖 동일\n트랙 B 규칙\ncontroller() 정의", fs=fs)
    box(ax, 34, 6, 28, 22, "물리 실행\nIsaac Sim 4.5\n60 Hz · 직렬", fs=fs, lw=0.9, ec=INK)
    box(ax, 2, 6, 28, 22, "판정\n위치 ≤ 0.75 m\nyaw ≤ 0.26 rad\n접촉 0", fs=fs)
    arrow(ax, (35, 56), (50, 56))                       # input -> generator
    arrow(ax, (78, 41), (78, 28))                       # generator -> static check
    arrow(ax, (66, 17), (62, 17))                       # static -> physics
    arrow(ax, (34, 17), (30, 17))                       # physics -> verdict
    # failure feedback: verdict -> generator (same conversation, cap 8)
    ax.plot([16, 16, 58], [28, 35.5, 35.5], color=INK2, lw=0.7, zorder=2)
    arrow(ax, (58, 35.5), (58, 41))
    ax.text(37, 36.6, "실패: 판정 결과 피드백 (같은 대화, 상한 8회차)", ha="center", va="bottom",
            fontsize=4.9, color=INK)
    # static disqualification: dashed back to the generator
    ax.plot([92, 92], [28, 33], color=INK2, lw=0.7, ls=(0, (2.5, 1.5)), zorder=2)
    arrow(ax, (92, 33), (92, 41), ls=(0, (2.5, 1.5)))
    ax.text(93.2, 34.5, "정적\n실격", ha="left", va="center", fontsize=4.6, color=INK2)
    arrow(ax, (16, 6), (16, 2.6))
    ax.text(19, 1.4, "통과 → 종료", ha="left", va="center", fontsize=5.0, color=INK, fontweight="bold")
    return fig

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # paper_fig1_task_iemek: drawn at the IEMEK column width (191 pt = 67.4 mm) so it is placed unscaled
    # paper_fig1_task_iemek_map: top view only, for the 3-page version
    for name, f in (("paper_fig1_task", fig1()), ("paper_fig1_task_iemek", fig1(67.4, 73, fs=1.12)),
                    ("paper_fig1_task_iemek_map", fig1(67.4, 47.3, fs=1.12, panel_b=False)),
                    ("paper_fig2_flow", fig2()), ("paper_fig2_flow_1col", fig2_compact())):
        for ext in ("png", "svg"):
            f.savefig(OUT / f"{name}.{ext}", dpi=600)
        print("wrote", OUT / f"{name}.png", OUT / f"{name}.svg")
