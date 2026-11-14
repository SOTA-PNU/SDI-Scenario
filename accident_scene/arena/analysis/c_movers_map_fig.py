"""Top view of the project C grading map: fixed colliders + the moving obstacles of a scenario.

Drawn in the same orientation as the recorded Isaac top-down view (north to the RIGHT, west UP):
screen x = world y, screen y = -world x. Colliders come from the measured scene listing
(a21_scene), movers from arena_movers.py, the dashed route is the A-4 seed (npu_gpt_oss_120b_pA4_a0)
with the points where a mover meets it (analysis/c_movers_design.py).

    ~/carter_ws/mamba/envs/isaacsim/bin/python analysis/c_movers_map_fig.py [scenario]
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene  # noqa: E402
import arena_movers as AM  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "v1"
OUT = ARENA / "results" / "analysis" / "figs" / f"c_movers_{NAME}_map.png"
from matplotlib import font_manager  # noqa: E402
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # Noto Sans CJK (has Hangul); registered by file, fontconfig names are not seen
font_manager.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
L = a21_scene.obstacle_listing(a21_scene.Scene())
W = L["walls_inner_face"]
src = (ARENA / "base_carter_run_a2.py").read_text()
MISSION = eval(src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
design = {d["name"]: d for d in json.loads((ARENA / "results" / "analysis" / f"c_movers_design_{NAME}.json").read_text())}
# label offset on screen (dx, dy) and alignment, chosen so no label covers a box, the start or the goal
LABEL = {"start_zone": (0.0, -0.75, "center", "top"), "west_corridor": (0.9, 0.0, "left", "center"),
         "center": (-0.6, 0.0, "right", "center"), "goal_west": (0.0, 0.9, "center", "bottom"),
         "goal_south": (-0.6, 0.0, "right", "center")}
KOR = {"start_zone": "출발 구역", "west_corridor": "서쪽 통로", "center": "가운데", "goal_west": "목표 서쪽", "goal_south": "목표 남쪽"}


def S(x, y):            # world -> screen (north right, west up)
    return y, -x


def rect(ax, x0, x1, y0, y1, **kw):
    sx0, sy0 = S(x1, y0)          # screen lower-left corner: max x (east, bottom), min y (south, left)
    ax.add_patch(Rectangle((sx0, sy0), y1 - y0, x1 - x0, **kw))


fig, ax = plt.subplots(figsize=(18.5, 10.5), dpi=150)
rect(ax, W["x"][0], W["x"][1], W["y"][0], W["y"][1], fill=False, lw=2.5, ec="#3a3a3a")
for name, scored, x0, x1, y0, y1, *_ in L["obstacles"]:
    if name.startswith("arena_box_"):
        continue
    rect(ax, x0, x1, y0, y1, fc="#e0a050" if scored else "#c8c8c8", ec="#8a6a3a" if scored else "#9a9a9a", lw=0.6)
for i, b in enumerate(MISSION["boxes"]):
    rect(ax, b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2,
         fc="#d23c2a", ec="#7a1a10", lw=0.8)
    ax.text(*S(b["x"], b["y"]), f"{i:02d}", ha="center", va="center", fontsize=8, color="white", weight="bold")

# A-4 seed route
wps = [(-9.0, -10.8), (-5.0, -10.8), (-5.0, 5.8), (0.0, 5.8), (0.0, 12.0), (1.5, 12.0), (1.5, 16.5), (8.0, 16.5)]
ax.plot(*zip(*[S(x, y) for x, y in wps]), ls="--", lw=1.6, color="#555555", zorder=3)

# movers
for k, m in enumerate(AM.scenario(NAME)):
    (x0, y0), (x1, y1) = m["p0"], m["p1"]
    sx, sy, sz = m["size"]
    ax.add_patch(FancyArrowPatch(S(x0, y0), S(x1, y1), arrowstyle="<|-|>", mutation_scale=14, lw=2.2,
                                 color="#1f5fd6", zorder=5))
    for (px, py), alpha in (((x0, y0), 0.25), ((x1, y1), 0.25), (AM.position(m, 0.0), 0.95)):
        rect(ax, px - sx / 2, px + sx / 2, py - sy / 2, py + sy / 2, fc="#1f5fd6", ec="#0b2f75", lw=0.8,
             alpha=alpha, zorder=6)
    d = design[m["name"]]
    period = 2 * math.hypot(x1 - x0, y1 - y0) / m["speed"]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy, ha, va = LABEL.get(m["name"], (0.9, 0.0, "left", "center"))
    ax.text(S(mx, my)[0] + dx, S(mx, my)[1] + dy,
            f"M{k + 1} {KOR.get(m['name'], m['name'])}\n{m['speed']} m/s, 왕복 {period:.1f} s",
            ha=ha, va=va, fontsize=9, color="#0b2f75", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9db7e8", lw=0.6, alpha=0.9))
    if d["seed_closest_m"] < 0.6:
        u = d["seed_u"]
        cx, cy = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        ax.plot(*S(cx, cy), marker="X", ms=11, color="#e01010", mec="white", mew=1.2, zorder=8)
        above = LABEL.get(m["name"], (0, 0, "", ""))[1] < 0      # name label below -> time above
        ax.text(S(cx, cy)[0] + (0.15 if above else -0.4), S(cx, cy)[1] + (0.45 if above else -0.9),
                f"{d['seed_t']:.1f} s", color="#c00000", fontsize=9, weight="bold",
                ha="left" if above else "right", va="bottom" if above else "baseline", zorder=8)

# start / goal
sx_, sy_, syaw = MISSION["start_xy_yaw"]
gx, gy, gyaw = MISSION["goal_xy_yaw"]
ax.plot(*S(sx_, sy_), marker="o", ms=11, color="#1a9a3a", mec="white", mew=1.5, zorder=9)
ax.annotate("", xy=S(sx_ + 1.2 * math.cos(syaw), sy_ + 1.2 * math.sin(syaw)), xytext=S(sx_, sy_),
            arrowprops=dict(arrowstyle="-|>", color="#1a9a3a", lw=2), zorder=9)
ax.text(S(sx_, sy_)[0], S(sx_, sy_)[1] + 0.45, "출발", color="#1a7a2a", fontsize=11, weight="bold", ha="center", va="bottom")
ax.add_patch(Circle(S(gx, gy), 0.75, fill=False, ec="#d4a000", lw=2, ls="-", zorder=9))
ax.plot(*S(gx, gy), marker="*", ms=17, color="#f2c200", mec="#8a6a00", zorder=9)
ax.text(S(gx, gy)[0] - 1.0, S(gx, gy)[1] + 1.0, "목표", color="#8a6a00", fontsize=11, weight="bold", ha="center")

ax.set_xlim(W["y"][0] - 0.8, W["y"][1] + 0.8)
ax.set_ylim(-(W["x"][1] + 0.8), -(W["x"][0] - 0.8))
ax.set_aspect("equal")
yt = range(-12, 19, 3)
xt = range(-10, 10, 2)
ax.set_xticks(list(yt)); ax.set_xticklabels([str(v) for v in yt])
ax.set_yticks([-v for v in xt]); ax.set_yticklabels([str(v) for v in xt])
ax.set_xlabel("y (m) — 북쪽이 오른쪽")
ax.set_ylabel("x (m) — 서쪽이 위")
ax.grid(alpha=0.25, lw=0.5)
from matplotlib.lines import Line2D  # noqa: E402
handles = [Rectangle((0, 0), 1, 1, fc="#d23c2a", ec="#7a1a10", label="고정 상자 14개 (충돌 집계)"),
           Rectangle((0, 0), 1, 1, fc="#e0a050", ec="#8a6a3a", label="지게차, 서쪽 선반 물건 (충돌 집계)"),
           Rectangle((0, 0), 1, 1, fc="#c8c8c8", ec="#9a9a9a", label="기둥, 랙, 팔레트 등 (막지만 집계 안 함)"),
           Rectangle((0, 0), 1, 1, fc="#1f5fd6", ec="#0b2f75", label="움직이는 상자 0.6×0.6×1.0 m (t = 0 위치, 충돌 집계)"),
           Line2D([], [], color="#1f5fd6", lw=2.2, label="왕복 구간"),
           Line2D([], [], color="#555555", ls="--", lw=1.6, label="A-4 a0 통과 코드의 경로"),
           Line2D([], [], color="#e01010", marker="X", ls="", ms=10, label="그 경로가 움직이는 상자를 만나는 곳과 시각")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, framealpha=0.95)
ax.set_title(f"채점 맵: 고정 장애물 + 움직이는 장애물 {NAME} (시험 주행 맵에는 파란 상자가 없음)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
