"""Render base vs solution trajectories of one level to results/trajectory.png.

Pure-python (matplotlib only, no Isaac):
  MAMBA_ROOT_PREFIX=$CARTER_WS/mamba $CARTER_WS/tools/bin/micromamba run -n isaacsim \
      python levels/common/plot_traj.py levels/level2
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


def load(folder, variant):
    csv_p = folder / "results" / f"{variant}_trajectory.csv"
    res_p = folder / "results" / f"{variant}_result.json"
    if not csv_p.exists() or not res_p.exists():
        return None
    xs, ys = [], []
    with open(csv_p) as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return xs, ys, json.load(open(res_p))


def main():
    folder = Path(sys.argv[1])
    fig, ax = plt.subplots(figsize=(5.2, 7))
    res = None
    for variant, color in (("base", "#c0392b"), ("solution", "#1a7f37")):
        data = load(folder, variant)
        if data is None:
            continue
        xs, ys, res = data
        m = res["metrics"]
        label = (f"{variant}: {res['verdict'].upper()}"
                 + (f", ttg={m['time_to_goal_s']}s" if m["time_to_goal_s"] else "")
                 + (f", col={m['collision_count']}" if res["criteria"]["no_collision"] else ""))
        ax.plot(xs, ys, color=color, lw=1.8, label=label)
        ax.plot(xs[-1], ys[-1], "s", color=color, ms=6)
    if res:
        g = res["goal"]
        tol = res["criteria"]["position_tolerance_m"]
        ax.add_patch(Circle((g["x"], g["y"]), tol, fill=False, ec="#2563eb", ls="--", lw=1.4))
        ax.plot(g["x"], g["y"], "*", color="#2563eb", ms=14, label="goal")
        if res.get("obstacle"):
            o = res["obstacle"]
            ax.add_patch(
                Rectangle(
                    (o["x"] - o["width"] / 2, o["y"] - o["depth"] / 2),
                    o["width"], o["depth"],
                    fc="#e67e22", ec="k", alpha=0.85, label="obstacle",
                )
            )
    ax.plot(-6.0, -1.0, "o", color="k", ms=7, label="spawn (-6,-1)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(folder.name)
    ax.legend(fontsize=8, loc="best")
    out = folder / "results" / "trajectory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
