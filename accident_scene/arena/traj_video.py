"""Top-down replay video of an arena run, drawn from its trajectory CSV.

Isaac's own viewport recording is unavailable on this host (driver 595 needs the
no-viewport patch), so this redraws the run from the logged ground-truth poses:
walls, the 14 MISSION boxes, the 43 fixed obstacles, the movers (when the run used
ARENA_MOVERS), START/GOAL, the robot footprint and its trail, with time and speed.

Run with the Isaac env python (it has matplotlib + OpenCV):
    $CARTER_WS/mamba/envs/isaacsim/bin/python accident_scene/arena/traj_video.py \
        ~/SDI-Scenario/accident_scene/arena/<variant>_carter_run.py [--movers v1] [--speedup 4] [--out x.mp4]
"""
import argparse
import ast
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_file", help="<variant>_carter_run.py (map, MISSION and fixed obstacles are read from it)")
ap.add_argument("--movers", default=None, help="ARENA_MOVERS scenario used in the run, e.g. v1")
ap.add_argument("--speedup", type=float, default=4.0, help="sim seconds per video second")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default=None)
ap.add_argument("--title", default=None)
ap.add_argument("--map-file", default=None,
                help="read WALLS / FIXED_OBSTACLES from this runner when the run file has none (e.g. A-4 code on the "
                     "thin base): keeps the whole warehouse in view instead of zooming to the trajectory")
a = ap.parse_args()

run = Path(a.run_file).resolve()
arena = run.parent
variant = run.name[: -len("_carter_run.py")]
traj_path = arena / "results" / f"{variant}_trajectory.csv"
res_path = arena / "results" / f"{variant}_result.json"
src = run.read_text()


def literal(name, text=None):
    """Evaluate a top-level `NAME = <literal>` assignment from the run file (or `text`)."""
    tree = ast.parse(src if text is None else text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
    return None


mission = eval(compile(ast.Expression(literal("MISSION")), "MISSION", "eval"), {})  # dict(...) call
map_src = Path(a.map_file).read_text() if a.map_file else None
def map_literal(name):
    node = literal(name)
    return node if node is not None or map_src is None else literal(name, map_src)
walls = ast.literal_eval(map_literal("WALLS")) if map_literal("WALLS") is not None else None
fixed = ast.literal_eval(map_literal("FIXED_OBSTACLES")) if map_literal("FIXED_OBSTACLES") is not None else []
selfbox = ast.literal_eval(literal("_BASE_SELF_BOX")) if literal("_BASE_SELF_BOX") is not None else (-0.62, 0.17, -0.31, 0.31)

movers = []
movers_file = arena / "results" / f"{variant}_movers.json"
sys.path.insert(0, str(arena))
import arena_movers as AM  # noqa: E402
if movers_file.exists():
    # randomised scenario (v1r ...): the speeds and phases this run drew, written by the harness
    movers = json.load(open(movers_file))["movers"]
elif a.movers:
    movers = AM.SCENARIOS[a.movers]

rows = list(csv.DictReader(open(traj_path)))
T = [float(r["t"]) for r in rows]
X = [float(r["x"]) for r in rows]
Y = [float(r["y"]) for r in rows]
YAW = [float(r["yaw"]) for r in rows]
result = json.load(open(res_path)) if res_path.exists() else {}
verdict = result.get("verdict", "?")
m = result.get("metrics", {})

# speed over a 0.1 s window
spd = [0.0] * len(T)
for i in range(len(T)):
    j = max(0, i - 6)
    dt = T[i] - T[j]
    spd[i] = math.hypot(X[i] - X[j], Y[i] - Y[j]) / dt if dt > 0 else 0.0

xmin, xmax, ymin, ymax = walls if walls else (min(X) - 2, max(X) + 2, min(Y) - 2, max(Y) + 2)
W_IN = 7.2
H_IN = W_IN * (ymax - ymin + 1.6) / (xmax - xmin + 1.6) + 0.9
fig = plt.figure(figsize=(W_IN, H_IN), dpi=110)
ax = fig.add_axes([0.04, 0.02, 0.92, 0.88])
ax.set_xlim(xmin - 0.8, xmax + 0.8)
ax.set_ylim(ymin - 0.8, ymax + 0.8)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=2.5, ec="#333"))
for _, x0, x1, y0, y1 in fixed:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#bbbbbb", ec="#888", lw=0.5))
for i, b in enumerate(mission["boxes"]):
    low = b["height"] < 0.9
    ax.add_patch(Rectangle((b["x"] - b["width"] / 2, b["y"] - b["depth"] / 2), b["width"], b["depth"],
                           fc="#e8a33c" if low else "#c0662b", ec="#7a3d12", lw=0.8))
    ax.text(b["x"], b["y"], f"{i:02d}", ha="center", va="center", fontsize=6, color="white")
sx, sy, _ = mission["start_xy_yaw"]
gx, gy, gyaw = mission["goal_xy_yaw"]
ax.plot(sx, sy, "s", color="#2a7", ms=8)
ax.add_patch(plt.Circle((gx, gy), mission["position_tolerance_m"], fc="#2a72", ec="#2a7", lw=1.2))
ax.annotate("", (gx + 0.9 * math.cos(gyaw), gy + 0.9 * math.sin(gyaw)), (gx, gy),
            arrowprops=dict(arrowstyle="->", color="#2a7", lw=1.5))
ax.text(sx, sy - 0.7, "START", ha="center", fontsize=7, color="#2a7")
ax.text(gx, gy + 1.0, "GOAL", ha="center", fontsize=7, color="#2a7")

trail, = ax.plot([], [], "-", color="#1f6fd1", lw=1.2, alpha=0.8)
robot = Polygon([[0, 0]] * 4, closed=True, fc="#1f6fd1aa", ec="#0b3a73", lw=1.2)
ax.add_patch(robot)
heading, = ax.plot([], [], "-", color="#0b3a73", lw=1.5)
mover_patches = [Rectangle((0, 0), mv["size"][0], mv["size"][1], fc="#d33a", ec="#a00", lw=1.0) for mv in movers]
for p in mover_patches:
    ax.add_patch(p)
title = a.title or variant
fig.text(0.04, 0.965, title, fontsize=10, weight="bold")
info = fig.text(0.04, 0.925, "", fontsize=9, family="monospace")
summary = (f"verdict={verdict}  ttg={m.get('time_to_goal_s')}  final_dist={m.get('final_dist_m')}  "
           f"collisions={m.get('collision_count')}" + (f"  movers={a.movers}" if a.movers else "  movers=off"))

fig.canvas.draw()
w, h = fig.canvas.get_width_height()
out = Path(a.out) if a.out else arena / "results" / f"{variant}_topdown.mp4"
# H.264 through the ffmpeg bundled in ~/venvs/ffmpeg (imageio-ffmpeg): OpenCV's pip build only has
# MPEG-4 Part 2, which browsers do not play. Even dimensions are required by yuv420p.
import glob  # noqa: E402
import subprocess  # noqa: E402

import shutil  # noqa: E402

FFMPEG = sorted(glob.glob(str(Path.home() / "venvs/ffmpeg/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))
if not FFMPEG and shutil.which("ffmpeg"):
    FFMPEG = [shutil.which("ffmpeg")]  # system ffmpeg (apt) has libx264
if not FFMPEG:
    sys.exit("ffmpeg not found: apt install ffmpeg, or uv venv ~/venvs/ffmpeg && uv pip install --python ~/venvs/ffmpeg imageio-ffmpeg")
ew, eh = w - w % 2, h - h % 2
proc = subprocess.Popen([FFMPEG[-1], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-", "-vf", f"crop={ew}:{eh}:0:0",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
                         str(out)], stdin=subprocess.PIPE)
codec = "h264"


class _Writer:
    def write(self, img):
        proc.stdin.write(img.tobytes())

    def release(self):
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")


writer = _Writer()

step = a.speedup / a.fps
t_end = T[-1]
k = 0
t = 0.0
n_frames = 0
while t <= t_end + 1e-9:
    while k + 1 < len(T) and T[k + 1] <= t:
        k += 1
    x, y, yaw = X[k], Y[k], YAW[k]
    c, s = math.cos(yaw), math.sin(yaw)
    bx0, bx1, by0, by1 = selfbox
    pts = [(x + c * px - s * py, y + s * px + c * py) for px, py in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1))]
    robot.set_xy(pts)
    heading.set_data([x, x + 0.6 * c], [y, y + 0.6 * s])
    trail.set_data(X[: k + 1], Y[: k + 1])
    for mv, p in zip(movers, mover_patches):
        mx, my = AM.position(mv, T[k])
        p.set_xy((mx - mv["size"][0] / 2, my - mv["size"][1] / 2))
    dist = math.hypot(gx - x, gy - y)
    info.set_text(f"t={T[k]:6.1f}s  v={spd[k]:4.2f} m/s  dist_to_goal={dist:5.2f} m   x{a.speedup:g}\n{summary}")
    fig.canvas.draw()
    img = cv2.cvtColor(__import__("numpy").asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    writer.write(img)
    n_frames += 1
    t += step
for _ in range(a.fps * 2):  # hold the last frame 2 s
    writer.write(img)
writer.release()
print(f"{out}  codec={codec} frames={n_frames} sim={t_end:.1f}s video={n_frames / a.fps:.1f}s")
