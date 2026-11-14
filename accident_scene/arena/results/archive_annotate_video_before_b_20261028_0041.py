"""Overlay labels on the recorded top-down arena frames and encode an MP4.

The frames come from base_carter_run.py with CARTER_RECORD=1 (path tracer,
orthographic top-down camera: centre (-0.45, 2.85), 36.4 m wide along +y,
20.5 m tall along x, rotated so north is RIGHT and west is UP).

    python3 accident_scene/arena/annotate_video.py FRAMES_DIR RESULT_JSON TRAJ_CSV OUT.mp4

Env: ANNOT_EVERY  physics steps per captured frame (= CARTER_RECORD_EVERY, default 2)
     ANNOT_MODE   "arena" (labels on the top-down arena view, default) or
                  "hud" (time/contacts/outcome only - for any other camera)
     ANNOT_CTRL   text for the controller caption
     ANNOT_LAYOUT layout JSON for labels (default layout.json = arena v1;
                  layout_v2.json for the v2 arena, where no box is below the scan plane)

Everything drawn comes from the run itself: the trail is the recorded
trajectory, the contact counter and the outcome banner come from the result
JSON, box labels from layout.json. The video plays in real time
(fps = 60 / ANNOT_EVERY).
"""

import os

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
frames_dir, result_json, traj_csv, out_mp4 = (Path(a) for a in sys.argv[1:5])
EVERY = int(os.environ.get("ANNOT_EVERY", "2"))   # physics steps per captured frame
MODE = os.environ.get("ANNOT_MODE", "arena")
CTRL = os.environ.get("ANNOT_CTRL", "base = levels_zero2level default controller (no control code)")
MAX_S = float(os.environ.get("ANNOT_MAX_S", "") or 0.0)  # cut the clip here [sim s]
FPS = 60 // EVERY
DT = 1.0 / 60.0

# camera -> pixel mapping (verified on frame 100: world (-7.38,-8.56) -> ~(358,175))
CX, CY, VW, VH, W, H = -0.45, 2.85, 36.4, 20.5, 1920, 1080


def px(x, y):
    return ((y - CY) / VW * W + W / 2.0, (x - CX) / VH * H + H / 2.0)


def m2px(m):
    return m / VW * W


layout = json.load(open(os.environ.get("ANNOT_LAYOUT") or (HERE / "layout.json")))
res = json.load(open(result_json))
timeout_s = res["criteria"]["timeout_s"]
final_dist = res["metrics"]["final_dist_m"]
ttg = res["metrics"]["time_to_goal_s"]
first_hit = res["metrics"]["first_collision_t"]
partner = (res["metrics"]["collision_partners"] or ["?"])[0]
total_contacts = res["metrics"]["collision_count"]
traj = []
with open(traj_csv) as f:
    for row in csv.DictReader(f):
        traj.append((float(row["t"]), float(row["x"]), float(row["y"]), float(row["yaw"])))

# DejaVu has no Hangul, so a Korean caption came out as boxes. Fall back to
# Noto Sans CJK for any label that is not pure Latin.
_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def _face(path, size, index=None):
    try:
        return (ImageFont.truetype(path, size, index=index) if index is not None
                else ImageFont.truetype(path, size))
    except OSError:
        return None


font = _face(_DEJAVU, 26) or ImageFont.load_default()
small = _face(_DEJAVU, 19) or ImageFont.load_default()
mono = _face(_MONO, 24) or ImageFont.load_default()
# index 1 of NotoSansCJK-Bold.ttc is the KR face on this host
cjk = _face(_CJK, 24, 1) or _face(_CJK, 24, 0) or small
cjk_small = _face(_CJK, 19, 1) or _face(_CJK, 19, 0) or small


def pick(text, latin_font, cjk_font):
    """Use the CJK face as soon as the text leaves ASCII."""
    return cjk_font if any(ord(c) > 0x2FF for c in text) else latin_font

sx, sy, _ = layout["start"]["x"], layout["start"]["y"], layout["start"]["yaw"]
gx, gy = layout["goal"]["x"], layout["goal"]["y"]
# Which boxes sit under the scan plane is a property of the scene, not a
# constant to retype: the old hard-coded {4, 5, 6} silently left box 13 - also
# below the plane - unlabelled in every recorded video. Derive it instead.
SCAN_Z = 0.35
_sub = [i for i, b in enumerate(layout["boxes"]) if b["height"] < SCAN_Z]
# "TRAP" invites the objection that the scene is a gotcha; the boxes are listed
# in the prompt with their heights. Name what they are.
_PREFIX = os.environ.get("ANNOT_LOW_LABEL", "LOW")
traps = {b: f"{_PREFIX} {n + 1}" for n, b in enumerate(_sub)}


def label(d, xy, text, fnt, fill, bg=(0, 0, 0, 170), anchor="mm"):
    box = d.textbbox(xy, text, font=fnt, anchor=anchor)
    d.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=bg)
    d.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def contacts_at(t):
    # contacts accumulate at 60 Hz from the first hit until the 5 s abort
    if first_hit is None or t < first_hit:
        return 0
    return min(total_contacts, int((t - first_hit) / DT) + 1)


def draw_world(d, t):
    """Arena top-down labels: goal disc, start, traps, trail."""
    gu, gv = px(gx, gy)
    r = m2px(0.75)
    d.ellipse((gu - r, gv - r, gu + r, gv + r), outline=(60, 220, 90, 255), width=4)
    label(d, (gu, gv - r - 22), "GOAL", font, (90, 240, 120, 255))
    su, sv = px(sx, sy)
    label(d, (su + 70, sv - 34), "START", font, (140, 200, 255, 255))
    for i, name in traps.items():
        b = layout["boxes"][i]
        u, v = px(b["x"], b["y"])
        label(d, (u, v), f"{name}", pick(name, small, cjk_small), (255, 225, 90, 255))
    pts = [px(x, y) for (tt, x, y, _) in traj if tt <= t]
    if len(pts) > 1:
        d.line(pts, fill=(120, 190, 255, 230), width=4)


def hud_and_banner(d, t, size):
    """Time/contacts HUD, controller caption, and the outcome banner."""
    Wd, Hd = size
    c = contacts_at(t)
    label(d, (24, Hd - 36), f" t = {t:5.1f} s    contacts = {c:3d} ", mono, (255, 255, 255, 255), anchor="lm")
    cap = CTRL + ("  |  yellow = box below the z=0.35 m scan plane" if MODE == "arena" and traps else "")
    label(d, (Wd - 24, Hd - 36), cap, pick(cap, small, cjk_small),
          (230, 230, 230, 255), anchor="rm")
    bx = Wd / 2 + (140 if MODE == "arena" else 0)   # arena: clear of the START label
    if MAX_S and t >= MAX_S - 3.0 and (first_hit is None or first_hit > MAX_S):
        # the clip stops early: say what the rest of the run did, so a short
        # video is not mistaken for a short run
        banner = (f"CLIP CUT AT {MAX_S:.0f} s  -  robot does not move again: "
                  f"TIMEOUT at {timeout_s:.0f} s, {total_contacts} contacts")
        label(d, (bx, 40), banner, font, (255, 200, 90, 255), bg=(40, 25, 0, 210))
        return
    if first_hit is not None and t >= first_hit:
        banner = f"COLLISION at t = {first_hit:.2f} s  ->  {partner.split('/')[-1]}"
        label(d, (bx, 40), banner, font, (255, 90, 80, 255), bg=(40, 0, 0, 210))
    elif res["verdict"] == "pass" and ttg is not None and t >= ttg:
        banner = f"GOAL REACHED at t = {ttg:.2f} s  -  {total_contacts} contacts"
        label(d, (bx, 40), banner, font, (120, 240, 140, 255), bg=(0, 35, 10, 210))
    elif res["verdict"] != "pass" and t >= timeout_s - 3.0:
        banner = (f"TIMEOUT at {timeout_s:.0f} s  -  goal not reached "
                  f"({final_dist:.1f} m away), {total_contacts} contacts")
        label(d, (bx, 40), banner, font, (255, 200, 90, 255), bg=(40, 25, 0, 210))


frames = sorted(frames_dir.glob("*.png"))
out_dir = frames_dir.parent / (frames_dir.name + "_annot")
out_dir.mkdir(exist_ok=True)
for k, fp in enumerate(frames):
    t = (k + 1) * EVERY * DT
    if MAX_S and t > MAX_S:
        print(f"[annot] cut at {MAX_S:.0f} s ({k} frames)", flush=True)
        break
    im = Image.open(fp).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if MODE == "arena":
        draw_world(d, t)
    hud_and_banner(d, t, ov.size)
    Image.alpha_composite(im, ov).convert("RGB").save(out_dir / fp.name)
    if (k + 1) % 150 == 0:
        print(f"[annot] {k + 1}/{len(frames)}", flush=True)

# hold the last frame for 2 s so the outcome can be read
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
       "-i", str(out_dir / "%05d.png"),
       "-vf", "tpad=stop_mode=clone:stop_duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "21", "-preset", "slow", str(out_mp4)]
subprocess.run(cmd, check=True)
print(f"[annot] wrote {out_mp4} ({out_mp4.stat().st_size / 2**20:.1f} MiB)", flush=True)
