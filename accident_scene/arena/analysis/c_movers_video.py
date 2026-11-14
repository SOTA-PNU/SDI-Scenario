"""Label the top-down Isaac frames of the moving-obstacle map (c_movers_view_carter_run.py, recorded
with CARTER_RECORD_EVERY=6 -> one frame per 0.1 s of mission time) and encode an MP4.

Pixel mapping of the recording camera (setup_topdown_camera: centre (-0.45, 2.85), 36.4 m along +y
across the 1280 px width, 20.5 m along x down the 720 px height, north right / west up):
    u = W/2 + (y - 2.85) * W / 36.4,   v = H/2 + (x + 0.45) * H / 20.5
Mover positions are arena_movers.position(m, t), checked against the frames (a few px).

    ~/carter_ws/mamba/envs/isaacsim/bin/python analysis/c_movers_video.py FRAMES_DIR OUT.mp4 [scenario]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import arena_movers as AM  # noqa: E402

FR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAME = sys.argv[3] if len(sys.argv) > 3 else "v1"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
f_big, f_small, f_tag = (ImageFont.truetype(FONT, s, index=1) for s in (22, 17, 18))
movers = AM.scenario(NAME)
frames = sorted(FR.glob("*.png"))
tmp = Path(tempfile.mkdtemp(prefix="cmovers_"))
for k, f in enumerate(frames):
    im = Image.open(f).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    t = (k + 1) * 0.1
    if t <= 20.0:
        for i, m in enumerate(movers):
            x, y = AM.position(m, t)
            u, v = W / 2 + (y - 2.85) * W / 36.4, H / 2 + (x + 0.45) * H / 20.5
            d.rectangle([u + 12, v - 26, u + 52, v - 4], fill=(15, 50, 140))
            d.text((u + 16, v - 27), f"M{i + 1}", font=f_tag, fill=(255, 255, 255))
    d.rectangle([0, 0, W, 34], fill=(20, 20, 20))
    d.text((10, 3), f"채점 맵: 움직이는 장애물 {NAME}   t = {min(t, 20.0):4.1f} s", font=f_big, fill=(255, 255, 255))
    d.rectangle([0, H - 30, W, H], fill=(20, 20, 20))
    d.text((10, H - 28), "하늘색 상자 5개(M1-M5)가 움직이는 장애물입니다. 로봇은 출발점(왼쪽 위)에 세워 두었습니다. "
                         "시험 주행 맵에는 이 상자가 없습니다.", font=f_small, fill=(230, 230, 230))
    im.save(tmp / f"{k:05d}.png")
    if k == 99:
        im.save(OUT.with_name(OUT.stem + "_t10s.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmp / "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(OUT)], check=True)
print("wrote", OUT, "and", OUT.with_name(OUT.stem + "_t10s.png"), f"({len(frames)} frames)")
