"""Render one straight-down still of the warehouse the Carter drives in.

An orthographic camera sits above the walled interior, north (+y) up. Because
neither the camera nor the scene moves, the renderer can accumulate over many
frames before the capture, which is what makes a still far cleaner than a
frame of the driving video. Two captures: the default RTX real-time mode after
warm-up, and the path tracer after accumulation.

    CARTER_RECORD_SIZE=1200x1830 CARTER_REPO=<repo> \\
      $CARTER_WS/tools/bin/carter-run accident_scene/arena/topdown_still.py

Optional: ARENA_LAYOUT=<json with "start"/"goal"/"boxes"> draws that layout in.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CARTER_REPO", "/home/jun/SDI-Scenario"))
sys.path.insert(0, str(REPO / "levels" / "common"))
sys.path.insert(0, str(HERE))
from carter_env import Harness  # noqa: E402
from arena_harness import ArenaHarness  # noqa: E402

OUT_DIR = Path(os.environ.get("STILL_OUT", str(HERE / "results")))
# the harness insists on a result dir; this run is a photo, so throw it away
RUN_DIR = os.environ.get("STILL_RUN_DIR",
                         str(Path(tempfile.gettempdir()) / "carter_still_run"))
# walled interior measured with an overlap sweep: x [-10.2, 9.3], y [-12.2, 17.9]
CX, CY = -0.45, 2.85
VIEW_W, VIEW_H = 21.0, 32.0          # metres covered, a little margin around the walls
CAM_Z = 40.0
# the warehouse has a roof: from above you only see ceiling. Clip everything
# higher than CUT_Z by pushing the near plane down to it (lights are not
# geometry, so the interior stays lit).
CUT_Z = float(os.environ.get("CUT_Z", "6.0"))
WARMUP_RT = int(os.environ.get("WARMUP_RT", "240"))
WARMUP_PT = int(os.environ.get("WARMUP_PT", "600"))

common = dict(level="still", variant="topdown", goal_xy_yaw=(0.0, 0.0, 0.0), timeout_s=5.0,
              position_tolerance_m=0.75, out_dir=RUN_DIR)
if os.environ.get("ARENA_LAYOUT"):
    layout = json.load(open(os.environ["ARENA_LAYOUT"]))
    st = layout["start"]   # layout.json stores start/goal as {"x","y","yaw",...}
    h = ArenaHarness(start_xy_yaw=(st["x"], st["y"], st["yaw"]), boxes=layout["boxes"], **common).boot()
else:
    h = Harness(**common).boot()

import carb  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

cam = UsdGeom.Camera.Define(h.stage, "/World/TopDownCam")
xf = UsdGeom.XformCommonAPI(cam)
xf.SetTranslate(Gf.Vec3d(CX, CY, CAM_Z))           # identity rotation: looks down -Z, +Y up
cam.CreateProjectionAttr("orthographic")
cam.CreateHorizontalApertureAttr(VIEW_W * 10.0)     # tenths of a stage unit (metres here)
cam.CreateVerticalApertureAttr(VIEW_H * 10.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(CAM_Z - CUT_Z, CAM_Z + 5.0))

vp = get_active_viewport()
vp.camera_path = "/World/TopDownCam"
s = carb.settings.get_settings()
OUT_DIR.mkdir(parents=True, exist_ok=True)
tag = os.environ.get("STILL_TAG", "warehouse")


def settle_and_capture(n, name):
    for _ in range(n):
        h.sim.step(render=True)
    path = str(OUT_DIR / name)
    capture_viewport_to_file(vp, path)
    for _ in range(8):             # let the async capture flush
        h.sim.step(render=True)
    print(f"[still] wrote {path}", flush=True)


print(f"[still] rendermode={s.get('/rtx/rendermode')!r}", flush=True)
settle_and_capture(WARMUP_RT, f"topdown_{tag}_rt.png")

s.set("/rtx/rendermode", "PathTracing")
s.set("/rtx/pathtracing/spp", 4)
s.set("/rtx/pathtracing/totalSpp", WARMUP_PT * 4)
s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
print(f"[still] rendermode={s.get('/rtx/rendermode')!r} spp={s.get('/rtx/pathtracing/spp')}", flush=True)
settle_and_capture(WARMUP_PT, f"topdown_{tag}_pt.png")

h.close()
sys.exit(0)
