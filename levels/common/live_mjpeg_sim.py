"""MJPEG live source: warehouse scene, captures rotating JPEGs to /dev/shm.

NVENC-free live view for A100 (no hardware encoder, so NVST/WebRTC streaming
cannot work on this host).  Pair with mjpeg_server.py which serves the frames
over HTTP.  Runs 3 hours or until killed; never writes benchmark results.
"""
import os
import time

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
)

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

stage_utils.open_stage(
    get_assets_root_path() + "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
)
for _ in range(60):
    app.update()

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)

set_camera_view(eye=[-1.0, -2.5, 7.5], target=[-6.0, 2.0, 0.3])
vp = get_active_viewport()
sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

OUT = "/dev/shm/carter_live"
os.makedirs(OUT, exist_ok=True)
ROTATE = 10
EVERY = 2
print("[mjpeg-sim] capturing to /dev/shm/carter_live (3h)", flush=True)
t0 = time.time()
i = 0
while time.time() - t0 < 3 * 3600:
    sim.step(render=True)
    i += 1
    if i % EVERY == 0:
        capture_viewport_to_file(vp, os.path.join(OUT, f"f{(i // EVERY) % ROTATE}.jpg"))
app.close()
