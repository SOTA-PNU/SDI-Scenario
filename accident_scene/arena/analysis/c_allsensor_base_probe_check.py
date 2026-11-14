"""Offline check of what the B-2 sensor base code returned in Isaac (c_allsensor_base_probe.py).

2D and 3D points (to_world output) against known geometry, 3D azimuth coverage and floor height,
camera_project of the MISSION box centres against the image colour, plus an overlay picture.
Writes results/analysis/c_allsensor_base_probe/check.json and snap{A,B}_overlay_small.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA / "analysis"))
from c_allsensor_probe_check import BOXES, near_known  # noqa: E402

D = ARENA / "results" / "analysis" / (sys.argv[sys.argv.index("--dir") + 1].split("/")[-1] if "--dir" in sys.argv
                                       else "c_allsensor_base_probe")
out = {}
for name in ("A", "B"):
    meta = json.load(open(D / f"snap{name}.json"))
    r = {"pose": meta["pose"], "sizes": meta["sizes"], "timing": meta["timing"]}
    for key in ("front2d", "back2d"):
        w = np.load(D / f"snap{name}_{key}_world.npy")
        p = np.load(D / f"snap{name}_{key}.npy")
        r[key] = dict(n=int(len(w)), near_known=round(float(near_known(w[:, 0], w[:, 1]).mean()), 4),
                      min_range_from_axle=round(float(np.hypot(p[:, 0], p[:, 1]).min()), 3))
    p3 = np.load(D / f"snap{name}_lidar3d.npy")
    w3 = np.load(D / f"snap{name}_lidar3d_world.npy")
    az = np.degrees(np.arctan2(p3[:, 1] - 0.0, p3[:, 0] + 0.232))
    hist, _ = np.histogram(az, bins=12, range=(-180, 180))
    floor = np.abs(p3[:, 2]) < 0.05
    above = (p3[:, 2] > 0.10) & (p3[:, 2] < 2.0)
    r["lidar3d"] = dict(n=int(len(p3)), azimuth_hist_30deg=hist.tolist(), floor_share=round(float(floor.mean()), 3),
                        floor_z_median=round(float(np.median(p3[floor, 2])), 4),
                        above_0p1_2m=int(above.sum()),
                        above_near_known=round(float(near_known(w3[above, 0], w3[above, 1], w3[above, 2]).mean()), 4))
    img = np.load(D / f"snap{name}_cam.npy")
    uv = np.array(meta["box_uv"])
    ok = np.array(meta["box_ok"])
    rows = []
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for i in np.where(ok)[0]:
        u, v = uv[i]
        patch = img[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4].reshape(-1, 3).mean(0)
        rows.append(dict(box=int(i), uv=[round(float(u), 1), round(float(v), 1)], rgb=[int(q) for q in patch],
                         reddish=bool(patch[0] > 1.5 * patch[1] and patch[0] > 1.5 * patch[2])))
        dr.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=4)
        dr.text((u + 14, v - 14), str(i), fill=(0, 255, 0))
    im.resize((960, 600)).save(D / f"snap{name}_overlay_small.png")
    r["camera"] = dict(shape=list(img.shape), boxes_in_view=len(rows), reddish=sum(x["reddish"] for x in rows),
                       rows=rows)
    out[name] = r
(D / "check.json").write_text(json.dumps(out, indent=1))
for name, r in out.items():
    print(name, "pose", [round(v, 3) for v in r["pose"]])
    print("  2D front", r["front2d"], "\n  2D back ", r["back2d"])
    print("  3D", {k: v for k, v in r["lidar3d"].items()})
    print("  camera", {k: v for k, v in r["camera"].items() if k != "rows"})
    for x in r["camera"]["rows"]:
        print("     box", x["box"], x["uv"], x["rgb"], "reddish" if x["reddish"] else "-")
    print("  timing", {k: (v["n"], round(v["mean_ms"], 3), round(v["p95_ms"], 3), round(v["max_ms"], 2)) for k, v in r["timing"].items()})
