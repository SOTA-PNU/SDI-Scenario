"""Project the MISSION boxes into the probe's camera images (offline) to check the camera model.

Draws every box's 12 edges with the /tf optical-frame mount and camera_info K, once as a plain
pinhole (green) and once through the camera_info rational_polynomial distortion (magenta), and
samples the pixel colour at each visible box's face centre. Writes snap{A,B}_cam_overlay.png.
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ARENA = Path(__file__).resolve().parents[1]
D = ARENA / "results" / "analysis" / "c_allsensor_probe"
base = (ARENA / "base_carter_run_a2.py").read_text()
BOXES = ast.literal_eval(re.search(r"boxes=(\[.*?\])\)", base, re.S).group(1))


def qmat(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def project(Pw, pose, tf, K, D=None):
    """World points (N,3) -> pixels (N,2) and depth; D = rational_polynomial coefficients or None."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    # world -> base_link
    dx, dy = Pw[:, 0] - x, Pw[:, 1] - y
    Pb = np.stack([c * dx + s * dy, -s * dx + c * dy, Pw[:, 2]], 1)
    t = np.array(tf[1:4])
    R = qmat(*tf[4:8])                  # optical -> base_link
    Pc = (Pb - t) @ R                    # base_link -> optical  (R^T (Pb - t))
    Z = Pc[:, 2]
    xn, yn = Pc[:, 0] / Z, Pc[:, 1] / Z
    if D is not None:
        k1, k2, p1, p2, k3, k4, k5, k6 = D
        r2 = xn * xn + yn * yn
        rad = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn, yn = xd, yd
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    return np.stack([fx * xn + cx, fy * yn + cy], 1), Z


def box_corners(b):
    xs = (b["x"] - b["width"] / 2, b["x"] + b["width"] / 2)
    ys = (b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2)
    zs = (0.0, b["height"])
    return np.array([(X, Y, Z) for X in xs for Y in ys for Z in zs])


EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def main(names):
    report = {}
    for name in names:
        meta = json.load(open(D / f"snap{name}.json"))
        img = np.load(D / f"snap{name}_cam.npy")
        ci = meta["topics"]["cam_info"]
        tf = meta["tf"]["front_stereo_camera_left_optical"]
        pose = meta["pose"]
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        rows = []
        for i, b in enumerate(BOXES):
            C = box_corners(b)
            for dist, colr in ((None, (0, 255, 0)), (ci["d"], (255, 0, 255))):
                uv, Z = project(C, pose, tf, ci["k"], dist)
                if (Z <= 0.2).any():
                    continue
                for a, bb in EDGES:
                    dr.line([tuple(uv[a]), tuple(uv[bb])], fill=colr, width=3)
            # colour at the box centre (half height), pinhole
            cuv, cz = project(np.array([[b["x"], b["y"], b["height"] / 2]]), pose, tf, ci["k"])
            u, v = cuv[0]
            if cz[0] > 0.2 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                patch = img[max(0, int(v) - 3):int(v) + 4, max(0, int(u) - 3):int(u) + 4].reshape(-1, 3).mean(0)
                rows.append(dict(box=i, uv=[round(float(u), 1), round(float(v), 1)], depth=round(float(cz[0]), 2),
                                 rgb=[int(q) for q in patch]))
        im.save(D / f"snap{name}_cam_overlay.png")
        small = im.resize((960, 600))
        small.save(D / f"snap{name}_cam_overlay_small.png")
        report[name] = dict(pose=pose, visible_box_centres=rows)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B"])
