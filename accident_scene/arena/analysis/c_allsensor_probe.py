"""Write the raw all-sensor probe runner (project C, B-2 revision; validation only, never shown to a model).

Collects, on the fixed-obstacle map, everything needed to write and check the B-2 sensor base
code for every Carter sensor that publishes: front and back 2D lidar (LaserScan), front 3D lidar
(PointCloud2), front stereo left camera (Image + CameraInfo), and the /tf transforms from
base_link to each sensor frame. The robot holds still at the start pose (snapshot A), turns
+90 deg in place, settles 1.5 s (snapshot B) and ends. Nothing here steers by sensor data.

    python3 analysis/c_allsensor_probe.py   -> c_allsensor_probe_carter_run.py
    output: results/analysis/c_allsensor_probe/snap{A,B}.json + .npy arrays
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUTDIR = ARENA / "results" / "analysis" / "c_allsensor_probe"

REGION = r'''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# RAW ALL-SENSOR PROBE (analysis/c_allsensor_probe.py) - validation only, not a model answer
import json as _json
import time as _time

_OUTDIR = "__OUTDIR__"
_TOPICS = {
    "front_2d": ("/front_2d_lidar/scan", "LaserScan"),
    "back_2d": ("/back_2d_lidar/scan", "LaserScan"),
    "lidar3d": ("/front_3d_lidar/lidar_points", "PointCloud2"),
    "cam": ("/front_stereo_camera/left/image_raw", "Image"),
    "cam_info": ("/front_stereo_camera/left/camera_info", "CameraInfo"),
}


class _St:
    pass


_P = _St()
_P.node = None
_P.msgs = {}
_P.n = {}
_P.first = {}
_P.tf = {}
_P.count = 0
_P.phase = 0
_P.t_phase = 0.0
_P.yaw0 = None
_P.drain = []


def _on(key, msg, env):
    _P.msgs[key] = msg
    _P.n[key] = _P.n.get(key, 0) + 1
    if key not in _P.first:
        _P.first[key] = round(env.sim.current_time - env._t0, 3)
    _P.count += 1


def _on_tf(msg):
    for tr in msg.transforms:
        t, q = tr.transform.translation, tr.transform.rotation
        _P.tf[tr.child_frame_id] = [tr.header.frame_id, t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    _P.count += 1


def _setup(env):
    import rclpy
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
    from tf2_msgs.msg import TFMessage
    types = {"LaserScan": LaserScan, "PointCloud2": PointCloud2, "Image": Image, "CameraInfo": CameraInfo}
    _P.rclpy = rclpy
    _P.node = rclpy.create_node("c_allsensor_probe")
    for key, (topic, typ) in _TOPICS.items():
        _P.node.create_subscription(types[typ], topic, lambda m, k=key: _on(k, m, env), qos_profile_sensor_data)
    _P.node.create_subscription(TFMessage, "/tf", _on_tf, 50)
    _P.node.create_subscription(TFMessage, "/tf_static", _on_tf,
                                QoSProfile(depth=100, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))


def _stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def _snap(name, t, pose, env):
    import os
    import numpy as np
    os.makedirs(_OUTDIR, exist_ok=True)
    meta = {"name": name, "t": t, "pose": list(pose), "sim_time": env.sim.current_time,
            "t0": env._t0, "counts": dict(_P.n), "first_t": dict(_P.first), "tf": dict(_P.tf),
            "drain_ms_mean": (sum(_P.drain) / len(_P.drain)) if _P.drain else None,
            "drain_ms_max": max(_P.drain) if _P.drain else None, "topics": {}}
    for key, m in _P.msgs.items():
        info = {"frame_id": m.header.frame_id, "stamp": _stamp(m)}
        if key in ("front_2d", "back_2d"):
            arr = np.array(m.ranges, dtype=np.float32)
            info.update(angle_min=m.angle_min, angle_max=m.angle_max, angle_increment=m.angle_increment,
                        range_min=m.range_min, range_max=m.range_max, n=int(arr.size))
            np.save(f"{_OUTDIR}/snap{name}_{key}.npy", arr)
        elif key == "lidar3d":
            fields = [(f.name, f.offset, f.datatype, f.count) for f in m.fields]
            info.update(width=m.width, height=m.height, point_step=m.point_step, row_step=m.row_step,
                        is_bigendian=m.is_bigendian, is_dense=m.is_dense, fields=fields)
            raw = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(-1, m.point_step)
            off = {f[0]: f[1] for f in fields}
            pts = np.stack([raw[:, off[c]:off[c] + 4].copy().view(np.float32)[:, 0] for c in ("x", "y", "z")], 1)
            np.save(f"{_OUTDIR}/snap{name}_{key}.npy", pts)
        elif key == "cam":
            info.update(width=m.width, height=m.height, encoding=m.encoding, step=m.step,
                        is_bigendian=m.is_bigendian)
            img = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(m.height, m.step)[:, :m.width * 3]
            np.save(f"{_OUTDIR}/snap{name}_{key}.npy", img.reshape(m.height, m.width, 3))
        elif key == "cam_info":
            info.update(width=m.width, height=m.height, distortion_model=m.distortion_model,
                        d=list(m.d), k=list(m.k), r=list(m.r), p=list(m.p))
        meta["topics"][key] = info
    with open(f"{_OUTDIR}/snap{name}.json", "w") as f:
        f.write(_json.dumps(meta, indent=1))
    print("PROBE snapshot", name, "t=%.2f" % t, "pose", [round(v, 3) for v in pose],
          "topics", sorted(_P.msgs), "counts", _P.n, "tf", len(_P.tf), flush=True)


def controller(t, pose, env):
    if _P.node is None:
        _setup(env)
    t_a = _time.perf_counter()
    for _ in range(30):
        c = _P.count
        _P.rclpy.spin_once(_P.node, timeout_sec=0.0)
        if _P.count == c:
            break
    _P.drain.append((_time.perf_counter() - t_a) * 1000.0)
    if _P.phase == 0:
        if t >= 3.0 and all(k in _P.msgs for k in _TOPICS):
            _snap("A", t, pose, env)
            _P.phase, _P.yaw0 = 1, pose[2]
        elif t >= 15.0:
            _snap("A_partial", t, pose, env)
            return 0.0, 0.0, True
        return 0.0, 0.0, False
    if _P.phase == 1:
        dyaw = math.atan2(math.sin(pose[2] - _P.yaw0), math.cos(pose[2] - _P.yaw0))
        if dyaw < 1.5708:
            return 0.0, 0.5, False
        _P.phase, _P.t_phase = 2, t
        return 0.0, 0.0, False
    if t - _P.t_phase >= 1.5:
        _snap("B", t, pose, env)
        return 0.0, 0.0, True
    return 0.0, 0.0, False


'''
base = (ARENA / "base_carter_run_a2.py").read_text()
b = base.index("# [EDIT REGION] mission controller")
end = "# ========================= [END EDIT REGION] ===============================\n"
e = base.index(end) + len(end)
out = ARENA / "c_allsensor_probe_carter_run.py"
out.write_text(base[:b] + REGION.replace("__OUTDIR__", str(OUTDIR)) + end + base[e:])
print("wrote", out)
