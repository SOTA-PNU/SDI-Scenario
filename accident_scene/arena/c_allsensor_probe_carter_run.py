"""Mission runner - ACCIDENT ARENA (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/arena/nova_carter_warehouse_arena.yaml): the whole warehouse is
the arena; the robot starts in one corner at START and must reach GOAL in the
opposite corner; static boxes stand in between.

PASS criteria are evaluated by the fixed harness (levels/common/carter_env.py):
  reached_goal      final distance <= position tolerance
                    (and |yaw error| <= yaw tolerance when it is set)
  no_collision      zero chassis contact events (when collision is scored)
  max_time_to_goal  first entry into the goal disc within the bound (when set)
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default - no control code, the robot never
moves.  Everything outside it is fixed plumbing.
"""

import math  # noqa: F401  (available to the controller)
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "levels" / "common"))
sys.path.insert(0, str(HERE))
from arena_harness import (  # noqa: E402
    ArenaHarness,
    apply_record_quality,
    follow_camera,
    recording,
    setup_topdown_camera,
)

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(HERE / "results")

# scenario givens (from accident_scene/arena/nova_carter_warehouse_arena.yaml - not part of the answer)
# boxes: x, y = centre; width along X, depth along Y; resting on the floor
MISSION = dict(start_xy_yaw=(-9.0, -10.8, 1.0138),
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# RAW ALL-SENSOR PROBE (analysis/c_allsensor_probe.py) - validation only, not a model answer
import json as _json
import time as _time

_OUTDIR = "/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/c_allsensor_probe"
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


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera
ctrl = controller
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(controller, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
