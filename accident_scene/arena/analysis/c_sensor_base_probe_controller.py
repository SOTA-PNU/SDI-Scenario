RAW = {}
TFS = {}
_probe_ready = False
_OUT = "/home/jun/SDI-Scenario/accident_scene/arena/results/analysis/c_sensor_base_probe.json"


def _probe_tf(msg):
    for tr in msg.transforms:
        cf = tr.child_frame_id
        if "lidar" in cf and cf not in TFS:
            t_, q = tr.transform.translation, tr.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            TFS[cf] = [tr.header.frame_id, t_.x, t_.y, t_.z, yaw]


def controller(t, pose, env):
    global _probe_ready
    sensor_setup()
    if not _probe_ready:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        from tf2_msgs.msg import TFMessage
        for topic in _LIDAR:
            _node.create_subscription(LaserScan, topic, lambda m, k=topic: RAW.__setitem__(k, m), qos_profile_sensor_data)
        _node.create_subscription(TFMessage, "/tf", _probe_tf, 10)
        _probe_ready = True
    sensor_points()
    ready = len(RAW) == len(_LIDAR) and "front_2d_lidar" in TFS and "rear_2d_lidar" in TFS
    if t >= 2.0 and (ready or t >= 6.0):
        import json
        out = {"pose": list(pose), "t": t, "lidar": _LIDAR, "body": list(_BODY),
               "raw": {k: {"angle_min": m.angle_min, "angle_increment": m.angle_increment,
                           "range_min": m.range_min, "range_max": m.range_max,
                           "ranges": [float(r) for r in m.ranges]} for k, m in RAW.items()},
               "base_scans": {k: [list(p) for p in v] for k, v in _scans.items()}, "tf": TFS}
        with open(_OUT, "w") as f:
            f.write(json.dumps(out))
        print("PROBE wrote", _OUT, {k: len(v) for k, v in _scans.items()}, flush=True)
        return 0.0, 0.0, True
    return 0.0, 0.0, False
