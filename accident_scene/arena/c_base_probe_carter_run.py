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

# ---- Carter 2D 라이다 기본 코드 ----
# 앞쪽과 뒤쪽 2D 라이다는 /front_2d_lidar/scan, /back_2d_lidar/scan 토픽(sensor_msgs/LaserScan)으로 발행됩니다.
# 360도, 빔 3200개, 0.05-30 m 입니다. 약 6 Hz 로 도착하므로 가장 최근 스캔이 0.2 s 가량 늦을 수 있습니다.

# 장착 위치와 방향: 로봇 기준 (x 앞, y 왼쪽, rad). /tf 에서 잰 값입니다.
_LIDAR = {
    "/front_2d_lidar/scan": (0.026, 0.0, 3.142),
    "/back_2d_lidar/scan": (-0.49, 0.0, 3.142),
}
# 로봇 외형(로봇 기준 x 최소, x 최대, y 최소, y 최대). 이 안에 맞은 점은 로봇 자신의 몸체입니다.
_BODY = (-0.70, 0.25, -0.34, 0.34)

_node = None
_rclpy = None
_scans = {}  # 토픽 -> 가장 최근 스캔의 점 목록 (로봇 기준 (x, y))


def sensor_setup():
    """ROS 2 노드와 구독을 한 번만 만듭니다. rclpy 는 여기서 가져옵니다.
    모듈 맨 위에서 import rclpy 를 하면 Isaac Sim 이 켜지기 전이라 실행이 실패합니다."""
    global _node, _rclpy
    if _node is not None:
        return
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    _rclpy = rclpy
    _node = rclpy.create_node("b2_lidar_reader")
    for topic in _LIDAR:
        _node.create_subscription(LaserScan, topic, lambda m, k=topic: _on_scan(k, m), qos_profile_sensor_data)


def _on_scan(topic, msg):
    """스캔의 빔마다 점을 만듭니다. 빔 방향 = 장착 방향 + 빔 각도, 점 = 장착 위치 + 거리 x 방향."""
    mx, my, myaw = _LIDAR[topic]
    pts = []
    a = msg.angle_min
    for r in msg.ranges:
        if msg.range_min <= r <= msg.range_max:
            bx = mx + r * math.cos(myaw + a)
            by = my + r * math.sin(myaw + a)
            if not (_BODY[0] <= bx <= _BODY[1] and _BODY[2] <= by <= _BODY[3]):
                pts.append((bx, by))
        a += msg.angle_increment
    _scans[topic] = pts


def sensor_points():
    """두 라이다의 가장 최근 스캔을 합친 점 목록 (로봇 기준 (x 앞, y 왼쪽)). controller() 에서 매 호출마다 부르십시오.
    처음 몇 번의 호출에서는 빈 목록이 나올 수 있습니다."""
    sensor_setup()
    _rclpy.spin_once(_node, timeout_sec=0.0)
    pts = []
    for v in _scans.values():
        pts.extend(v)
    return pts


def to_world(points, pose):
    """로봇 기준 점을 월드 (x, y) 로 바꿉니다. pose 는 controller() 가 준 (x, y, yaw) 입니다."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in points]


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
