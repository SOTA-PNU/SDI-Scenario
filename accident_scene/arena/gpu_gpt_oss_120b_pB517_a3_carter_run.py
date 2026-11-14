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
  choose_command(v_follow, w_follow, target_xy, horizon_s=3.0) returns
  (v, w, info) after checking all named candidate commands and applying
  the fixed safety/progress priority; use its returned command every step.

  check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) checks the robot body
  along a constant command against tracked lidar surface points and the static map;
  it returns minimum gaps, first hit times relative to now, and the end pose.

  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   None - the controller gets no harness handle; it reads the robot's
          own sensors with the functions above the [EDIT REGION]
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

# ---- Carter 센서 기본 코드 ----
# 좌표는 로봇 기준 x 앞, y 왼쪽, z 바닥에서의 높이 (m) 입니다. 장착값은 /tf 의 base_link -> 센서 프레임 값입니다.
import numpy as np

_LIDAR2D = {"front": ("/front_2d_lidar/scan", (0.026, 0.0, math.pi)),   # (토픽, 장착 (x, y, 방향 rad))
            "back": ("/back_2d_lidar/scan", (-0.489, 0.0, 0.0))}
_LIDAR3D = ("/front_3d_lidar/lidar_points", (-0.232, 0.0, 0.526))      # (토픽, 장착 (x, y, z)), 회전 없음
_CAMERA_TOPIC = "/front_stereo_camera/left/image_raw"
_CAMERA_POS = (0.100, 0.075, 0.346)                                    # 카메라 광학 중심 (x, y, z)
_CAMERA_K = (957.81, 957.81, 960.0, 600.0)                             # fx, fy, cx, cy (camera_info)
_CAMERA_D = (0.147811, -0.032313, -0.000194, -0.000035,                # 렌즈 왜곡 (camera_info, rational_polynomial)
             0.008823, 0.517913, -0.06708, 0.01695)                    # k1 k2 p1 p2 k3 k4 k5 k6
_CAMERA_R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # 로봇 기준 -> 카메라 (x 오른쪽, y 아래, z 앞)
_BASE_SELF_BOX = (-0.70, 0.25, -0.34, 0.34)                            # 자기 점을 지우는 상자(몸체보다 큼) (x 최소, x 최대, y 최소, y 최대)

_BASE_SENSOR_STATE = {"node": None, "rclpy": None, "subs": set(), "count": 0, "scan": {}, "cloud": [], "image": None}


def _subscribe(key, msg_type, topic, callback):
    """센서 하나를 처음 쓸 때 한 번만 구독합니다. rclpy 는 여기서 가져옵니다
    (모듈 맨 위에서 가져오면 Isaac Sim 이 켜지기 전이라 실패합니다)."""
    if key in _BASE_SENSOR_STATE["subs"]:
        return
    if _BASE_SENSOR_STATE["node"] is None:
        import rclpy
        _BASE_SENSOR_STATE["rclpy"] = rclpy
        _BASE_SENSOR_STATE["node"] = rclpy.create_node("b2_sensor_reader")
    from rclpy.qos import qos_profile_sensor_data
    _BASE_SENSOR_STATE["node"].create_subscription(msg_type, topic, callback, qos_profile_sensor_data)
    _BASE_SENSOR_STATE["subs"].add(key)


def _spin():
    """쌓여 있는 센서 메시지를 모두 처리합니다. 아래 센서 함수들이 부릅니다."""
    for _ in range(20):
        n = _BASE_SENSOR_STATE["count"]
        _BASE_SENSOR_STATE["rclpy"].spin_once(_BASE_SENSOR_STATE["node"], timeout_sec=0.0)
        if _BASE_SENSOR_STATE["count"] == n:
            break


def _drop_body(p):
    """로봇 자신의 몸체에 맞은 점을 뺍니다."""
    body = (p[:, 0] >= _BASE_SELF_BOX[0]) & (p[:, 0] <= _BASE_SELF_BOX[1]) & (p[:, 1] >= _BASE_SELF_BOX[2]) & (p[:, 1] <= _BASE_SELF_BOX[3])
    return p[~body]


def _on_scan(which, msg):
    mx, my, myaw = _LIDAR2D[which][1]
    r = np.asarray(msg.ranges, dtype=np.float64)
    a = msg.angle_min + np.arange(r.size) * msg.angle_increment + myaw
    ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    pts = np.stack([mx + r[ok] * np.cos(a[ok]), my + r[ok] * np.sin(a[ok])], axis=1)
    _BASE_SENSOR_STATE["scan"][which] = _drop_body(pts)
    _BASE_SENSOR_STATE["count"] += 1


def _on_cloud(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}
    xyz = np.stack([raw[:, off[c]:off[c] + 4].copy().view(np.float32)[:, 0] for c in "xyz"], axis=1)
    pts = xyz.astype(np.float64) + np.array(_LIDAR3D[1])
    _BASE_SENSOR_STATE["cloud"] = (_BASE_SENSOR_STATE["cloud"] + [_drop_body(pts)])[-6:]
    _BASE_SENSOR_STATE["count"] += 1


def _on_image(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    _BASE_SENSOR_STATE["image"] = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    _BASE_SENSOR_STATE["count"] += 1


def lidar2d_points(which="front"):
    """2D 라이다 which ("front" 앞쪽, "back" 뒤쪽) 의 가장 최근 스캔을 로봇 기준 점 배열 (N, 2) [x, y] 로 돌려줍니다.
    두 라이다 모두 바닥 위 0.42 m 높이의 평면을 360° 로, 0.1 s 마다 한 바퀴 훑습니다(빔 3,200개, 0.05–30 m). 새 스캔은 처음 돌려줄 때 이미 약 0.1 s 전에 잰 것이고, 같은 스캔이 다음 호출에 새 배열로 다시 올 수 있습니다(내용은 직전과 같음).
    controller() 에서 매번 부르십시오. 처음 몇 번은 빈 배열일 수 있습니다."""
    from sensor_msgs.msg import LaserScan
    _subscribe("lidar2d_" + which, LaserScan, _LIDAR2D[which][0], lambda m: _on_scan(which, m))
    _spin()
    return _BASE_SENSOR_STATE["scan"].get(which, np.zeros((0, 2)))


def lidar3d_points():
    """3D 라이다 (Hesai XT-32, 바닥 위 0.53 m, 32채널) 의 최근 한 바퀴를 로봇 기준 점 배열 (N, 3) [x, y, z] 로 돌려줍니다.
    이 라이다는 메시지 하나에 약 60° 조각만 담아 1/60 s 마다 보내므로, 최근 6 조각(약 0.1 s)을 모아 360° 를 만듭니다.
    z 는 바닥에서의 높이이며 바닥에 맞은 점은 z 가 0 근처입니다. 처음 몇 번은 빈 배열일 수 있습니다."""
    from sensor_msgs.msg import PointCloud2
    _subscribe("lidar3d", PointCloud2, _LIDAR3D[0], _on_cloud)
    _spin()
    return np.concatenate(_BASE_SENSOR_STATE["cloud"]) if _BASE_SENSOR_STATE["cloud"] else np.zeros((0, 3))


def camera_image():
    """앞쪽 스테레오 카메라의 왼쪽 카메라 영상을 배열 (1200, 1920, 3) uint8 [RGB] 로 돌려줍니다. 아직 없으면 None.
    1/60 s 마다 새 영상이 옵니다. 수평 약 ±60°, 수직 약 ±36° 를 보는 광각이고, 영상에 렌즈 왜곡이 들어 있습니다
    (camera_project, camera_ray 가 이 왜곡을 반영합니다)."""
    from sensor_msgs.msg import Image
    _subscribe("camera", Image, _CAMERA_TOPIC, _on_image)
    _spin()
    return _BASE_SENSOR_STATE["image"]


def camera_project(points):
    """로봇 기준 점 (N, 3) [x, y, z] 를 영상 픽셀 좌표 (N, 2) [u 오른쪽, v 아래] 로 바꿉니다.
    둘째 값은 그 점이 영상 안에 보이는 위치인지 나타내는 (N,) bool 배열입니다(가림은 따지지 않습니다)."""
    p = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.array(_CAMERA_POS)) @ _CAMERA_R.T
    z = p[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    x, y = p[:, 0] / zs, p[:, 1] / zs
    k1, k2, p1, p2, k3, k4, k5, k6 = _CAMERA_D
    r2 = x * x + y * y
    rad = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    fx, fy, cx, cy = _CAMERA_K
    uv = np.stack([fx * xd + cx, fy * yd + cy], axis=1)
    ok = (z > 0.05) & (r2 < 9.0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1200)
    return uv, ok


def camera_ray(uv):
    """영상 픽셀 좌표 (N, 2) [u, v] 를 로봇 기준 단위 방향 (N, 3) 으로 바꿉니다. 광선은 _CAMERA_POS 에서 출발합니다."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    fx, fy, cx, cy = _CAMERA_K
    k1, k2, p1, p2, k3, k4, k5, k6 = _CAMERA_D
    xd, yd = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x * x + y * y
        ic = (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3) / (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) * ic, (yd - dy) * ic
    d = np.stack([x, y, np.ones_like(x)], axis=1) @ _CAMERA_R
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def to_world(points, pose):
    """로봇 기준 점 (N, 2) 또는 (N, 3) 을 월드 좌표로 바꿉니다(z 는 그대로). pose 는 그 점을 잰 때의 로봇 자세 (x, y, yaw) 입니다. 라이다 점은 controller() 가 받은 지금 pose 보다 앞서 잰 것입니다."""
    p = np.array(points, dtype=np.float64)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * p[:, 0] - s * p[:, 1]
    wy = y + s * p[:, 0] + c * p[:, 1]
    p[:, 0], p[:, 1] = wx, wy
    return p


# ---- 지도 기본 코드 ----
# 지도는 이 장면에서 자리가 변하지 않는 물체입니다. 직사각형은 월드 좌표 축에 나란한
# (x 최소, x 최대, y 최소, y 최대) [m] 이고, 값은 이 장면의 USD 에서 읽은 경계 상자입니다.
# MISSION 상자는 실행할 때 MISSION["boxes"] 에서 읽습니다. 움직이는 물체는 지도에 없습니다.
# 경로 계산에 쓸 표준 라이브러리 heapq, collections, deque 는 여기서 import 해 둡니다.
import collections  # noqa: E402,F401
import heapq  # noqa: E402,F401
from collections import deque  # noqa: E402,F401

WALLS = (-10.30, 9.35, -12.15, 17.95)        # 벽 안쪽 면: x 최소, x 최대, y 최소, y 최대

FIXED_OBSTACLES = [                          # MISSION 상자 말고 바닥에 놓인 고정물 43개: (이름, x 최소, x 최대, y 최소, y 최대)
    ("SM_PillarPartA_9M3_1445", -10.45, -10.00, -9.25, -8.80),
    ("SM_PillarPartA_9M4_1446", 9.10, 9.55, -9.20, -8.75),
    ("SM_Rackshield_43", -9.95, -8.70, -3.80, -3.40),
    ("SM_RackFrame_167", -9.85, -8.80, -3.80, -3.60),
    ("SM_PaletteA_357", -9.90, -8.80, -3.45, -2.20),
    ("Forklift_01", -9.60, -6.05, -3.45, -2.25),
    ("Forklift", 1.85, 5.30, -3.45, -0.70),
    ("Extras_BoxSetA", -9.60, -8.85, -3.35, -1.05),
    ("SM_PillarPartA_9M5_1454", -10.45, -10.00, -3.25, -2.80),
    ("SM_PillarPartA_9M6_1455", 9.10, 9.55, -3.20, -2.75),
    ("SM_PaletteA_358", -9.85, -8.80, -2.25, -0.95),
    ("SM_PaletteA_359", -9.90, -8.80, -1.00, 0.25),
    ("Extras_BoxSetA", -9.60, -8.85, -0.90, 0.20),
    ("SM_RackFrame_159", -9.85, -8.80, 0.20, 0.40),
    ("SM_PaletteA_11", -9.85, -8.80, 0.45, 1.70),
    ("Extras_BoxSetB", -9.85, -8.75, 0.60, 4.05),
    ("SM_PaletteA_12", -9.85, -8.80, 1.65, 2.95),
    ("SM_PillarPartA_9M7_1463", -10.45, -10.00, 2.75, 3.20),
    ("SM_PillarPartA_9M8_1464", 9.10, 9.55, 2.80, 3.25),
    ("SM_PaletteA_319", -9.85, -8.80, 2.90, 4.15),
    ("SM_RackFrame_161", -9.85, -8.80, 4.20, 4.40),
    ("SM_PaletteA_5", -9.75, -8.70, 4.45, 5.70),
    ("Extras_Misc", -9.35, -8.70, 4.75, 5.40),
    ("SM_PaletteA_6", -9.75, -8.70, 5.65, 6.95),
    ("Extras_Misc", -9.70, -8.75, 5.65, 8.25),
    ("SM_PaletteA_318", -9.75, -8.70, 6.90, 8.15),
    ("SM_RackFrame_163", -9.85, -8.80, 8.20, 8.40),
    ("SM_PaletteA_78", -9.75, -8.70, 8.45, 9.70),
    ("KLT_Bins", -9.75, -9.00, 8.70, 9.65),
    ("SM_PillarPartA_9M9_1472", -10.45, -10.00, 8.75, 9.20),
    ("SM_PillarPartA_9M10_1473", 9.10, 9.55, 8.80, 9.25),
    ("SM_PaletteA_77", -9.75, -8.70, 9.65, 10.90),
    ("KLT_Bins", -9.75, -9.00, 9.85, 10.80),
    ("SM_PaletteA_76", -9.75, -8.70, 10.85, 12.15),
    ("KLT_Bins", -9.75, -9.00, 11.15, 12.10),
    ("SM_Rackshield_50", -9.95, -8.70, 12.10, 12.50),
    ("SM_RackFrame_165", -9.85, -8.80, 12.20, 12.40),
    ("SM_PushcartA_02_22", -7.05, -5.20, 12.90, 14.35),
    ("SM_PaletteA_360", -6.85, -5.35, 13.00, 14.40),
    ("KLT_Bins", -6.60, -5.50, 13.20, 14.15),
    ("SM_PillarPartA_9M11_1481", -10.45, -10.00, 14.75, 15.20),
    ("SM_PillarPartA_9M12_1482", 9.10, 9.55, 14.80, 15.25),
    ("SM_FuseBox_24", -4.15, -4.05, 17.75, 18.00),
]


_MAP_RECTS = []  # 처음 쓸 때 한 번 만들어 둡니다


def _map_rects():
    if not _MAP_RECTS:
        boxes = [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0, b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0)
                 for b in MISSION["boxes"]]
        _MAP_RECTS.append(np.array(boxes + [r[1:] for r in FIXED_OBSTACLES], dtype=np.float64))
    return _MAP_RECTS[0]


def map_rects():
    """지도 물체의 직사각형 배열 (K, 4): MISSION 상자와 FIXED_OBSTACLES. 벽은 WALLS 에 따로 있습니다."""
    return _map_rects().copy()


def map_clearance(x, y):
    """월드 좌표 (x, y) 에서 가장 가까운 지도 물체(상자, 고정물, 벽 안쪽 면)까지의 거리 [m]. 물체 안이면 0 입니다.
    숫자 입력은 float, numpy 배열 입력은 x, y 를 방송한 모양의 배열을 돌려줍니다.

    배열은 모든 점의 최근접 거리가 될 수 없는 먼 사각형을 먼저 빼 계산합니다. 거리값은 같은 식으로 잽니다."""
    r = _map_rects()
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    wx0, wx1, wy0, wy1 = WALLS
    wall_gap = np.maximum(np.minimum(np.minimum(x - wx0, wx1 - x), np.minimum(y - wy0, wy1 - y)), 0.0)
    if x.ndim and x.size:
        # For any point p in the bounding box, distance to the rectangle nearest
        # its centre is <= distance(centre, rectangle) + radius of that box.
        # A rectangle farther from the whole box than this upper bound cannot win.
        x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dx = np.maximum(np.maximum(r[:, 0] - cx, cx - r[:, 1]), 0.0)
        dy = np.maximum(np.maximum(r[:, 2] - cy, cy - r[:, 3]), 0.0)
        upper = min(float(wall_gap.max()), float(np.hypot(dx, dy).min()) + math.hypot(x1 - x0, y1 - y0) * 0.5)
        bx = np.maximum(np.maximum(r[:, 0] - x1, x0 - r[:, 1]), 0.0)
        by = np.maximum(np.maximum(r[:, 2] - y1, y0 - r[:, 3]), 0.0)
        r = r[bx * bx + by * by <= upper * upper + 1e-12]
    if len(r) == 0 or x.size == 0:
        return float(wall_gap) if wall_gap.ndim == 0 else wall_gap
    dx = np.maximum(np.maximum(r[:, 0] - x[..., None], x[..., None] - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - y[..., None], y[..., None] - r[:, 3]), 0.0)
    gap = np.minimum(np.hypot(dx, dy).min(axis=-1), wall_gap)
    return float(gap) if gap.ndim == 0 else gap


def segment_clearance(p, q):
    """점 p=(x, y) 에서 q 까지 곧게 갈 때 map_clearance 의 최솟값 [m] 입니다(0.05 m 간격으로 잽니다)."""
    n = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05)))
    k = np.arange(n + 1, dtype=np.float64)
    xs, ys = p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n
    r = _map_rects()
    dx = np.maximum(np.maximum(r[None, :, 0] - xs[:, None], xs[:, None] - r[None, :, 1]), 0.0)
    dy = np.maximum(np.maximum(r[None, :, 2] - ys[:, None], ys[:, None] - r[None, :, 3]), 0.0)
    wx0, wx1, wy0, wy1 = WALLS
    d = np.minimum(np.hypot(dx, dy).min(axis=1), np.minimum(np.minimum(xs - wx0, wx1 - xs), np.minimum(ys - wy0, wy1 - ys)))
    return float(max(0.0, float(d.min())))


def unmapped_points(points_world, margin):
    """월드 좌표 점 (N, 2) 또는 (N, 3) 가운데 지도에 없는 물체의 점만 돌려줍니다: 지도 물체(상자, 고정물)에서
    margin [m] 넘게 떨어져 있고 벽 안쪽 면보다 margin 넘게 안쪽에 있는 점. x, y 만 봅니다. 3D 라이다 점은
    바닥이나 높은 곳의 점을 먼저 높이(z)로 거르십시오."""
    p = np.asarray(points_world, dtype=np.float64)
    if len(p) == 0:
        return p
    r = map_rects()
    px, py = p[:, 0:1], p[:, 1:2]
    near = ((px >= r[:, 0] - margin) & (px <= r[:, 1] + margin)
            & (py >= r[:, 2] - margin) & (py <= r[:, 3] + margin)).any(axis=1)
    wx0, wx1, wy0, wy1 = WALLS
    inside = (p[:, 0] > wx0 + margin) & (p[:, 0] < wx1 - margin) & (p[:, 1] > wy0 + margin) & (p[:, 1] < wy1 - margin)
    return p[~near & inside]


# ---- 감지 기본 코드 (물체 추적) ----
# 앞뒤 2D 라이다로 지도에 없는 물체를 묶어 추적합니다. base 가 controller() 를 부르기 직전마다 로봇 자세를 적고(_BASE_POSES)
# 추적을 갱신하므로(_base_track_step), controller() 안에서는 observed_objects() 로 결과만 받으면 됩니다.
# 이 부분도 [EDIT REGION] 밖이므로 고치지 마십시오.
_BASE_OBJ_CFG = dict(age=0.1, near=8.0, margin=0.25, voxel=0.1, gap=0.45, gate=1.0, vwin=0.8, vmin_n=4, vmin_span=0.3,
                     vmax=2.0, keep=1.0, stale=0.3, frag_n=3, frag_join=0.6)
_BASE_POSES = []                                  # (t, x, y, yaw), 최근 1 s
_BASE_OBJ_ST = {"prev": {"front": None, "back": None}, "pts": {"front": None, "back": None},
                "pts_t": {"front": None, "back": None}, "tracks": [], "nid": 0, "error": None}


def _base_obj_pose_at(t):
    """적어 둔 자세에서 시각 t 의 자세를 직선으로 보간합니다."""
    h = _BASE_POSES
    if not h:
        return None
    if t <= h[0][0]:
        return h[0][1:]
    for (ta, xa, ya, wa), (tb, xb, yb, wb) in zip(reversed(h[:-1]), reversed(h[1:])):
        if ta <= t:
            f = 0.0 if tb == ta else (t - ta) / (tb - ta)
            dw = math.atan2(math.sin(wb - wa), math.cos(wb - wa))
            return xa + f * (xb - xa), ya + f * (yb - ya), wa + f * dw
    return h[-1][1:]


def _base_obj_cluster(P):
    """gap 이하로 떨어진 점끼리 묶습니다."""
    n = len(P)
    if n == 0:
        return []
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) <= _BASE_OBJ_CFG["gap"]
    label = -np.ones(n, dtype=int)
    groups = []
    for i in range(n):
        if label[i] >= 0:
            continue
        label[i] = len(groups)
        stack, members = [i], []
        while stack:
            k = stack.pop()
            members.append(k)
            for m in np.nonzero(d[k] & (label < 0))[0]:
                label[m] = label[i]
                stack.append(m)
        groups.append(P[members])
    return groups


def _base_obj_velocity(hist):
    """최근 vwin 초의 중심 위치를 직선으로 맞춘 속도 (vx, vy, 맞춤 여부). 자료가 모자라면 (0, 0, False)."""
    cfg = _BASE_OBJ_CFG
    t_end = hist[-1][0]
    h = np.array([q for q in hist if t_end - q[0] <= cfg["vwin"]])
    if len(h) < cfg["vmin_n"] or h[-1, 0] - h[0, 0] < cfg["vmin_span"]:
        return 0.0, 0.0, False
    tt = h[:, 0] - h[:, 0].mean()
    den = float((tt * tt).sum())
    vx = float((tt * (h[:, 1] - h[:, 1].mean())).sum() / den)
    vy = float((tt * (h[:, 2] - h[:, 2].mean())).sum() / den)
    sp = math.hypot(vx, vy)
    if sp > cfg["vmax"]:
        vx, vy = vx * cfg["vmax"] / sp, vy * cfg["vmax"] / sp
    return vx, vy, True


def _base_track_step(t):
    """base 가 controller() 를 부르기 직전마다 부릅니다. 새 스캔이 왔으면(내용이 직전과 다르면) 그 스캔을 잰 때의 자세로
    점을 바꿔 물체를 다시 묶고 추적을 이어 붙입니다. 1.0 s 넘게 보이지 않은 물체는 매 스텝 지웁니다."""
    st, cfg = _BASE_OBJ_ST, _BASE_OBJ_CFG
    t_scan = t - cfg["age"]
    new = False
    for which in ("front", "back"):
        p = lidar2d_points(which)
        q = st["prev"][which]
        if q is not None and p.shape == q.shape and np.array_equal(p, q):
            continue
        st["prev"][which] = np.array(p, copy=True)
        pose = _base_obj_pose_at(t_scan)
        st["pts"][which] = to_world(p, pose)[:, :2] if len(p) and pose is not None else np.zeros((0, 2))
        st["pts_t"][which] = t_scan
        new = True
    tracks = st["tracks"]
    if new and _BASE_POSES:
        x0, y0 = _BASE_POSES[-1][1], _BASE_POSES[-1][2]
        parts = [P for w, P in st["pts"].items()
                 if P is not None and len(P) and t_scan - st["pts_t"][w] <= cfg["stale"]]
        P = np.concatenate(parts) if parts else np.zeros((0, 2))
        if len(P):
            P = P[np.hypot(P[:, 0] - x0, P[:, 1] - y0) < cfg["near"]]
        if len(P):
            P = unmapped_points(P, cfg["margin"])
        if len(P):
            _, keep = np.unique(np.floor(P / cfg["voxel"]).astype(np.int64), axis=0, return_index=True)
            P = P[keep]
        groups = _base_obj_cluster(P)
        big = [g for g in groups if len(g) >= cfg["frag_n"]]
        for g in (g for g in groups if len(g) < cfg["frag_n"]):
            if big:
                dmin = [float(np.hypot(B[:, None, 0] - g[None, :, 0], B[:, None, 1] - g[None, :, 1]).min()) for B in big]
                k = int(np.argmin(dmin))
                if dmin[k] <= cfg["frag_join"]:
                    big[k] = np.concatenate([big[k], g])
                    continue
            big.append(g)
        for tr in tracks:
            tr["hit"] = False
        for grp in big:
            c = grp.mean(axis=0)
            best, bd = None, cfg["gate"]
            for tr in tracks:
                if tr["hit"]:
                    continue
                lt, lx, ly = tr["hist"][-1]
                dd = math.hypot(c[0] - (lx + tr["vx"] * (t_scan - lt)), c[1] - (ly + tr["vy"] * (t_scan - lt)))
                if dd < bd:
                    best, bd = tr, dd
            if best is None:
                best = {"id": st["nid"], "hist": [], "vx": 0.0, "vy": 0.0, "vel_ok": False, "born_t": t_scan}
                st["nid"] += 1
                tracks.append(best)
            best["hit"], best["seen_t"] = True, t_scan
            best["radius"] = float(np.hypot(grp[:, 0] - c[0], grp[:, 1] - c[1]).max())
            best["points"] = grp.tolist()  # observed world points at seen_t; no hidden shape model
            best["n_points"] = int(len(grp))
            best["hist"] = [h for h in best["hist"] if t_scan - h[0] <= 2.0] + [(t_scan, float(c[0]), float(c[1]))]
            best["vx"], best["vy"], best["vel_ok"] = _base_obj_velocity(best["hist"])
    st["tracks"] = [tr for tr in tracks if t_scan - tr["seen_t"] <= cfg["keep"]]


def observed_objects():
    """지도에 없는 물체를 추적한 결과를 리스트로 돌려줍니다. 가까운 것부터 정렬되어 있습니다.

    base 가 controller() 를 부르기 직전마다 추적을 갱신하므로, controller() 안에서 언제 몇 번 불러도 같은 결과를 줍니다.
    갱신: 앞뒤 2D 라이다에 새 스캔이 오면(내용이 직전과 다를 때만), 그 스캔을 잰 때(받기 0.1 s 전)의 로봇 자세로 점을
    월드 좌표로 바꿉니다. 로봇에서 8 m 안이면서 지도 물체와 벽에서 0.25 m 넘게 떨어진 점만 남겨 0.1 m 칸으로 솎고, 0.45 m
    이하로 떨어진 점끼리 묶어 물체 하나로 봅니다(점이 3개보다 적은 조각은 0.6 m 안의 큰 묶음에 합칩니다). 묶음은 직전 물체의
    예측 위치에서 1.0 m 안이면 같은 물체로 이어 붙이고, 최근 0.8 s 의 중심 위치를 직선으로 맞춰 속도를 구합니다.

    물체마다 dict: id(같은 물체면 같은 번호), x, y(보인 점들의 중심, 월드 좌표 [m]), vx, vy, speed([m/s]),
    vel_ok(속도를 맞출 자료가 충분했는지: 최근 0.8 s 안에 4번 넘게, 0.3 s 넘게 보였을 때 True), radius(중심에서 가장 먼
    점까지 [m]), n_points(솎은 뒤 점 수), seen_t(그 물체를 마지막으로 본 스캔의 시각 [s]), age_s(지금 시각 - seen_t),
    track_age_s(처음 본 스캔부터 seen_t 까지 [s]), dist(지금 pose 의 점에서 중심까지 [m]).
    points 는 seen_t 때 관측한 실제 라이다 점들의 월드 (x, y) 리스트입니다. 반환한 리스트는 추적 상태와 분리되어 있습니다.
    x, y, vx, vy 와 points 는 seen_t 때의 값입니다. 1.0 s 넘게 보이지 않은 물체는 빠집니다.

    주의: 라이다는 로봇 쪽을 향한 면만 보므로 중심은 물체의 실제 중심과 다르고, 보이는 면이 바뀌면 중심이 옮겨 가 속도에
    잡음이 섞입니다. 속도는 최근 0.8 s 위치의 직선 맞춤이라 실제 변화보다 늦게 따라갑니다. vel_ok 가 False 면 vx, vy 는
    0 이지만 정지했다는 뜻이 아닙니다. 지도에 없는 정지 물체도 나옵니다."""
    if not _BASE_POSES:
        return []
    t_now, x0, y0, _ = _BASE_POSES[-1]
    out = []
    for tr in _BASE_OBJ_ST["tracks"]:
        lt, lx, ly = tr["hist"][-1]
        out.append({"id": tr["id"], "x": lx, "y": ly, "vx": tr["vx"], "vy": tr["vy"], "speed": math.hypot(tr["vx"], tr["vy"]),
                    "vel_ok": tr["vel_ok"], "radius": tr["radius"], "n_points": tr["n_points"], "seen_t": lt,
                    "age_s": t_now - lt, "track_age_s": lt - tr["born_t"], "dist": math.hypot(lx - x0, ly - y0),
                    "points": [list(p) for p in tr["points"]]})
    out.sort(key=lambda o: o["dist"])
    return out


# ---- 명령별 몸체 검사 ----
# 실제 검사 몸체 (x 최소, x 최대, y 최소, y 최대). _BASE_SELF_BOX 는 라이다 자기 점 제거용이며 이 검사에 쓰지 않습니다.
_BASE_BODY_BOX = (-0.59, 0.13, -0.25, 0.25)


def _base_body_perimeter():
    """몸체 둘레를 간격 0.1 m 이하로 나눕니다. 꼭짓점도 모두 포함합니다."""
    x0, x1, y0, y1 = _BASE_BODY_BOX
    xx = np.linspace(x0, x1, int(math.ceil((x1 - x0) / 0.1)) + 1)
    yy = np.linspace(y0, y1, int(math.ceil((y1 - y0) / 0.1)) + 1)
    return np.concatenate((np.column_stack((xx, np.full_like(xx, y0))),
                           np.column_stack((xx, np.full_like(xx, y1))),
                           np.column_stack((np.full_like(yy[1:-1], x0), yy[1:-1])),
                           np.column_stack((np.full_like(yy[1:-1], x1), yy[1:-1]))))


_BASE_BODY_PERIMETER = _base_body_perimeter()


_BASE_MOTION_UNCERTAINTY = {'dynamic_margin': 0.15, 'uncertainty_gain': 1.0, 'uncertainty_cap': 1.0, 'response_window': 1.2, 'unknown_margin': 0.15, 'fast_preference': True, 'fast_scale': 2.0, 'fast_min': 1.0, 'emergency_gap_first': True}


def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:
    """직선 예측한 관측 점과 몸체를 검사합니다. 지도 여유는 기존대로 유지합니다.

    min_gap_obj/min_gap_map 은 원래 기하 간격입니다. 관측 물체의 속도와 관측 나이로 제한된 불확실성 여유를
    더한 판정이 t_hit_obj/t_hit 입니다. raw_t_hit_obj/raw_t_hit 은 추가 여유 없는 판정입니다.
    robust_min_gap_obj 는 원래 점 간격에서 불확실성 여유를 뺀 최솟값입니다. 멈춤도 같은 방식으로 검사합니다.
    불확실성은 관측 속도만 사용하며, 보이지 않은 물체의 위치나 이후 운동 형태를 가정하지 않습니다."""
    v, w, horizon_s, dt, margin_m = map(float, (v, w, horizon_s, dt, margin_m))
    if not all(math.isfinite(q) for q in (v, w, horizon_s, dt, margin_m)):
        raise ValueError("check_motion requires finite inputs")
    if horizon_s < 0.0 or dt <= 0.0 or margin_m < 0.0:
        raise ValueError("check_motion requires horizon_s >= 0, dt > 0, margin_m >= 0")
    if not _BASE_POSES:
        raise RuntimeError("check_motion requires the current controller pose")
    t_now, x0, y0, yaw0 = _BASE_POSES[-1]
    if not all(math.isfinite(q) for q in (t_now, x0, y0, yaw0)):
        raise ValueError("check_motion requires a finite current pose and timestamp")
    tt = np.minimum(np.arange(int(math.ceil(horizon_s / dt)) + 1, dtype=np.float64) * dt, horizon_s)
    half_turn = 0.5 * w * tt
    travel = v * tt * np.sinc(half_turn / math.pi)
    xx, yy = x0 + travel * np.cos(yaw0 + half_turn), y0 + travel * np.sin(yaw0 + half_turn)
    yaw = yaw0 + w * tt
    cc, ss = np.cos(yaw), np.sin(yaw)
    objects = observed_objects()
    min_gap_obj, t_hit_obj, hit_obj_id = math.inf, None, None
    raw_t_hit_obj, robust_min_gap_obj = None, math.inf
    end_gap_obj = robust_end_gap_obj = future_min_gap_obj = math.inf
    if objects:
        if not all(math.isfinite(o[key]) for o in objects for key in ("vx", "vy", "seen_t")):
            raise ValueError("check_motion requires finite observed velocities and timestamps")
        groups = [np.asarray(o["points"], dtype=np.float64).reshape(-1, 2) for o in objects]
        if any(len(p) == 0 or not np.isfinite(p).all() for p in groups):
            raise ValueError("check_motion requires nonempty finite observed points")
        which = np.repeat(np.arange(len(objects)), [len(p) for p in groups])
        points = np.concatenate(groups, axis=0)
        vel_ok = np.array([bool(o["vel_ok"]) for o in objects])
        ovx = np.array([o["vx"] if o["vel_ok"] else 0.0 for o in objects])
        ovy = np.array([o["vy"] if o["vel_ok"] else 0.0 for o in objects])
        seen = np.array([o["seen_t"] for o in objects])
        age = t_now + tt[:, None] - seen[which]
        dx = points[:, 0] + age * ovx[which] - xx[:, None]
        dy = points[:, 1] + age * ovy[which] - yy[:, None]
        bx = cc[:, None] * dx + ss[:, None] * dy
        by = -ss[:, None] * dx + cc[:, None] * dy
        xmin, xmax, ymin, ymax = _BASE_BODY_BOX
        qx = np.maximum(xmin - bx, bx - xmax)
        qy = np.maximum(ymin - by, by - ymax)
        gaps = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
        min_gap_obj, end_gap_obj = float(gaps.min()), float(gaps[-1].min())
        raw_hit = np.flatnonzero(np.any(gaps <= margin_m + 1e-12, axis=1))
        raw_t_hit_obj = float(tt[int(raw_hit[0])]) if raw_hit.size else None
        cfg = _BASE_MOTION_UNCERTAINTY
        speed = np.hypot(ovx, ovy)
        moving = vel_ok & (speed > 0.05)
        elapsed = np.maximum(t_now - seen, 0.0)[None, :] + np.minimum(tt[:, None], cfg["response_window"])
        extra = cfg["dynamic_margin"] + np.minimum(cfg["uncertainty_cap"], cfg["uncertainty_gain"] * elapsed * speed)
        extra = np.where(moving[None, :], extra, np.where(vel_ok[None, :], 0.0, cfg["unknown_margin"]))
        robust = gaps - extra[:, which]
        robust_min_gap_obj = float(robust.min())
        robust_end_gap_obj = float(robust[-1].min())
        future_min_gap_obj = float(robust[1:].min()) if len(tt) > 1 else robust_min_gap_obj
        hit = np.flatnonzero(np.any(robust <= margin_m + 1e-12, axis=1))
        if hit.size:
            i = int(hit[0])
            j = int(np.flatnonzero(robust[i] <= margin_m + 1e-12)[0])
            t_hit_obj, hit_obj_id = float(tt[i]), objects[int(which[j])]["id"]
    perimeter = _BASE_BODY_PERIMETER
    px = xx[:, None] + cc[:, None] * perimeter[:, 0] - ss[:, None] * perimeter[:, 1]
    py = yy[:, None] + ss[:, None] * perimeter[:, 0] + cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(tt[int(hit_map[0])]) if hit_map.size else None
    hits = [q for q in (t_hit_obj, t_hit_map) if q is not None]
    raw_hits = [q for q in (raw_t_hit_obj, t_hit_map) if q is not None]
    return {"min_gap_obj": min_gap_obj, "t_hit_obj": t_hit_obj, "hit_obj_id": hit_obj_id,
            "min_gap_map": min_gap_map, "t_hit_map": t_hit_map, "t_hit": min(hits) if hits else None,
            "end_pose": (float(xx[-1]), float(yy[-1]), float(yaw[-1])),
            "raw_t_hit_obj": raw_t_hit_obj, "raw_t_hit": min(raw_hits) if raw_hits else None,
            "robust_min_gap_obj": robust_min_gap_obj, "end_gap_obj": end_gap_obj,
            "robust_end_gap_obj": robust_end_gap_obj, "future_min_gap_obj": future_min_gap_obj}


# ---- 후보 명령 평가와 선택 ----
_BASE_CHOOSE_DBG_WINDOW = None


_BASE_ESCAPE_CONFIG = {'slow_v': 0.15, 'fast_v': 1.0, 'curve_v': 1.0, 'curve_gain': 1.5, 'curve_max_w': 0.4}


def _base_command_candidates(v_follow, w_follow, target_xy=None):
    """추종이 회전뿐이어도 검사할 전진 후보를 유지합니다."""
    candidates = [("follow", v_follow, w_follow), ("slow", 0.5 * v_follow, w_follow), ("stop", 0.0, 0.0)]
    if v_follow > 0.0:
        cfg = _BASE_MOTION_UNCERTAINTY
        candidates.append(("fast", min(max(cfg["fast_scale"] * v_follow, cfg["fast_min"]), 1.0), w_follow))
    candidates.extend((("turn_l", 0.0, 0.4), ("turn_r", 0.0, -0.4), ("back", -0.2, 0.0)))
    if v_follow == 0.0:
        escape = _BASE_ESCAPE_CONFIG
        curve_w = w_follow
        if target_xy is not None:
            _, x, y, yaw = _BASE_POSES[-1]
            error = math.atan2(target_xy[1] - y, target_xy[0] - x) - yaw
            error = math.atan2(math.sin(error), math.cos(error))
            curve_w = max(-escape["curve_max_w"], min(escape["curve_max_w"], escape["curve_gain"] * error))
        candidates.extend((("forward_slow", escape["slow_v"], 0.0), ("forward_fast", escape["fast_v"], 0.0), ("toward", escape["curve_v"], curve_w)))
    return candidates


def _base_choose_rank(candidates, target_xy):
    """지도와 관측 물체의 간격을 독립된 순위 기준으로 비교합니다."""
    def distance(name):
        p = candidates[name]["end_pose"]
        return math.hypot(p[0] - target_xy[0], p[1] - target_xy[1])
    def hit_time(value):
        return math.inf if value is None else value
    # An unobstructed requested alignment or goal stop must not initiate travel.
    if candidates["follow"]["v"] == 0.0 and candidates["follow"]["t_hit"] is None:
        return "follow", "follow_safe"
    if _BASE_MOTION_UNCERTAINTY["fast_preference"] and "fast" in candidates and candidates["fast"]["t_hit"] is None:
        other = [n for n in ("follow", "slow") if candidates[n]["t_hit"] is None]
        current = _BASE_POSES[-1]
        reference = min((distance(n) for n in other), default=math.hypot(current[1] - target_xy[0], current[2] - target_xy[1]))
        if distance("fast") + 1e-9 < reference:
            return "fast", "fast_progress"
    for name in ("follow", "slow", "stop"):
        if candidates[name]["t_hit"] is None:
            return name, name + "_safe"
    safe = [name for name, item in candidates.items() if item["t_hit"] is None]
    if safe:
        return min(safe, key=lambda n: (distance(n), -candidates[n]["robust_min_gap_obj"], -candidates[n]["min_gap_map"], n)), "safe_progress"
    clear_map = [name for name, item in candidates.items() if item["t_hit_map"] is None]
    positive_map = [name for name, item in candidates.items() if item["min_gap_map"] > 0.0]
    pool = clear_map or positive_map or list(candidates)
    filtered = len(pool) < len(candidates)
    chosen = min(pool, key=lambda n: (-candidates[n]["robust_min_gap_obj"],
                                     -hit_time(candidates[n]["t_hit_obj"]),
                                     -candidates[n]["min_gap_map"],
                                     -hit_time(candidates[n]["t_hit_map"]), n))
    return chosen, "map_clearance_object_gap" if clear_map else ("map_filtered_object_gap" if filtered else "emergency_object_gap")


def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):
    """지도 여유를 지키는 후보를 우선하고 물체/지도 간격을 분리 비교합니다."""
    v_follow, w_follow, horizon_s = map(float, (v_follow, w_follow, horizon_s))
    target_xy = tuple(float(p) for p in target_xy)
    if len(target_xy) != 2 or not all(math.isfinite(q) for q in (*target_xy, v_follow, w_follow, horizon_s)):
        raise ValueError("choose_command requires finite commands and a target (x,y)")
    if horizon_s < 0.0:
        raise ValueError("choose_command requires horizon_s >= 0")
    candidates = {}
    for name, v, w in _base_command_candidates(v_follow, w_follow, target_xy):
        result = check_motion(v, w, horizon_s)
        candidates[name] = {"v": v, "w": w, "t_hit": result["t_hit"],
                            "min_gap_obj": result["min_gap_obj"], "min_gap_map": result["min_gap_map"],
                            "end_pose": result["end_pose"], "raw_t_hit": result["raw_t_hit"],
                            "raw_t_hit_obj": result["raw_t_hit_obj"], "t_hit_obj": result["t_hit_obj"],
                            "t_hit_map": result["t_hit_map"], "robust_min_gap_obj": result["robust_min_gap_obj"],
                            "future_min_gap_obj": result["future_min_gap_obj"], "robust_end_gap_obj": result["robust_end_gap_obj"]}
    chosen, reason = _base_choose_rank(candidates, target_xy)
    pick = candidates[chosen]
    map_filter = "clearance" if reason == "map_clearance_object_gap" else ("positive" if reason == "map_filtered_object_gap" else "none")
    excluded = sorted(n for n, c in candidates.items() if (map_filter == "clearance" and c["t_hit_map"] is not None) or (map_filter == "positive" and c["min_gap_map"] <= 0.0))
    filtered = bool(excluded)
    info = {"chosen": chosen, "reason": reason, "candidates": candidates, "min_gap_obj": pick["min_gap_obj"],
            "robust_min_gap_obj": pick["robust_min_gap_obj"], "min_gap_map": pick["min_gap_map"],
            "map_filtered": filtered, "map_filter": map_filter,
            "excluded_map_names": excluded}
    global _BASE_CHOOSE_DBG_WINDOW
    t = _BASE_POSES[-1][0]
    window = int(t // 0.5)
    if window != _BASE_CHOOSE_DBG_WINDOW:
        _BASE_CHOOSE_DBG_WINDOW = window
        fh, sh = candidates["follow"]["t_hit"], candidates["stop"]["t_hit"]
        fhs, shs = ("-" if fh is None else f"{fh:.1f}"), ("-" if sh is None else f"{sh:.1f}")
        builtins.print(f"DBG choose t={t:.2f} pick={chosen} why={reason} f={fhs} s={shs} obj_min={pick['min_gap_obj']:.2f} rob={pick['robust_min_gap_obj']:.2f} map={pick['min_gap_map']:.2f}")
    return pick["v"], pick["w"], info


# ---- DBG 기록 ----
# controller() 안에서 print("DBG", ...) 처럼 DBG 로 시작하게 찍은 줄은 아래 print 를 거칩니다. 시뮬레이션 시각
# 0.5 s 구간마다 그 구간의 첫 DBG 줄만 내보내고 나머지는 버립니다(처음 0.5 s 구간만 열 줄까지 내보냅니다).
# DBG 로 시작하지 않는 줄은 그대로 내보냅니다. 시각은 controller() 를 부를 때 base 가 적어 둡니다.
import builtins  # noqa: E402

_BASE_DBG_WINDOW_S, _BASE_DBG_FIRST_WINDOW_LINES = 0.5, 10
_BASE_DBG_STATE = {"t": 0.0, "window": None, "kept": 0, "dropped": 0}


def print(*args, **kwargs):  # noqa: A001
    if args and str(args[0]).startswith("DBG"):
        s = _BASE_DBG_STATE
        w = int(s["t"] // _BASE_DBG_WINDOW_S)
        if w != s["window"]:
            s["window"], s["kept"] = w, 0
        if s["kept"] >= (_BASE_DBG_FIRST_WINDOW_LINES if w == 0 else 1):
            s["dropped"] += 1
            return
        s["kept"] += 1
    builtins.print(*args, **kwargs)


# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import heapq
import numpy as np

# ---------- helper utilities ----------
def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------- path planning ----------
# Global cache for the static path
_PATH = None          # list of (x, y) points from START to GOAL
_PATH_IDX = 0         # next waypoint index to head for
_SAFETY_MARGIN = 0.60  # required clearance from any map object (m)


def _plan_path():
    """Compute a static collision‑free polyline from START to GOAL using a visibility graph."""
    global _PATH, _PATH_IDX

    # collect free‑space nodes: start, goal, and offset corners of all map rectangles
    nodes = [(START[0], START[1]), (GOAL[0], GOAL[1])]
    rects = map_rects()          # (K,4): xmin,xmax,ymin,ymax
    for (xmin, xmax, ymin, ymax) in rects:
        cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
        # four corners, nudged a little outward so they lie in free space
        for (cxn, cyn) in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
            # direction from rectangle centre to corner
            dx, dy = cxn - cx, cyn - cy
            norm = math.hypot(dx, dy) or 1.0
            # offset 0.2 m outward
            off = 0.20
            nx, ny = cxn + off * dx / norm, cyn + off * dy / norm
            # keep only points that are inside the arena walls and have some clearance
            if map_clearance(nx, ny) > 0.0:
                nodes.append((nx, ny))

    # build visibility edges (undirected)
    N = len(nodes)
    edges = [[] for _ in range(N)]
    for i in range(N):
        pi = nodes[i]
        for j in range(i + 1, N):
            pj = nodes[j]
            # quick bounding‑box test: if the segment is longer than the arena, skip
            # (not really needed but cheap)
            if segment_clearance(pi, pj) >= _SAFETY_MARGIN:
                d = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
                edges[i].append((j, d))
                edges[j].append((i, d))

    # Dijkstra from start (index 0) to goal (index 1)
    start_idx, goal_idx = 0, 1
    dist = [math.inf] * N
    prev = [-1] * N
    dist[start_idx] = 0.0
    heap = [(0.0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == goal_idx:
            break
        for v, w in edges[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # reconstruct path
    if dist[goal_idx] == math.inf:
        # no path found – fall back to direct line (will be unsafe)
        _PATH = [(START[0], START[1]), (GOAL[0], GOAL[1])]
    else:
        rev = []
        cur = goal_idx
        while cur != -1:
            rev.append(nodes[cur])
            cur = prev[cur]
        _PATH = list(reversed(rev))

    _PATH_IDX = 1   # first waypoint after the start point
    # DBG: report basic path stats
    seg_mins = [segment_clearance(_PATH[i], _PATH[i + 1]) for i in range(len(_PATH) - 1)]
    min_clear = min(seg_mins) if seg_mins else float('inf')
    total_len = sum(math.hypot(_PATH[i + 1][0] - _PATH[i][0],
                               _PATH[i + 1][1] - _PATH[i][1]) for i in range(len(_PATH) - 1))
    print(f"DBG path pts={len(_PATH)} len={total_len:.2f} min_clear={min_clear:.2f}")


def _next_target(pose):
    """Return the current waypoint the robot should head for."""
    global _PATH_IDX
    if _PATH is None:
        _plan_path()

    # advance index if we are close enough to the current waypoint
    while _PATH_IDX < len(_PATH):
        tx, ty = _PATH[_PATH_IDX]
        dx, dy = tx - pose[0], ty - pose[1]
        if math.hypot(dx, dy) < 0.35:   # within pass‑through radius
            _PATH_IDX += 1
            continue
        break

    # if we have exhausted waypoints, head for the final GOAL position
    if _PATH_IDX >= len(_PATH):
        return GOAL[:2]
    return _PATH[_PATH_IDX]


# ---------- main controller ----------
def controller(t, pose, env):
    """Follow a pre‑computed safe path while delegating dynamic‑obstacle handling to choose_command."""
    # keep the global pose history up‑to‑date (base already does this before calling us)
    # (nothing to do here – the harness updates _BASE_POSES)

    # compute the target point on the static path
    target_xy = _next_target(pose)

    x, y, yaw = pose
    dx, dy = target_xy[0] - x, target_xy[1] - y
    dist = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    e = _wrap_angle(angle_to_target - yaw)

    # final‑goal alignment handling
    at_goal = (abs(x - GOAL[0]) < 0.25 and abs(y - GOAL[1]) < 0.25)
    if at_goal:
        # rotate in place to match goal yaw
        e_goal = _wrap_angle(GOAL[2] - yaw)
        v_f, w_f = 0.0, _clamp(1.5 * e_goal, -0.4, 0.4)
        # ask choose_command for the zero‑velocity stop condition
        v, w, info = choose_command(v_f, w_f, GOAL[:2])
        # if the chosen command is truly zero we can finish
        if abs(v) < 1e-6 and abs(w) < 1e-6 and abs(e_goal) < 0.05:
            return 0.0, 0.0, True
        return v, w, False

    # normal path‑following law (same as the reference in the prompt)
    if abs(e) > 0.35:
        v_f, w_f = 0.0, _clamp(1.5 * e, -0.4, 0.4)
    else:
        v_f = min(0.3, 0.5 * dist + 0.1)
        w_f = _clamp(1.5 * e, -0.4, 0.4)

    # hand the command to the safety layer
    v, w, info = choose_command(v_f, w_f, target_xy)

    # optional debug line – will be throttled by the harness
    print(f"DBG ctrl t={t:.2f} pose=({x:.2f},{y:.2f},{yaw:.2f}) target=({target_xy[0]:.2f},{target_xy[1]:.2f})"
          f" v_f={v_f:.2f} w_f={w_f:.2f} chosen=({v:.2f},{w:.2f})")

    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera


def _robot_only(t, pose, env):
    """The controller sees no harness handle (env is None): only the robot's own sensors."""
    _BASE_DBG_STATE["t"] = t
    _BASE_POSES.append((t, pose[0], pose[1], pose[2]))       # observed_objects(): poses of the last 1 s
    while _BASE_POSES and t - _BASE_POSES[0][0] > 1.0:
        _BASE_POSES.pop(0)
    try:
        _base_track_step(t)                                  # tracking updated every step, before the controller
    except Exception as e:                                   # noqa: BLE001  a tracking fault must not stop the run
        if _BASE_OBJ_ST["error"] is None:
            _BASE_OBJ_ST["error"] = repr(e)
            print("DBG [base] observed_objects() 갱신 오류:", repr(e))
    return controller(t, pose, None)


ctrl = _robot_only
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(ctrl, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
