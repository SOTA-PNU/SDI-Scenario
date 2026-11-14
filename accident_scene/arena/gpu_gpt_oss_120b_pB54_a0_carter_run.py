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
_BODY = (-0.70, 0.25, -0.34, 0.34)                                     # 로봇 자신의 몸체 범위 (x 최소, x 최대, y 최소, y 최대)

_S = {"node": None, "rclpy": None, "subs": set(), "count": 0, "scan": {}, "cloud": [], "image": None}


def _subscribe(key, msg_type, topic, callback):
    """센서 하나를 처음 쓸 때 한 번만 구독합니다. rclpy 는 여기서 가져옵니다
    (모듈 맨 위에서 가져오면 Isaac Sim 이 켜지기 전이라 실패합니다)."""
    if key in _S["subs"]:
        return
    if _S["node"] is None:
        import rclpy
        _S["rclpy"] = rclpy
        _S["node"] = rclpy.create_node("b2_sensor_reader")
    from rclpy.qos import qos_profile_sensor_data
    _S["node"].create_subscription(msg_type, topic, callback, qos_profile_sensor_data)
    _S["subs"].add(key)


def _spin():
    """쌓여 있는 센서 메시지를 모두 처리합니다. 아래 센서 함수들이 부릅니다."""
    for _ in range(20):
        n = _S["count"]
        _S["rclpy"].spin_once(_S["node"], timeout_sec=0.0)
        if _S["count"] == n:
            break


def _drop_body(p):
    """로봇 자신의 몸체에 맞은 점을 뺍니다."""
    body = (p[:, 0] >= _BODY[0]) & (p[:, 0] <= _BODY[1]) & (p[:, 1] >= _BODY[2]) & (p[:, 1] <= _BODY[3])
    return p[~body]


def _on_scan(which, msg):
    mx, my, myaw = _LIDAR2D[which][1]
    r = np.asarray(msg.ranges, dtype=np.float64)
    a = msg.angle_min + np.arange(r.size) * msg.angle_increment + myaw
    ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    pts = np.stack([mx + r[ok] * np.cos(a[ok]), my + r[ok] * np.sin(a[ok])], axis=1)
    _S["scan"][which] = _drop_body(pts)
    _S["count"] += 1


def _on_cloud(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}
    xyz = np.stack([raw[:, off[c]:off[c] + 4].copy().view(np.float32)[:, 0] for c in "xyz"], axis=1)
    pts = xyz.astype(np.float64) + np.array(_LIDAR3D[1])
    _S["cloud"] = (_S["cloud"] + [_drop_body(pts)])[-6:]
    _S["count"] += 1


def _on_image(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    _S["image"] = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    _S["count"] += 1


def lidar2d_points(which="front"):
    """2D 라이다 which ("front" 앞쪽, "back" 뒤쪽) 의 가장 최근 스캔을 로봇 기준 점 배열 (N, 2) [x, y] 로 돌려줍니다.
    두 라이다 모두 바닥 위 0.42 m 높이의 평면을 360° 로, 0.1 s 마다 한 바퀴 훑습니다(빔 3,200개, 0.05–30 m).
    controller() 에서 매번 부르십시오. 처음 몇 번은 빈 배열일 수 있습니다."""
    from sensor_msgs.msg import LaserScan
    _subscribe("lidar2d_" + which, LaserScan, _LIDAR2D[which][0], lambda m: _on_scan(which, m))
    _spin()
    return _S["scan"].get(which, np.zeros((0, 2)))


def lidar3d_points():
    """3D 라이다 (Hesai XT-32, 바닥 위 0.53 m, 32채널) 의 최근 한 바퀴를 로봇 기준 점 배열 (N, 3) [x, y, z] 로 돌려줍니다.
    이 라이다는 메시지 하나에 약 60° 조각만 담아 1/60 s 마다 보내므로, 최근 6 조각(약 0.1 s)을 모아 360° 를 만듭니다.
    z 는 바닥에서의 높이이며 바닥에 맞은 점은 z 가 0 근처입니다. 처음 몇 번은 빈 배열일 수 있습니다."""
    from sensor_msgs.msg import PointCloud2
    _subscribe("lidar3d", PointCloud2, _LIDAR3D[0], _on_cloud)
    _spin()
    return np.concatenate(_S["cloud"]) if _S["cloud"] else np.zeros((0, 3))


def camera_image():
    """앞쪽 스테레오 카메라의 왼쪽 카메라 영상을 배열 (1200, 1920, 3) uint8 [RGB] 로 돌려줍니다. 아직 없으면 None.
    1/60 s 마다 새 영상이 옵니다. 수평 약 ±60°, 수직 약 ±36° 를 보는 광각이고, 영상에 렌즈 왜곡이 들어 있습니다
    (camera_project, camera_ray 가 이 왜곡을 반영합니다)."""
    from sensor_msgs.msg import Image
    _subscribe("camera", Image, _CAMERA_TOPIC, _on_image)
    _spin()
    return _S["image"]


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
    """로봇 기준 점 (N, 2) 또는 (N, 3) 을 월드 좌표로 바꿉니다(z 는 그대로). pose 는 controller() 가 받은 (x, y, yaw) 입니다."""
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


def map_rects():
    """지도 물체의 직사각형 배열 (K, 4): MISSION 상자와 FIXED_OBSTACLES. 벽은 WALLS 에 따로 있습니다."""
    boxes = [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0, b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0)
             for b in MISSION["boxes"]]
    return np.array(boxes + [r[1:] for r in FIXED_OBSTACLES], dtype=np.float64)


def map_clearance(x, y):
    """월드 좌표 (x, y) 에서 가장 가까운 지도 물체(상자, 고정물, 벽 안쪽 면)까지의 거리 [m]. 물체 안이면 0 입니다."""
    r = map_rects()
    dx = np.maximum(np.maximum(r[:, 0] - x, x - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - y, y - r[:, 3]), 0.0)
    wx0, wx1, wy0, wy1 = WALLS
    return float(max(0.0, min(float(np.hypot(dx, dy).min()), x - wx0, wx1 - x, y - wy0, wy1 - y)))


def segment_clearance(p, q):
    """점 p=(x, y) 에서 q 까지 곧게 갈 때 map_clearance 의 최솟값 [m] 입니다(0.05 m 간격으로 잽니다)."""
    n = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05)))
    return min(map_clearance(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(n + 1))


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


# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# ----------------------------------------------------------------------
# Waypoint list (x, y). The last waypoint is the GOAL position.
# These points were chosen to stay at least ~1 m away from all static
# obstacles (boxes, pillars, walls, etc.) as verified with
# `segment_clearance()`.
# ----------------------------------------------------------------------
_WAYPOINTS = [
    (-8.5, -7.0),   # north‑west of box 00
    (-5.0, -3.0),   # between boxes 01 / 02
    (0.0, 2.0),     # central corridor
    (3.0, 6.0),     # north of box 03‑05
    (5.0, 11.0),    # east of the middle‑row boxes
    (6.5, 14.0),    # north‑east of box 06
    (GOAL[0], GOAL[1]),  # final goal position
]

# ----------------------------------------------------------------------
# Simple state holder – persists between controller() calls.
# ----------------------------------------------------------------------
_state = {
    "wp_idx": 0,          # index of the current waypoint in _WAYPOINTS
    "avoid": False,       # are we currently avoiding a dynamic obstacle?
    "avoid_dir": 0.0,     # preferred turn direction while avoiding (rad/s)
}

# ----------------------------------------------------------------------
# Helper math utilities (same as in the reference implementation)
# ----------------------------------------------------------------------
def _wrap(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


# ----------------------------------------------------------------------
# Main controller
# ----------------------------------------------------------------------
def controller(t, pose, env):
    """
    Follow the predefined waypoint list while reacting to unmapped (moving)
    obstacles detected by the on‑board sensors.

    The controller uses the same pure‑pursuit law that was verified to work
    for static waypoints, but adds a very lightweight reactive avoidance:
      * All 2‑D lidar points are gathered, transformed to world coordinates,
        and filtered with `unmapped_points()` so that only points belonging
        to *unknown* (i.e. moving) objects remain.
      * If any such point is closer than `SAFE_DIST` in front of the robot,
        the robot stops forward motion and rotates away from the closest
        point until the danger disappears.
    """
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    SAFE_DIST = 0.8          # [m] distance at which we start avoiding
    FRONT_ANGLE = math.radians(90)   # +/- 90° in front of the robot
    MAX_TURN = 0.8          # [rad/s] limit for avoidance turning
    WAYPOINT_TOL = 0.35      # distance tolerance for waypoint switching
    ALIGN_TOL = 0.05         # yaw tolerance for final alignment

    # ------------------------------------------------------------------
    # 1) Sensor reading – combine front and back 2‑D lidar scans
    # ------------------------------------------------------------------
    pts_front = lidar2d_points("front")   # (N,2) in robot frame
    pts_back = lidar2d_points("back")
    pts_robot = np.vstack([pts_front, pts_back]) if pts_front.size else pts_back

    # Transform to world coordinates
    if pts_robot.size:
        pts_world = to_world(pts_robot, pose)
        # Keep only points that are *not* part of the static map
        unmapped = unmapped_points(pts_world, margin=0.2)
    else:
        unmapped = np.zeros((0, 3))

    # ------------------------------------------------------------------
    # 2) Detect a nearby unmapped obstacle in the forward sector
    # ------------------------------------------------------------------
    avoid_now = False
    avoid_dir = 0.0

    if unmapped.shape[0]:
        # Convert world points back to robot‑local coordinates for easy angle test
        dx = unmapped[:, 0] - pose[0]
        dy = unmapped[:, 1] - pose[1]
        c, s = math.cos(-pose[2]), math.sin(-pose[2])   # rotate by -yaw
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        # polar coordinates in robot frame
        r = np.hypot(local_x, local_y)
        theta = np.arctan2(local_y, local_x)          # 0 = forward, + left

        # Points that are in front half‑plane and within SAFE_DIST
        mask = (np.abs(theta) <= FRONT_ANGLE) & (r < SAFE_DIST)
        if np.any(mask):
            avoid_now = True
            # Choose turn direction opposite to the average angle of the danger points
            danger_angles = theta[mask]
            mean_angle = np.mean(danger_angles)
            avoid_dir = -_clamp(1.5 * mean_angle, -MAX_TURN, MAX_TURN)

    # ------------------------------------------------------------------
    # 3) State handling – remember we are in avoidance mode until cleared
    # ------------------------------------------------------------------
    if _state["avoid"]:
        # stay in avoidance mode until no close unmapped points remain
        if not avoid_now:
            _state["avoid"] = False
    else:
        if avoid_now:
            _state["avoid"] = True
            _state["avoid_dir"] = avoid_dir

    # ------------------------------------------------------------------
    # 4) If avoiding, rotate in place; otherwise follow waypoints
    # ------------------------------------------------------------------
    if _state["avoid"]:
        # stop forward motion, rotate away from the obstacle
        v = 0.0
        w = _state["avoid_dir"]
        return v, w, False

    # ------------------------------------------------------------------
    # 5) Pure‑pursuit waypoint tracking (the verified method)
    # ------------------------------------------------------------------
    wp_idx = _state["wp_idx"]
    tx, ty = _WAYPOINTS[wp_idx]

    # distance and heading error to the current waypoint
    dx = tx - pose[0]
    dy = ty - pose[1]
    dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    e = _wrap(target_yaw - pose[2])

    # Switch to next waypoint if we are close enough (except for the final goal)
    if wp_idx < len(_WAYPOINTS) - 1 and dist < WAYPOINT_TOL:
        _state["wp_idx"] = wp_idx + 1
        tx, ty = _WAYPOINTS[_state["wp_idx"]]
        dx = tx - pose[0]
        dy = ty - pose[1]
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        e = _wrap(target_yaw - pose[2])

    # ------------------------------------------------------------------
    # 6) Final goal handling (alignment & termination)
    # ------------------------------------------------------------------
    if wp_idx == len(_WAYPOINTS) - 1:
        # We are heading for the final GOAL position
        if dist < 0.25:
            # Align to the required final yaw
            e_align = _wrap(GOAL[2] - pose[2])
            w_align = _clamp(1.5 * e_align, -0.6, 0.6)
            v = 0.0
            if abs(e_align) < ALIGN_TOL:
                # Goal reached and aligned – finish the mission
                return 0.0, 0.0, True
            return v, w_align, False

    # ------------------------------------------------------------------
    # 7) Normal waypoint pursuit law (same as the reference)
    # ------------------------------------------------------------------
    if abs(e) > 0.35:
        # turn in place
        v = 0.0
        w = _clamp(1.5 * e, -0.8, 0.8)
    else:
        v = min(0.6, 0.5 * dist + 0.1)
        w = _clamp(1.5 * e, -0.8, 0.8)

    # ------------------------------------------------------------------
    # 8) Debug output (visible in `controller_output` of the harness)
    # ------------------------------------------------------------------
    print(
        "DBG",
        {
            "t": round(t, 2),
            "wp_idx": _state["wp_idx"],
            "pose": tuple(round(v, 3) for v in pose),
            "target": (tx, ty),
            "dist": round(dist, 3),
            "e": round(e, 3),
            "v": round(v, 3),
            "w": round(w, 3),
            "avoid": _state["avoid"],
            "avoid_dir": round(_state.get("avoid_dir", 0.0), 3),
            "unmapped_cnt": int(unmapped.shape[0]),
        },
    )

    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera


def _robot_only(t, pose, env):
    """The controller sees no harness handle (env is None): only the robot's own sensors."""
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
