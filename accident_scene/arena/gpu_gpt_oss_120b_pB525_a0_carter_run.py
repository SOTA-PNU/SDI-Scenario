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

ROBOT API available to the model's three hooks:
  plan_path(start_xy, goal_xy, clearance_m) -> [(x,y), ...]
  lookahead_target(path, pose) -> (x,y)
  follow_cmd(pose, target_xy) -> (v,w)

  The base owns each 1/60 s control step, goal alignment, planning triggers,
  path validation, command clipping and the single choose_command call.
  The EDIT controller is a compatibility stub; _robot_only dispatches directly
  to _base_control_loop and does not call the model's controller implementation.
  The base has no path planner: plan_path is supplied by the model.

  Planning tries centre clearances 0.5, 0.45, 0.4, 0.35 m in order, initially
  and after continuous map stop longer than 10 s. check_path subtracts robot
  half-width 0.25 m, so the corresponding margin is clearance_m - 0.25 m.
  Endpoint gaps up to 0.5 m are connected to the actual current start and
  GOAL before check_path validates the entire path, including both connectors.
  validate_plan(path) returns the same accepted/path/reason, endpoint gaps and
  check_path result without state changes or logs. During plan_path it uses that
  attempt's start/GOAL/clearance; otherwise the latest pose and adopted clearance
  (default 0.5 m) are used. Before the first pose, the MISSION start is used;
  start_source reports plan_context, latest_pose or mission_start.
  Exact endpoint paths are retained without duplicate endpoint vertices.
  Rejections log DBG plan_reject, at most once per reason per 10 s. A separate
  DBG plan_summary line reports every clearance actually attempted in a batch,
  whether a path was accepted, absent, or retained, with every attempted
  clearance and its accepted/reason outcome first. Detailed failed segment,
  minimum gap and required margin follow in JSON; a shortened tool excerpt may
  truncate this detail. Adopted clearance, path point count and next retry time
  are also reported. Summaries are limited to one per 10 s.
  validate_plan includes first_bad_segment_points on the connected path. The
  existing check_path min_location remains the global minimum location.
  A planning exception rejects only that clearance and tries the next clearance.
  Failed replanning preserves a previous valid path; with no path, checked zero
  input is requested and planning retries every 10 s. A target is used only
  when its path projection is 0.3 to 2.0 m forward of the robot projection
  along path arc length, and its path distance is at most 1.0 m. Otherwise
  a base forward path point replaces it, with one DBG per reason per 10 s.
  Equal-distance projection branches prefer previous robot arc position
  (the earliest arc on the first tick); target ties prefer the robot arc.
  Fallback advances at most 2 m on the current segment, or 0.3 m along
  following segments when needed. With less than 0.3 m path remaining
  outside goal mode, a checked zero is requested. GOAL is accepted only
  when it satisfies the same target range; goal alignment is unchanged.
  Nominal commands are clipped to v in [0,0.3], w in [-0.4,0.4]; the unchanged
  chooser may select its fast/backward escape candidates. Hooks do not choose.
  Non-planning hook exceptions or nonfinite outputs request checked zero and DBG.

  choose_command retains the v9 geometry, candidates and priority; object
  prediction/end_pose use 3 s by default and map prediction uses a short horizon.
  Public check_motion(v,w,horizon_s=3.0,dt=0.1,margin_m=0.2) remains unchanged.
  check_path(points,margin_m=0.25) checks polyline centre clearance minus 0.25 m.

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
contains three empty model hooks and a compatibility controller stub.
Everything outside it is fixed base control and plumbing.
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


_BASE_PATH_HALF_WIDTH = 0.25
_BASE_PATH_SAMPLE_STEP = 0.1


def check_path(points, margin_m=0.25):
    """꺾은선 중심의 지도 거리에서 몸체 반폭 0.25 m를 뺀 여유를 검사합니다.

    각 선분의 양끝을 포함해 최대 0.1 m 간격으로 검사합니다. 기본 기준 여유는
    0.25 m(검사 여유 0.2 + 추가 0.05)이므로 중심의 지도 거리는 0.5 m 이상이어야 합니다.
    first_bad_segment 는 여유가 기준보다 작은 첫 선분의 0-based 번호이며 없으면 None입니다.
    points 는 유한한 (x,y) 두 점 이상이어야 합니다. 중복점은 허용하고 그 위치를 검사합니다.
    min_location 은 전체 샘플 중 최저 여유 위치이며 방향이나 회전 궤적은 가정하지 않습니다."""
    margin_m = float(margin_m)
    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("check_path requires a finite nonnegative margin")
    try:
        path = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("check_path requires finite points with shape (N,2), N >= 2") from exc
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("check_path requires finite points with shape (N,2), N >= 2")
    samples, segments = [], []
    for index, (p, q) in enumerate(zip(path[:-1], path[1:])):
        count = max(1, int(math.ceil(math.hypot(*(q - p)) / _BASE_PATH_SAMPLE_STEP)))
        part = p + np.linspace(0.0, 1.0, count + 1)[:, None] * (q - p)
        samples.append(part)
        segments.append(np.full(count + 1, index, dtype=np.int64))
    samples, segments = np.concatenate(samples), np.concatenate(segments)
    gaps = map_clearance(samples[:, 0], samples[:, 1]) - _BASE_PATH_HALF_WIDTH
    lowest = int(np.argmin(gaps))
    bad = np.flatnonzero(gaps < margin_m - 1e-12)
    return {"min_gap_map": float(gaps[lowest]),
            "min_location": (float(samples[lowest, 0]), float(samples[lowest, 1])),
            "first_bad_segment": int(segments[int(bad[0])]) if bad.size else None,
            "margin_m": margin_m, "sampling_step_m": _BASE_PATH_SAMPLE_STEP}


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
                     vmax=2.0, keep=3.0, stale=0.3, frag_n=3, frag_join=0.6)
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
    점을 바꿔 물체를 다시 묶고 추적을 이어 붙입니다. 스캔 기준 시각(t - 0.1 s)에서 마지막 관측 후 3.0 s가 넘은 물체는 매 스텝 지웁니다."""
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
    x, y, vx, vy 와 points 는 seen_t 때의 값입니다. 스캔 기준 시각(t - 0.1 s)에서 마지막 관측 후 3.0 s가 넘은 물체는 빠집니다. 현재 age_s는 최대 약 3.1 s입니다.

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


# CPU 가속 모형 기준 완충 시간 + 정지시간의 휴리스틱이며 실제 제동을 보장하지 않습니다.
_BASE_MAP_HORIZON_CONFIG = {'minimum_s': 0.8, 'buffer_s': 0.5, 'linear_decel_m_s2': 2.0, 'angular_decel_rad_s2': 2.68}


def _base_map_horizon(v, w, horizon_s):
    cfg = _BASE_MAP_HORIZON_CONFIG
    stop_time = max(abs(v) / cfg["linear_decel_m_s2"], abs(w) / cfg["angular_decel_rad_s2"])
    return min(horizon_s, max(cfg["minimum_s"], cfg["buffer_s"] + stop_time))


def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:
    """지도와 물체 모두 horizon_s 동안 검사합니다. 기존 공개 API의 의미를 유지합니다.

    choose_command 내부만 지도 검사 시간을 별도로 줄입니다. 이 함수의 기본 검사 시간은 양쪽 모두 3 s이며,
    몸체·관측 점 기하, 속도 불확실성, 반환 키는 이전과 같습니다."""
    return _base_check_motion(v, w, horizon_s, dt, margin_m)


def _base_check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2, map_horizon_s=None) -> dict:
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
    if map_horizon_s is not None:
        map_horizon_s = float(map_horizon_s)
        if not math.isfinite(map_horizon_s) or not 0.0 <= map_horizon_s <= horizon_s:
            raise ValueError("private map horizon must be finite and between zero and horizon_s")
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
    # Object prediction, uncertainty and end_pose above still use the full tt.
    # Use a separate endpoint-inclusive grid only for the static-map component.
    map_tt, map_xx, map_yy, map_cc, map_ss = tt, xx, yy, cc, ss
    if map_horizon_s is not None:
        map_tt = np.minimum(np.arange(int(math.ceil(map_horizon_s / dt)) + 1, dtype=np.float64) * dt, map_horizon_s)
        map_half_turn = 0.5 * w * map_tt
        map_travel = v * map_tt * np.sinc(map_half_turn / math.pi)
        map_xx = x0 + map_travel * np.cos(yaw0 + map_half_turn)
        map_yy = y0 + map_travel * np.sin(yaw0 + map_half_turn)
        map_yaw = yaw0 + w * map_tt
        map_cc, map_ss = np.cos(map_yaw), np.sin(map_yaw)
    perimeter = _BASE_BODY_PERIMETER
    px = map_xx[:, None] + map_cc[:, None] * perimeter[:, 0] - map_ss[:, None] * perimeter[:, 1]
    py = map_yy[:, None] + map_ss[:, None] * perimeter[:, 0] + map_cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(map_tt[int(hit_map[0])]) if hit_map.size else None
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


_BASE_MAP_STOP_DBG_STATE = {"since": None, "last_t": None, "last_warn_t": None}


def _base_note_map_stop(t, chosen, candidates):
    """지도 때문에 선택한 정지가 연속 10 s를 넘으면 별도 진단 한 줄을 냅니다."""
    state = _BASE_MAP_STOP_DBG_STATE
    if state["last_t"] is not None and t < state["last_t"]:
        state["since"], state["last_warn_t"] = None, None
    state["last_t"] = t
    pick = candidates[chosen]
    blocked = (chosen == "stop" and pick["v"] == 0.0 and pick["w"] == 0.0
               and candidates["follow"]["t_hit_map"] is not None)
    if not blocked:
        state["since"], state["last_warn_t"] = None, None
        return
    if state["since"] is None:
        state["since"] = t
    elapsed = t - state["since"]
    if elapsed > 10.0 and (state["last_warn_t"] is None or t - state["last_warn_t"] >= 10.0 - 1e-12):
        builtins.print(f"DBG follow 가 지도에 막힘: 경로 여유 부족 t={t:.2f} stop_for={elapsed:.2f}")
        state["last_warn_t"] = t


def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):
    """물체와 끝점은 horizon_s, 지도는 후보별 짧은 시간 동안 검사해 기존 순위로 선택합니다.

    기본 물체 예측은 3 s입니다. 지도 시간은 min(horizon_s, max(0.8, 0.5 +
    max(abs(v)/2.0, abs(w)/2.68))) s이며 CPU 가속 모형에 기반한 휴리스틱입니다.
    info 및 각 후보의 object_horizon_s/map_horizon_s 에 실제 검사 시간을 기록합니다.
    end_pose 는 짧은 지도 시간이 아닌 물체 검사 시간 끝의 자세입니다."""
    v_follow, w_follow, horizon_s = map(float, (v_follow, w_follow, horizon_s))
    target_xy = tuple(float(p) for p in target_xy)
    if len(target_xy) != 2 or not all(math.isfinite(q) for q in (*target_xy, v_follow, w_follow, horizon_s)):
        raise ValueError("choose_command requires finite commands and a target (x,y)")
    if horizon_s < 0.0:
        raise ValueError("choose_command requires horizon_s >= 0")
    candidates = {}
    for name, v, w in _base_command_candidates(v_follow, w_follow, target_xy):
        map_horizon_s = _base_map_horizon(v, w, horizon_s)
        result = _base_check_motion(v, w, horizon_s, map_horizon_s=map_horizon_s)
        candidates[name] = {"v": v, "w": w, "t_hit": result["t_hit"],
                            "object_horizon_s": horizon_s, "map_horizon_s": map_horizon_s,
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
            "object_horizon_s": horizon_s, "map_horizon_s": pick["map_horizon_s"],
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
    _base_note_map_stop(t, chosen, candidates)
    return pick["v"], pick["w"], info


# ---- base 제어 루프: EDIT REGION 의 세 함수만 사용 ----
_BASE_LOOP_CFG = {"clearances_m": (0.5, 0.45, 0.4, 0.35), "target_path_limit_m": 1.0,
                  "target_min_ahead_m": 0.3, "target_max_ahead_m": 2.0, "target_note_s": 10.0,
                  "v_min": 0.0, "v_max": 0.3, "w_max": 0.4,
                  "goal_distance_m": 0.5, "goal_yaw_rad": 0.15, "map_stop_s": 10.0,
                  "plan_endpoint_limit_m": 0.5, "plan_retry_s": 10.0, "plan_note_s": 10.0, "plan_summary_s": 10.0}
_BASE_LOOP_ST = {"path": None, "clearance_m": None, "replan_pending": True,
                 "block_since": None, "last_t": None, "fault_key": None, "last": None,
                 "path_progress_s": None, "target_note_t": {}, "plan_context": None,
                 "plan_note_t": {}, "next_plan_t": None, "plan_summary_t": None}


def _base_loop_pair(value, name):
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(name + " must return two finite numbers")
    return float(values[0]), float(values[1])


def _base_loop_hook(name, *args):
    function = globals().get(name)
    if not callable(function):
        raise TypeError(name + " is missing or not callable")
    return function(*args)


def _base_loop_note_fault(t, stage, error):
    key = (stage, type(error).__name__, str(error)[:160])
    builtins.print(f"DBG base_loop t={t:.2f} hook={stage} error={key[1]}: {key[2]}; checked stop requested")
    _BASE_LOOP_ST["fault_key"] = key


def _base_plan_result(clearance_m):
    return {"accepted": False, "path": None, "reason": None,
            "start_gap_m": None, "goal_gap_m": None, "check_path": None,
            "clearance_m": float(clearance_m), "endpoint_limit_m": _BASE_LOOP_CFG["plan_endpoint_limit_m"],
            "connected_start": False, "connected_goal": False, "start_source": None, "error": None,
            "first_bad_segment_points": None}


def _base_plan_exception(clearance_m, error):
    result = _base_plan_result(clearance_m)
    result.update(reason="exception", error={"type": type(error).__name__, "message": str(error)[:160]})
    return result


def _base_validate_plan(value, start_xy, goal_xy, clearance_m):
    """The single pure plan acceptance rule, including exact endpoint connectors."""
    result = _base_plan_result(clearance_m)
    if value is None:
        result["reason"] = "empty_path"
        return result
    try:
        path = np.asarray(value, dtype=np.float64)
        if path.size == 0:
            result["reason"] = "empty_path"
            return result
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
            result["reason"] = "invalid_path"
            return result
        start_xy, goal_xy = tuple(map(float, start_xy)), tuple(map(float, goal_xy))
        start_gap, goal_gap = math.dist(path[0], start_xy), math.dist(path[-1], goal_xy)
        result.update(start_gap_m=start_gap, goal_gap_m=goal_gap)
        if max(start_gap, goal_gap) > result["endpoint_limit_m"] + 1e-12:
            result["reason"] = "endpoint_distance"
            return result
        connected = [(float(p[0]), float(p[1])) for p in path]
        # Preserve exact reference paths without duplicate endpoint vertices.
        if start_gap > 1e-12:
            connected.insert(0, start_xy)
            result["connected_start"] = True
        else:
            connected[0] = start_xy
        if goal_gap > 1e-12:
            connected.append(goal_xy)
            result["connected_goal"] = True
        else:
            connected[-1] = goal_xy
        margin = max(0.0, float(clearance_m)-0.25)
        checked = check_path(connected, margin_m=margin)
        result["check_path"] = checked
        if checked["first_bad_segment"] is not None:
            index = checked["first_bad_segment"]
            result["first_bad_segment_points"] = [connected[index], connected[index+1]]
        if checked["first_bad_segment"] is not None or checked["min_gap_map"] < margin - 1e-12:
            result["reason"] = "check_path"
            return result
        result.update(accepted=True, path=connected)
        return result
    except Exception as error:  # noqa: BLE001 - public validation reports malformed input
        result.update(reason="exception", error={"type": type(error).__name__, "message": str(error)[:160]})
        return result


def validate_plan(path):
    """Return the same acceptance result used by the base, without logs or state changes.

    During plan_path this uses that call's actual start, GOAL and clearance.
    Otherwise it uses the latest robot pose and adopted clearance (default 0.5 m).
    Before the first pose it uses MISSION start; start_source identifies this.
    path is the connected path on success, or None on rejection. check_path
    segment indices refer to the connected path, including endpoint connectors.
    first_bad_segment_points contains that segment's endpoints, or None. The
    check_path min_location is the global minimum, not necessarily on that segment.
    """
    context = _BASE_LOOP_ST["plan_context"]
    if context is not None:
        start_xy, goal_xy, clearance = context
        start_source = "plan_context"
    else:
        clearance = _BASE_LOOP_ST["clearance_m"]
        clearance = _BASE_LOOP_CFG["clearances_m"][0] if clearance is None else clearance
        if _BASE_POSES:
            start_xy, start_source = tuple(_BASE_POSES[-1][1:3]), "latest_pose"
        else:
            start_xy, start_source = tuple(MISSION["start_xy_yaw"][:2]), "mission_start"
        goal_xy = tuple(GOAL[:2])
    result = _base_validate_plan(path, start_xy, goal_xy, clearance)
    result["start_source"] = start_source
    return result


def _base_loop_valid_path(value, start_xy, goal_xy, clearance_m):
    result = _base_validate_plan(value, start_xy, goal_xy, clearance_m)
    return tuple(result["path"]) if result["accepted"] else None


def _base_loop_note_plan(t, result):
    reason = result["reason"]
    previous = _BASE_LOOP_ST["plan_note_t"].get(reason)
    if previous is not None and t >= previous and t-previous < _BASE_LOOP_CFG["plan_note_s"]:
        return
    checked = result["check_path"] or {}
    error = result["error"]
    builtins.print(f"DBG plan_reject t={t:.2f} reason={reason} clearance={result['clearance_m']:.2f} "
                   f"start_gap={result['start_gap_m']} goal_gap={result['goal_gap_m']} "
                   f"segment={checked.get('first_bad_segment')} min_gap={checked.get('min_gap_map')} "
                   f"margin={checked.get('margin_m')} error={error}")
    _BASE_LOOP_ST["plan_note_t"][reason] = t


def _base_loop_note_plan_summary(t, attempts):
    """One complete batch result; logging is separate from plan acceptance."""
    import json
    state = _BASE_LOOP_ST
    status = ("accepted" if any(a["accepted"] for a in attempts) else
              "no_path" if state["path"] is None else "retained")
    rows = []
    for attempt in attempts:
        checked = attempt["check_path"] or {}
        rows.append({"c": attempt["clearance_m"], "accepted": attempt["accepted"],
                     "reason": attempt["reason"], "segment": checked.get("first_bad_segment"),
                     "min_gap": checked.get("min_gap_map"), "margin": checked.get("margin_m"),
                     "segment_points": attempt["first_bad_segment_points"]})
    summary = {"status": status, "adopted_clearance": state["clearance_m"],
               "path_points": len(state["path"]) if state["path"] is not None else 0,
               "next_retry": state["next_plan_t"], "attempts": rows}
    previous = state["plan_summary_t"]
    if previous is None or t < previous or t-previous >= _BASE_LOOP_CFG["plan_summary_s"]:
        encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        outcomes = "|".join(f"{row['c']:.2f}:" + ("accepted" if row["accepted"] else row["reason"]) for row in rows)
        builtins.print(f"DBG plan_summary t={t:.2f} status={status} attempts={outcomes} "
                       f"adopted_clearance={summary['adopted_clearance']} path_points={summary['path_points']} "
                       f"next_retry={summary['next_retry']} details={encoded}")
        state["plan_summary_t"] = t
    return summary


def _base_loop_path_distance(target, path):
    closest = float("inf")
    for p, q in zip(path, path[1:]):
        dx, dy = q[0]-p[0], q[1]-p[1]
        length2 = dx*dx + dy*dy
        fraction = 0.0 if length2 <= 1e-18 else max(0.0, min(1.0,
            ((target[0]-p[0])*dx + (target[1]-p[1])*dy) / length2))
        closest = min(closest, math.hypot(target[0]-p[0]-fraction*dx,
                                         target[1]-p[1]-fraction*dy))
    return closest


def _base_loop_projection(path, point, hint_s=None):
    """Nearest path projection; equal-distance branches use prior arc position."""
    records, offset = [], 0.0
    for index, (p, q) in enumerate(zip(path, path[1:])):
        dx, dy = q[0]-p[0], q[1]-p[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        fraction = max(0.0, min(1.0, ((point[0]-p[0])*dx + (point[1]-p[1])*dy) / (length*length)))
        projected = (p[0] + fraction*dx, p[1] + fraction*dy)
        records.append({"distance": math.dist(point[:2], projected), "s": offset+fraction*length,
                        "index": index, "fraction": fraction, "length": length, "point": projected})
        offset += length
    if not records:
        return {"distance": math.dist(point[:2], path[0]), "s": 0.0, "index": 0,
                "fraction": 0.0, "length": 0.0, "point": tuple(path[0]), "total": 0.0}
    nearest = min(record["distance"] for record in records)
    tied = [record for record in records if record["distance"] <= nearest + 1e-9]
    chosen = min(tied, key=lambda record: (record["s"], record["index"]) if hint_s is None
                 else (abs(record["s"]-hint_s), record["s"], record["index"]))
    return dict(chosen, total=offset)


def _base_loop_point_at_s(path, path_s):
    remaining = max(0.0, float(path_s))
    for p, q in zip(path, path[1:]):
        length = math.dist(p, q)
        if length <= 1e-12:
            continue
        if remaining <= length:
            fraction = remaining / length
            return (p[0]+fraction*(q[0]-p[0]), p[1]+fraction*(q[1]-p[1]))
        remaining -= length
    return tuple(path[-1])


def _base_loop_forward_target(path, pose, projection=None):
    """Forward point on supplied polyline only; no path search or new route."""
    cfg = _BASE_LOOP_CFG
    current = projection or _base_loop_projection(path, pose[:2], _BASE_LOOP_ST["path_progress_s"])
    remaining = current["total"] - current["s"]
    if remaining < cfg["target_min_ahead_m"] - 1e-12:
        return None
    same_segment = (1.0-current["fraction"]) * current["length"]
    ahead = max(cfg["target_min_ahead_m"], min(cfg["target_max_ahead_m"], same_segment))
    return _base_loop_point_at_s(path, min(current["total"], current["s"]+ahead))


def _base_loop_note_target(t, reason, ahead, path_distance):
    previous = _BASE_LOOP_ST["target_note_t"].get(reason)
    if previous is None or t < previous or t-previous >= _BASE_LOOP_CFG["target_note_s"]:
        action = "checked_zero" if reason == "no_forward_range" else "forward_target"
        builtins.print(f"DBG target_guard t={t:.2f} reason={reason} ahead={ahead:.3f} path_distance={path_distance:.3f} action={action}")
        _BASE_LOOP_ST["target_note_t"][reason] = t


def _base_loop_guard_target(path, pose, requested, t):
    cfg, state = _BASE_LOOP_CFG, _BASE_LOOP_ST
    robot = _base_loop_projection(path, pose[:2], state["path_progress_s"])
    state["path_progress_s"] = robot["s"]
    projected = _base_loop_projection(path, requested, robot["s"])
    ahead = projected["s"] - robot["s"]
    reason = ("off_path" if projected["distance"] > cfg["target_path_limit_m"] + 1e-12 else
              "behind" if ahead < -1e-12 else
              "too_near" if ahead < cfg["target_min_ahead_m"] - 1e-12 else
              "too_far" if ahead > cfg["target_max_ahead_m"] + 1e-12 else None)
    detail = {"target_path_distance_m": projected["distance"], "target_ahead_m": ahead,
              "robot_path_s": robot["s"], "target_path_s": projected["s"],
              "target_reject_reason": reason, "target_corrected": reason is not None,
              "target_corrected_ahead_m": None, "target_fallback_unavailable": False}
    if reason is None:
        return requested, detail
    target = _base_loop_forward_target(path, pose, robot)
    if target is None:
        detail.update(target_reject_reason="no_forward_range", requested_reject_reason=reason,
                      target_fallback_unavailable=True)
    else:
        # The chosen arc is known even at a self-intersection: do not reproject it.
        same_segment = (1.0-robot["fraction"]) * robot["length"]
        detail["target_corrected_ahead_m"] = min(robot["total"]-robot["s"],
            max(cfg["target_min_ahead_m"], min(cfg["target_max_ahead_m"], same_segment)))
    _base_loop_note_target(t, detail["target_reject_reason"], ahead, projected["distance"])
    return target, detail


def _base_control_loop(t, pose, env=None):
    """Own the loop; call model plan/target/follow hooks, then one common chooser.

    Planning occurs initially and on the tick after continuous map stop >10 s.
    All rejected attempts preserve a previous valid path. Without a path,
    checked zero is requested and planning retries after 10 s. A planning hook
    exception rejects that clearance and tries the next one. Other hook faults
    retain checked-zero behavior. validate_plan uses the same adoption rule.
    """
    state, cfg = _BASE_LOOP_ST, _BASE_LOOP_CFG
    t = float(t)
    pose = tuple(float(q) for q in pose)
    if len(pose) != 3 or not math.isfinite(t) or not all(math.isfinite(q) for q in pose):
        raise ValueError("base loop requires a finite simulation time and pose")
    if state["last_t"] is not None and t < state["last_t"]:
        state.update(path=None, clearance_m=None, replan_pending=True,
                     block_since=None, fault_key=None, last=None, path_progress_s=None, target_note_t={},
                     plan_context=None, plan_note_t={}, next_plan_t=None, plan_summary_t=None)
    state["last_t"] = t
    target, nominal = tuple(GOAL[:2]), (0.0, 0.0)
    distance = math.dist(pose[:2], GOAL[:2])
    yaw_error = math.atan2(math.sin(GOAL[2]-pose[2]), math.cos(GOAL[2]-pose[2]))
    goal_mode = distance <= cfg["goal_distance_m"]
    goal_stop = goal_mode and abs(yaw_error) <= cfg["goal_yaw_rad"]
    detail = {"t": t, "pose": pose, "phase": "waiting_for_path", "plan_attempts": [],
              "clearance_m": state["clearance_m"], "target_requested": None,
              "target_corrected": False, "target_reject_reason": None, "target_ahead_m": None,
              "robot_path_s": None, "target_path_s": None, "error": None,
              "plan_rejection_count": 0, "plan_error_count": 0, "plan_errors": [], "plan_summary": None}
    stage = "base"
    try:
        if goal_mode:
            detail["phase"] = "goal_stop" if goal_stop else "goal_align"
            nominal = (0.0, 0.0 if goal_stop else max(-cfg["w_max"], min(cfg["w_max"], 1.5*yaw_error)))
        else:
            if (state["path"] is None and state["next_plan_t"] is not None
                    and t >= state["next_plan_t"]):
                state["replan_pending"] = True
            if state["replan_pending"]:
                state["replan_pending"], state["block_since"] = False, None
                state["next_plan_t"] = None
                stage = "plan_path"
                for clearance in cfg["clearances_m"]:
                    attempt = {"clearance_m": clearance, "accepted": False}
                    detail["plan_attempts"].append(attempt)
                    state["plan_context"] = (pose[:2], tuple(GOAL[:2]), clearance)
                    try:
                        proposed = _base_loop_hook("plan_path", pose[:2], tuple(GOAL[:2]), clearance)
                        validation = validate_plan(proposed)
                    except Exception as error:  # noqa: BLE001 - one rejected clearance, then try the next
                        validation = _base_plan_exception(clearance, error)
                        validation["start_source"] = "plan_context"
                    finally:
                        state["plan_context"] = None
                    attempt.update({key: value for key, value in validation.items() if key != "path"})
                    if validation["accepted"]:
                        checked_path = tuple(validation["path"])
                        state["path"], state["clearance_m"] = checked_path, clearance
                        state["path_progress_s"] = None
                        attempt["points"] = len(checked_path)
                        break
                    detail["plan_rejection_count"] += 1
                    if validation["error"] is not None:
                        detail["plan_error_count"] += 1
                        detail["plan_errors"].append({"clearance_m": clearance, **validation["error"]})
                    _base_loop_note_plan(t, validation)
                if not any(a["accepted"] for a in detail["plan_attempts"]):
                    if state["path"] is not None:
                        detail["phase"] = "old_path_after_plan_failure"
                    else:
                        state["next_plan_t"] = t + cfg["plan_retry_s"]
                detail["plan_summary"] = _base_loop_note_plan_summary(t, detail["plan_attempts"])
            if state["path"] is not None:
                stage = "lookahead_target"
                # A fresh list protects the cached path from accidental hook mutation.
                requested = _base_loop_hook("lookahead_target", list(state["path"]), pose)
                target = _base_loop_pair(requested, "lookahead_target")
                detail["target_requested"] = target
                guarded, target_detail = _base_loop_guard_target(state["path"], pose, target, t)
                detail.update(target_detail)
                if guarded is None:
                    target = _base_loop_point_at_s(state["path"], detail["robot_path_s"])
                    detail["phase"] = "waiting_for_forward_target"
                else:
                    target = guarded
                    stage = "follow_cmd"
                    raw_v, raw_w = _base_loop_pair(_base_loop_hook("follow_cmd", pose, target), "follow_cmd")
                    nominal = (max(cfg["v_min"], min(cfg["v_max"], raw_v)),
                               max(-cfg["w_max"], min(cfg["w_max"], raw_w)))
                    if detail["phase"] != "old_path_after_plan_failure":
                        detail["phase"] = "path_follow"
        state["fault_key"] = None
    except Exception as error:  # noqa: BLE001 - model hook faults cannot bypass the common chooser
        nominal = (0.0, 0.0)
        detail["phase"], detail["error"] = "hook_error", {"hook": stage, "type": type(error).__name__, "message": str(error)[:160]}
        _base_loop_note_fault(t, stage, error)
    v, w, info = choose_command(nominal[0], nominal[1], target)
    blocked = (info["chosen"] == "stop" and v == 0.0 and w == 0.0
               and info["candidates"]["follow"]["t_hit_map"] is not None)
    if blocked:
        if state["block_since"] is None:
            state["block_since"] = t
        elif t - state["block_since"] > cfg["map_stop_s"]:
            state["replan_pending"], state["block_since"] = True, None
    else:
        state["block_since"] = None
    done = goal_stop and v == 0.0 and w == 0.0
    detail.update(clearance_m=state["clearance_m"], target_xy=target, nominal=nominal,
                  command=(v, w), info=info, done=done, replan_pending=state["replan_pending"],
                  next_plan_t=state["next_plan_t"])
    state["last"] = detail
    return v, w, done


# ---- DBG 기록 ----
# choose_command 자동 선택 줄과 10 s 지도 정지 경고는 builtins.print 로 따로 출력하며 이 제한을 쓰지 않습니다.
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

def plan_path(start_xy, goal_xy, clearance_m):
    """
    Simple deterministic grid A* planner.

    * Grid resolution 0.2 m (covers the whole arena).
    * 8‑connectivity, each edge is accepted only if
      ``segment_clearance`` between the two cell centres ≥ ``clearance_m``.
    * Cell centres are kept only if their own ``map_clearance`` ≥ ``clearance_m``.
    * The returned polyline must start / end ≤0.5 m from the real start / goal
      (otherwise ``validate_plan`` will reject it).  If no free cell satisfies
      that requirement the function returns ``None`` so the base will retry with
      a smaller clearance.
    """

    import heapq

    # ------------------------------------------------------------------
    # grid definition
    # ------------------------------------------------------------------
    res = 0.2  # metres – a compromise between resolution and speed
    wx0, wx1, wy0, wy1 = WALLS

    # number of cells in each direction
    nx = int(math.ceil((wx1 - wx0) / res))
    ny = int(math.ceil((wy1 - wy0) / res))

    # helper: world coordinate of a cell centre
    def cell_center(ix, iy):
        return (wx0 + (ix + 0.5) * res, wy0 + (iy + 0.5) * res)

    # ------------------------------------------------------------------
    # build set of traversable cells (centre clearance ≥ clearance_m)
    # ------------------------------------------------------------------
    free = set()
    for ix in range(nx):
        cx = wx0 + (ix + 0.5) * res
        for iy in range(ny):
            cy = wy0 + (iy + 0.5) * res
            if map_clearance(cx, cy) >= clearance_m:
                free.add((ix, iy))

    # ------------------------------------------------------------------
    # locate start / goal cells (must be ≤0.5 m from the true points)
    # ------------------------------------------------------------------
    def nearest_free(point):
        px, py = point
        best = None
        best_dist = 1e9
        # examine cells whose centre lies inside a 0.5 m disc around the point
        ix0 = int(math.floor((px - 0.5 - wx0) / res))
        ix1 = int(math.ceil((px + 0.5 - wx0) / res))
        iy0 = int(math.floor((py - 0.5 - wy0) / res))
        iy1 = int(math.ceil((py + 0.5 - wy0) / res))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if (ix, iy) not in free:
                    continue
                cx, cy = cell_center(ix, iy)
                d = math.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist = d
                    best = (ix, iy)
        return best, best_dist

    start_cell, start_dist = nearest_free(start_xy)
    goal_cell, goal_dist = nearest_free(goal_xy)

    # if we cannot find a free cell within the allowed 0.5 m, give up
    if start_cell is None or goal_cell is None:
        return None

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (1, -1), (-1, 1), (1, 1)]

    def heuristic(ix, iy):
        cx, cy = cell_center(ix, iy)
        gx, gy = cell_center(*goal_cell)
        return math.hypot(cx - gx, cy - gy)

    open_set = []
    heapq.heappush(open_set, (heuristic(*start_cell), 0.0, start_cell))
    came_from = {}
    g_score = {start_cell: 0.0}
    found = False

    while open_set:
        _, cur_g, current = heapq.heappop(open_set)
        if current == goal_cell:
            found = True
            break

        cx, cy = cell_center(*current)

        for dx, dy in neighbor_offsets:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt not in free:
                continue

            # edge clearance test
            nx, ny = cell_center(*nxt)
            if segment_clearance((cx, cy), (nx, ny)) < clearance_m:
                continue

            # cost = Euclidean distance of the edge
            edge_len = math.hypot(nx - cx, ny - cy)
            tentative_g = cur_g + edge_len

            if nxt not in g_score or tentative_g < g_score[nxt] - 1e-12:
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(*nxt)
                heapq.heappush(open_set, (f_score, tentative_g, nxt))
                came_from[nxt] = current

    if not found:
        return None

    # ------------------------------------------------------------------
    # reconstruct path (list of world (x,y) points)
    # ------------------------------------------------------------------
    path_cells = []
    cur = goal_cell
    while cur != start_cell:
        path_cells.append(cur)
        cur = came_from[cur]
    path_cells.append(start_cell)
    path_cells.reverse()

    path = [cell_center(ix, iy) for ix, iy in path_cells]

    # ``validate_plan`` will add the exact start / goal connectors if they are
    # slightly off, so we simply return the cell‑centre polyline.
    return path


def lookahead_target(path, pose):
    """
    Return a point on *path* that lies 0.3 – 2.0 m ahead of the robot’s
    projection onto the path.  If the remaining forward distance is less than
    the minimum required, ``None`` is returned so the base will request a
    checked‑zero.
    """

    # find the nearest projection of the robot onto the polyline
    # ------------------------------------------------------------------
    robot_x, robot_y = pose[0], pose[1]
    best = None
    best_dist = 1e9
    total_s = 0.0
    best_record = None

    for i, (p, q) in enumerate(zip(path, path[1:])):
        px, py = p
        qx, qy = q
        dx, dy = qx - px, qy - py
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-12:
            continue
        # projection fraction on the segment
        t = ((robot_x - px) * dx + (robot_y - py) * dy) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        proj_x = px + t * dx
        proj_y = py + t * dy
        dist = math.hypot(robot_x - proj_x, robot_y - proj_y)
        if dist < best_dist - 1e-12 or (abs(dist - best_dist) < 1e-12 and total_s + t * seg_len < best_record["s"]):
            best_dist = dist
            best_record = {
                "point": (proj_x, proj_y),
                "s": total_s + t * seg_len,
                "segment_len": seg_len,
                "fraction": t,
                "index": i,
                "total": total_s + seg_len,
            }
        total_s += seg_len

    if best_record is None:
        return None

    cfg_min = _BASE_LOOP_CFG["target_min_ahead_m"]
    cfg_max = _BASE_LOOP_CFG["target_max_ahead_m"]

    # remaining distance on the current segment after the projection
    remaining_on_seg = (1.0 - best_record["fraction"]) * best_record["segment_len"]

    # Desired forward distance respecting the limits
    if remaining_on_seg >= cfg_min:
        ahead = min(cfg_max, remaining_on_seg)
    else:
        # need to look into following segments
        needed = cfg_min - remaining_on_seg
        forward = remaining_on_seg
        # walk forward through subsequent segments until we have at least cfg_min
        idx = best_record["index"] + 1
        while needed > 0.0 and idx < len(path) - 1:
            seg_len = math.hypot(path[idx + 1][0] - path[idx][0],
                                 path[idx + 1][1] - path[idx][1])
            take = min(seg_len, needed)
            forward += take
            needed -= take
            idx += 1
        if forward < cfg_min - 1e-12:
            return None
        ahead = min(cfg_max, forward)

    target_s = best_record["s"] + ahead
    # Use the helper that converts an arc‑length to a point on the path
    target_pt = _base_loop_point_at_s(path, target_s)
    return target_pt


def follow_cmd(pose, target_xy):
    """
    Pure pursuit style controller without any object handling.
    Returns (v, w) before clipping – the base will clip to the allowed limits.
    """

    x, y, yaw = pose
    tx, ty = target_xy

    # distance and bearing error
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    e = bearing - yaw
    # wrap to [-π, π]
    e = math.atan2(math.sin(e), math.cos(e))

    # speed command
    if abs(e) > 0.35:
        v = 0.0
    else:
        v = min(0.3, 0.5 * dist + 0.1)

    # angular command (clamped later by the base)
    w = max(-0.4, min(0.4, 1.5 * e))

    return v, w


def controller(t, pose, env): return _base_control_loop(t, pose, env)


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
    return _base_control_loop(t, pose, None)


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
