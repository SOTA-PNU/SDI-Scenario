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


# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# 작성: Claude (Anthropic), 2026-10-29.
# 정적 지도(벽, MISSION 상자, 장면의 고정물)로 첫 호출에 경로를 계산하고 그 경로를 따라간다.
# 지도에 없는 물체는 앞뒤 2D 라이다로만 감지한다. 움직이는 물체의 위치, 경로, 시각은 코드에 없다.
# 매 호출: 라이다 점 -> 지도에 없는 점 -> 묶음 -> 추적과 속도 추정 -> 후보 명령마다 몇 초 앞을 굴려
# 몸체와 예측한 점의 간격을 보고, 안전한 것 가운데 가장 앞으로 가는 명령을 고른다.
import heapq

_WALLS = (-10.30, 9.35, -12.15, 17.95)        # 벽 안쪽 면: x 최소, x 최대, y 최소, y 최대
_FIXED = np.array([                           # 고정물 43개의 경계 상자: x 최소, x 최대, y 최소, y 최대
    (-10.45, -10.00, -9.25, -8.80),
    (9.10, 9.55, -9.20, -8.75),
    (-9.95, -8.70, -3.80, -3.40),
    (-9.85, -8.80, -3.80, -3.60),
    (-9.90, -8.80, -3.45, -2.20),
    (-9.60, -6.05, -3.45, -2.25),
    (1.85, 5.30, -3.45, -0.70),
    (-9.60, -8.85, -3.35, -1.05),
    (-10.45, -10.00, -3.25, -2.80),
    (9.10, 9.55, -3.20, -2.75),
    (-9.85, -8.80, -2.25, -0.95),
    (-9.90, -8.80, -1.00, 0.25),
    (-9.60, -8.85, -0.90, 0.20),
    (-9.85, -8.80, 0.20, 0.40),
    (-9.85, -8.80, 0.45, 1.70),
    (-9.85, -8.75, 0.60, 4.05),
    (-9.85, -8.80, 1.65, 2.95),
    (-10.45, -10.00, 2.75, 3.20),
    (9.10, 9.55, 2.80, 3.25),
    (-9.85, -8.80, 2.90, 4.15),
    (-9.85, -8.80, 4.20, 4.40),
    (-9.75, -8.70, 4.45, 5.70),
    (-9.35, -8.70, 4.75, 5.40),
    (-9.75, -8.70, 5.65, 6.95),
    (-9.70, -8.75, 5.65, 8.25),
    (-9.75, -8.70, 6.90, 8.15),
    (-9.85, -8.80, 8.20, 8.40),
    (-9.75, -8.70, 8.45, 9.70),
    (-9.75, -9.00, 8.70, 9.65),
    (-10.45, -10.00, 8.75, 9.20),
    (9.10, 9.55, 8.80, 9.25),
    (-9.75, -8.70, 9.65, 10.90),
    (-9.75, -9.00, 9.85, 10.80),
    (-9.75, -8.70, 10.85, 12.15),
    (-9.75, -9.00, 11.15, 12.10),
    (-9.95, -8.70, 12.10, 12.50),
    (-9.85, -8.80, 12.20, 12.40),
    (-7.05, -5.20, 12.90, 14.35),
    (-6.85, -5.35, 13.00, 14.40),
    (-6.60, -5.50, 13.20, 14.15),
    (-10.45, -10.00, 14.75, 15.20),
    (9.10, 9.55, 14.80, 15.25),
    (-4.15, -4.05, 17.75, 18.00),
], dtype=np.float64)
_BODY_C = np.array([[-0.05, 0.0], [-0.41, 0.0]])   # 몸체(x -0.59..0.13, 바퀴 y +-0.25)를 덮는 두 원의 중심
_BODY_R = 0.32
_CFG = dict(res=0.1, c_min=0.75, c_pref=1.3, c_los=1.0,                       # 경로 계산
            near=12.0, margin=0.25, voxel=0.1, gap=0.45, gate=1.0, keep=5.0, fresh=1.0,  # 감지와 추적
            vwin=0.6, hist=30.0, axis_spread=1.0, slack=0.15, r_min=0.30, r_pad=0.05,     # 운동 모형
            T=3.0, T_resume=4.0, dt=0.1, d_safe=0.15)                            # 예측과 고르기
_ST = {"rects": None, "path": None, "wp": 1, "align": False, "tracks": [], "nid": 0,
       "scans": (None, None), "mode": "go", "dbg_t": -9.0}


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clear(xs, ys):
    """점(배열)에서 가장 가까운 지도 물체나 벽 안쪽 면까지의 거리 [m]. 물체 안이면 0."""
    r = _ST["rects"]
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    dx = np.maximum(np.maximum(r[:, 0] - xs[..., None], xs[..., None] - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - ys[..., None], ys[..., None] - r[:, 3]), 0.0)
    d = np.hypot(dx, dy).min(axis=-1)
    x0, x1, y0, y1 = _WALLS
    return np.maximum(0.0, np.minimum(d, np.minimum(np.minimum(xs - x0, x1 - xs), np.minimum(ys - y0, y1 - ys))))


def _seg_clear(p, q):
    n = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05)))
    k = np.linspace(0.0, 1.0, n + 1)
    return float(_clear(p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k).min())


def _plan(start, goal):
    """격자 A*. 여유 c_min 미만인 칸은 막고, c_pref 미만이면 비용을 더한다. 그 뒤 여유 c_los 이상인 직선으로 줄인다."""
    res = _CFG["res"]
    x0, x1, y0, y1 = _WALLS
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    C = np.stack([_clear(np.full(ys.size, x), ys) for x in xs])        # (nx, ny)
    nx, ny = C.shape

    def cell(p):
        return (min(nx - 1, max(0, int((p[0] - x0) / res))), min(ny - 1, max(0, int((p[1] - y0) / res))))

    s, g = cell(start), cell(goal)
    free = C >= _CFG["c_min"]
    free[s] = free[g] = True
    pen = 1.0 + 4.0 * np.maximum(0.0, _CFG["c_pref"] - C)
    dist, prev, done = {s: 0.0}, {}, set()
    heap = [(0.0, s)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while heap:
        _, u = heapq.heappop(heap)
        if u in done:
            continue
        if u == g:
            break
        done.add(u)
        for dx, dy, length in steps:
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < nx and 0 <= v[1] < ny and free[v]:
                nd = dist[u] + length * res * 0.5 * (pen[u] + pen[v])
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd + res * math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g != s and g not in prev:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    pts = [(float(xs[i]), float(ys[j])) for i, j in reversed(cells)]
    pts[0], pts[-1] = (start[0], start[1]), (goal[0], goal[1])
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and _seg_clear(pts[i], pts[j + 1]) >= _CFG["c_los"]:
            j += 1
        out.append(pts[j])
        i = j
    return out


def _cluster(P):
    """가까운 점끼리(gap 이하) 묶는다."""
    n = len(P)
    if n == 0:
        return []
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) <= _CFG["gap"]
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


def _fit_vel(hist, win):
    """추적 묶음 중심의 최근 win 초를 직선으로 맞춘 속도. 자료가 모자라면 0."""
    t_end = hist[-1][0]
    h = np.array([q for q in hist if t_end - q[0] <= win])
    if len(h) < 4 or h[-1, 0] - h[0, 0] < 0.3:
        return np.zeros(2)
    tt = h[:, 0] - h[:, 0].mean()
    den = float((tt * tt).sum())
    v = np.array([(tt * (h[:, 1] - h[:, 1].mean())).sum() / den, (tt * (h[:, 2] - h[:, 2].mean())).sum() / den])
    sp = float(np.hypot(*v))
    if sp < 0.08:
        return np.zeros(2)
    return v * min(1.0, 1.5 / sp)


def _update_motion(tr):
    """관찰한 중심으로 운동축(한번 정하면 고정), 축 위 위치, 최고 속력, 실제로 방향을 바꾼 지점(양 끝)을 갱신한다."""
    h = np.array(tr["hist"])
    xy = h[:, 1:3]
    if tr["axis"] is None and len(h) >= 6:
        c = xy - xy.mean(axis=0)
        w, vecs = np.linalg.eigh(c.T @ c / len(c))
        if 2.0 * math.sqrt(max(w[-1], 0.0)) * 2.0 >= _CFG["axis_spread"]:     # 주축 방향 퍼짐(약 +-2 sigma)
            tr["axis"] = vecs[:, -1] / np.linalg.norm(vecs[:, -1])
            tr["origin"] = xy.mean(axis=0)
    if tr["axis"] is None:
        return
    u, o = tr["axis"], tr["origin"]
    s_now = float((xy[-1] - o) @ u)
    ds = float(tr["vel"] @ u)
    tr["speed"] = max(tr.get("speed", 0.0) * 0.995, abs(ds))
    sign = 1 if ds > 0.1 else (-1 if ds < -0.1 else 0)
    if sign:
        if tr.get("sign") and sign != tr["sign"]:
            if tr["sign"] > 0:
                tr["hi"] = tr["ext"]                   # + 로 가다 - 로: 그 사이 가장 큰 값에서 돌았다
            else:
                tr["lo"] = tr["ext"]
            tr["ext"] = s_now
        tr["sign"] = sign
        prev = tr.get("ext", s_now)
        tr["ext"] = max(prev, s_now) if sign > 0 else min(prev, s_now)
    tr["s"] = s_now


def _sense(t, pose):
    """새 스캔이 왔을 때만 지도에 없는 점을 묶어 추적을 갱신한다."""
    front, back = lidar2d_points("front"), lidar2d_points("back")
    if front is _ST["scans"][0] and back is _ST["scans"][1]:
        return
    _ST["scans"] = (front, back)
    parts = [to_world(p, pose)[:, :2] for p in (front, back) if len(p)]
    P = np.concatenate(parts) if parts else np.zeros((0, 2))
    if len(P):
        P = P[np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]) < _CFG["near"]]
    if len(P):
        P = P[_clear(P[:, 0], P[:, 1]) > _CFG["margin"]]
    if len(P):
        _, keep = np.unique(np.floor(P / _CFG["voxel"]).astype(np.int64), axis=0, return_index=True)
        P = P[keep]
    groups = _cluster(P)
    tracks = _ST["tracks"]
    for tr in tracks:
        tr["hit"] = False
    for grp in groups:
        c = grp.mean(axis=0)
        best, bd = None, _CFG["gate"]
        for tr in tracks:
            if tr["hit"]:
                continue
            last_t, lx, ly = tr["hist"][-1]
            dd = math.hypot(c[0] - (lx + tr["vel"][0] * (t - last_t)), c[1] - (ly + tr["vel"][1] * (t - last_t)))
            if dd < bd:
                best, bd = tr, dd
        if best is None:
            best = {"id": _ST["nid"], "hist": [], "vel": np.zeros(2), "axis": None}
            _ST["nid"] += 1
            tracks.append(best)
        best["hit"], best["seen"], best["pts"] = True, t, grp
        best["r"] = max(_CFG["r_min"], float(np.hypot(*(grp - c).T).max())) + _CFG["r_pad"]
        best["hist"] = [h for h in best["hist"] if t - h[0] <= _CFG["hist"]] + [(t, float(c[0]), float(c[1]))]
        best["vel"] = _fit_vel(best["hist"], _CFG["vwin"])
        _update_motion(best)
    _ST["tracks"] = [tr for tr in tracks if t - tr["seen"] <= _CFG["keep"]]


def _obstacle_shapes(t, K, dt):
    """판단에 쓸 장애물: 정지 묶음은 점 그대로, 움직이는 묶음은 예측 구간을 덮는 캡슐 (k 마다 선분 a-b, 반지름 r)."""
    pts, caps = [], []
    for tr in _ST["tracks"]:
        if t - tr["seen"] > _CFG["fresh"]:
            continue
        c = tr["pts"].mean(axis=0)
        if tr["axis"] is None and float(np.hypot(*tr["vel"])) < 0.15:
            pts.append(tr["pts"])
            continue
        tk = np.arange(K + 1) * dt + (t - tr["seen"])
        if tr["axis"] is None:                     # 아직 축을 모름: 지금 속도로 직진하거나, 언제든 되돌아올 수 있다
            u = tr["vel"] / max(1e-6, float(np.hypot(*tr["vel"])))
            sp = float(np.hypot(*tr["vel"]))
            lo, hi = -sp * tk, sp * tk
        else:
            u, sp = tr["axis"], max(0.1, tr.get("speed", 0.0))
            s0 = float((c - tr["origin"]) @ u)
            if "lo" in tr and "hi" in tr and tr["hi"] - tr["lo"] > 0.5:
                L = tr["hi"] - tr["lo"]                # 양 끝을 봤다: 그 사이를 오간다고 예측(앞뒤 slack 여유)
                d = tr.get("sign", 1) or 1
                ph = (s0 - tr["lo"]) if d > 0 else (2 * L - (s0 - tr["lo"]))
                q = np.mod(ph + sp * tk, 2 * L)
                s_pred = tr["lo"] + np.where(q <= L, q, 2 * L - q)
                lo, hi = s_pred - s0 - _CFG["slack"], s_pred - s0 + _CFG["slack"]
            else:                                  # 언제든 되돌아올 수 있다: +- 속력 x 시간, 본 끝 너머로는 가지 않는다
                lo, hi = -sp * tk, sp * tk
                if "lo" in tr:
                    lo = np.maximum(lo, tr["lo"] - _CFG["slack"] - s0)
                if "hi" in tr:
                    hi = np.minimum(hi, tr["hi"] + _CFG["slack"] - s0)
                lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
        a = c[None, :] + lo[:, None] * u[None, :]
        b = c[None, :] + hi[:, None] * u[None, :]
        caps.append((a, b, tr["r"]))
    P0 = np.concatenate(pts) if pts else np.zeros((0, 2))
    return P0, caps


def _follow(pose):
    """경로 추종(시험으로 통과를 확인한 방법): (v, w, done)."""
    x, y, yaw = pose
    path = _ST["path"]
    if _ST["align"]:
        e = _wrap(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False
    i = _ST["wp"]
    tx, ty = path[i]
    dist = math.hypot(tx - x, ty - y)
    if i < len(path) - 1 and dist < 0.35:
        _ST["wp"] = i = i + 1
        tx, ty = path[i]
        dist = math.hypot(tx - x, ty - y)
    if i == len(path) - 1 and dist < 0.25:
        _ST["align"] = True
        return _follow(pose)
    e = _wrap(math.atan2(ty - y, tx - x) - yaw)
    w = _clamp(1.5 * e, -0.8, 0.8)
    if abs(e) > 0.35:
        return 0.0, w, False
    return min(0.6, 0.5 * dist + 0.1), w, False


def _margin(pose, v, w, T, P0, caps, static):
    """(v, w) 로 T 초 굴렸을 때 몸체 두 원과 장애물 사이의 가장 작은 여유(안전 거리를 뺀 값)와 지도 여유."""
    dt = _CFG["dt"]
    K = int(round(T / dt))
    x, y, th = pose
    poses = [(x, y, th)]
    for _ in range(K):
        th += w * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        poses.append((x, y, th))
    Q = np.array(poses)
    c, s = np.cos(Q[:, 2]), np.sin(Q[:, 2])
    cx = Q[:, 0:1] + c[:, None] * _BODY_C[None, :, 0] - s[:, None] * _BODY_C[None, :, 1]     # (K+1, 2)
    cy = Q[:, 1:2] + s[:, None] * _BODY_C[None, :, 0] + c[:, None] * _BODY_C[None, :, 1]
    worst = float("inf")
    if len(P0):
        d = np.hypot(P0[None, None, :, 0] - cx[:, :, None], P0[None, None, :, 1] - cy[:, :, None]).min()
        worst = float(d) - _BODY_R - _CFG["d_safe"]
    for a, b, r in caps:
        a, b = a[:K + 1], b[:K + 1]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(axis=1), 1e-9)                                         # (K+1,)
        tq = ((cx - a[:, None, 0]) * ab[:, None, 0] + (cy - a[:, None, 1]) * ab[:, None, 1]) / L2[:, None]
        tq = np.clip(tq, 0.0, 1.0)
        dx = cx - (a[:, None, 0] + tq * ab[:, None, 0])
        dy = cy - (a[:, None, 1] + tq * ab[:, None, 1])
        worst = min(worst, float(np.hypot(dx, dy).min()) - r - _BODY_R - _CFG["d_safe"])
    st = float(_clear(cx, cy).min()) - _BODY_R if static else float("inf")
    return worst, st


def _choose(t, pose, vn, wn):
    """후보 명령 가운데 안전한 것을 고른다. 반환: (이름, v, w, 여유)."""
    K = int(round(_CFG["T_resume"] / _CFG["dt"]))
    P0, caps = _obstacle_shapes(t, K, _CFG["dt"])
    if not len(P0) and not caps:
        return "go", vn, wn, float("inf")
    near = [float(np.hypot(P0[:, 0] - pose[0], P0[:, 1] - pose[1]).min())] if len(P0) else []
    near += [float(np.hypot(*(a[0] - np.array(pose[:2])))) for a, _, _ in caps]
    if min(near) > 7.0:
        return "go", vn, wn, float("inf")
    on_path = [("go", vn, wn)] + ([("slow", 0.6 * vn, wn), ("creep", 0.3 * vn, wn)] if vn > 0 else []) + [("hold", 0.0, 0.0)]
    escape = [("back", -0.25, 0.0), ("backL", -0.2, 0.6), ("backR", -0.2, -0.6), ("arcL", 0.3, 0.7), ("arcR", 0.3, -0.7)]
    scored = []
    for name, v, w in on_path:
        T = _CFG["T_resume"] if (name == "go" and _ST["mode"] != "go") else _CFG["T"]
        m, _ = _margin(pose, v, w, T, P0, caps, False)
        if m >= 0.0:
            return name, v, w, m
        scored.append((m, name, v, w))
    for name, v, w in escape:
        m, st = _margin(pose, v, w, _CFG["T"], P0, caps, True)
        if st >= 0.05:
            scored.append((m, name, v, w))
    m, name, v, w = max(scored)
    return name, v, w, m


def controller(t, pose, env):
    if _ST["rects"] is None:
        boxes = [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0, b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0)
                 for b in MISSION["boxes"]]
        _ST["rects"] = np.concatenate([np.array(boxes, dtype=np.float64), _FIXED])
        _ST["path"] = _plan((pose[0], pose[1]), (GOAL[0], GOAL[1]))
        p = _ST["path"]
        print("DBG plan %d pts, length %.2f m, min segment clearance %.2f m" % (
            len(p), sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])),
            min(_seg_clear(a, b) for a, b in zip(p, p[1:]))), [(round(a, 2), round(b, 2)) for a, b in p])
    _sense(t, pose)
    vn, wn, done = _follow(pose)
    if done:
        print("DBG done t=%.2f pose=(%.2f, %.2f, %.2f)" % (t, pose[0], pose[1], pose[2]))
        return 0.0, 0.0, True
    name, v, w, m = _choose(t, pose, vn, wn)
    if name != _ST["mode"] or t - _ST["dbg_t"] >= 2.0:
        near = min(((math.hypot(tr["pts"][:, 0].mean() - pose[0], tr["pts"][:, 1].mean() - pose[1]), tr) for tr in _ST["tracks"]),
                   key=lambda z: z[0], default=(None, None))
        print("DBG t=%.2f pose=(%.2f, %.2f, %.2f) wp=%d mode=%s v=%.2f w=%.2f margin=%s tracks=%d near=%s" % (
            t, pose[0], pose[1], pose[2], _ST["wp"], name, v, w, "inf" if m == float("inf") else "%.2f" % m,
            len(_ST["tracks"]), "-" if near[1] is None else "%.2f m v=(%.2f, %.2f) axis=%s lo=%s hi=%s" % (
                near[0], near[1]["vel"][0], near[1]["vel"][1], near[1]["axis"] is not None,
                "%.1f" % near[1]["lo"] if "lo" in near[1] else "-", "%.1f" % near[1]["hi"] if "hi" in near[1] else "-")))
        _ST["dbg_t"] = t
    _ST["mode"] = name
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
