# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    """Clamp v to the interval [lo, hi]."""
    return max(lo, min(hi, v))


# Way‑points that avoid all boxes and the fixed structures.
# The last point is the goal (GOAL) itself.
_WAYPOINTS = [
    (-5.0, -10.8),   # move east away from the left wall
    (-5.0, 5.8),     # go north, staying below the left‑side shelf
    (0.0, 5.8),      # move east below the large central boxes
    (0.0, 12.0),     # go north, staying left of the middle obstacles
    (1.5, 12.0),     # shift east but still left of box 08
    (1.5, 16.5),     # go north past all remaining boxes
    (GOAL[0], GOAL[1])  # final goal position
]

def controller(t, pose, env):
    """Follow the way-points; stop/slow when the front lidar sees an object that is not in the static map."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.done_align = False
        controller.hold = 0

    x, y, yaw = pose

    # --- 지도 외 물체 감지 (매 스텝) ---
    stop, scale, backup, rear_clear = _sense_movers(pose)
    if stop:
        controller.hold = _HOLD_STEPS
    elif controller.hold > 0:
        controller.hold -= 1
        stop, scale = True, 0.0

    # 1) Goal-alignment phase
    if controller.idx >= len(_WAYPOINTS):
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, w, False

    # 2) Normal waypoint-following
    tx, ty = _WAYPOINTS[controller.idx]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # 3) Control law (unchanged)
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        v = min(1.5, 2.5 * dist + 0.5)
        w = _clamp(7.5 * e, -2.0, 2.0)

    # 4) 감지 결과 반영: 감속 / 정지(회전은 유지) / 아주 가까우면 후방이 빌 때만 후진
    if backup and rear_clear:
        return -0.3, 0.0, False
    if stop:
        v = 0.0
    else:
        v = v * scale

    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1

    return v, w, False




_SAFE_MARGIN = 0.30          # 정적 지도 영역에 더하는 여유 [m] (자세·스캔 지연 흡수)


_STOP_RANGE = 2.0            # 정면 콘 안 지도 외 물체가 이 거리 이내면 정지 [m]


_SLOW_RANGE = 3.5            # 이 거리부터 정지 거리까지 선형 감속 [m]


_BACKUP_RANGE = 1.0          # 이 거리 이내로 다가오면 (후방이 비었을 때) 천천히 후진 [m]


_CONE = math.radians(45.0)   # 정면 감지 콘 반각


_MIN_PTS = 4                 # 잡음 제거용 최소 점 수


_SENSE_MAX = 6.0             # 이보다 먼 점은 무시 [m]


_HOLD_STEPS = 15             # 감지가 사라진 뒤 정지를 유지할 스텝 수 (0.25 s)


_STATIC_RECTS = np.array(
    [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0,
      b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0) for b in MISSION["boxes"]]
    + [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES],
    dtype=np.float64)


def _off_map_points(pts, pose):
    """로봇 기준 점 (N, 2) 중 정적 지도(벽, 바닥 박스, 고정물)에 속하지 않는 점만 로봇 기준으로 돌려줍니다."""
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= _SENSE_MAX]
    if len(pts) == 0:
        return pts
    w = to_world(pts, pose)
    wx, wy = w[:, 0], w[:, 1]
    m = _SAFE_MARGIN
    static = ((wx <= WALLS[0] + m) | (wx >= WALLS[1] - m) |
              (wy <= WALLS[2] + m) | (wy >= WALLS[3] - m))
    R = _STATIC_RECTS
    inside = ((wx[:, None] >= R[:, 0] - m) & (wx[:, None] <= R[:, 1] + m) &
              (wy[:, None] >= R[:, 2] - m) & (wy[:, None] <= R[:, 3] + m)).any(axis=1)
    static = static | inside
    return pts[~static]


def _sense_movers(pose):
    """전방 라이다로 지도 외 물체를 보고 (정지 여부, 속도 배율, 후진 필요 여부, 후방 비었는지) 를 돌려줍니다."""
    front = lidar2d_points("front")
    movers = _off_map_points(front, pose)
    stop, scale, backup = False, 1.0, False
    if len(movers) > 0:
        r = np.hypot(movers[:, 0], movers[:, 1])
        ang = np.arctan2(movers[:, 1], movers[:, 0])
        cone = (np.abs(ang) <= _CONE) & (movers[:, 0] > 0.0)
        rc = r[cone]
        if np.count_nonzero(rc <= _STOP_RANGE) >= _MIN_PTS:
            stop = True
            scale = 0.0
        elif np.count_nonzero(rc <= _SLOW_RANGE) >= _MIN_PTS:
            dmin = float(np.sort(rc[rc <= _SLOW_RANGE])[_MIN_PTS - 1])
            scale = _clamp((dmin - _STOP_RANGE) / (_SLOW_RANGE - _STOP_RANGE), 0.0, 1.0)
        if np.count_nonzero(rc <= _BACKUP_RANGE) >= _MIN_PTS:
            backup = True
    # 후방이 비었는지: 정적·동적 구분 없이 뒤쪽 0.9 m, 폭 ±0.5 m 안에 점이 없어야 함
    back = lidar2d_points("back")
    rear_clear = True
    if len(back) > 0:
        b = np.asarray(back, dtype=np.float64)
        rear = (b[:, 0] < _BODY[0]) & (b[:, 0] >= _BODY[0] - 0.9) & (np.abs(b[:, 1]) <= 0.5)
        rear_clear = np.count_nonzero(rear) < _MIN_PTS
    return stop, scale, backup, rear_clear


# ========================= [END EDIT REGION] ===============================
