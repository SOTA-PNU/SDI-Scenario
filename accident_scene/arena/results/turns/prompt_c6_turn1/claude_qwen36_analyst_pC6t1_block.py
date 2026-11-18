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
    """경유점 추종 + 2D 라이다 기반 반응형 정지/후진 회피."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
    dyn_w, dyn_r, raw = _update_sensors(pose)
    if not _NAV["got"]:
        return 0.0, 0.0, False          # 라이다 데이터가 오기 전에는 움직이지 않음

    v, w, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True

    fx, rx = _BODY[1], _BODY[0]
    ax, ay = dyn_r[:, 0], np.abs(dyn_r[:, 1])
    in_f = (ax > fx - 0.05) & (ay < _HW)
    gap_f = _kth_smallest(ax[in_f] - fx)                                  # 앞 통로의 움직이는 물체까지 거리
    gap_r = _kth_smallest(rx - ax[(ax < rx + 0.05) & (ay < _HW)])        # 뒤 통로의 움직이는 물체까지 거리
    near = _kth_smallest(np.hypot(dyn_r[:, 0], dyn_r[:, 1]))             # 가장 가까운 움직이는 물체
    bx, by = raw[:, 0], np.abs(raw[:, 1])
    rear_free = np.count_nonzero((bx < rx) & (bx > rx - 0.9) & (by < 0.5)) < 3
    front_free = np.count_nonzero((bx > fx) & (bx < fx + _RESUME_GAP) & (by < 0.5)) < 3

    # 지도에 없지만 오래 움직이지 않는 점(지도 오차)은 정적 장애물로 등록해 영구 정지를 막음
    in_f2 = in_f & (ax - fx < _RESUME_GAP)
    if np.count_nonzero(in_f2) >= 3:
        c = dyn_w[in_f2].mean(axis=0)
        a = _NAV["anchor"]
        if a is None or math.hypot(c[0] - a[0], c[1] - a[1]) > 0.15:
            _NAV["anchor"], _NAV["anchor_t"] = c, t
        elif t - _NAV["anchor_t"] > 8.0:
            cells = np.unique(np.round(dyn_w[in_f2] / 0.1) * 0.1, axis=0)
            _NAV["learned"] = np.concatenate([_NAV["learned"], cells])
            _NAV["anchor"] = None
    else:
        _NAV["anchor"] = None

    # 1) 물체가 앞 1.0 m 안: 뒤가 비어 있으면 후진, 아니면 정지 (진단 외 확장: 정지 중 피격 방지)
    retreat = gap_f < _RETREAT_GAP or (_NAV["retreat"] and gap_f < _RETREAT_GAP + 0.3)
    _NAV["retreat"] = retreat
    if retreat:
        _NAV["blocked"], _NAV["clear_since"] = True, None
        if rear_free and gap_r > 0.8:
            return -_clamp(0.4 + (_RETREAT_GAP - gap_f), 0.4, 1.0), 0.0, False
        return 0.0, 0.0, False

    # 2) 뒤에서 물체가 다가오고 앞 2.5 m 가 완전히 비어 있을 때만 앞으로 비켜남
    if gap_r < 0.8 and gap_f >= _RESUME_GAP and front_free:
        return 0.5, 0.0, False

    # 3) 앞 2.0 m 안에 물체: 정지(v=0, w=0), 2.5 m 밖으로 0.3 s 벗어나면 재개
    if gap_f < _RESUME_GAP:
        _NAV["clear_since"] = None
    elif _NAV["clear_since"] is None:
        _NAV["clear_since"] = t
    if gap_f < _STOP_GAP:
        _NAV["blocked"] = True
    elif _NAV["blocked"] and _NAV["clear_since"] is not None and t - _NAV["clear_since"] >= 0.3:
        _NAV["blocked"] = False
    if _NAV["blocked"]:
        return 0.0, 0.0, False

    # 4) 정상 주행: 앞 물체가 가까우면 감속, 제자리 회전 중 물체가 몸 가까이 있으면 회전 보류
    if v > 0.0 and gap_f < 3.5:
        v = min(v, 0.5 + 0.4 * (gap_f - _STOP_GAP))
    if v == 0.0 and near < 0.95:
        w = 0.0
    return v, w, False




_STATIC_MARGIN = 0.30   # 정적 지도와의 여유 [m]


_SENSE_RANGE = 4.0      # 움직이는 물체로 판정할 점의 최대 거리 [m]


_HW = 0.60              # 앞/뒤 감시 통로 반폭 [m]


_STOP_GAP = 2.0         # 앞 통로에 물체가 이 거리 안이면 정지


_RESUME_GAP = 2.5       # 이 거리 밖으로 벗어나야 주행 재개


_RETREAT_GAP = 1.0      # 이 거리 안으로 다가오면 후진


def _build_static_rects():
    """MISSION 바닥 박스 + FIXED_OBSTACLES 를 여유만큼 넓힌 사각형 배열 (M, 4) [xmin, xmax, ymin, ymax]."""
    rects = [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"],
              b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"]) for b in MISSION["boxes"]]
    rects += [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES]
    r = np.array(rects, dtype=np.float64)
    r += np.array([-_STATIC_MARGIN, _STATIC_MARGIN, -_STATIC_MARGIN, _STATIC_MARGIN])
    return r


_STATIC_RECTS = _build_static_rects()


_NAV = {"scan": {}, "dyn": {}, "got": False, "blocked": False, "clear_since": None,
        "retreat": False, "anchor": None, "anchor_t": 0.0, "learned": np.zeros((0, 2))}


def _dynamic_world(pts, pose):
    """로봇 기준 라이다 점에서 정적 지도(벽, 고정물, 바닥 박스)에 해당하는 점을 빼고, 남은 점을 월드 좌표로 돌려줍니다."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = p[np.hypot(p[:, 0], p[:, 1]) < _SENSE_RANGE]
    if p.shape[0] == 0:
        return np.zeros((0, 2))
    w = to_world(p, pose)
    x, y = w[:, 0], w[:, 1]
    m = _STATIC_MARGIN
    static = (x <= WALLS[0] + m) | (x >= WALLS[1] - m) | (y <= WALLS[2] + m) | (y >= WALLS[3] - m)
    R = _STATIC_RECTS
    static |= ((x[:, None] >= R[:, 0]) & (x[:, None] <= R[:, 1]) &
               (y[:, None] >= R[:, 2]) & (y[:, None] <= R[:, 3])).any(axis=1)
    L = _NAV["learned"]
    if L.shape[0]:
        d2 = (x[:, None] - L[:, 0]) ** 2 + (y[:, None] - L[:, 1]) ** 2
        static |= (d2 < 0.25 ** 2).any(axis=1)
    return w[~static]


def _update_sensors(pose):
    """앞/뒤 2D 라이다를 읽어 새 스캔이 오면 그때의 자세로 정적 점을 거릅니다.
    (움직이는 점 월드, 움직이는 점 로봇 기준, 전체 점 로봇 기준) 을 돌려줍니다."""
    for which in ("front", "back"):
        pts = lidar2d_points(which)
        if pts.shape[0] and _NAV["scan"].get(which) is not pts:
            _NAV["scan"][which] = pts
            _NAV["dyn"][which] = _dynamic_world(pts, pose)
            _NAV["got"] = True
    dl = [d for d in _NAV["dyn"].values() if d.shape[0]]
    dyn_w = np.concatenate(dl) if dl else np.zeros((0, 2))
    rl = [r for r in _NAV["scan"].values() if r.shape[0]]
    raw = np.concatenate(rl) if rl else np.zeros((0, 2))
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = dyn_w[:, 0] - x, dyn_w[:, 1] - y
    dyn_r = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
    return dyn_w, dyn_r, raw


def _kth_smallest(a, k=3):
    """k 번째로 작은 값(낱개 잡음 점에 덜 민감). 점이 k 개보다 적으면 inf."""
    if a.size < k:
        return float("inf")
    return float(np.partition(a, k - 1)[k - 1])


def _nominal(pose):
    """기존 경유점 추종 + 목표 정렬 로직 (명령 v, w, done)."""
    x, y, yaw = pose
    n = len(_WAYPOINTS)
    if controller.idx >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.5:
        controller.idx = n - 1          # 회피 후진 등으로 목표에서 밀려났으면 다시 접근
    if controller.idx >= n:
        e = _wrap_angle(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False

    tx, ty = _WAYPOINTS[controller.idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)
    if controller.idx < n - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if abs(e) > 0.35:
        v = 0.0
    else:
        v = min(1.5, 2.5 * dist + 0.5)
    w = _clamp(7.5 * e, -2.0, 2.0)

    if controller.idx == n - 1 and dist < 0.25:
        controller.idx += 1
    return v, w, False


# ========================= [END EDIT REGION] ===============================
