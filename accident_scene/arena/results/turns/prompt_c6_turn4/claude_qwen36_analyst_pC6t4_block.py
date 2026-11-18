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
    """경유점 추종 + 2D 라이다 기반 단순 정지/크리프 회피 (C-2 시도 4, 진단 반영).
    - 정적 지도 사각형 '내부' 점만 거르고(_STATIC_MARGIN=0.0) 남은 점을 움직이는 물체로 본다.
    - 앞 통로(x > -0.2, |y| < 0.8) 최단 거리 d_f < 1.2 m 이면 정지(blocked), 그 시각을 blocked_t 로 기록.
    - d_f > 1.5 m 이면 바로 정지 해제(접근 속도 계산·재개 타이머 없음).
    - 정지 중(blocked 또는 명령 v=0) 가장 가까운 움직이는 점이 0.8 m 안이면 후진 v=-0.2
      (물체가 몸 뒤쪽이면 앞이 비었을 때 전진 0.2, 둘 다 막히면 정지).
    - 정지가 2.0 s 넘게 이어지면 v=0.1 크리프 전진(앞 범퍼 여유 d_f > 0.7 m 일 때만)."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
    dyn_w, dyn_r, raw = _update_sensors(pose)
    if not _NAV["got"]:
        return 0.0, 0.0, False          # 라이다 데이터가 오기 전에는 움직이지 않음

    v, w, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True           # 목표 도착·정렬 완료: 미션 종료

    fx, rx = _BODY[1], _BODY[0]
    d_all = np.hypot(dyn_r[:, 0], dyn_r[:, 1])
    ok = d_all >= _NOISE_R                                    # 0.2 m 미만 잡음 제외
    ax, ay, d = dyn_r[ok, 0], np.abs(dyn_r[ok, 1]), d_all[ok]

    # 앞 통로 최단 거리, 전 방향 최단 거리와 그 점의 앞뒤 위치
    d_f = _min_or_inf(d[(ax > _FRONT_X) & (ay < _HW)])
    near = _min_or_inf(d)
    ox = float(ax[int(np.argmin(d))]) if d.size else 0.0

    # 뒤/앞 빈 공간 확인(전체 점 기준 + 뒤쪽 움직이는 점)
    gap_r = _min_or_inf(rx - ax[(ax < rx + 0.05) & (ay < _HW)])
    bx, by = raw[:, 0], np.abs(raw[:, 1])
    rear_free = np.count_nonzero((bx < rx) & (bx > rx - 0.9) & (by < 0.45)) < 3 and gap_r > 0.5
    front_free = np.count_nonzero((bx > fx) & (bx < fx + 0.9) & (by < 0.45)) < 3

    # 정지 / 재개: 임계값 하나씩 (정지 1.2 m, 재개 1.5 m)
    if not _NAV["blocked"] and d_f < _STOP_GAP:
        _NAV["blocked"] = True
        _NAV["blocked_t"] = t
    elif _NAV["blocked"] and d_f > _RESUME_GAP:
        _NAV["blocked"] = False
        _NAV["blocked_t"] = None
    hold = _NAV["blocked"]

    # 정지 중 물체가 0.8 m 안: 후진 (물체가 뒤쪽이면 전진), 둘 다 막히면 정지
    if (hold or v == 0.0) and near < _EVADE_R:
        if ox >= 0.5 * (fx + rx):
            if rear_free:
                return -_EVADE_V, 0.0, False
        elif front_free and d_f > _CREEP_MIN:
            return _EVADE_V, 0.0, False
        return 0.0, 0.0, False

    # 정지 상태: 2.0 s 넘게 이어지면 크리프(정적 혼동·가림으로 인한 무한 정지 방지)
    if hold:
        bt = _NAV["blocked_t"] if _NAV["blocked_t"] is not None else t
        if t - bt > _CREEP_T:
            cv = _CREEP_V if (v > 0.0 and d_f > _CREEP_MIN) else 0.0
            cw = _clamp(w, -_EVADE_W, _EVADE_W) if near > _SWEEP_R else 0.0
            return cv, cw, False
        return 0.0, 0.0, False

    # 정상 주행: 앞에 움직이는 물체가 보이면 감속
    if v > 0.0:
        if d_f < 3.5:
            v = min(v, 0.5 + 0.5 * (d_f - _STOP_GAP))
        if np.any((ax > 0.0) & (ax < 3.5) & (ay < 2.0)):
            v = min(v, 1.0)
    return v, w, False




_STATIC_MARGIN = 0.0    # 정적 지도와의 여유 [m] (진단: 0.2 -> 0.0, 정적 사각형 내부 점만 제거해 인접 물체 감지)


_SENSE_RANGE = 4.0      # 움직이는 물체로 판정할 점의 최대 거리 [m]


_HW = 0.80              # 앞 감시 통로 반폭 [m] (진단: 0.54 -> 0.8, 정면-측면 경계 |y|≈0.55 포함)


_STOP_GAP = 1.2         # 앞 통로 물체가 로봇 중심에서 이 거리 안이면 정지 (진단: 1.2 m)


_RESUME_GAP = 1.5       # 앞 통로 최단 거리가 이보다 크면 바로 주행 재개 (진단: 1.5 m)


_RETREAT_GAP = 1.1      # 앞 통로 물체가 로봇 중심에서 이 거리 안이면 (뒤가 비었을 때) 후진


def _build_static_rects():
    """MISSION 바닥 박스 + FIXED_OBSTACLES 를 여유만큼 넓힌 사각형 배열 (M, 4) [xmin, xmax, ymin, ymax]."""
    rects = [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"],
              b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"]) for b in MISSION["boxes"]]
    rects += [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES]
    r = np.array(rects, dtype=np.float64)
    r += np.array([-_STATIC_MARGIN, _STATIC_MARGIN, -_STATIC_MARGIN, _STATIC_MARGIN])
    return r


_STATIC_RECTS = _build_static_rects()


_NAV = {"scan": {}, "dyn": {}, "got": False, "fresh": False, "blocked": False, "blocked_t": None,
        "clear_since": None, "hist": [], "esc": 0, "nhist": []}


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
    return w[~static]


def _update_sensors(pose):
    """앞/뒤 2D 라이다를 읽어 새 스캔이 오면 그때의 자세로 정적 점을 거릅니다.
    (움직이는 점 월드, 움직이는 점 로봇 기준, 전체 점 로봇 기준) 을 돌려줍니다. 새 스캔 여부는 _NAV["fresh"]."""
    _NAV["fresh"] = False
    for which in ("front", "back"):
        pts = lidar2d_points(which)
        if pts.shape[0] and _NAV["scan"].get(which) is not pts:
            _NAV["scan"][which] = pts
            _NAV["dyn"][which] = _dynamic_world(pts, pose)
            _NAV["got"] = True
            _NAV["fresh"] = True
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




_FRONT_X = -0.2         # 앞 감시 통로의 뒤쪽 경계(로봇 기준 x): 앞 측면 접근도 포함


_NOISE_R = 0.2          # 로봇 중심에서 이 거리 미만의 점은 잡음으로 제외


_RESUME_HOLD = 0.5      # 재개 전 통로가 비어 있어야 하는 시간 [s]


_FLANK_W = 1.2          # 몸 옆 감시 폭 [m] (몸 옆면에서 바깥으로)


_FLANK_HARD = 0.12      # 옆면 여유가 이보다 작으면 무조건 비켜남 [m]


_FLANK_TTC = 1.6        # 옆에서 다가오는 물체가 이 시간 안에 닿을 것 같으면 비켜남 [s]


_ESC_V = 1.0            # 비켜나기 속도 [m/s]


def _min_or_inf(a):
    """점이 하나라도 있으면 그 최솟값, 없으면 inf (1개 점에도 반응)."""
    return float(np.min(a)) if a.size else float("inf")




_EVADE_R = 0.8          # 정지 중 가장 가까운 움직이는 점이 이 거리 안이면 후진 [m] (진단: 0.8 m)


_CLEAR_R = 1.5          # 정지 해제 전 모든 움직이는 점이 이 거리 밖이어야 함 [m]


_EVADE_V = 0.2          # 정지 중 회피 후진(또는 전진) 속도 [m/s] (진단: v=-0.2)


_EVADE_W = 0.8          # 앞뒤 모두 막혔을 때 측면 회피 회전 속도 [rad/s]


_APPROACH_RATE = 0.05   # 가장 가까운 물체와의 거리가 이 속도보다 빨리 줄면 '접근 중'으로 보고 재개 지연 [m/s]




_CREEP_T = 2.0          # 정지가 이 시간보다 길게 이어지면 크리프 전진 [s] (진단: 2.0 s)


_CREEP_V = 0.1          # 크리프 전진 속도 [m/s] (진단: v=0.1)


_CREEP_MIN = 0.7        # 크리프·전진 허용 최소 앞 통로 거리 [m] (앞 범퍼 여유 약 0.45 m)


_SWEEP_R = 0.9          # 정지 중 제자리 회전을 허용할 최소 물체 거리 [m] (몸체 회전 반경 약 0.78 m + 여유)


# ========================= [END EDIT REGION] ===============================
