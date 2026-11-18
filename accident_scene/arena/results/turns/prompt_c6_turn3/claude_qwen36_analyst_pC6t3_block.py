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
    """경유점 추종 + 2D 라이다 기반 반응형 회피.
    - 앞 통로(반폭 0.8 m) 물체가 1.5 m 안이면 정지(blocked).
    - 정지 상태(blocked 또는 명령 v=0)에서 가장 가까운 움직이는 점이 1.0 m 안이면 무조건 회피:
      물체가 앞쪽이면 후진(기본 0.3 m/s, 접근이 더 빠르면 접근 속도+0.1, 최대 1.0), 뒤쪽이면 전진,
      둘 다 막혔으면 가장 가까운 몸체 점이 물체에서 멀어지는 쪽으로 w=±0.8 회전.
    - 정지 해제: 앞 통로 3.0 m 이상 + 모든 물체 1.5 m 이상 + 접근 속도 양수 아님 이 0.5 s 이어질 때."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
    dyn_w, dyn_r, raw = _update_sensors(pose)
    if not _NAV["got"]:
        return 0.0, 0.0, False          # 라이다 데이터가 오기 전에는 움직이지 않음

    v, w, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True           # 목표 도착·정렬 완료: 즉시 미션 종료

    fx, rx = _BODY[1], _BODY[0]
    d_all = np.hypot(dyn_r[:, 0], dyn_r[:, 1])
    ok = d_all >= _NOISE_R                                                # 0.2 m 미만 잡음 제외
    ax, sy_all, d = dyn_r[ok, 0], dyn_r[ok, 1], d_all[ok]
    ay = np.abs(sy_all)

    # 앞 통로(앞 측면 포함, 반폭 0.8 m): 로봇 중심 기준 최단 거리
    in_f = (ax > _FRONT_X) & (ay < _HW)
    d_f = _min_or_inf(d[in_f])
    # 뒤 통로: 뒤 범퍼에서 물체까지 거리
    in_r = (ax < rx + 0.05) & (ay < _HW)
    gap_r = _min_or_inf(rx - ax[in_r])
    # 전 방향 최단 거리와 그 점(로봇 기준, y 부호 유지)
    near = _min_or_inf(d)
    if d.size:
        k = int(np.argmin(d))
        ox, oy = float(ax[k]), float(sy_all[k])
    else:
        ox, oy = 0.0, 0.0

    bx, by = raw[:, 0], np.abs(raw[:, 1])
    rear_free = np.count_nonzero((bx < rx) & (bx > rx - 0.9) & (by < 0.45)) < 3 and gap_r > 0.5
    front_free = np.count_nonzero((bx > fx) & (bx < fx + 0.9) & (by < 0.45)) < 3

    # 옆 물체 관측과 접근 속도 기록(정지 중에도 이력이 끊기지 않도록 항상 갱신)
    side = (ax > rx - 0.2) & (ax < fx + 0.7) & (ay > _BODY[3] - 0.05) & (ay < _BODY[3] + _FLANK_W)
    if np.any(side):
        sy, sx = ay[side], ax[side]
        clr = max(0.0, float(np.min(sy)) - _BODY[3])
        grp = sy < float(np.min(sy)) + 0.6
        xs_min, xs_max = float(np.min(sx[grp])), float(np.max(sx[grp]))
    else:
        clr, xs_min, xs_max = float("inf"), 0.0, 0.0
    h = _NAV["hist"]
    nh = _NAV["nhist"]
    if _NAV["fresh"]:
        if math.isinf(clr):
            h.clear()
        else:
            h.append((t, clr))
            while h and t - h[0][0] > 0.5:
                h.pop(0)
        if math.isinf(near):
            nh.clear()
        else:
            nh.append((t, near))
            while nh and t - nh[0][0] > 0.5:
                nh.pop(0)

    # 가장 가까운 물체의 접근 속도(closing speed, 양수면 다가옴)
    n_rate = 0.0
    if len(nh) >= 2 and nh[-1][0] - nh[0][0] >= 0.15:
        n_rate = (nh[0][1] - nh[-1][1]) / (nh[-1][0] - nh[0][0])
    approaching = n_rate > _APPROACH_RATE

    # 정지 판정: 앞 통로 최단 거리 < 1.5 m
    if d_f < _STOP_GAP:
        _NAV["blocked"] = True

    # 재개 판단: 앞 통로 3.0 m 이상, 모든 물체 1.5 m 이상, 접근 중 아님 이 함께 유지된 시각부터 잼
    if d_f < _RESUME_GAP or near < _CLEAR_R or approaching:
        _NAV["clear_since"] = None
    elif _NAV["clear_since"] is None:
        _NAV["clear_since"] = t
    if _NAV["blocked"] and _NAV["clear_since"] is not None and t - _NAV["clear_since"] >= _RESUME_HOLD:
        _NAV["blocked"] = False
    hold = _NAV["blocked"]              # True 이면 정상 전진 금지

    # A) 정지 상태(blocked 또는 명령 v=0)에서 물체가 1.0 m 안: 서 있지 않고 무조건 회피
    if (hold or v == 0.0) and near < _EVADE_R:
        _NAV["esc"] = 0
        spd = _clamp(n_rate + 0.1, _EVADE_V, _ESC_V)
        if ox >= 0.5 * (fx + rx):                      # 물체가 몸 중앙보다 앞쪽: 후진
            if rear_free:
                return -spd, 0.0, False
        else:                                          # 물체가 뒤쪽: 앞 통로가 비었을 때만 전진
            if front_free and d_f >= _STOP_GAP:
                return spd, 0.0, False
        # 앞뒤 모두 막힘: 가장 가까운 몸체 점이 물체에서 멀어지는 방향으로 제자리 회전
        px = _clamp(ox, rx, fx)
        py = _clamp(oy, _BODY[2], _BODY[3])
        torque = px * (oy - py) - py * (ox - px)
        if abs(torque) < 1e-6:
            return 0.0, _EVADE_W, False
        return 0.0, (-_EVADE_W if torque > 0.0 else _EVADE_W), False

    # B) 앞 통로 1.5 m 안 물체: 정지 (v=0, w=0)
    if d_f < _STOP_GAP:
        _NAV["esc"] = 0
        return 0.0, 0.0, False

    # 0) 몸 옆으로 다가오는 물체: 앞/뒤 중 비어 있는 쪽으로 비켜남. 정지 해제 전에는 후진만 허용
    rate = 0.0
    if len(h) >= 2 and h[-1][0] - h[0][0] >= 0.15:
        rate = (h[0][1] - h[-1][1]) / (h[-1][0] - h[0][0])          # 옆면 여유가 줄어드는 속도 [m/s]
    threat = clr < _FLANK_HARD or (rate > 0.15 and clr / rate < _FLANK_TTC)
    if threat:
        if hold:
            _NAV["esc"] = -1
            if rear_free:
                return -_ESC_V, 0.0, False
            return 0.0, 0.0, False
        if _NAV["esc"] == 0:
            back = fx - (xs_min - 0.3) + 0.15                          # 뒤로 비켜야 할 거리
            fwd = (xs_max + 0.3) - rx + 0.15                           # 앞으로 비켜야 할 거리
            _NAV["esc"] = -1 if back <= fwd else 1
        if _NAV["esc"] < 0 and not rear_free and front_free:
            _NAV["esc"] = 1
        elif _NAV["esc"] > 0 and not front_free and rear_free:
            _NAV["esc"] = -1
        if _NAV["esc"] < 0 and rear_free:
            return -_ESC_V, 0.0, False
        if _NAV["esc"] > 0 and front_free:
            return _ESC_V, 0.0, False
        return 0.0, 0.0, False
    _NAV["esc"] = 0

    # 정지 해제 전: 그 자리에서 대기 (1.0 m 안 접근은 위 A 에서 회피)
    if hold:
        return 0.0, 0.0, False

    # 2) 뒤에서 물체가 다가오고 앞 통로가 3.0 m 이상 비어 있을 때만 앞으로 비켜남
    if gap_r < 0.8 and d_f >= _RESUME_GAP and front_free:
        return 0.6, 0.0, False

    # 4) 정상 주행: 앞에 움직이는 물체가 보이면 감속
    if v > 0.0:
        if d_f < 3.5:
            v = min(v, 0.5 + 0.5 * (d_f - _STOP_GAP))
        if np.any((ax > 0.0) & (ax < 3.5) & (ay < 2.0)):
            v = min(v, 1.0)
    return v, w, False




_STATIC_MARGIN = 0.20   # 정적 지도와의 여유 [m] (진단: 0.3 -> 0.2 로 축소해 불필요한 정지 방지)


_SENSE_RANGE = 4.0      # 움직이는 물체로 판정할 점의 최대 거리 [m]


_HW = 0.80              # 앞 감시 통로 반폭 [m] (진단: 0.54 -> 0.8, 정면-측면 경계 |y|≈0.55 포함)


_STOP_GAP = 1.5         # 앞 통로 물체가 로봇 중심에서 이 거리 안이면 정지(v=0, w=0)


_RESUME_GAP = 3.0       # 앞 통로가 이 거리 밖인 상태가 _RESUME_HOLD 동안 이어져야 주행 재개 (진단: 3.0 m)


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


_NAV = {"scan": {}, "dyn": {}, "got": False, "fresh": False, "blocked": False, "clear_since": None,
        "hist": [], "esc": 0, "nhist": []}


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




_EVADE_R = 1.0          # 정지 상태에서 가장 가까운 움직이는 점이 이 거리 안이면 무조건 회피 [m]


_CLEAR_R = 1.5          # 정지 해제 전 모든 움직이는 점이 이 거리 밖이어야 함 [m]


_EVADE_V = 0.3          # 정지 중 회피 후진(또는 전진) 기본 속도 [m/s]


_EVADE_W = 0.8          # 앞뒤 모두 막혔을 때 측면 회피 회전 속도 [rad/s]


_APPROACH_RATE = 0.05   # 가장 가까운 물체와의 거리가 이 속도보다 빨리 줄면 '접근 중'으로 보고 재개 지연 [m/s]


# ========================= [END EDIT REGION] ===============================
