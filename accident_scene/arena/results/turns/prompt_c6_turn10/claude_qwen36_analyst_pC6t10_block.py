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
    """경유점 추종 + bearing 반응형 계단식 거리 게이트 (C-2 시도 10, 진단·검토 반영).
    - 움직이는 점의 bearing b = atan2(|ay|, ax) 로 전방(b < 1.05 rad)과 측면(1.05 <= b < 2.09 rad)을 나눈다.
    - 정지: 전방 최단 거리 < 1.5 m 이거나 측면 최단 거리 < 2.0 m.
    - 재개: 감시 영역(b < 2.09) 의 최단 거리가 2.2 m 이상인 상태가 0.5 s 지속될 때만.
    - 주행 중 감속: 전방은 2.0 -> 1.5 m, 측면은 2.5 -> 2.0 m 에서 v 가 0 이 되도록 선형 감속.
    - 정지 중에는 어떤 움직임도 하지 않는다(v = 0, w = 0)."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
    dyn_w, dyn_r, raw = _update_sensors(pose)
    if not _NAV["got"]:
        return 0.0, 0.0, False          # 라이다 데이터가 오기 전에는 움직이지 않음

    v, w, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True           # 목표 도착·정렬 완료: 미션 종료

    d_all = np.hypot(dyn_r[:, 0], dyn_r[:, 1])
    ok = d_all >= _NOISE_R                                    # 0.2 m 미만 잡음 제외
    ax, ays, d = dyn_r[ok, 0], dyn_r[ok, 1], d_all[ok]
    ay = np.abs(ays)
    b = np.arctan2(ay, ax)                                    # |bearing| [0, pi]

    sec_front = b < _FRONT_B                                  # 전방 ±60°
    sec_side = (b >= _FRONT_B) & (b < _SIDE_B)                # 측면 ±60°~120°
    watch = b < _SIDE_B                                       # 감시 영역 (재개 판단)
    d_fs = _min_or_inf(d[sec_front])
    d_sd = _min_or_inf(d[sec_side])
    d_watch = _min_or_inf(d[watch])

    stop_now = (d_fs < _HARD_STOP) or (d_sd < _SIDE_HARD)

    # ---- 정지/재개 상태 기계 ----
    if not _NAV["blocked"]:
        if stop_now:
            _NAV["blocked"] = True
            _NAV["blocked_t"] = t
            _NAV["clear_since"] = None
    else:
        resume = False
        if d_watch >= _RESUME_D:
            if _NAV["clear_since"] is None:
                _NAV["clear_since"] = t
            if t - _NAV["clear_since"] >= _RESUME_HOLD:
                resume = True
        else:
            _NAV["clear_since"] = None
        if resume:
            _NAV["blocked"] = False
            _NAV["last_blocked_t"] = _NAV["blocked_t"]
            _NAV["resume_t"] = t
            _NAV["blocked_t"] = None
            _NAV["clear_since"] = None

    if _NAV["blocked"]:
        return 0.0, 0.0, False          # 정지 중에는 무조건 정지 유지

    # ---- 주행: 섹터별 최단 거리에 따른 선형 감속 ----
    if v > 0.0:
        k_front = _clamp((d_fs - _HARD_STOP) / (_SLOW_GAP - _HARD_STOP), 0.0, 1.0)
        k_side = _clamp((d_sd - _SIDE_HARD) / (_SIDE_SLOW - _SIDE_HARD), 0.0, 1.0)
        v = v * min(k_front, k_side)
        if np.any((ax > 0.0) & (ax < 3.5) & (ay < 2.0)):
            v = min(v, 1.0)
    return v, w, False




_STATIC_MARGIN = 0.15   # 정적 지도와의 여유 [m] (진단: 0.0 -> 0.15, 라이다 잡음·자세 오차로 경계 밖에 찍힌 정적 점 오탐 방지)


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
        "clear_since": None, "hist": [], "esc": 0, "nhist": [], "bhist": [],
        "resume_t": None, "last_blocked_t": None}


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




_FRONT_X = -0.3         # 전방 감시 영역의 뒤쪽 경계(로봇 기준 x): 이보다 앞의 점은 모두 감시 (진단: 전방 180°, x > -0.3)


_NOISE_R = 0.2          # 로봇 중심에서 이 거리 미만의 점은 잡음으로 제외


_RESUME_HOLD = 0.5      # 재개 전 감시 영역이 _RESUME_D 밖으로 비어 있어야 하는 시간 [s] (진단: 0.3 -> 0.5)


_FLANK_W = 1.2          # 몸 옆 감시 폭 [m] (몸 옆면에서 바깥으로)


_FLANK_HARD = 0.12      # 옆면 여유가 이보다 작으면 무조건 비켜남 [m]


_FLANK_TTC = 1.6        # 옆에서 다가오는 물체가 이 시간 안에 닿을 것 같으면 비켜남 [s]


_ESC_V = 1.0            # 비켜나기 속도 [m/s]


def _min_or_inf(a):
    """점이 하나라도 있으면 그 최솟값, 없으면 inf (1개 점에도 반응)."""
    return float(np.min(a)) if a.size else float("inf")




_EVADE_R = 0.9          # 정지 중 가장 가까운 움직이는 점이 이 거리 안이면 직선 회피 [m] (진단: 0.9 m)


_CLEAR_R = 1.5          # 정지 해제 전 모든 움직이는 점이 이 거리 밖이어야 함 [m]


_EVADE_V = 0.2          # 정지 중 회피 후진(또는 전진) 속도 [m/s] (진단: v=-0.2)


_EVADE_W = 0.8          # 앞뒤 모두 막혔을 때 측면 회피 회전 속도 [rad/s]


_APPROACH_RATE = 0.05   # 가장 가까운 물체와의 거리가 이 속도보다 빨리 줄면 '접근 중'으로 보고 재개 지연 [m/s]




_CREEP_T = 2.0          # 정지가 이 시간보다 길게 이어지면 크리프 전진 [s] (진단: 2.0 s)


_CREEP_V = 0.15         # 강제 탈출 크리프 전진 속도 [m/s] (진단: v=0.15)


_CREEP_MIN = 0.7        # 크리프·전진 허용 최소 앞 통로 거리 [m] (앞 범퍼 여유 약 0.45 m)


_SWEEP_R = 0.9          # 정지 중 제자리 회전을 허용할 최소 물체 거리 [m] (몸체 회전 반경 약 0.78 m + 여유)




_SLOW_GAP = 2.0         # 전방 최단 거리가 이보다 작으면 _HARD_STOP 에서 0 이 되도록 선형 감속 [m] (진단: 2.0)


_HARD_STOP = 1.5        # 전방 최단 거리가 이보다 작으면 blocked 정지 v = 0 [m] (진단: 1.5)


_SIDE_Y = 0.5           # |ay| 가 이보다 크면 측면 물체로 본다 [m] (진단)


_SIDE_STOP = 1.0        # 측면 물체가 이 거리 안이면 즉시 v = 0 [m] (진단)




_RESUME_D = 2.2         # 정지 해제 조건: 감시 영역(|b| < 2.09 rad) 최단 거리가 이보다 커야 함 [m] (진단: 1.7 -> 2.2)


_ESCAPE_T = 2.0         # 정지 누적 시간이 이보다 길면 강제 탈출 [s] (진단: 2.0 s)


_ESCAPE_W = 0.6         # 강제 탈출 제자리 회전 속도 [rad/s] (진단: w=0.6)


_CREEP_LANE = 0.45      # 강제 탈출 시 크리프 가능 여부를 볼 몸 앞 통로 반폭 [m]


_CREEP_GAP = 0.35       # 몸 앞면에서 이 거리 안에 점이 있으면 크리프 대신 회전으로 탈출 [m]




_CREEP_D = 1.2          # 강제 탈출 시 d_front 가 이보다 크면 크리프, 아니면 제자리 회전 [m] (진단: 1.2 m)


_BEARING_WIN = 0.5      # bearing 변화를 볼 최근 시간 창 [s] (진단: 0.5 s)


_BEARING_PASS = math.radians(30.0)   # 이 이상 bearing 이 바뀌면 측면 통과로 판단 (진단: 30°)


_REBLOCK_T = 1.0        # 재개 후 이 시간 안에 다시 막히면 정지 시간을 누적 [s]


def _update_bearing(t, pose, wx, wy, d):
    """가장 가까운 전방 움직이는 점의 월드 기준 bearing(로봇 위치에서, 자기 yaw 와 무관)을 기록하고,
    최근 _BEARING_WIN 초 동안 현재 값과의 차이가 _BEARING_PASS 이상이면 True(측면 통과 중)."""
    h = _NAV["bhist"]
    if d.size:
        i = int(np.argmin(d))
        h.append((t, math.atan2(float(wy[i]) - pose[1], float(wx[i]) - pose[0])))
    while h and t - h[0][0] > _BEARING_WIN:
        h.pop(0)
    if len(h) < 2 or h[-1][0] - h[0][0] < 0.6 * _BEARING_WIN:
        return False
    b_now = h[-1][1]
    return max(abs(_wrap_angle(b - b_now)) for _, b in h) >= _BEARING_PASS




_FRONT_B = 1.05         # 전방 섹터 |bearing| 상한 [rad] (진단: ±60°)


_SIDE_B = 2.09          # 측면 섹터 |bearing| 상한 [rad] (진단: ±120°), 이 안이 감시 영역


_SIDE_HARD = 2.0        # 측면 섹터 최단 거리가 이보다 작으면 정지 [m] (진단: 2.0)


_SIDE_SLOW = 2.5        # 측면 섹터 최단 거리가 이보다 작으면 _SIDE_HARD 에서 0 이 되도록 선형 감속 [m]


# ========================= [END EDIT REGION] ===============================
