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
    """경유점 추종 + 하드 안전 정지 + 연속 거리 기반 속도 제한기 + 반발 조향 (C-2 시도 28, 진단 반영).
    - 상태 래치·타이머 없음: 매 스텝 _watch_min 의 d_min, b_min 으로 v, w 를 바로 계산합니다.
    - d_min < 0.8 m: v = 0, w = 0 무조건 정지 (제자리 회전 충돌 차단).
    - 0.8 m <= d_min < 1.5 m: v_cmd = v_nom * max(0, min(1, (d_min - 0.6) / 0.9)),
      w_cmd = clamp(w_nom - 1.5 * sign(b_min) / max(d_min, 0.5), -1.0, 1.0).
    - d_min >= 1.5 m (또는 감시 영역에 움직이는 점 없음): 경유점 추종 (v_nom, w_nom) 그대로.
    - _front_blocked 분기는 제거하고 d_min 기준으로 통합했습니다."""
    if not hasattr(controller, "idx"):
        controller.idx = 0

    dyn_w, dyn_r, raw = _update_sensors(pose)
    if not _NAV["got"]:
        return 0.0, 0.0, False          # 라이다 데이터가 오기 전에는 움직이지 않음

    v_nom, w_nom, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True           # 목표 도착·정렬 완료: 미션 종료

    d_min, b_min, _x_min = _watch_min(dyn_r)

    # ---- 가까운 움직이는 물체 없음: 경유점 추종 그대로 ----
    if not math.isfinite(d_min) or d_min >= _GOV_FULL_D:
        return v_nom, w_nom, False

    # ---- 하드 안전 레이어: 0.8 m 안이면 전진·회전 모두 정지 ----
    if d_min < _SAFE_HALT_D:
        return 0.0, 0.0, False

    # ---- 0.8 ~ 1.5 m: 기존 연속 속도 제한 (0.6 m 에서 0, 1.5 m 에서 1) ----
    scale = max(0.0, min(1.0, (d_min - _GOV_D0) / _GOV_SPAN))
    v_cmd = max(0.0, v_nom * scale)

    # ---- 반발 조향: 물체가 왼쪽(+)이면 오른쪽(-)으로, |w| <= 1.0 rad/s ----
    side = 1.0 if b_min >= 0.0 else -1.0
    w_rep = -_GOV_REP_K * side / max(d_min, _GOV_REP_DFLOOR)
    w_cmd = _clamp(w_nom + w_rep, -_GOV_W_MAX, _GOV_W_MAX)
    return v_cmd, w_cmd, False




_STATIC_MARGIN = 0.05   # 정적 지도와의 여유 [m] (진단 시도 27: 0.15 -> 0.05, 정적 장애물 근처 움직이는 물체 점의 과다 필터링 방지)


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


_NAV = {"scan": {}, "dyn": {}, "got": False, "fresh": False, "vhist": []}


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


_EVADE_V = 0.15         # 능동 회피 후진 속도 [m/s] (진단 시도 20: d_min < 0.7 m 시 v=-0.15)


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




_CLOSE_STOP = 0.2       # 닫힘 속도가 이 이상이고 거리가 _CLOSE_D 미만이면 k = 0 [m/s] (진단: 0.2)


_CLOSE_SLOW = 0.1       # 닫힘 속도가 이 미만(멀어짐 포함)이면 서행 k = _SLOW_K [m/s] (진단: 0.1)


_CLOSE_D = 1.5          # 접근 정지 판정 거리 [m] (진단: 1.5)


_SLOW_K = 0.3           # 멈춰 있거나 멀어지는 물체 곁에서의 서행 가중치 (진단: k = 0.3)


_VEL_DT = 0.1           # 물체 평균 위치 기록 간격 [s] (진단: 0.1 s)


_VEL_WIN = 0.4          # 물체 속도 추정에 쓰는 최근 시간 창 [s]


_TRACK_R = 0.7          # 가장 가까운 점에서 이 거리 안의 움직이는 점을 같은 물체로 보고 평균 [m]


_TRACK_JUMP = 0.5       # 평균 위치가 한 기록 사이에 이보다 크게 튀면 다른 물체로 보고 기록을 새로 시작 [m]


def _update_closing(t, pose, wx, wy, d):
    """감시 영역에서 가장 가까운 움직이는 물체의 평균 위치(월드)를 새 스캔마다 _VEL_DT 간격으로 _NAV["vhist"] 에 기록하고,
    최근 _VEL_WIN 초의 위치 변화로 물체의 월드 속도를 구해 로봇 쪽으로 다가오는 성분(닫힘 속도, + 가 접근)을 돌려줍니다.
    물체 자신의 속도만 쓰므로 로봇의 전진이 닫힘 속도에 섞이지 않습니다. 아직 추정할 수 없으면 None."""
    h = _NAV["vhist"]
    if d.size == 0:
        del h[:]
        return None
    if _NAV["fresh"] and (not h or t - h[-1][0] >= _VEL_DT - 0.01):
        i = int(np.argmin(d))
        near = np.hypot(wx - wx[i], wy - wy[i]) < _TRACK_R
        cx, cy = float(np.mean(wx[near])), float(np.mean(wy[near]))
        if h and math.hypot(cx - h[-1][1], cy - h[-1][2]) > _TRACK_JUMP:
            del h[:]
        h.append((t, cx, cy))
    while len(h) > 2 and t - h[0][0] > _VEL_WIN:
        h.pop(0)
    if len(h) < 2:
        return None
    dt = h[-1][0] - h[0][0]
    if dt < 0.15 or t - h[-1][0] > 0.3:
        return None
    vx = (h[-1][1] - h[0][1]) / dt
    vy = (h[-1][2] - h[0][2]) / dt
    rx, ry = h[-1][1] - pose[0], h[-1][2] - pose[1]
    r = math.hypot(rx, ry)
    if r < 1e-6:
        return None
    return -(vx * rx + vy * ry) / r




_BLOCK_HOLD = 0.5       # k = 0 이 이 시간보다 오래 이어지면 능동 회피 [s] (진단: t - blocked_t > 0.5)


_EVADE_D = 1.2          # 감시 영역 최단 거리가 이보다 작으면 회피 시작 [m] (진단: d_watch < 1.2)


_EVADE_CLEAR = 1.5      # 감시 영역 최단 거리가 이보다 크면 blocked_t 초기화, 정상 주행 복귀 [m] (진단: d_watch > 1.5)


_REAR_MIN = 1.0         # 뒤쪽 라이다 기준 후진 통로 거리가 이보다 작으면 후진 대신 회전 [m] (진단: 1.0 m)


_REAR_LANE = 0.45       # 후진 통로 반폭 [m] (몸 반폭 0.34 m + 여유)


_REAR_X = -0.489        # 뒤쪽 2D 라이다 장착 x [m] (로봇 기준)


def _rear_gap(raw):
    """뒤쪽 라이다 위치에서 몸 뒤 통로(|y| < _REAR_LANE)의 가장 가까운 점까지의 거리. 정적·움직이는 점을 모두 봅니다."""
    if raw.shape[0] == 0:
        return float("inf")
    m = (raw[:, 0] < _REAR_X) & (np.abs(raw[:, 1]) < _REAR_LANE)
    return _min_or_inf(_REAR_X - raw[m, 0])




_BUB_STOP = 1.5         # 감시 영역 최단 거리가 이보다 작으면 정지 v = 0 [m] (진단: 1.5 m)


_BUB_HOLD = 0.5         # 정지가 이 시간보다 오래 이어지고 아직 _BUB_STOP 안이면 제자리 회전 [s] (진단: 0.5 s)


_BUB_W = 0.8            # 정지 중 제자리 회전 속도 크기 [rad/s] (진단: w = ±0.8)


_BUB_CLEAR = 2.0        # 정지 해제 조건: 감시 영역 최단 거리가 이보다 커야 함 [m] (진단: 2.0 m)


_BUB_CLEAR_T = 1.0      # 정지 해제 전 _BUB_CLEAR 밖 상태가 이어져야 하는 시간 [s] (진단: 1.0 s)




_REP_D = 1.5            # 반발 조향·감속을 시작하는 감시 영역 최단 거리 [m] (진단: d < 1.5 m)


_REP_K = 1.5            # 반발 조향 이득 K [rad/s · m] (d = 0.6 m 에서 약 1.5 rad/s)


_REP_W_MAX = 2.0        # 반발 조향을 더한 뒤 회전 속도 상한 [rad/s] (경유점 추종 상한과 같음)


_REP_D0 = 0.5           # 감속 식 (d - 0.5)/1.0 의 0.5 [m] (진단)


_REP_DSPAN = 1.0        # 감속 식 (d - 0.5)/1.0 의 1.0 [m] (진단)


_REP_SCALE_MIN = 0.2    # 감속 비율 하한 max(0.2, ...) (진단)


_REP_VMIN = 0.0         # 반발 구간 최소 전진 속도 하한 제거 [m/s] (진단: 0.2 -> 0.0, controller 에서 더 이상 쓰지 않음)




_HALT_D = 0.5           # 감시 영역 최단 거리가 이보다 작으면 무조건 v = 0, w = 0 정지 [m] (진단 시도 24: 0.8 -> 0.5 m)




def _evade_cmd(raw, dyn_r, d_min, x_min, b_min):
    """이진 정지-회피의 회피 동작 (전진 속도는 항상 0 이하).
    - 감시 영역에 움직이는 점이 없으면(d_min 이 inf) x_min, b_min 은 관측값이 아니므로 v = 0, w = 0
      (재개 타이머가 돌고 있으므로 _CONT_RESUME_T 뒤 재개됨).
    - 가장 가까운 물체가 앞쪽(x > _FRONT_X)이고, 뒤 통로(_rear_gap)가 _REAR_MIN(1.0 m) 이상 비었으며,
      뒤쪽(x < _FRONT_X) _REAR_BACK_R 안에 움직이는 점이 없으면 v = -_EVADE_V(-0.15) 직선 후진.
    - 아니면 v = 0, |w| = _EVADE_TURN_W(0.6) 제자리 회전. 방향은 물체에 가까운 몸 부분이 물체에서 멀어지는 쪽:
      물체가 앞쪽이면 코를 반대로(w = -0.6 * sign(b)), 뒤쪽이면 꼬리를 반대로(w = +0.6 * sign(b))."""
    if not math.isfinite(d_min):
        return 0.0, 0.0, False
    side = 1.0 if b_min >= 0.0 else -1.0
    front_obj = x_min > _FRONT_X
    rear_dyn = False
    if dyn_r.shape[0]:
        m = (dyn_r[:, 0] < _FRONT_X) & (np.hypot(dyn_r[:, 0], dyn_r[:, 1]) < _REAR_BACK_R)
        rear_dyn = bool(np.any(m))
    if front_obj and not rear_dyn and _rear_gap(raw) >= _REAR_MIN:
        return -_EVADE_V, 0.0, False
    if front_obj:
        return 0.0, -_EVADE_TURN_W * side, False
    return 0.0, _EVADE_TURN_W * side, False


_EVADE_EXIT = 1.0       # 회피 해제 조건: 감시 영역 최단 거리가 이보다 크면 evade_flag=False, blocked_t 초기화 [m] (진단: d_min > 1.0)


_EVADE_DEC = 0.05       # 정지 중 최단 거리가 정지 시작 때보다 이만큼 줄면 '계속 줄어듦'으로 보고 즉시 회피 [m] (진단)




_CAUTION_D = 0.9        # 감시 영역 최단 거리가 이보다 작으면 caution_t = t 로 갱신 [m] (진단: d_min < 0.9 m)


_CAUTION_HOLD = 0.2     # caution_t 이후 이 시간 동안 v = 0, w = 0 강제 [s] (진단: 0.2 s)


_CAUTION_CLEAR = 1.2    # 감시 영역 최단 거리가 이 이상이면 caution_t = None 으로 초기화 [m] (진단: d_min >= 1.2 m)


_GOV_D0 = 0.6           # 속도 제한 식 (d_min - 0.6) / 1.0 의 0.6 [m] (진단)


_GOV_SPAN = 0.9         # 속도 제한 식 (d_min - 0.6) / 0.9 의 0.9 [m] (진단 시도 27: 0.6 m 에서 0, 1.5 m 에서 정상 속도)




_STOP_ESC_T = 0.5       # v 가 0 으로 강제된 뒤 이 시간이 지나면 강제 탈출 후보 [s] (진단: t - stop_t > 0.5)


_STOP_ESC_D = 1.0       # 강제 탈출 진입 조건: 감시 영역 최단 거리가 이보다 작아야 함 [m] (진단: d_min < 1.0 m)


_STOP_ESC_CLEAR = 1.2   # 감시 영역 최단 거리가 이 이상이면 stop_t 초기화, 탈출 해제 [m] (진단: d_min >= 1.2 m)




_CONT_STOP_D = 1.2      # 감시 영역 최단 거리가 이보다 작으면 v = 0, w = 0 정지 래치 [m] (진단: 1.0 -> 1.2 m)


_CONT_FREE_D = 1.8      # 재개 조건: 감시 영역 최단 거리가 이보다 커야 함 [m] (진단: 1.5 -> 1.8 m)


_CONT_SPAN = 0.5        # 감속 식 (d_min - 1.0) / 0.5 의 0.5 [m] (진단)


_CONT_RESUME_T = 0.5    # 정지 해제 조건: d_min > 1.8 m 가 이 시간 이상 지속 [s] (진단: 0.5 s)




_LAT_T = 1.0            # 정지 시작 후 이 시간이 지나면 측면 천이 후보 [s] (진단: 1.0 s)


_LAT_B = math.radians(30.0)   # |b_min| 이 이보다 크면 측면 천이 허용 [rad] (진단: 30°)


_LAT_V = 0.1            # 측면 천이 전진 속도 [m/s] (진단: v = 0.1)


_LAT_W = 0.3            # 측면 천이 회전 속도 크기 [rad/s] (진단: |w| = 0.3, 물체 반대쪽)


def _front_blocked(raw):
    """몸 앞면 바로 앞 통로(|y| < _CREEP_LANE, 앞면에서 _CREEP_GAP 이내)에 정적·움직이는 점이 하나라도 있으면 True."""
    if raw.shape[0] == 0:
        return False
    m = ((raw[:, 0] > _BODY[1]) & (raw[:, 0] < _BODY[1] + _CREEP_GAP) &
         (np.abs(raw[:, 1]) < _CREEP_LANE))
    return bool(np.any(m))




_EVADE_HOLD = 0.4       # 정지 래치 후 이 시간이 지나면 _evade_cmd [s] (진단: 0.5 -> 0.4 s)


_EVADE_TURN_W = 0.6     # 뒤가 막혔을 때 제자리 회전 속도 크기 [rad/s] (진단: w = ±0.6)




_REAR_WATCH = 0.5       # 뒤쪽(x <= _FRONT_X) 점은 몸체 사각형에서 이 거리 안이면 감시 영역에 포함 [m] (후면-측면 접근 포착)


_REAR_BACK_R = 1.5      # 뒤쪽에 이 거리(로봇 중심 기준) 안의 움직이는 점이 있으면 후진하지 않음 [m]


def _watch_min(dyn_r):
    """감시 영역의 움직이는 점 중 로봇 중심에서 가장 가까운 점의 (거리, bearing, 로봇 기준 x) 를 돌려줍니다.
    감시 영역: 전방 180°(x > _FRONT_X) 의 점 + 뒤쪽 점 중 몸체 사각형 _BODY 에서 _REAR_WATCH 안의 점.
    _NOISE_R 미만의 점은 잡음으로 뺍니다. 점이 없으면 (inf, 0.0, 0.0)."""
    if dyn_r.shape[0] == 0:
        return float("inf"), 0.0, 0.0
    ax, ay = dyn_r[:, 0], dyn_r[:, 1]
    d = np.hypot(ax, ay)
    dxb = np.maximum(np.maximum(_BODY[0] - ax, ax - _BODY[1]), 0.0)
    dyb = np.maximum(np.maximum(_BODY[2] - ay, ay - _BODY[3]), 0.0)
    clr = np.hypot(dxb, dyb)
    watch = (d >= _NOISE_R) & ((ax > _FRONT_X) | (clr < _REAR_WATCH))
    if not np.any(watch):
        return float("inf"), 0.0, 0.0
    dw, xw, yw = d[watch], ax[watch], ay[watch]
    i = int(np.argmin(dw))
    return float(dw[i]), float(math.atan2(yw[i], xw[i])), float(xw[i])




_BR_D = 1.0             # 분기 회피를 시작하는 감시 영역 최단 거리 [m] (진단: d_min < 1.0 m), 이상이면 경유점 추종


_BR_FRONT_B = 1.05      # |b_min| 이 이보다 작으면 정면, 크거나 같으면 측면 [rad] (진단: 1.05 rad)


_BR_BACK_V = 0.2        # 정면 물체 회피 후진 속도 [m/s] (진단: v = -0.2)


_BR_TURN_W = 0.8        # 회피 회전 속도 크기 [rad/s] (진단: w = ±0.8)


_BR_SIDE_V = 0.2        # 측면 물체 회피 시 회전 전진 속도 [m/s] (진단: v = 0.2)


_BR_W_MAX = 1.5         # 모든 회피 명령의 회전 속도 상한 [rad/s] (진단: 1.5 rad/s)




_BR_EXIT = 1.2          # 회피 해제 거리: 감시 영역 최단 거리가 이 이상이면 즉시 경유점 추종으로 복귀 [m] (진단: d_min >= 1.2 m)




_GOV_FULL_D = 1.5       # 감시 영역 최단 거리가 이 이상이면 감속·반발 없이 경유점 추종 그대로 [m] (진단: d_min >= 1.5 m)


_GOV_REP_K = 1.5        # 반발 조향 이득: w_rep = -1.5 * sign(b_min) / max(d_min, 0.5) [rad/s · m] (진단)


_GOV_REP_DFLOOR = 0.5   # 반발 조향 분모 하한 max(d_min, 0.5) [m] (진단)


_GOV_W_MAX = 1.0        # 0.8 m <= d_min < 1.5 m 구간에서 반발 조향을 더한 회전 속도의 절대값 상한 [rad/s] (진단 시도 28: 1.5 -> 1.0)




_SAFE_HALT_D = 0.8      # 감시 영역 최단 거리가 이보다 작으면 v = 0, w = 0 무조건 정지 [m] (진단 시도 28: 하드 안전 레이어)


# ========================= [END EDIT REGION] ===============================
