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
    """경유점 추종 + 2D 라이다 기반 정지·대기·우회(go / stop / detour) + 다가오는 물체 회피.
    전방 gap 스케일 f 에 가장 가까운 물체 거리 스케일 g 를 곱해 감속하고,
    전방 gap 이나 물체 최소 거리가 _STOP_D 미만이면 stop 으로 전환한다."""
    c = controller
    if not hasattr(c, "idx"):
        c.idx = 0
        c.state = "go"
        c.stop_t = 0.0
        c.detour_t = 0.0
        c.detour_w = 0.5
        c.prev = None
        c.vm = 0.0
        c.wm = 0.0

    # 실제 속도 추정(예측 굴림의 초기값)
    if c.prev is not None:
        pt, px, py, pyaw = c.prev
        dt = t - pt
        if dt > 1e-4:
            vm = ((pose[0] - px) * math.cos(pose[2]) + (pose[1] - py) * math.sin(pose[2])) / dt
            wm = _wrap_angle(pose[2] - pyaw) / dt
            c.vm = 0.7 * c.vm + 0.3 * vm
            c.wm = 0.7 * c.wm + 0.3 * wm
    c.prev = (t, pose[0], pose[1], pose[2])

    _perceive(t, pose)
    if not _P["got"] and t < 2.0:                    # 첫 스캔 전에는 움직이지 않는다
        return 0.0, 0.0, False

    movers = _movers(t, pose)
    v_nom, w_nom, done, target = _nominal(pose)
    if done:
        return 0.0, 0.0, True

    gap = _front_gap(pose)
    f = 1.0 if gap >= 2.5 else _clamp((gap - _STOP_D) / 1.5, 0.15, 1.0)
    d_min = _min_mover_dist(pose, movers)            # 진단: 가장 가까운 물체 거리(측면·후방 포함)
    g = _clamp((d_min - 0.5) / 1.5, 0.15, 1.0)
    scale = f * g

    # 경유점 추종은 go 일 때만(물체 예측으로 감속)
    if c.state == "go":
        if gap < _STOP_D or d_min < _STOP_D:
            c.state, c.stop_t = "stop", t
        else:
            for k in (1.0, 0.7, 0.45, 0.25, 0.1):
                cmd = (v_nom * scale * k, w_nom)
                if _is_safe(pose, cmd, movers, _SAFE):
                    return cmd[0], cmd[1], False
            # 안전한 스케일 명령이 하나도 없을 때만 정지 대신 회피 이동
            v, w = _evade(pose, movers, target, False)
            return v, w, False

    resume = gap >= _GO_D and _is_safe(pose, (v_nom * f, w_nom), movers, _SAFE_GO)

    if c.state == "stop":
        if resume:
            c.state = "go"
            return v_nom * f, w_nom, False
        if t - c.stop_t >= _STOP_WAIT:               # 대기 뒤 우회
            c.state, c.detour_t = "detour", t
            c.detour_w = _detour_side(pose)
        else:
            if _is_safe(pose, (0.0, 0.0), movers, _SAFE):
                return 0.0, 0.0, False
            v, w = _evade(pose, movers, target, True)    # 멈춰 있어도 물체가 다가오면 비켜선다
            return v, w, False

    # detour
    if resume:
        c.state = "go"
        return v_nom * f, w_nom, False
    el = t - c.detour_t
    if el > _DETOUR_T:
        c.state, c.stop_t = "stop", t
        if _is_safe(pose, (0.0, 0.0), movers, _SAFE):
            return 0.0, 0.0, False
        v, w = _evade(pose, movers, target, True)
        return v, w, False
    if el < _DETOUR_ROT:
        cmd = (0.0, c.detour_w)
        if _is_safe(pose, cmd, movers, _SAFE) and _static_ok(pose, cmd):
            return cmd[0], cmd[1], False
    v, w = _evade(pose, movers, target, False)
    return v, w, False




_STOP_D = 1.5          # 전방 정지 거리 [m] (앞 범퍼 기준) - 진단 지시 1 (1.0 -> 1.5)


_GO_D = 1.2            # 재출발 거리 [m] - 진단 지시 4


_STOP_WAIT = 2.0       # 정지가 이 시간 이상이면 우회 [s] - 진단 지시 1 (5.0 -> 2.0)


_DETOUR_T = 6.0        # 우회 상태 최대 지속 [s], 지나면 다시 정지·대기


_DETOUR_ROT = 1.2      # 우회 시작 시 제자리 회전(±0.5 rad/s) 시간 [s]


_MAP_MARGIN = 0.30     # 이 거리 안의 라이다 점은 정적 지도에 속한 것으로 본다 [m]


_MOVER_R = 0.42        # 0.6 x 0.6 상자의 외접원 반지름 [m]


_SAFE = 0.25           # 예측 여유 거리 (진행 허용) [m]


_SAFE_GO = 0.6         # 정지 -> 진행 복귀 시 요구 여유 [m] - 진단 지시 6 (0.40 -> 0.6)


_RELAX = 0.05          # 먼 미래 예측일수록 여유 요구를 줄이는 비율 [m/s]


_HORIZON = 3.0         # 예측 시간 [s]


_DT = 0.15             # 예측 간격 [s]


_TAU = 0.30            # 속도 명령 응답 시정수 [s]


_K_STATIC = 14         # 정적 장애물 검사는 앞 2 s 만


_ST_CLEAR = 0.12       # 정적 장애물과의 최소 여유 [m]


def _static_rects():
    """MISSION 바닥 박스와 FIXED_OBSTACLES 를 (x0, x1, y0, y1) 사각형 목록으로."""
    rects = [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0,
              b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0) for b in MISSION["boxes"]]
    rects += [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES]
    return rects


_RECTS = _static_rects()


_RECT_ARR = np.array(_RECTS, dtype=np.float64)


def _map_points(step=0.1):
    """정적 지도(사각형 테두리, 벽 안쪽 면)를 step 간격 점으로 표본화한다(회피 시 정적 충돌 검사용)."""
    pts = []

    def seg(x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        for i in range(n):
            a = i / (n - 1)
            pts.append((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a))

    for x0, x1, y0, y1 in _RECTS:
        seg(x0, y0, x1, y0)
        seg(x1, y0, x1, y1)
        seg(x1, y1, x0, y1)
        seg(x0, y1, x0, y0)
    xmin, xmax, ymin, ymax = WALLS
    seg(xmin, ymin, xmax, ymin)
    seg(xmax, ymin, xmax, ymax)
    seg(xmax, ymax, xmin, ymax)
    seg(xmin, ymax, xmin, ymin)
    return np.array(pts, dtype=np.float64)


_MAP_PTS = _map_points()


_CANDS = np.array([(v, w) for v in (-0.6, -0.3, 0.0, 0.3, 0.7, 1.1, 1.5)
                   for w in (-1.5, -0.8, -0.35, 0.0, 0.35, 0.8, 1.5)], dtype=np.float64)


_CANDS_MOVE = _CANDS[~((_CANDS[:, 0] == 0.0) & (_CANDS[:, 1] == 0.0))]


_P = {"front": None, "back": None, "wf": np.zeros((0, 2)), "wb": np.zeros((0, 2)),
      "pts": np.zeros((0, 2)), "tracks": [], "got": False}


def _is_static(w):
    """월드 점 (N, 2) 중 정적 지도(박스, 고정물, 벽 사각형 바깥)에 속하는 점의 bool 마스크."""
    m = _MAP_MARGIN
    x = w[:, 0:1]
    y = w[:, 1:2]
    R = _RECT_ARR
    inr = ((x > R[:, 0] - m) & (x < R[:, 1] + m) & (y > R[:, 2] - m) & (y < R[:, 3] + m)).any(axis=1)
    xmin, xmax, ymin, ymax = WALLS
    wall = (w[:, 0] < xmin + m) | (w[:, 0] > xmax - m) | (w[:, 1] < ymin + m) | (w[:, 1] > ymax - m)
    return inr | wall


def _to_robot(w, pose):
    """월드 점 (N, 2) 를 현재 로봇 기준 좌표로."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = w[:, 0] - x, w[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def _scan_world(which, pose):
    """새 스캔이 왔으면 정적 지도에 없는 점만 월드 좌표로 저장하고 True."""
    p = lidar2d_points(which)
    if p is _P[which]:
        return False
    _P[which] = p
    w = np.zeros((0, 2))
    if len(p):
        _P["got"] = True
        q = p[np.hypot(p[:, 0], p[:, 1]) < 8.0]
        if len(q):
            w = to_world(q, pose)
            w = w[~_is_static(w)]
    _P["w" + which[0]] = w
    return True


def _clusters(pts):
    """움직이는 물체 점을 0.9 m 반경으로 묶는다(4점 미만은 잡음으로 버림)."""
    out = []
    rest = pts
    for _ in range(40):
        if len(rest) == 0:
            break
        d = np.hypot(rest[:, 0] - rest[0, 0], rest[:, 1] - rest[0, 1])
        m = d < 0.9
        if int(m.sum()) >= 4:
            out.append(rest[m])
        rest = rest[~m]
    return out


def _box_center(c, rx, ry):
    """보이는 면의 점들로 0.6 m 상자의 중심을 추정한다(한 축 폭이 좁으면 보이는 면 뒤로 0.3 m)."""
    cen = []
    for k, r in ((0, rx), (1, ry)):
        lo, hi = float(c[:, k].min()), float(c[:, k].max())
        mid = 0.5 * (lo + hi)
        if hi - lo > 0.4:
            cen.append(mid)
        else:
            cen.append(mid + (0.3 if mid > r else -0.3))
    return cen[0], cen[1]


def _update_tracks(t, pose, pts):
    """물체 묶음을 이전 추적과 짝지어 0.6 s 이력으로 속도를 추정한다."""
    old = _P["tracks"]
    new = []
    used = set()
    for cl in _clusters(pts):
        cx, cy = _box_center(cl, pose[0], pose[1])
        best, bd = None, 0.8
        for i, tr in enumerate(old):
            if i in used:
                continue
            d = math.hypot(cx - tr["c"][0], cy - tr["c"][1])
            if d < bd:
                best, bd = i, d
        hist = [(t, cx, cy)]
        vx = vy = 0.0
        if best is not None:
            used.add(best)
            tr = old[best]
            hist = [h for h in tr["hist"] if t - h[0] <= 0.6] + hist
            vx, vy = tr["v"]
            h0 = hist[0]
            dt = t - h0[0]
            if dt >= 0.2:
                mx, my = (cx - h0[1]) / dt, (cy - h0[2]) / dt
                sp = math.hypot(mx, my)
                if sp > 2.0:
                    mx, my = mx * 2.0 / sp, my * 2.0 / sp
                vx, vy = 0.6 * vx + 0.4 * mx, 0.6 * vy + 0.4 * my
        new.append({"c": (cx, cy), "v": (vx, vy), "hist": hist, "t": t})
    _P["tracks"] = new


def _perceive(t, pose):
    """앞·뒤 2D 라이다를 읽어 움직이는 물체 점과 추적을 갱신한다."""
    nf = _scan_world("front", pose)
    nb = _scan_world("back", pose)
    if not (nf or nb):
        return
    pts = np.concatenate([_P["wf"], _P["wb"]])
    _P["pts"] = pts
    _update_tracks(t, pose, pts)


def _movers(t, pose):
    """6 m 안의 물체 (cx, cy, vx, vy) 목록(중심은 현재 시각으로 외삽)."""
    out = []
    for tr in _P["tracks"]:
        vx, vy = tr["v"]
        if math.hypot(vx, vy) < 0.1:
            vx = vy = 0.0
        cx = tr["c"][0] + vx * (t - tr["t"])
        cy = tr["c"][1] + vy * (t - tr["t"])
        if math.hypot(cx - pose[0], cy - pose[1]) > 6.0:
            continue
        out.append((cx, cy, vx, vy))
    return out


def _front_gap(pose):
    """진단 지시 2: 전방 ±30° (또는 차체 폭 통로) 안 물체 점까지 앞 범퍼로부터의 최소 거리."""
    p = _P["pts"]
    if len(p) == 0:
        return 99.0
    r = _to_robot(p, pose)
    ahead = r[:, 0] > _BODY[1] - 0.05
    ang = np.abs(np.arctan2(r[:, 1], r[:, 0]))
    m = ahead & ((ang < math.radians(30.0)) | (np.abs(r[:, 1]) < 0.60))
    if not m.any():
        return 99.0
    return float((r[m, 0] - _BODY[1]).min())


def _rollout(pose, cands, v0, w0):
    """(v, w) 후보들을 응답 지연(시정수 _TAU)을 넣어 _HORIZON 동안 굴린 궤적."""
    cands = np.asarray(cands, dtype=np.float64).reshape(-1, 2)
    K = int(round(_HORIZON / _DT))
    tau = np.arange(1, K + 1) * _DT
    a = np.exp(-tau / _TAU)[None, :]
    vc, wc = cands[:, 0:1], cands[:, 1:2]
    v = vc + (v0 - vc) * a
    w = wc + (w0 - wc) * a
    yaw = pose[2] + np.cumsum(w * _DT, axis=1)
    x = pose[0] + np.cumsum(v * np.cos(yaw) * _DT, axis=1)
    y = pose[1] + np.cumsum(v * np.sin(yaw) * _DT, axis=1)
    C = cands.shape[0]
    tau = np.concatenate([[0.0], tau])
    x = np.concatenate([np.full((C, 1), pose[0]), x], axis=1)
    y = np.concatenate([np.full((C, 1), pose[1]), y], axis=1)
    yaw = np.concatenate([np.full((C, 1), pose[2]), yaw], axis=1)
    return tau, x, y, yaw


def _rect_dist(x, y, yaw, px, py):
    """로봇 몸체 사각형(_BODY) 에서 점까지의 거리(안이면 0)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = px - x, py - y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    ex = np.maximum(np.maximum(_BODY[0] - lx, lx - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - ly, ly - _BODY[3]), 0.0)
    return np.hypot(ex, ey)


def _mover_clear(ro, movers):
    """궤적별 물체와의 예측 최소 여유 (C,) 와 현재 여유."""
    tau, x, y, yaw = ro
    C = x.shape[0]
    if len(movers) == 0:
        return np.full(C, 9.0), 9.0
    M = np.asarray(movers, dtype=np.float64)
    px = M[None, None, :, 0] + M[None, None, :, 2] * tau[None, :, None]
    py = M[None, None, :, 1] + M[None, None, :, 3] * tau[None, :, None]
    d = _rect_dist(x[:, :, None], y[:, :, None], yaw[:, :, None], px, py) - _MOVER_R
    eff = d + _RELAX * tau[None, :, None]
    return eff.min(axis=(1, 2)), float(eff[0, 0, :].min())


def _static_clear(ro, pose):
    """궤적별 정적 지도와의 최소 여유 (C,) (앞 2 s) 와 현재 여유."""
    tau, x, y, yaw = ro
    C = x.shape[0]
    P = _MAP_PTS[np.hypot(_MAP_PTS[:, 0] - pose[0], _MAP_PTS[:, 1] - pose[1]) < 3.5]
    if len(P) == 0:
        return np.full(C, 9.0), 9.0
    k = min(x.shape[1], _K_STATIC)
    d = _rect_dist(x[:, :k, None], y[:, :k, None], yaw[:, :k, None], P[None, None, :, 0], P[None, None, :, 1])
    return d.min(axis=(1, 2)), float(d[0, 0, :].min())


def _is_safe(pose, cmd, movers, need):
    """진단 지시 1~4: 시간-대-충돌(TTC) 검사.
    cmd (v, w) 로 로봇의 세계 속도를 구하고, 각 물체와의 최소 접근 시간 t_c = -(d·dv)/|dv|^2 를 계산한다.
    t_c 를 [0, _TTC_H] 로 제한한 시점의 거리와 현재 거리 중 하나라도 _MOVER_R + need 보다 작으면 False.
    거리는 로봇 몸체 사각형(_BODY)에서 물체 중심까지 재고, 속도 명령 응답 지연(_TAU)을 반영한다."""
    if not movers:
        return True
    v, w = float(cmd[0]), float(cmd[1])
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    v0 = float(controller.vm)
    lag = (v0 - v) * _TAU                       # v0 에서 v 로 바뀌는 동안 더 가는 거리
    bxc = 0.5 * (_BODY[0] + _BODY[1])           # 몸체 중심 (로봇 기준 x)
    # 지연을 반영한 등가 출발점(몸체 중심)과 세계 속도 - 진단 지시 1
    cx0 = x + c * (bxc + lag)
    cy0 = y + s * (bxc + lag)
    rvx, rvy = v * c, v * s
    lim = _MOVER_R + need
    for mx, my, mvx, mvy in movers:
        # 진단 지시 2: 상대 위치·상대 속도와 최소 접근 시간
        dx, dy = cx0 - mx, cy0 - my
        dvx, dvy = rvx - mvx, rvy - mvy
        dv2 = dvx * dvx + dvy * dvy
        tc = 0.0 if dv2 < 1e-9 else -(dx * dvx + dy * dvy) / dv2
        tc = _clamp(tc, 0.0, _TTC_H)            # t_c < 0 이면 현재 거리만, 창 밖이면 창 끝에서의 거리
        # 현재 거리
        d_now = float(_rect_dist(x, y, yaw, mx, my))
        if d_now < lim:
            return False
        if tc <= 0.0:
            continue
        # 진단 지시 3: t_c 시점의 거리 d_c (로봇 위치는 지연 응답을 넣은 직선 이동)
        disp = v * tc + (v0 - v) * _TAU * (1.0 - math.exp(-tc / _TAU))
        rx = x + c * disp
        ry = y + s * disp
        ryaw = yaw + w * tc
        d_c = float(_rect_dist(rx, ry, ryaw, mx + mvx * tc, my + mvy * tc))
        if d_c < lim:
            return False
    return True                                  # 진단 지시 4: 모든 물체 통과


def _static_ok(pose, cmd):
    ro = _rollout(pose, [cmd], controller.vm, controller.wm)
    sc, now = _static_clear(ro, pose)
    return bool(sc[0] >= min(_ST_CLEAR, now - 0.01))


def _evade(pose, movers, target, allow_stop):
    """정적 지도와 겹치지 않는 후보 중 물체 여유가 가장 크고(1 m 까지) 목표 쪽으로 가는 명령."""
    cands = _CANDS if allow_stop else _CANDS_MOVE
    ro = _rollout(pose, cands, controller.vm, controller.wm)
    mc, _ = _mover_clear(ro, movers)
    sc, sc_now = _static_clear(ro, pose)
    _, x, y, _ = ro
    tx, ty = target
    prog = np.clip(math.hypot(tx - pose[0], ty - pose[1]) - np.hypot(tx - x[:, -1], ty - y[:, -1]), -2.0, 2.0)
    score = 10.0 * np.minimum(mc, 1.0) + prog - 0.3 * (cands[:, 0] < 0.0)
    ok = sc >= min(_ST_CLEAR, sc_now - 0.01)
    score = np.where(ok, score, score - 100.0 + 10.0 * sc)
    i = int(np.argmax(score))
    return float(cands[i, 0]), float(cands[i, 1])


def _detour_side(pose):
    """진단 지시 5: 우회 회전 방향(±0.5 rad/s). 물체 점이 많은 쪽의 반대로, 정적 장애물이 막으면 다른 쪽."""
    side = 1.0
    p = _P["pts"]
    if len(p):
        r = _to_robot(p, pose)
        r = r[np.hypot(r[:, 0], r[:, 1]) < 3.0]
        if len(r) and float(r[:, 1].mean()) > 0.0:
            side = -1.0
    for s in (side, -side):
        if _static_ok(pose, (0.0, 0.5 * s)):
            return 0.5 * s
    return 0.5 * side


def _nominal(pose):
    """기존 경유점 추종 법칙(그대로). 회피로 밀려난 경우의 경유점 건너뛰기·목표 복귀만 더함.
    (v, w, done, 현재 목표점) 을 돌려준다."""
    x, y, yaw = pose
    n = len(_WAYPOINTS)
    gx, gy = GOAL[0], GOAL[1]
    if controller.idx >= n and math.hypot(gx - x, gy - y) > 0.4:
        controller.idx = n - 1          # 정렬 중 물체를 피해 밀려났으면 목표로 다시 간다
    if controller.idx >= n:
        e = _wrap_angle(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True, (gx, gy)
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False, (gx, gy)

    tx, ty = _WAYPOINTS[controller.idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if controller.idx < n - 1:
        px, py = _WAYPOINTS[controller.idx - 1] if controller.idx > 0 else (START[0], START[1])
        sx, sy = tx - px, ty - py
        L = math.hypot(sx, sy)
        past = L > 1e-6 and ((x - tx) * sx + (y - ty) * sy) / L > -0.1
        if dist < 0.35 or (past and dist < 1.0):
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
    return v, w, False, (tx, ty)




_TTC_H = 1.0           # TTC 검사 시간 창 [s] - 진단 지시 3 (0~1 s)




def _min_mover_dist(pose, movers):
    """로봇 중심과 각 움직이는 물체 중심 사이 유클리드 거리의 최소값(물체가 없으면 99.0)."""
    if not movers:
        return 99.0
    return min(math.hypot(mx - pose[0], my - pose[1]) for mx, my, _, _ in movers)


# ========================= [END EDIT REGION] ===============================
