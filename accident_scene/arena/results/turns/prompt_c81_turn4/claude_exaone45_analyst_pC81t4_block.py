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
    """경유점 추종 + 2D 라이다로 지도에 없는 물체 감지.
    3.5 m 안(d_eff)에서는 거리 비례 감속(하한 0.2 m/s, 회전 ±1.0 rad/s), 예측 충돌이 있을 때만 정지,
    멈춰도 부딪히면 물체 진행 방향의 수직 후보를 포함한 회피."""
    S = _ST
    x, y, yaw = pose
    if not S:
        S.update(idx=0, prev_wp=(START[0], START[1]), rejoin=None, esc=None, esc_start=None,
                 scan_obj={}, dyn={}, tracks=[], track_t=-1.0,
                 P=np.zeros((0, 2)), U=np.zeros((0, 2)), seen=False,
                 last=(t, x, y, yaw), speed=0.0, yawrate=0.0,
                 hold=None, armed=True, rel_d=99.0)
    t0, x0, y0, yaw0 = S["last"]
    dt = t - t0
    if dt > 1e-4:                                       # 자기 이동 속도 (스캔 지연 여유 계산용)
        S["speed"] = 0.7 * S["speed"] + 0.3 * math.hypot(x - x0, y - y0) / dt
        S["yawrate"] = 0.7 * S["yawrate"] + 0.3 * abs(_wrap_angle(yaw - yaw0)) / dt
    S["last"] = (t, x, y, yaw)

    _perceive(t, pose)
    if not S["seen"] and t < 2.0:                       # 첫 스캔이 들어올 때까지 출발하지 않음
        return 0.0, 0.0, False
    if S["esc"] is not None:
        S["hold"], S["armed"], S["rel_d"] = None, False, 0.0
        return _escape_step(t, pose)

    v_nom, w_nom, done = _nominal(pose)
    if done:
        return 0.0, 0.0, True
    P, U = S["P"], S["U"]

    d_thr, clr_min = 99.0, 99.0
    if len(P):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rx, ry = c * dx + s * dy, -s * dx + c * dy
        clr = _rect_clear(rx, ry)                       # 몸체 가장자리까지 거리
        rng = np.maximum(np.hypot(dx, dy), 1e-6)
        closing = -(U[:, 0] * dx + U[:, 1] * dy) / rng  # 물체 자신의 접근 속도
        ahead = (v_nom > 0.0) & (rx > 0.0) & (np.abs(ry) < 1.0)
        coming = (closing > 0.15) & (rx > -0.3)
        rel = ahead | coming
        d_eff = clr - _CLOSE_W * np.maximum(closing, 0.0)   # 지시 1: 가중치 0.7 -> 1.0
        d_thr = float(d_eff[rel].min()) if rel.any() else 99.0
        clr_min = float(clr.min())

    # 감속 구간 (지시 3, 5): d_thr < 3.5 m 이면 거리 비례 감속, 전진 하한 0.2 m/s, 회전 ±1.0 rad/s
    slow = d_thr < _HOLD_ON
    if slow and v_nom > 0.0:
        v_go = max(_V_FLOOR, min(v_nom, 0.2 + 0.5 * (d_thr - 0.8)))
    else:
        v_go = v_nom
    w_go = _clamp(w_nom, -_W_SLOW, _W_SLOW) if slow else w_nom
    look = _lookahead(pose)
    pred = _rollout(P, U, x, y, yaw, yaw, v_go, 0.0, cap=look)   # 이대로 갈 때의 예측 여유
    threat = pred < _CONF_ON

    # 정지 상태(H): 3.5 m 안이면서 예측 충돌이 있을 때만 (지시 4), 히스테리시스 유지
    H = S["hold"]
    if H is None:
        S["rel_d"] = max(S["rel_d"], d_thr)
        if d_thr >= _HOLD_ON:
            S["armed"] = True
        thr = _HOLD_ON if S["armed"] else min(_HOLD_REARM, S["rel_d"] - _HOLD_GAIN)
        if d_thr < thr and threat:
            S["hold"] = {"t0": t, "d": d_thr}
    else:
        if threat and d_thr < min(_HOLD_REARM, H["d"] - _HOLD_GAIN):
            S["hold"] = {"t0": t, "d": d_thr}           # 물체가 더 다가옴: H 재설정
        elif _hold_release(H, d_thr, t, pred):
            # 예측 경로가 비어서 풀렸으면 바로 다시 무장(0.5/0.7 m 히스테리시스 + 최소 정지 시간)
            S["hold"], S["armed"], S["rel_d"] = None, pred >= _CONF_OFF, d_thr

    if S["hold"] is not None:
        v, w = 0.0, 0.0                                 # 정지
    else:
        v, w = v_go, w_go
    if clr_min < 0.45:
        w = 0.0                                         # 바로 옆 물체가 있으면 회전 금지

    if _rollout(P, U, x, y, yaw, yaw, v, 0.0, cap=look) < 0.3:
        if v > 0.0 and _rollout(P, U, x, y, yaw, yaw, 0.0, 0.0) >= 0.3:
            return 0.0, 0.0, False
        S["hold"], S["armed"], S["rel_d"] = None, False, 0.0
        return _escape_step(t, pose)                    # 멈춰도 부딪힘: 비켜섬
    return v, w, False




_FP = (-0.72, 0.30, -0.36, 0.36)      # 로봇 몸체 범위 (로봇 기준 x 최소, x 최대, y 최소, y 최대)


_CIRC_OFF = (-0.465, 0.045)           # 몸체를 덮는 원 두 개의 중심 (로봇 x 축 위)


_CIRC_R = 0.45                        # 그 원의 반지름


_STATIC_RECTS = np.array(
    [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"],
      b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"]) for b in MISSION["boxes"]]
    + [tuple(o[1:]) for o in FIXED_OBSTACLES], dtype=np.float64)


_ST = {}                              # 제어기 상태 (첫 호출 때 채움)


def _static_mask(P, margin):
    """월드 점 (N, 2) 중 정적 지도(벽 바깥, 바닥 박스, 고정물)에서 margin 이내인 점이면 True."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    m = np.broadcast_to(np.asarray(margin, dtype=np.float64), (len(P),))
    x, y = P[:, 0], P[:, 1]
    out = (x <= WALLS[0] + m) | (x >= WALLS[1] - m) | (y <= WALLS[2] + m) | (y >= WALLS[3] - m)
    R, mm = _STATIC_RECTS, m[:, None]
    inside = ((x[:, None] >= R[:, 0] - mm) & (x[:, None] <= R[:, 1] + mm)
              & (y[:, None] >= R[:, 2] - mm) & (y[:, None] <= R[:, 3] + mm)).any(axis=1)
    return out | inside


def _poses_free(bx, by, h, extra=0.03):
    """base_link 위치들(bx, by), 방향 h 에서 몸체가 정적 지도와 겹치지 않으면 True."""
    bx = np.atleast_1d(np.asarray(bx, dtype=np.float64))
    by = np.atleast_1d(np.asarray(by, dtype=np.float64))
    c, s = math.cos(h), math.sin(h)
    C = np.concatenate([np.stack([bx + o * c, by + o * s], axis=1) for o in _CIRC_OFF])
    return not _static_mask(C, _CIRC_R + extra).any()


def _segment_free(a, b):
    """a 에서 b 로 직진할 때 몸체가 정적 지도와 겹치지 않으면 True."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L < 0.1:
        return True
    h = math.atan2(b[1] - a[1], b[0] - a[0])
    s = np.arange(0.1, L + 1e-6, 0.1)
    return _poses_free(a[0] + s * math.cos(h), a[1] + s * math.sin(h), h)


def _path_free(x, y, h, v, delay):
    """회피 후보(방향 h 로 회전 후 속도 v)의 최대 1.5 m 이동 경로가 정적 지도상 비어 있으면 True."""
    L = min(1.5, abs(v) * max(3.0 - delay, 0.0))
    s = np.arange(0.1, L + 1e-6, 0.15)
    if s.size == 0:
        return True
    sg = 1.0 if v > 0 else -1.0
    return _poses_free(x + sg * s * math.cos(h), y + sg * s * math.sin(h), h)


def _rect_clear(rx, ry):
    """로봇 기준 점들의 몸체 사각형까지 거리."""
    ddx = np.maximum(np.maximum(_FP[0] - rx, rx - _FP[1]), 0.0)
    ddy = np.maximum(np.maximum(_FP[2] - ry, ry - _FP[3]), 0.0)
    return np.hypot(ddx, ddy)


def _rollout(P, U, x, y, yaw, h, v, delay, T=3.0, cap=99.0):
    """물체 점 P 가 속도 U 로 움직이고, 로봇이 delay 초 동안 h 로 돌고 나서 속도 v 로 h 방향 직진할 때
    T 초 동안의 최소 여유 거리 [m] (몸체를 덮는 두 원 기준)."""
    if len(P) == 0:
        return 99.0
    ts = np.arange(0.0, T + 1e-6, 0.1)
    s = np.clip(v * np.maximum(ts - delay, 0.0), -cap, cap)
    hh = np.where(ts < delay, yaw, h)
    bx = x + s * math.cos(h)
    by = y + s * math.sin(h)
    qx = P[None, :, 0] + U[None, :, 0] * ts[:, None]
    qy = P[None, :, 1] + U[None, :, 1] * ts[:, None]
    best = 99.0
    for o in _CIRC_OFF:
        cx = (bx + o * np.cos(hh))[:, None]
        cy = (by + o * np.sin(hh))[:, None]
        best = min(best, float(np.hypot(qx - cx, qy - cy).min()))
    return best - _CIRC_R


def _update_tracks(t, pose):
    """지도에 없는 점들을 물체별로 묶고(가까운 것부터), 이전 위치와 비교해 물체 속도를 추정합니다."""
    S = _ST
    P = np.concatenate([S["dyn"].get(k, np.zeros((0, 2))) for k in ("front", "back")])
    if len(P):
        P = P[np.argsort(np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]))]
    old = [tr for tr in S["tracks"] if t - tr["t"] < 0.5]
    hit = set()
    used = np.zeros(len(P), dtype=bool)
    tracks, keepP, keepU = [], [], []
    for _ in range(40):
        free = np.flatnonzero(~used)
        if free.size == 0 or len(keepP) >= 12:
            break
        sx, sy = P[free[0]]
        m = ~used & (np.hypot(P[:, 0] - sx, P[:, 1] - sy) < 0.9)
        used |= m
        if m.sum() < 5:                      # 점 몇 개짜리 잡음은 버림
            continue
        pts = P[m]
        c = pts.mean(axis=0)
        vel, age, best, bd = np.zeros(2), 1, None, 0.8
        for i, tr in enumerate(old):
            d = math.hypot(c[0] - tr["c"][0], c[1] - tr["c"][1])
            if d < bd:
                best, bd = i, d
        if best is not None:
            tr = old[best]
            hit.add(best)
            dt = t - tr["t"]
            vel = tr["v"].copy() if dt < 0.03 else 0.6 * tr["v"] + 0.4 * (c - tr["c"]) / dt
            sp = math.hypot(vel[0], vel[1])
            if sp > 2.0:
                vel = vel * (2.0 / sp)
            age = tr["age"] + 1
        tracks.append({"c": c, "v": vel, "t": t, "age": age})
        if len(pts) > 40:
            pts = pts[::int(math.ceil(len(pts) / 40.0))]
        keepP.append(pts)
        keepU.append(np.tile(vel, (len(pts), 1)))
    tracks += [tr for i, tr in enumerate(old) if i not in hit]
    S["tracks"] = tracks
    S["P"] = np.concatenate(keepP) if keepP else np.zeros((0, 2))
    S["U"] = np.concatenate(keepU) if keepU else np.zeros((0, 2))


def _perceive(t, pose):
    """앞·뒤 2D 라이다의 새 스캔을 받은 순간의 자세로 월드 좌표로 바꾸고, 정적 지도에 없는 점만 남깁니다."""
    S = _ST
    fresh = False
    for which in ("front", "back"):
        pts = lidar2d_points(which)
        if len(pts) == 0 or pts is S["scan_obj"].get(which):
            continue
        S["scan_obj"][which] = pts
        S["seen"] = True
        fresh = True
        rng = np.hypot(pts[:, 0], pts[:, 1])
        keep = rng < 7.0
        pts, rng = pts[keep], rng[keep]
        if len(pts) == 0:
            S["dyn"][which] = np.zeros((0, 2))
            continue
        W = to_world(pts, pose)
        # 스캔 지연 동안의 이동·회전으로 정적 점이 밀려 보이는 것을 여유로 흡수
        margin = 0.25 + 0.1 * S["speed"] + np.minimum(0.1 * S["yawrate"] * rng, 0.6)
        S["dyn"][which] = W[~_static_mask(W, margin)]
    if fresh and t - S["track_t"] >= 0.08:
        _update_tracks(t, pose)
        S["track_t"] = t


def _nominal(pose):
    """경유점(또는 회피 후 복귀점) 추종: 기존 제어 법칙 그대로. (v, w, done) 을 돌려줍니다."""
    S = _ST
    x, y, yaw = pose
    n = len(_WAYPOINTS)
    if S["rejoin"] is not None and math.hypot(S["rejoin"][0] - x, S["rejoin"][1] - y) < 0.35:
        S["rejoin"] = None
    if S["rejoin"] is None and S["idx"] >= n:
        e = _wrap_angle(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False
    if S["rejoin"] is not None:
        tx, ty = S["rejoin"]
    else:
        tx, ty = _WAYPOINTS[S["idx"]]
        if S["idx"] < n - 1 and math.hypot(tx - x, ty - y) < 0.35:
            S["prev_wp"] = (tx, ty)
            S["idx"] += 1
            tx, ty = _WAYPOINTS[S["idx"]]
    dist = math.hypot(tx - x, ty - y)
    e = _wrap_angle(math.atan2(ty - y, tx - x) - yaw)
    w = _clamp(7.5 * e, -2.0, 2.0)
    v = 0.0 if abs(e) > 0.35 else min(1.5, 2.5 * dist + 0.5)
    if S["rejoin"] is None and S["idx"] == n - 1 and dist < 0.25:
        S["idx"] = n
    return v, w, False


def _set_rejoin(pose):
    """회피가 끝나면 현재 경로 구간 위 0.5 m 앞 지점으로 복귀합니다(직선이 막혀 있으면 회피 시작점으로)."""
    S = _ST
    x, y, _ = pose
    n = len(_WAYPOINTS)
    if S["idx"] >= n:                         # 정렬 중 회피했으면 GOAL 로 다시 접근
        S["idx"] = n - 1
        S["rejoin"] = None
        return
    ax, ay = S["prev_wp"]
    bx, by = _WAYPOINTS[S["idx"]]
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy)
    if L < 1e-6:
        r = (bx, by)
    else:
        u = _clamp(((x - ax) * sx + (y - ay) * sy) / (L * L) + 0.5 / L, 0.0, 1.0)
        r = (ax + u * sx, ay + u * sy)
    if math.hypot(r[0] - x, r[1] - y) < 0.35:
        S["rejoin"] = None
    elif _segment_free((x, y), r) or S["esc_start"] is None:
        S["rejoin"] = r
    else:
        S["rejoin"] = S["esc_start"]


def _plan_escape(pose, E):
    """정지로는 물체를 피할 수 없을 때: 정지/후진/전진/여러 방향 비켜서기 중 예측 여유가 가장 큰 것을 고릅니다.
    지시 2: 가까운 움직이는 물체의 속도 벡터에 수직인 두 방향을 후보로 추가합니다."""
    S = _ST
    x, y, yaw = pose
    P, U = S["P"], S["U"]
    if S["rejoin"] is not None:
        tx, ty = S["rejoin"]
    elif S["idx"] < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[S["idx"]]
    else:
        tx, ty = GOAL[0], GOAL[1]
    ph = math.atan2(ty - y, tx - x)
    cands = [(yaw, 0.0, 0.0, 0.0), (yaw, -0.6, 0.0, 0.0), (yaw, 1.0, 0.0, 0.0)]
    if not _static_mask([[x, y]], 0.82).any():          # 제자리 회전해도 몸체가 정적 장애물을 쓸지 않음
        heads = [(_wrap_angle(ph + k * math.pi / 4), 0.0) for k in range(8)]
        if E["v"] > 0:
            heads.append((E["h"], 0.0))
        movers = []
        for tr in S["tracks"]:
            sp = math.hypot(float(tr["v"][0]), float(tr["v"][1]))
            dd = math.hypot(float(tr["c"][0]) - x, float(tr["c"][1]) - y)
            if sp > 0.15 and dd < 4.0:
                movers.append((dd, math.atan2(float(tr["v"][1]), float(tr["v"][0]))))
        for _, a in sorted(movers)[:3]:                 # 물체 진행 방향과 수직으로 비켜서기
            heads.append((_wrap_angle(a + math.pi / 2), 0.1))
            heads.append((_wrap_angle(a - math.pi / 2), 0.1))
        for h, bonus in heads:
            cands.append((h, 1.0, abs(_wrap_angle(h - yaw)) / 1.6, bonus))
    best = None
    for h, v, delay, bonus in cands:
        if v != 0.0 and not _path_free(x, y, h, v, delay):
            continue
        sc = min(_rollout(P, U, x, y, yaw, h, v, delay, cap=1.5), 1.2)
        if v > 0.0:
            sc += 0.15 * math.cos(_wrap_angle(h - ph)) + bonus
        sc -= 0.05 * delay
        if v == E["v"] and abs(_wrap_angle(h - E["h"])) < 0.05:
            sc += 0.15                                  # 방금 고른 동작 유지 (흔들림 방지)
        if best is None or sc > best[0]:
            best = (sc, h, v)
    return best[1], best[2]


def _escape_step(t, pose):
    """경유점 추종을 멈추고 회피 동작을 실행합니다. 안전 거리가 확보되면 복귀점을 정하고 추종을 재개합니다.
    지시 6: 회피 후 복귀 전 최소 대기 시간 0.6 s -> 0.5 s."""
    S = _ST
    x, y, yaw = pose
    E = S["esc"]
    if E is None:
        if S["rejoin"] is None or S["esc_start"] is None:
            S["esc_start"] = (x, y)
        E = S["esc"] = {"t0": t, "replan": t, "h": yaw, "v": 0.0}
    if t >= E["replan"]:
        E["h"], E["v"] = _plan_escape(pose, E)
        E["replan"] = t + 0.15
    P, U = S["P"], S["U"]
    if t - E["t0"] > _ESC_MIN:
        stay = _rollout(P, U, x, y, yaw, yaw, 0.0, 0.0)
        dmin = 99.0
        if len(P):
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = P[:, 0] - x, P[:, 1] - y
            dmin = float(_rect_clear(c * dx + s * dy, -s * dx + c * dy).min())
        if stay >= 0.8 and dmin >= 0.9:
            S["esc"] = None
            _set_rejoin(pose)
            return 0.0, 0.0, False
    h, v = E["h"], E["v"]
    if v == 0.0:
        return 0.0, 0.0, False
    rel = _wrap_angle(h - yaw)
    if abs(rel) > 0.25:
        return 0.0, _clamp(4.0 * rel, -1.8, 1.8), False
    sg = 1.0 if v > 0 else -1.0
    if not _poses_free(x + sg * 0.35 * math.cos(yaw), y + sg * 0.35 * math.sin(yaw), yaw):
        E["replan"] = t                                 # 정적 장애물에 막힘: 다시 고름
        return 0.0, 0.0, False
    return v, _clamp(3.0 * rel, -1.0, 1.0), False




_HOLD_ON = 3.5       # 정지/감속 구간 시작 거리 [m] (지시 4: 2.0 -> 3.5)


_HOLD_REARM = 1.8    # 재출발 직후 다시 정지하는 낮은 임계값 [m] (지시 6: 히스테리시스)


_HOLD_MIN = 0.5      # 최소 정지 시간 [s] (지시 6: 1.0 -> 0.5)


_HOLD_GAIN = 0.1     # 재출발에 필요한 거리 증가 [m] (지시 2: 0.05 -> 0.1)


def _hold_release(H, d_now, t, pred=99.0):
    """정지 상태 H 를 풀어도 되면 True: 최소 정지 시간(0.5 s)이 지났고,
    예측 여유가 0.7 m 이상으로 회복됐거나, 거리가 0.1 m 이상 늘었거나, 3.5 m 밖으로 벗어났을 때."""
    if t - H["t0"] < _HOLD_MIN:
        return False
    return pred >= _CONF_OFF or d_now >= H["d"] + _HOLD_GAIN or d_now >= _HOLD_ON




_CLOSE_W = 1.0       # d_eff 의 접근 속도 가중치 (지시 1: 0.7 -> 1.0)


_V_FLOOR = 0.2       # 감속 구간 전진 속도 하한 [m/s] (지시 5)


_W_SLOW = 1.0        # 감속 구간 회전 속도 한도 [rad/s] (지시 3: 0.6 -> 1.0)


_CONF_ON = 0.5       # 이 예측 여유 [m] 미만이면 정지 대상


_CONF_OFF = 0.7      # 이 예측 여유 [m] 이상이면 정지 해제 (히스테리시스)


_ESC_MIN = 0.5       # 회피 후 복귀 판단 전 최소 시간 [s] (지시 6)


def _lookahead(pose):
    """현재 추종 목표(복귀점 또는 경유점)까지 거리 + 0.3 m: 예측 롤아웃이 경로 꺾임 너머로 직진하지 않게 합니다."""
    S = _ST
    x, y, _ = pose
    if S["rejoin"] is not None:
        tx, ty = S["rejoin"]
    elif S["idx"] < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[S["idx"]]
    else:
        return 0.5
    return max(0.5, math.hypot(tx - x, ty - y) + 0.3)


# ========================= [END EDIT REGION] ===============================
