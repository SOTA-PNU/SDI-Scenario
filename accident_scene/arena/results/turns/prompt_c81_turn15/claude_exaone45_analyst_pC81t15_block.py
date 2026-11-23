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
    """경유점 추종 + 2D 라이다로 지도에 없는 물체를 감지하고, MTC(충돌까지 남은 시간)로 감속/정지/회피합니다.
    정지 해제: (a) 감지된 물체 점 전체(정지를 일으킨 점 포함)가 모두 멀어지는(closing0 < 0) 상태가 _RESUME_GO 초
    지속되면 MTC 와 무관하게 즉시 재출발, (b) 최소 정지 시간(_HOLD_MIN) 뒤 MTC >= _HOLD_ON, (c) 물체 거리 증가 추세.
    감지 점이 없으면 (a) 는 시작하지 않고 (b) 경로에 맡깁니다.
    재출발 시 _path_free 확인 없이 곧바로 경유점 추종 속도를 적용합니다. 정지 중 물체가 다가오면 즉시 회피합니다."""
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

    d_thr, clr_min = 99.0, 99.0                         # d_thr: 최소 MTC [s]
    approach = 0.0                                      # 정지한 로봇 쪽으로 다가오는 근접 물체의 최대 접근 속도 [m/s]
    recede = False                                      # 감지 점이 없으면 멀어짐 판정을 하지 않음 (검토 지시 2)
    if len(P):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rx, ry = c * dx + s * dy, -s * dx + c * dy
        clr = _rect_clear(rx, ry)                       # 몸체 가장자리까지 거리
        rng = np.maximum(np.hypot(dx, dy), 1e-6)
        vrx, vry = v_nom * c, v_nom * s                 # 로봇 속도 벡터
        closing = -((U[:, 0] - vrx) * dx + (U[:, 1] - vry) * dy) / rng   # 상대 접근 속도
        closing0 = -(U[:, 0] * dx + U[:, 1] * dy) / rng  # 로봇 속도 0 (정지 중 실제 접근 속도)
        rel = closing > _CLOSE_MIN
        if rel.any():
            mtc = clr[rel] / closing[rel]               # MTC = 거리 / 접근 속도
            d_thr = float(mtc.min())
        clr_min = float(clr.min())
        near = clr < _ESC_NEAR
        if near.any():
            approach = float(closing0[near].max())
        # 검토 지시 1: S["P"] 전체(rel 이 True 인 점, 곧 MTC 를 만든 점 포함)의 closing0 가 모두 < 0 일 때만 멀어짐
        recede = bool(np.all(closing0 < 0.0))

    # 정지 상태(H) 관리: 시작 MTC _HOLD_ON(2.0 s), 재출발 직후에는 0.8 s 이하이고 더 줄어들 때만 다시 정지
    H = S["hold"]
    resume = False
    if H is None:
        S["rel_d"] = max(S["rel_d"], d_thr)
        if d_thr >= _HOLD_ON:
            S["armed"] = True
        thr = _HOLD_ON if S["armed"] else min(_HOLD_REARM, S["rel_d"] - _HOLD_GAIN)
        if d_thr < thr:
            S["hold"] = {"t0": t, "d": d_thr, "dmin": clr_min, "inc_t": None, "rec_t": None}
    else:
        H["dmin"] = min(H["dmin"], clr_min)
        if d_thr < min(_HOLD_REARM, H["d"] - _HOLD_GAIN):
            # MTC 가 더 줄어듦: 기준 재설정 (멀어짐 타이머는 물체가 다시 다가올 때만 초기화 - 지시 4)
            S["hold"] = {"t0": t, "d": d_thr, "dmin": clr_min, "inc_t": None,
                         "rec_t": H.get("rec_t") if recede else None}
        elif _hold_release(H, d_thr, t, recede) or _hold_trend(H, clr_min, t):
            S["hold"], S["armed"], S["rel_d"] = None, False, d_thr   # 재출발: 멀어짐 타이머도 함께 사라짐 (지시 4)
            resume = True

    # 정지 중 물체가 스스로 다가오면(closing0 > _ESC_CLOSE = 0.05 m/s) 정지 대신 즉시 회피 (지시 5: 유지)
    if S["hold"] is not None and approach > _ESC_CLOSE:
        S["hold"], S["armed"], S["rel_d"] = None, False, 0.0
        return _escape_step(t, pose)

    if S["hold"] is not None:
        v, w = 0.0, 0.0                                 # 정지
    elif resume:
        v, w = v_nom, w_nom                             # 진단 지시 3: _path_free 확인 없이 즉시 경유점 추종 속도
    elif d_thr < _SLOW_T:
        k = (d_thr - _HOLD_ON) / (_SLOW_T - _HOLD_ON)
        v = min(v_nom, max(_V_SLOW, 1.5 * k))
        w = _clamp(w_nom, -1.0, 1.0)                    # 감속 구간 회전 제한 ±1.0 rad/s
    else:
        v, w = v_nom, w_nom
    if clr_min < 0.45:
        w = 0.0                                         # 바로 옆 물체가 있으면 회전 금지

    if _rollout(P, U, x, y, yaw, yaw, v, 0.0) < 0.3:
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
    """정지로는 물체를 피할 수 없을 때: 정지/후진/전진/여러 방향 비켜서기 중 예측 여유가 가장 큰 것을 고릅니다."""
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
    cands = [(yaw, 0.0, 0.0), (yaw, -0.6, 0.0), (yaw, 1.0, 0.0)]
    if not _static_mask([[x, y]], 0.82).any():          # 제자리 회전해도 몸체가 정적 장애물을 쓸지 않음
        heads = [_wrap_angle(ph + k * math.pi / 4) for k in range(8)]
        if E["v"] > 0:
            heads.append(E["h"])
        for h in heads:
            cands.append((h, 1.0, abs(_wrap_angle(h - yaw)) / 1.6))
    best = None
    for h, v, delay in cands:
        if v != 0.0 and not _path_free(x, y, h, v, delay):
            continue
        sc = min(_rollout(P, U, x, y, yaw, h, v, delay, cap=1.5), 1.2)
        if v > 0.0:
            sc += 0.15 * math.cos(_wrap_angle(h - ph))
        sc -= 0.05 * delay
        if v == E["v"] and abs(_wrap_angle(h - E["h"])) < 0.05:
            sc += 0.15                                  # 방금 고른 동작 유지 (흔들림 방지)
        if best is None or sc > best[0]:
            best = (sc, h, v)
    return best[1], best[2]


def _escape_step(t, pose):
    """경유점 추종을 멈추고 회피 동작을 실행합니다. 안전 거리가 확보되면 복귀점을 정하고 추종을 재개합니다."""
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
    if t - E["t0"] > 0.6:
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




_HOLD_ON = 2.0       # 정지 시작·해제 MTC [s] (진단 지시 1: 1.0 -> 2.0)


_HOLD_REARM = 0.8    # 재출발 직후 다시 정지하는 낮은 MTC 임계값 [s] (히스테리시스)


_HOLD_MIN = 1.0      # 최소 정지 시간 [s] (진단 지시 5: 0.5 -> 1.0)


_HOLD_GAIN = 0.05    # 재출발에 필요한 MTC 증가 [s] (지시 4: 0.1 -> 0.05)


def _hold_release(H, d_now, t, recede):
    """정지 상태 H 를 풀어도 되면 True.
    진단 지시 2: 근접 물체가 모두 멀어지는(closing0 < 0) 상태가 _RESUME_GO 초 이상 이어지면 MTC 와 무관하게 True.
    진단 지시 4: 물체가 다시 다가오면(recede False) 그 지속 시간 H["rec_t"] 를 초기화합니다.
    그 밖에는 기존대로 최소 정지 시간(_HOLD_MIN)이 지났고 MTC 가 _HOLD_ON 이상일 때 True."""
    if recede:
        if H.get("rec_t") is None:
            H["rec_t"] = t
    else:
        H["rec_t"] = None
    if H["rec_t"] is not None and t - H["rec_t"] >= _RESUME_GO:
        return True
    if t - H["t0"] < _HOLD_MIN:
        return False
    return d_now >= _HOLD_ON




_SLOW_T = 3.5        # 감속 시작 MTC [s] (지시 1: 2.5 -> 3.5)


_CLOSE_MIN = 0.15    # 위험으로 보는 최소 상대 접근 속도 [m/s]


_V_SLOW = 0.2        # 감속 구간 최저 전진 속도 [m/s] (지시 3: 0.3 -> 0.2)




_RESUME_T = 2.0      # 정지 중 주행 재개 판단의 예측 시간 [s] (지시 5)


_RESUME_GO = 0.5     # 정지 중 근접 물체가 모두 멀어지는(closing0 < 0) 상태가 이만큼 이어지면 재출발 [s] (진단 지시 2·6: 0.8 -> 0.5)


_RESUME_MARGIN = 0.5 # 정지 중 주행 재개: 계속 갈 때의 여유가 멈출 때보다 이만큼 커야 함 [m] (지시 3: 0.2 -> 0.5)




_ESC_CLOSE = 0.05    # 정지 중 회피로 전환하는 물체 접근 속도 closing0 [m/s] (진단 지시 4: 0.1 -> 0.05)


_ESC_NEAR = 2.0      # 정지 중 접근 여부를 볼 물체의 몸체까지 거리 범위 [m]




_HOLD_DIST_INC = 0.5 # 정지 중 가장 가까운 물체 거리가 정지 중 최소값보다 이만큼 늘면 재출발 후보 [m] (진단 지시 6)


_HOLD_INC_T = 0.2    # 그 거리 증가가 이 시간 이상 이어져야 재출발 허용 [s] (진단 지시 6)


def _hold_trend(H, clr_now, t):
    """진단 지시 6: 정지 중 가장 가까운 물체까지 거리(clr_now)가 정지 중 최소값 H["dmin"] 보다 _HOLD_DIST_INC 이상
    늘어난 상태가 _HOLD_INC_T 이상 이어지면 True. 최소 정지 시간(_HOLD_MIN)은 반드시 지나야 합니다."""
    if clr_now >= H["dmin"] + _HOLD_DIST_INC:
        if H["inc_t"] is None:
            H["inc_t"] = t
    else:
        H["inc_t"] = None
    if t - H["t0"] < _HOLD_MIN or H["inc_t"] is None:
        return False
    return t - H["inc_t"] >= _HOLD_INC_T


# ========================= [END EDIT REGION] ===============================
