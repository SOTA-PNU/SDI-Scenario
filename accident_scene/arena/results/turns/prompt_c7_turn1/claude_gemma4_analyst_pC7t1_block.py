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
    """Follow the pre-computed way-points; forward speed is overridden by the lidar safety layer."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.done_align = False
    if not _SAFE:
        _safety_reset()
    _sense(t, pose)

    x, y, yaw = pose
    # 1. Goal-alignment phase
    if controller.idx >= len(_WAYPOINTS):
        if math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.5:
            controller.idx = len(_WAYPOINTS) - 1      # pushed off the goal while dodging: drive back
        else:
            e = _wrap_angle(GOAL[2] - yaw)
            w = _clamp(1.5 * e, -0.6, 0.6)
            v = _safe_speed(t, pose, 0.0)
            if abs(e) < 0.05 and v == 0.0:
                return 0.0, 0.0, True
            return v, w, False

    # 2. Normal waypoint-following (unchanged)
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

    # 3. Control law (unchanged)
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        v = min(1.5, 2.5 * dist + 0.5)
        w = _clamp(7.5 * e, -2.0, 2.0)

    # 4. Sensor override of v only (stop-and-wait, back off if a mover would still hit)
    v = _safe_speed(t, pose, v)

    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1

    return v, w, False




_MAP_MARGIN = 0.30


_STATIC_RECTS = np.array(
    [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"], b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"])
     for b in MISSION["boxes"]] + [tuple(o[1:5]) for o in FIXED_OBSTACLES], dtype=np.float64)


_PRED_T = np.arange(0.0, 2.05, 0.1)


_SAFE = {}


def _safety_reset():
    """센서 기반 안전 계층의 상태를 초기화합니다."""
    _SAFE.clear()
    _SAFE.update(hist=[], t_prev=None, raw={}, world={}, mov_w=np.zeros((0, 2)), all_w=np.zeros((0, 2)),
                 tracks=[], t_trk=-1.0, v_meas=0.0, yaw_rate=0.0, blocked=False, have_scan=False)


def _static_mask(pw):
    """월드 점 (N, 2) 중 정적 지도(벽, 고정물, 바닥 박스)에 해당하는 점이면 True."""
    x, y = pw[:, 0], pw[:, 1]
    m = _MAP_MARGIN
    st = (x <= WALLS[0] + m) | (x >= WALLS[1] - m) | (y <= WALLS[2] + m) | (y >= WALLS[3] - m)
    r = _STATIC_RECTS
    inr = ((x[:, None] >= r[None, :, 0] - m) & (x[:, None] <= r[None, :, 1] + m) &
           (y[:, None] >= r[None, :, 2] - m) & (y[:, None] <= r[None, :, 3] + m)).any(axis=1)
    return st | inr


def _to_robot(pw, pose):
    """월드 점 (N, 2) 를 현재 로봇 기준 좌표로 바꿉니다."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pw[:, 0] - x, pw[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def _clusters(p, cell=0.35):
    """점들을 격자 이웃 연결로 묶습니다."""
    if len(p) == 0:
        return []
    keys = np.floor(p / cell).astype(int).tolist()
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault((k[0], k[1]), []).append(i)
    seen, out = set(), []
    for c0 in cells:
        if c0 in seen:
            continue
        seen.add(c0)
        stack, idx = [c0], []
        while stack:
            cc = stack.pop()
            idx.extend(cells[cc])
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    nb = (cc[0] + ddx, cc[1] + ddy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(p[idx])
    return out


def _update_tracks(t, mov):
    """지도에 없는 점 무리를 추적해 위치와 속도를 추정합니다."""
    S = _SAFE
    tracks = [tr for tr in S["tracks"] if t - tr["t"] <= 1.2]
    for cl in _clusters(mov):
        if len(cl) < 3:
            continue
        c = cl.mean(axis=0)
        best, bd = None, 0.9
        for tr in tracks:
            if tr["hit"] == t:
                continue
            pred = tr["c"] + tr["u"] * (t - tr["t"])
            dd = float(np.hypot(c[0] - pred[0], c[1] - pred[1]))
            if dd < bd:
                best, bd = tr, dd
        if best is None:
            tracks.append({"c": c, "u": np.zeros(2), "t": t, "pts": cl, "n": 0, "hit": t})
            continue
        dt = max(t - best["t"], 1e-3)
        raw = (c - best["c"]) / dt
        a = 0.6 if best["n"] == 0 else 0.35
        u = (1.0 - a) * best["u"] + a * raw
        sp = float(np.hypot(u[0], u[1]))
        if sp > 2.5:
            u = u * (2.5 / sp)
        best.update(c=c, u=u, t=t, pts=cl, n=best["n"] + 1, hit=t)
    S["tracks"] = tracks


def _sense(t, pose):
    """앞·뒤 2D 라이다를 읽어 월드 점, 지도에 없는 점, 추적 상태를 갱신합니다."""
    S = _SAFE
    h = S["hist"]
    if h and S["t_prev"] is not None and t - S["t_prev"] > 1e-6:
        dt = t - S["t_prev"]
        px, py, pyaw = h[-1]
        vm = ((pose[0] - px) * math.cos(pose[2]) + (pose[1] - py) * math.sin(pose[2])) / dt
        yr = _wrap_angle(pose[2] - pyaw) / dt
        S["v_meas"] = 0.7 * S["v_meas"] + 0.3 * vm
        S["yaw_rate"] = 0.7 * S["yaw_rate"] + 0.3 * yr
    S["t_prev"] = t
    h.append(tuple(pose))
    if len(h) > 8:
        del h[0]
    lag = h[max(0, len(h) - 4)]          # 스캔 한 바퀴(0.1 s)의 중간 시점 자세로 변환
    new = False
    for which in ("front", "back"):
        try:
            p = lidar2d_points(which)
        except Exception:
            continue
        if len(p) == 0 or p is S["raw"].get(which):
            continue
        S["raw"][which] = p
        S["world"][which] = to_world(p, lag)[:, :2]
        new = True
        if which == "front":
            S["have_scan"] = True
    if not new:
        return
    pts = np.concatenate(list(S["world"].values()))
    d2 = (pts[:, 0] - pose[0]) ** 2 + (pts[:, 1] - pose[1]) ** 2
    pts = pts[d2 < 64.0]
    S["all_w"] = pts
    S["mov_w"] = pts[~_static_mask(pts)]
    if abs(S["yaw_rate"]) < 0.8 and t - S["t_trk"] >= 0.09:
        _update_tracks(t, S["mov_w"])
        S["t_trk"] = t


def _front_blocked(pose, v_ref):
    """진단 지시 3: 지도에 없는 점이 정면 ±30° 또는 몸통 폭 통로 안, 앞면에서 2.0 m 이내에 있는가."""
    S = _SAFE
    m = S["mov_w"]
    if len(m) == 0:
        return False
    loc = _to_robot(m, pose)
    px, py = loc[:, 0], loc[:, 1]
    L = max(2.0, 0.6 + 1.2 * max(v_ref, 0.0)) + (0.3 if S["blocked"] else 0.0)
    ahead = (px > _BODY[1] - 0.05) & (px < _BODY[1] + L)
    cone = np.abs(np.arctan2(py, px)) <= math.radians(30.0)
    corr = np.abs(py) <= 0.6
    return bool(np.any(ahead & (cone | corr)))


def _rear_clear(pose):
    """후진할 때 뒤쪽 0.6 m 안에 아무 점(정적 포함)도 없는가."""
    a = _SAFE["all_w"]
    if len(a) == 0:
        return True
    loc = _to_robot(a, pose)
    px, py = loc[:, 0], loc[:, 1]
    return not bool(np.any((px < _BODY[0] + 0.05) & (px > _BODY[0] - 0.6) & (np.abs(py) < 0.5)))


def _clearance(t, pose, v_cand):
    """전진 속도 v_cand 로 2 s 동안 갈 때 추적 중인 물체와 몸체 사이의 예상 최소 거리."""
    S = _SAFE
    if not S["tracks"]:
        return 9.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    ts = _PRED_T
    v0, acc = S["v_meas"], 1.2
    if v_cand >= v0:
        vs = np.minimum(v_cand, v0 + acc * ts)
    else:
        vs = np.maximum(v_cand, v0 - acc * ts)
    sx = np.concatenate([[0.0], np.cumsum(0.5 * (vs[1:] + vs[:-1]) * np.diff(ts))])
    best = 9.0
    for tr in S["tracks"]:
        u = tr["u"]
        p = tr["pts"] + u * (t - tr["t"]) - np.array([x, y])
        px = c * p[:, 0] + s * p[:, 1]
        py = -s * p[:, 0] + c * p[:, 1]
        ux = c * u[0] + s * u[1]
        uy = -s * u[0] + c * u[1]
        X = px[None, :] + ux * ts[:, None] - sx[:, None]
        Y = py[None, :] + uy * ts[:, None]
        ddx = np.maximum(np.maximum(_BODY[0] - X, X - _BODY[1]), 0.0)
        ddy = np.maximum(np.maximum(_BODY[2] - Y, Y - _BODY[3]), 0.0)
        best = min(best, float(np.hypot(ddx, ddy).min()))
    return best


def _safe_speed(t, pose, v_law):
    """경유점 추종이 낸 v 를 센서 결과로 덮어씁니다(진단 지시 4~6). w 는 건드리지 않습니다."""
    S = _SAFE
    if not S["have_scan"] and t < 3.0:
        return 0.0
    blocked = _front_blocked(pose, max(v_law, S["v_meas"]))
    S["blocked"] = blocked
    fwd_ok = v_law > 0.05 and not blocked
    if fwd_ok:
        for f in (1.0, 0.6, 0.3):
            if _clearance(t, pose, v_law * f) >= 0.40:
                return v_law * f
    c0 = _clearance(t, pose, 0.0)
    if c0 >= 0.30:
        return 0.0                                   # 정지 후 대기
    best_v, best_c = 0.0, c0                         # 멈춰 있어도 닿을 상황: 가장 멀어지는 쪽을 고름
    if _rear_clear(pose):
        for vb in (-0.4, -0.8):
            cb = _clearance(t, pose, vb)
            if cb > best_c + 0.05:
                best_v, best_c = vb, cb
    if fwd_ok:
        cf = _clearance(t, pose, v_law)
        if cf > best_c + 0.05:
            best_v, best_c = v_law, cf
    return best_v


# ========================= [END EDIT REGION] ===============================
