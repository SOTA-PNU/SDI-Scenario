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
    (1.5, 12.0),     # shift east but still left of box 08
    (1.5, 16.5),     # go north past all remaining boxes
    (GOAL[0], GOAL[1])  # final goal position
]

def controller(t, pose, env):
    """Follow the pre-computed way-points; v (and w while dodging) is overridden by the sensor safety layer.
    진단 지시(시도 24): 0.35 < |e| < 0.7 rad 이면 0.8 m/s 로 전진하며 회전하고(0.6 -> 0.8), |e| >= 0.7 일 때만 제자리 회전합니다."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.done_align = False
    if not _SAFE:
        _safety_reset()
    _sense(t, pose)

    x, y, yaw = pose
    # 1. Goal-alignment phase: finish as soon as we are inside tolerance (with margin) and aligned
    if controller.idx >= len(_WAYPOINTS):
        dg = math.hypot(GOAL[0] - x, GOAL[1] - y)
        if dg > 0.6:
            controller.idx = len(_WAYPOINTS) - 1      # pushed off the goal while dodging: drive back
        else:
            e = _wrap_angle(GOAL[2] - yaw)
            if abs(e) < 0.10 and abs(_SAFE["v_meas"]) < 0.3:
                return 0.0, 0.0, True
            w = _clamp(2.0 * e, -1.2, 1.2)
            v, w = _safe_speed(t, pose, 0.0, w)
            return v, w, False

    # 2. Normal waypoint-following
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

    # 3. Control law (진단 지시(시도 24): 중간 오차에서는 0.8 m/s 로 전진하며 회전, 큰 오차에서만 정지 회전)
    last = controller.idx == len(_WAYPOINTS) - 1
    w = _clamp(7.5 * e, -2.0, 2.0)
    if abs(e) >= 0.7:
        v = 0.0
    elif abs(e) > 0.35:
        v = min(0.8, 1.2 * dist + 0.25) if last else 0.8
    else:
        v = min(1.5, 1.2 * dist + 0.25) if last else min(1.5, 2.5 * dist + 0.5)

    # 4. Sensor override (stop / creep / active dodge forward or backward)
    v, w = _safe_speed(t, pose, v, w)

    if last and dist < 0.3:
        controller.idx += 1

    return v, w, False




_MAP_MARGIN = 0.30


_STATIC_RECTS = np.array(
    [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"], b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"])
     for b in MISSION["boxes"]] + [tuple(o[1:5]) for o in FIXED_OBSTACLES], dtype=np.float64)


_PRED_T = np.arange(0.0, 2.55, 0.1)


_SAFE = {}


def _safety_reset():
    """센서 기반 안전 계층의 상태를 초기화합니다."""
    _SAFE.clear()
    _SAFE.update(hist=[], t_prev=None, raw={}, world={}, mov_w=np.zeros((0, 2)), all_w=np.zeros((0, 2)),
                 tracks=[], t_trk=-1.0, v_meas=0.0, yaw_rate=0.0, blocked=False, have_scan=False,
                 near_hist=[], approach=None, thr_t=None, thr=[], wait=0.0, t_cmd=None, esc_v=None)


def _static_mask(pw, pose=None, yaw_rate=0.0):
    """월드 점 (N, 2) 중 정적 지도(벽, 고정물, 바닥 박스)에 해당하는 점이면 True.
    회전 중에는 스캔 번짐(약 ±0.05 s x 회전속도 x 거리)만큼 점마다 여유를 넓힙니다."""
    x, y = pw[:, 0], pw[:, 1]
    m = np.full(len(pw), _MAP_MARGIN)
    if pose is not None and abs(yaw_rate) > 0.2:
        m = m + 0.06 * abs(yaw_rate) * np.hypot(x - pose[0], y - pose[1])
    st = (x <= WALLS[0] + m) | (x >= WALLS[1] - m) | (y <= WALLS[2] + m) | (y >= WALLS[3] - m)
    r = _STATIC_RECTS
    mm = m[:, None]
    inr = ((x[:, None] >= r[None, :, 0] - mm) & (x[:, None] <= r[None, :, 1] + mm) &
           (y[:, None] >= r[None, :, 2] - mm) & (y[:, None] <= r[None, :, 3] + mm)).any(axis=1)
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


def _update_tracks(t, mov, yaw_rate=0.0):
    """지도에 없는 점 무리를 추적해 위치와 속도를 추정합니다.
    진단 지시 2: 회전 중에도 갱신하되, 번짐 때문에 연결 범위는 넓히고 속도 갱신 이득은 낮춥니다."""
    S = _SAFE
    spin = abs(yaw_rate) > 0.8
    gate = 1.1 if spin else 0.9
    tracks = [tr for tr in S["tracks"] if t - tr["t"] <= 1.2]
    for cl in _clusters(mov):
        if len(cl) < 3:
            continue
        c = cl.mean(axis=0)
        best, bd = None, gate
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
        if spin:
            a *= 0.4
        u = (1.0 - a) * best["u"] + a * raw
        sp = float(np.hypot(u[0], u[1]))
        if sp > 2.5:
            u = u * (2.5 / sp)
        best.update(c=c, u=u, t=t, pts=cl, n=best["n"] + 1, hit=t)
    S["tracks"] = tracks


def _sense(t, pose):
    """앞·뒤 2D 라이다를 읽어 월드 점, 지도에 없는 점, 추적 상태, 접근 속도 추정을 갱신합니다."""
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
    S["mov_w"] = pts[~_static_mask(pts, pose, S["yaw_rate"])]
    if t - S["t_trk"] >= 0.09:                      # 진단 지시 2: 회전 속도 제한 없이 추적
        _update_tracks(t, S["mov_w"], S["yaw_rate"])
        S["t_trk"] = t
    _approach_update(t, pose)
    S["thr_t"] = None


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


def _rear_clear(pose, v=-0.4):
    """진단 지시 2: 후진할 때 뒤쪽이 비었는가. 범위 폭 ±0.40 m, 길이 0.15 + 0.3|v| m (0.2 + 0.4|v| 에서 축소)."""
    a = _SAFE["all_w"]
    if len(a) == 0:
        return True
    loc = _to_robot(a, pose)
    px, py = loc[:, 0], loc[:, 1]
    L = 0.15 + 0.3 * abs(v)
    return not bool(np.any((px < _BODY[0] + 0.05) & (px > _BODY[0] - L) & (np.abs(py) < 0.40)))


def _clearance(t, pose, v_cand):
    """전진 속도 v_cand 로 2.5 s 동안 직진할 때 위협(추적 + 접근 속도 추정)과 몸체 사이의 예상 최소 거리.
    진단 지시 1: 물체는 0.6 m 폭 상자이므로 계산된 최소 거리에서 반폭 _MOVER_HALF (0.3 m) 를 빼 외곽 간 거리로 봅니다.
    속도에 비례하는 안전 여유 _CLR_K * |v_cand| 도 빼서 고속일수록 보수적으로 판단합니다."""
    S = _SAFE
    thr = _threats(t, pose)
    if not thr:
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
    for P, u, n in thr:
        p = P - np.array([x, y])
        if float(np.min(np.hypot(p[:, 0], p[:, 1]))) > 7.0:
            continue
        px = c * p[:, 0] + s * p[:, 1]
        py = -s * p[:, 0] + c * p[:, 1]
        ux = c * u[0] + s * u[1]
        uy = -s * u[0] + c * u[1]
        X = px[None, :] + ux * ts[:, None] - sx[:, None]
        Y = py[None, :] + uy * ts[:, None]
        ddx = np.maximum(np.maximum(_BODY[0] - X, X - _BODY[1]), 0.0)
        ddy = np.maximum(np.maximum(_BODY[2] - Y, Y - _BODY[3]), 0.0)
        best = min(best, float(np.hypot(ddx, ddy).min()))
    return best - _MOVER_HALF - _CLR_K * abs(v_cand)


def _safe_speed(t, pose, v_law, w_law=0.0):
    """경유점 추종이 낸 (v, w) 를 센서 결과로 덮어씁니다.
    속도 상한 cap (최저 _CAP_MIN) 을 추종 속도와 회피 속도 후보에 적용합니다.
    일반 주행 구간의 출력은 _static_guard 로 정적 장애물 하드 스톱·회전 쓸림 검사를 거칩니다.
    - 위험 없음: 정면이 비고 예상 거리(물체 반폭 반영)가 _CLR_GO 이상이면 전진.
      대기가 _WAIT_RELAX_S 를 넘으면 몸통 폭 통로가 빌 때 _RELAX_V 또는 그 절반으로 저속 전진.
    - 위험(옆면 접근 또는 멈춰 있어도 예상 거리 < _ESC_C):
      정적 검사를 critical=True (probe 0.05 m) 로 완화해, 전진·후진 후보 중
      가장 멀어지는 쪽으로 이동합니다 (_ESC_GAIN 이득 이상일 때 채택).
      진단 지시(시도 27) 1: 측면 위협이면 정지의 기준값을 c0 - _SIDE_STOP_PEN (0.35 m) 으로 낮춰 이동을 강하게 선호합니다.
      진단 지시(시도 27) 4: c0 < _SIDE_CLOSE_C 로 매우 가깝고 물체가 다가오면(측면 위협 또는 접근 속도 > _FRONT_CLOSE)
      정지에 _CLOSE_STOP_PEN 을 추가로 줍니다. 이동 후보는 기준값 c0 - pen 과만 비교합니다(별도 하한 필터 없음;
      approaching 이 False 이면 pen = 0 이므로 정지보다 나쁜 후보는 원래 채택되지 않습니다).
      진단 지시(시도 27) 5: 매우 근접 시 전진 후보에 필요한 이득 _CLOSE_FWD_GAIN 을 0.03 m 로 낮춰 유리한 방향 이동을 촉진합니다.
      진단 지시(시도 27) 2·3: 회피 후보는 |v| >= 0.5 m/s 만 쓰고, 저속 회피 허용 이득 _ESC_SURE_GAIN 은 0.05 m 입니다.
      진단 지시(시도 22) 3: 측면 위협이고 측정 속도 S['v_meas'] >= _MOMENTUM_V 이며
      c0 >= _MOMENTUM_C0 일 때만, 현재 속도의 80% 이상인 후보의 예상 거리에 _SIDE_MOMENTUM_BONUS 를 더합니다.
      진단 지시(시도 27) 6: 회피 후보의 정적 검사는 critical=True 를 유지합니다."""
    S = _SAFE
    dt = 0.0 if S["t_cmd"] is None else min(max(t - S["t_cmd"], 0.0), 0.1)
    S["t_cmd"] = t
    if not S["have_scan"] and t < 3.0:
        return 0.0, 0.0
    cap = _speed_cap(t, pose)
    v_law = min(v_law, cap)
    want = v_law > 0.05
    dz = _is_danger(t, pose, max(v_law, S["v_meas"]))
    S["blocked"] = dz["front"]
    front_ok = not dz["front"]
    c0 = _clearance(t, pose, 0.0)

    if not (dz["side"] or c0 < _ESC_C):
        S["esc_v"] = None
        if want:
            if not dz["front"]:
                for f in (1.0, 0.6, 0.3):
                    vf = v_law * f
                    if _clearance(t, pose, vf) >= _CLR_GO:
                        S["wait"] = 0.0
                        return _static_guard(pose, _clamp(vf, -cap, cap), w_law, front_ok)
            if S["wait"] >= _WAIT_RELAX_S and not dz["front_near"]:
                vr = min(v_law, _RELAX_V)
                for vf in (vr, 0.5 * vr):
                    if _clearance(t, pose, vf) >= _CLR_GO:
                        return _static_guard(pose, _clamp(vf, -cap, cap), w_law, front_ok)
            S["wait"] += dt
        return _static_guard(pose, 0.0, w_law, front_ok)

    # 능동 회피: 접근 방향의 반대(전진 또는 후진)로 이동, 후보 속도는 상한 cap 으로 제한
    # 긴급 회피이므로 정적 검사는 critical=True (probe 0.05 m)
    cands = []
    for vc in _ESC_V:
        vc = _clamp(vc, -cap, cap)
        if vc not in cands:
            cands.append(vc)
    close = c0 < _SIDE_CLOSE_C
    ap = S["approach"]
    approaching = dz["side"] or (ap is not None and ap[1] > _FRONT_CLOSE)
    # 진단 지시(시도 27) 1·4: 정지 페널티 강화 (측면 위협 + 매우 근접하고 다가오는 물체면 추가 페널티)
    pen = _SIDE_STOP_PEN if dz["side"] else 0.0
    if close and approaching:
        pen += _CLOSE_STOP_PEN
    best_v = 0.0
    best_c = c0 - pen
    v_now = S["v_meas"]
    momentum = dz["side"] and v_now >= _MOMENTUM_V and c0 >= _MOMENTUM_C0
    for vc in cands:
        if not _static_free(pose, vc, critical=True):
            continue
        cv = _clearance(t, pose, vc) + (0.05 if vc == S["esc_v"] else 0.0)
        if momentum and vc >= v_now * 0.8:
            cv += _SIDE_MOMENTUM_BONUS
        need = _CLOSE_FWD_GAIN if (close and vc > 0.0) else _ESC_GAIN
        if cv > best_c + need:
            best_v, best_c = vc, cv
    # 아주 느린 회피이면서 이득도 크지 않으면 정지 (후보는 모두 |v| >= _ESC_MIN_V 이므로 실제로는 거의 쓰이지 않음)
    if best_v != 0.0 and abs(best_v) < _ESC_MIN_V and best_c < c0 + _ESC_SURE_GAIN:
        best_v = 0.0
    if best_v == 0.0:
        S["esc_v"] = None
        return 0.0, w_law
    S["esc_v"] = best_v
    return _clamp(best_v, -cap, cap), 0.0            # 직진 회피(예측과 같은 궤적, 뒤쪽 몸체 휘두름 방지)




_ESC_C = 0.45            # 진단 지시(시도 24): 멈춰 있어도 예상 최소거리가 이보다 작으면 능동 회피 (0.6 -> 0.45 m), 불필요한 회피 진입을 줄임


_SIDE_GAP = 0.6          # 진단 지시 4: 몸체 옆면에서 이 거리 안으로 다가오면 위험 (0.8 -> 0.6 m)


_SIDE_WARN = 2.0         # 진단 지시 3: 옆면 감시 범위 (1.5 -> 2.0 m)


_WAIT_RELAX_S = 0.3      # 진단 지시(시도 24): 대기가 이보다 길면 저속 전진 시도 (0.5 -> 0.3 s), 정체 후 더 빨리 재출발


_ESC_V = (-0.8, -0.5, 0.5, 1.0, 1.5)   # 진단 지시(시도 27) 3: 너무 느린 ±0.3 m/s 후보 제거, 확실히 경로를 벗어나는 속도만 사용


def _approach_update(t, pose):
    """진단 지시 5: 가장 가까운 '지도에 없는 점' 까지 거리의 최근 2~3 스캔 변화로 물체의 접근 속도를 추정합니다.
    로봇 자신의 전진 성분은 뺍니다. 제자리 회전은 이 거리를 바꾸지 않으므로 회전 직후에도 바로 쓸 수 있습니다."""
    S = _SAFE
    m = S["mov_w"]
    h = S["near_hist"]
    if len(m) == 0:
        h.clear()
        S["approach"] = None
        return
    d = np.hypot(m[:, 0] - pose[0], m[:, 1] - pose[1])
    k = int(np.argmin(d))
    dk = float(d[k])
    p = m[k].copy()
    if dk > 4.0:
        h.clear()
        S["approach"] = None
        return
    h.append((t, dk, p))
    del h[:-3]
    if len(h) < 2:
        return
    t0, d0, p0 = h[0]
    if math.hypot(p[0] - p0[0], p[1] - p0[1]) > 0.8:   # 가장 가까운 물체가 바뀜
        del h[:-1]
        S["approach"] = None
        return
    if t - t0 < 0.04:
        return
    b = math.atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
    rate = (d0 - dk) / (t - t0) - S["v_meas"] * math.cos(b)
    old = S["approach"]
    if old is not None and math.hypot(old[0][0] - p[0], old[0][1] - p[1]) < 0.8:
        rate = 0.5 * old[1] + 0.5 * rate
    S["approach"] = (p, rate)


def _threats(t, pose):
    """현재 시각으로 옮긴 위협 목록 [(월드 점 (N, 2), 속도 (2,), 추적 횟수)].
    추적이 덜 된 가장 가까운 물체는 거리 변화로 얻은 접근 속도로 로봇 쪽을 향해 움직인다고 봅니다."""
    S = _SAFE
    if S["thr_t"] == t:
        return S["thr"]
    out = [(tr["pts"] + tr["u"] * (t - tr["t"]), tr["u"], tr["n"]) for tr in S["tracks"]]
    ap = S["approach"]
    if ap is not None and ap[1] > 0.15:
        p0 = ap[0]
        covered = any(n >= 2 and float(np.min(np.hypot(P[:, 0] - p0[0], P[:, 1] - p0[1]))) < 0.6
                      for P, u, n in out)
        if not covered:
            mv = S["mov_w"]
            near = mv[np.hypot(mv[:, 0] - p0[0], mv[:, 1] - p0[1]) < 0.8] if len(mv) else mv
            if len(near) == 0:
                near = p0[None, :]
            dv = np.array([pose[0] - p0[0], pose[1] - p0[1]])
            nn = float(np.hypot(dv[0], dv[1]))
            if nn > 1e-6:
                out.append((near, dv / nn * min(ap[1], 1.5), 0))
    S["thr_t"] = t
    S["thr"] = out
    return out


def _is_danger(t, pose, v_ref):
    """정면/옆면 위험 판단.
    front      진단 지시 1·2: 지도에 없는 점이 정면 ±30° 또는 폭 ±0.6 m 통로 안, 앞면에서 2.0 m 이내에 있고,
               그 물체가 로봇 쪽으로 다가오거나(접근 속도 양수) 아직 추적되지 않았을 때만 True
    front_near 몸통 폭(±0.45 m) 통로 안, 앞면에서 1.2 m 이내 (오래 기다린 뒤 저속 전진 판단용)
    side       옆면 _SIDE_GAP 안에서 _SIDE_CLOSE_NEAR 이상 접근하거나,
               _SIDE_WARN 안에서 _SIDE_CLOSE_FAR 이상으로 _SIDE_HORIZON 안에 닿을 속도로 접근"""
    S = _SAFE
    out = {"front": False, "front_near": False, "side": False}
    m = S["mov_w"]
    if len(m):
        loc = _to_robot(m, pose)
        px, py = loc[:, 0], loc[:, 1]
        L = max(2.0, 0.6 + 1.2 * max(v_ref, 0.0)) + (0.3 if S["blocked"] else 0.0)
        ahead = (px > _BODY[1] - 0.05) & (px < _BODY[1] + L)
        cone = np.abs(np.arctan2(py, px)) <= math.radians(_FRONT_CONE_DEG)
        corr = np.abs(py) <= _FRONT_CORR
        zone = ahead & (cone | corr)
        if np.any(zone):
            out["front"] = _front_closing(t, pose, m[zone])
        near = (px > _BODY[1] - 0.05) & (px < _BODY[1] + 1.2) & (np.abs(py) <= 0.45)
        out["front_near"] = bool(np.any(near))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for P, u, n in _threats(t, pose):
        loc = _to_robot(P, pose)
        px, py = loc[:, 0], loc[:, 1]
        gap = np.maximum(_BODY[2] - py, py - _BODY[3])
        zone = (px >= _BODY[0] - 0.3) & (px <= _BODY[1] + 0.3) & (gap > -0.02) & (gap <= _SIDE_WARN)
        if not np.any(zone):
            continue
        idx = np.flatnonzero(zone)
        k = int(idx[int(np.argmin(gap[idx]))])
        g = max(float(gap[k]), 0.0)
        uy = -s * u[0] + c * u[1]
        closing = -uy if py[k] > 0 else uy
        if ((g <= _SIDE_GAP and closing >= _SIDE_CLOSE_NEAR) or
                (closing >= _SIDE_CLOSE_FAR and g <= _SIDE_HORIZON * closing)):
            out["side"] = True
            break
    return out


def _static_free(pose, v, critical=False):
    """회피 속도 v 로 직진해도 정적 장애물(라이다 점 + 정적 지도)에 닿지 않는가.
    일반: probe 거리 전진 0.15 + 0.4v, 후진 0.15 + 0.3|v|.
    진단 지시 1: critical=True(긴급 회피)이면 probe 거리 L 을 _CRIT_PROBE (0.05 m) 로 줄여
    정적 마진을 조금 희생하더라도 다가오는 물체에서 멀어지는 미세한 움직임을 허용합니다."""
    S = _SAFE
    a = S["all_w"]
    loc = _to_robot(a, pose) if len(a) else None
    if v < 0:
        L = _CRIT_PROBE if critical else 0.15 + 0.3 * abs(v)
        if loc is not None and np.any((loc[:, 0] < _BODY[0] + 0.05) & (loc[:, 0] > _BODY[0] - L) &
                                      (np.abs(loc[:, 1]) < 0.40)):
            return False
        ex = _BODY[0] - L
    else:
        L = _CRIT_PROBE if critical else 0.15 + 0.4 * v
        if loc is not None and np.any((loc[:, 0] > _BODY[1] - 0.05) & (loc[:, 0] < _BODY[1] + L) &
                                      (np.abs(loc[:, 1]) < 0.40)):
            return False
        ex = _BODY[1] + L
    probe = np.array([[ex, _BODY[2]], [ex, 0.0], [ex, _BODY[3]]])
    return not bool(_static_mask(to_world(probe, pose)[:, :2]).any())




_FRONT_CONE_DEG = 30.0   # 진단 지시 1: 정면 감지 원뿔 반각 (45 -> 30 deg)


_FRONT_CORR = 0.6        # 진단 지시 1: 정면 통로 반폭 (0.8 -> 0.6 m)


_CLR_K = 0.10            # 진단 지시 3: 예상 최소거리에서 빼는 속도 비례 여유 (0.15 -> 0.10 m per m/s)


_CAP_HYST = 0.2          # 속도 상한을 올릴 때만 쓰는 거리 히스테리시스 [m]


def _nearest_mover_dist(t, pose):
    """진단 지시 2: 지도에 없는 물체 중 가장 가까운 것까지의 거리(로봇 중심 기준).
    낱점 잡음은 무시합니다(0.3 m 안에 점이 3개 이상 모인 점만 인정). 최근 0.6 s 안에 본 추적은 예측 위치로 함께 봅니다."""
    S = _SAFE
    d = 99.0
    m = S["mov_w"]
    if len(m):
        dd = np.hypot(m[:, 0] - pose[0], m[:, 1] - pose[1])
        for k in np.argsort(dd)[:60]:
            nb = int(np.count_nonzero(np.hypot(m[:, 0] - m[k, 0], m[:, 1] - m[k, 1]) < 0.3))
            if nb >= 3:
                d = float(dd[k])
                break
    for tr in S["tracks"]:
        if t - tr["t"] <= 0.6 and tr["n"] >= 1:
            P = tr["pts"] + tr["u"] * (t - tr["t"])
            if len(P):
                d = min(d, float(np.min(np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]))))
    return d


def _speed_cap(t, pose):
    """진단 지시 5: d_min 에 따른 속도 상한. d_min > 3 m: 1.5 m/s, 그 밖: _CAP_MIN (0.8 m/s).
    상한을 올릴 때만 경계에 _CAP_HYST 를 더해 경계에서 떨지 않게 합니다."""
    S = _SAFE
    d = _nearest_mover_dist(t, pose)
    prev = S.get("cap", 1.5)
    need = 3.0 + (_CAP_HYST if prev < 1.5 else 0.0)
    cap = 1.5 if d > need else _CAP_MIN
    S["cap"] = cap
    S["d_min"] = d
    return cap




_SIDE_CLOSE_NEAR = 0.05  # 진단 지시 6: 옆면 _SIDE_GAP 안 접근 속도 임계값 (0.10 -> 0.05 m/s)


_SIDE_CLOSE_FAR = 0.08   # 진단 지시 6: 옆면 _SIDE_WARN 안 접근 속도 임계값 (0.15 -> 0.08 m/s)


_SIDE_HORIZON = 2.5      # 진단 지시 6: 이 시간 안에 옆면에 닿을 속도면 위험 (2.0 -> 2.5 s)


_ESC_GAIN = 0.02         # 진단 지시(시도 30) 2: 회피 속도 채택에 필요한 예상 거리 이득 (0.05 -> 0.02 m), 정지보다 조금만 나아도 이동 채택


_CAP_MIN = 1.0           # 진단 지시(시도 18): 속도 상한의 최저값 (0.8 -> 1.0 m/s)


_STATIC_STOP = 0.40      # 진단 지시 4: 정적 장애물 하드 스톱 거리 (앞면 기준, + 0.3 x 현재 속도)


_SWING_R = 0.82          # 진단 지시 4: 제자리 회전 시 몸체가 쓸고 지나가는 반경 (뒤 모서리 0.78 m + 여유)


def _map_hit(pw, margin=0.0):
    """월드 점 (N, 2) 이 정적 지도(벽 바깥, 바닥 박스, 고정물)에 margin 여유 안으로 들어가면 True."""
    x, y = pw[:, 0], pw[:, 1]
    st = (x <= WALLS[0] + margin) | (x >= WALLS[1] - margin) | (y <= WALLS[2] + margin) | (y >= WALLS[3] - margin)
    r = _STATIC_RECTS
    inr = ((x[:, None] >= r[None, :, 0] - margin) & (x[:, None] <= r[None, :, 1] + margin) &
           (y[:, None] >= r[None, :, 2] - margin) & (y[:, None] <= r[None, :, 3] + margin)).any(axis=1)
    return st | inr


def _static_ahead(pose, gap):
    """진단 지시 4: 앞면에서 gap 안(몸통 폭 통로)에 라이다 점이나 정적 지도 장애물이 있는가."""
    a = _SAFE["all_w"]
    if len(a):
        loc = _to_robot(a, pose)
        if np.any((loc[:, 0] > _BODY[1] - 0.05) & (loc[:, 0] < _BODY[1] + gap) & (np.abs(loc[:, 1]) < 0.40)):
            return True
    xs = _BODY[1] + gap * np.array([0.35, 0.7, 1.0])
    probe = np.array([[px, py] for px in xs for py in (_BODY[2] - 0.03, 0.0, _BODY[3] + 0.03)])
    return bool(_map_hit(to_world(probe, pose)[:, :2]).any())


def _swing_hit(pose, w):
    """진단 지시 4: 회전 속도 w 로 제자리 회전할 때 몸체가 쓸고 지나갈 반경(_SWING_R) 안,
    회전 방향 쪽에 있는 가장 가까운 라이다 점(로봇 기준). 없으면 None."""
    a = _SAFE["all_w"]
    if len(a) == 0:
        return None
    loc = _to_robot(a, pose)
    px, py = loc[:, 0], loc[:, 1]
    r = np.hypot(px, py)
    inb = (px >= _BODY[0] - 0.02) & (px <= _BODY[1] + 0.02) & (np.abs(py) <= _BODY[3] + 0.02)
    side = np.where(px < 0.0, py * w <= 0.0, py * w >= 0.0)   # 반시계 회전: 뒤쪽은 -y, 앞쪽은 +y 로 쓸림
    hit = (r < _SWING_R) & ~inb & side
    if not np.any(hit):
        return None
    idx = np.flatnonzero(hit)
    return loc[int(idx[int(np.argmin(r[idx]))])]


def _static_guard(pose, v, w, front_ok):
    """진단 지시 4: 일반 주행의 하드 스톱.
    - 전진 중 앞면에서 _STATIC_STOP + 0.3 x 속도 안에 장애물이 있으면 v = 0.
    - 제자리 회전이 장애물을 쓸 것 같으면 회전을 멈추고, 그 점의 반대쪽으로 0.25 m/s 로 빠져나갑니다."""
    S = _SAFE
    if v > 0.0 and _static_ahead(pose, _STATIC_STOP + 0.3 * max(S["v_meas"], 0.0)):
        v = 0.0
    if v == 0.0 and abs(w) > 0.3:
        hp = _swing_hit(pose, w)
        if hp is not None:
            if hp[0] < -0.2:
                if front_ok and not _static_ahead(pose, _STATIC_STOP):
                    return 0.25, 0.0
            elif _static_free(pose, -0.25):
                return -0.25, 0.0
    return v, w




_MOVER_HALF = 0.30       # 진단 지시 1: 움직이는 상자(0.6 x 0.6 m)의 반폭, 예상 최소거리에서 뺌


_CLR_GO = 0.35           # 진단 지시(시도 24): 일반 주행에서 전진을 허용하는 예상 최소거리 (0.5 -> 0.35 m), 불필요한 정지 시점을 늦춤




_RELAX_V = 1.2           # 진단 지시(시도 24): 오래 기다린 뒤 전진 속도 (1.0 -> 1.2 m/s, 실제 값은 cap 으로 제한), 돌파력 강화


_FRONT_CLOSE = 0.05      # 진단 지시 2: '접근 속도 양수' 판정 임계값 (속도 추정 잡음 여유) [m/s]


def _front_closing(t, pose, pts):
    """_is_danger['front'] 판정: 정면 범위 안의 '지도에 없는 점' pts (월드 (N, 2)) 가 실제 위험인가.
    - 가장 가까운 물체의 접근 속도(_approach rate)가 양수(> _FRONT_CLOSE)이고 그 점이 pts 근처면 위험.
    - 추적된(n >= 2, 최근 0.6 s) 물체의 속도가 로봇 쪽 성분 > _FRONT_CLOSE 이면 위험.
    - 진단 지시 3: 어떤 추적에도 속하지 않는 점은 로봇 중심에서 _FRONT_UNTRACKED_R (1.0 m) 이내에 있을 때만 위험.
    - 그 밖은 정면 막힘으로 보지 않습니다(예상 경로 충돌은 _clearance, 근접 정지는 _static_guard 가 막습니다)."""
    S = _SAFE
    if len(pts) == 0:
        return False
    ap = S["approach"]
    if ap is not None and ap[1] > _FRONT_CLOSE:
        if float(np.min(np.hypot(pts[:, 0] - ap[0][0], pts[:, 1] - ap[0][1]))) < 0.8:
            return True
    covered = np.zeros(len(pts), dtype=bool)
    for tr in S["tracks"]:
        if tr["n"] < 2 or t - tr["t"] > 0.6:
            continue
        P = tr["pts"] + tr["u"] * (t - tr["t"])
        if len(P) == 0:
            continue
        dmin = np.min(np.hypot(pts[:, 0, None] - P[None, :, 0], pts[:, 1, None] - P[None, :, 1]), axis=1)
        hit = dmin < 0.5
        if not np.any(hit):
            continue
        cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
        rx, ry = pose[0] - cx, pose[1] - cy
        nn = math.hypot(rx, ry)
        closing = (tr["u"][0] * rx + tr["u"][1] * ry) / nn if nn > 1e-6 else 0.0
        if closing > _FRONT_CLOSE:
            return True
        covered |= hit
    unc = ~covered
    if not np.any(unc):
        return False
    du = np.hypot(pts[unc, 0] - pose[0], pts[unc, 1] - pose[1])
    return bool(np.any(du <= _FRONT_UNTRACKED_R))




_CRIT_PROBE = 0.05       # 진단 지시 1: 긴급 회피(critical) 때 정적 장애물 probe 거리 [m]


_FRONT_UNTRACKED_R = 0.6  # 진단 지시(시도 14): 추적 안 된 점을 정면 위험으로 보는 최대 거리(로봇 중심 기준) (1.0 -> 0.6 m)




_SIDE_STOP_PEN = 0.20    # 진단 지시(시도 30) 1: 측면 위협 시 '정지' 에 주는 페널티 [m] (0.05 -> 0.20), 측면 접근 물체 앞에서 멈춰 서지 않고 이동 후보를 강하게 선호




_ESC_MIN_V = 0.50        # 진단 지시(시도 27) 3: 능동 회피로 채택할 최소 속도 [m/s] (0.30 -> 0.50)


_ESC_SURE_GAIN = 0.10    # 진단 지시(시도 28) 2: 저속 회피를 허용하는 데 필요한 예상 거리 이득 [m] (0.05 -> 0.10)




_SIDE_MOMENTUM_BONUS = 0.3   # 진단 지시(시도 20): 측면 위협 + 고속 주행 중 현재 속도 유지 후보에 주는 예상 거리 보너스 [m] (0.2 -> 0.3), 교차 지점을 빠르게 통과하도록 유도 (_MOMENTUM_C0 = 0.6 은 유지)




_MOMENTUM_C0 = 0.6       # 진단 지시(시도 17): 측면 momentum 보너스를 주려면 현재 예상 최소거리 c0 가 이 값 이상이어야 함 [m] (0.4 -> 0.6, 시도 14 값으로 환원)




_CAP_BEHIND = 0.3        # 로봇 뒷면보다 이 거리 이상 뒤에 있고 따라오지 않는 물체는 속도 상한 계산에서 뺌 [m]




_MOMENTUM_V = 0.30       # 진단 지시(시도 22) 3: 측면 momentum 보너스를 주는 최소 측정 속도 [m/s] (0.5 -> 0.3)




_SIDE_CLOSE_C = 0.5      # 진단 지시(시도 25): 예상 최소거리 c0 가 이보다 작으면 '매우 근접' (측면 정지 페널티 0, 전진 후보 이득 강화) [m]


_CLOSE_FWD_GAIN = 0.10   # 진단 지시(시도 30) 3: 매우 근접 시 전진 후보 채택에 필요한 이득 [m] (0.15 -> 0.10), 근접 상황에서도 전진 회피 가능성 확보




_CLOSE_STOP_PEN = 0.0    # 진단 지시(시도 28) 1: 매우 근접 + 접근 시 정지 추가 페널티 제거 [m] (0.20 -> 0.0)


_CLOSE_FLOOR_TOL = 0.05  # 진단 지시(시도 27) 4: 매우 근접 시 정지보다 예상 거리가 이만큼 넘게 나빠지는 이동 후보는 버림 [m]


# ========================= [END EDIT REGION] ===============================
