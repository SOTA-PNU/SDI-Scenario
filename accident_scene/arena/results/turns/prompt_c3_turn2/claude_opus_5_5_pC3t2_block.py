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
    """경유점 추종 + 물체 추적·등속 예측 기반 속도 선택(후보 v∈{v_wp, 0.5, 0.3, 0, -0.3}, 제자리 회전 시 w∈{w, 0})."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.ready = 0

    x, y, yaw = pose
    n = len(_WAYPOINTS)

    tracks, raw, stat, ok = _track_movers(pose)
    controller.ready = controller.ready + 1 if ok else 0
    if controller.ready < 12:          # 라이다가 들어오고 트랙이 잡힐 때까지 대기
        return 0.0, 0.0, False

    # ---- 현재 시점으로 예측한 물체 (x, y, vx, vy, 반경 0.45, 요구 간격 0.35) : 모든 트랙 동일 ----
    step = _TRK["step"]
    movers = []
    for tr in tracks:
        age = (step - tr["seen"]) / 60.0
        vx, vy = tr["v"]
        movers.append((tr["p"][0] + vx * age, tr["p"][1] + vy * age, vx, vy, _MOVER_R, _MOVER_GAP))

    # ---- 경유점 진행 ----
    if controller.idx >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.45:
        controller.idx = n - 1         # 피하느라 밀려났으면 GOAL 로 복귀
    if controller.idx < n:
        _, _, controller.idx = _pursuit_cmd(x, y, yaw, controller.idx, 0.8)
    align = controller.idx >= n

    rear_blocked = raw.shape[0] > 0 and bool(
        ((raw[:, 0] < _BODY[0]) & (raw[:, 0] > _BODY[0] - 0.45) & (np.abs(raw[:, 1]) < 0.45)).any())

    # ---- GOAL 정렬 완료: 정지한 채 2.5 s 동안 간격 0.35 m 가 지켜질 때만 끝냄 ----
    if align and abs(_wrap_angle(GOAL[2] - yaw)) < 0.05:
        cmin, _ = _rollout(pose, ("fix", 0.0, 0.0), controller.idx, movers)
        if cmin >= 0.0:
            return 0.0, 0.0, True

    # ---- 후보: 경유점 추종 v_wp/0.5/0.3 (제자리 회전 중이면 (0, w)), 정지 (0, 0), 후진 (-0.3, 0) ----
    if align:
        prim = [("align", 0.0, 0.0), ("fix", 0.0, 0.0), ("fix", -0.3, 0.0)]
    else:
        prim = [("path", 0.8, 0.0), ("path", 0.5, 0.0), ("path", 0.3, 0.0),
                ("fix", 0.0, 0.0), ("fix", -0.3, 0.0)]

    chosen = None
    best = None
    for cand in prim:
        kind, v, w = cand
        if v < 0.0 and rear_blocked:
            continue
        cmin, traj = _rollout(pose, cand, controller.idx, movers)
        if kind == "fix" and v < 0.0 and not _static_ok(traj, stat):
            continue
        if cmin >= 0.0:                # 간격 0.35 m 를 지키는 가장 빠른 후보
            chosen = cand
            break
        if best is None or cmin > best[0] + 1e-6:
            best = (cmin, cand)
    if chosen is None:                 # 모두 위험하면 prim 전체에서 최소 간격이 가장 큰 후보
        chosen = best[1] if best is not None else ("fix", 0.0, 0.0)

    kind, v, w = chosen
    if kind == "path":
        v, w, _ = _pursuit_cmd(x, y, yaw, controller.idx, v)
    elif kind == "align":
        e = _wrap_angle(GOAL[2] - yaw)
        v, w = 0.0, (0.0 if abs(e) < 0.05 else _clamp(1.5 * e, -0.6, 0.6))
    return float(v), float(w), False




_STATIC_RECTS = np.array(
    [(b["x"] - b["width"] / 2.0 - 0.2, b["x"] + b["width"] / 2.0 + 0.2,
      b["y"] - b["depth"] / 2.0 - 0.2, b["y"] + b["depth"] / 2.0 + 0.2) for b in MISSION["boxes"]]
    + [(x0 - 0.2, x1 + 0.2, y0 - 0.2, y1 + 0.2) for _, x0, x1, y0, y1 in FIXED_OBSTACLES],
    dtype=np.float64)


def _body_dist(p):
    """로봇 기준 점 (N, 2) 각각의 몸체 사각형 _BODY 바깥면까지 거리."""
    dx = np.maximum(np.maximum(_BODY[0] - p[:, 0], p[:, 0] - _BODY[1]), 0.0)
    dy = np.maximum(np.maximum(_BODY[2] - p[:, 1], p[:, 1] - _BODY[3]), 0.0)
    return np.hypot(dx, dy)


def _movers_robot_frame(pose):
    """앞·뒤 2D 라이다 점에서 정적 지도(WALLS 띠·바깥, 0.2 m 부풀린 boxes/FIXED_OBSTACLES)를 빼고,
    0.3 m 안에 3점 이상 모인 점만 움직이는 물체로 남겨 로봇 기준 (K, 2) 로 돌려줍니다.
    반환: (물체 점, 3 m 안의 전체 라이다 점(로봇 기준), 스캔 수신 여부)."""
    scans = [lidar2d_points("front"), lidar2d_points("back")]
    scans = [s for s in scans if s is not None and len(s) > 0]
    empty = np.zeros((0, 2))
    if not scans:
        return empty, empty, False
    rob = np.concatenate(scans, axis=0)
    rob = rob[np.hypot(rob[:, 0], rob[:, 1]) < 3.0]
    if rob.shape[0] == 0:
        return empty, rob, True
    wpts = to_world(rob, pose)
    wx, wy = wpts[:, 0], wpts[:, 1]
    keep = ((wx > WALLS[0] + 0.2) & (wx < WALLS[1] - 0.2) &
            (wy > WALLS[2] + 0.2) & (wy < WALLS[3] - 0.2))
    R = _STATIC_RECTS
    inside = ((wx[:, None] >= R[None, :, 0]) & (wx[:, None] <= R[None, :, 1]) &
              (wy[:, None] >= R[None, :, 2]) & (wy[:, None] <= R[None, :, 3])).any(axis=1)
    keep &= ~inside
    m = rob[keep]
    if m.shape[0] > 600:
        m = m[::int(math.ceil(m.shape[0] / 600.0))]
    if m.shape[0] < 3:
        return empty, rob, True
    d2 = ((m[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
    m = m[(d2 < 0.09).sum(axis=1) >= 3]
    return m, rob, True




_TRK = {"tracks": [], "step": 0, "f": None, "b": None,
        "raw": np.zeros((0, 2)), "static": np.zeros((0, 2))}


def _clusters(m, adj):
    """월드 점 m (K, 2) 를 인접 행렬 adj(0.5 m 연결)로 덩어리로 나눕니다. 반환: [(중심 x, 중심 y, 반경 _MOVER_R)]."""
    n = m.shape[0]
    lab = np.full(n, -1)
    out = []
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = 0
        mem = [i]
        stack = [i]
        while stack:
            j = stack.pop()
            nb = np.nonzero(adj[j] & (lab < 0))[0]
            if nb.size:
                lab[nb] = 0
                nl = nb.tolist()
                mem.extend(nl)
                stack.extend(nl)
        if len(mem) >= 3:
            c = m[mem].mean(axis=0)
            out.append((float(c[0]), float(c[1]), _MOVER_R))
    return out


def _update_tracks(clusters, step):
    """덩어리 중심을 직전 트랙(예측 위치) 중 0.8 m 안 최근접과 짝짓고 속도를 갱신합니다."""
    tracks = _TRK["tracks"]
    pairs = []
    for i, tr in enumerate(tracks):
        age = (step - tr["seen"]) / 60.0
        px = tr["p"][0] + tr["v"][0] * age
        py = tr["p"][1] + tr["v"][1] * age
        for j, (cx, cy, _r) in enumerate(clusters):
            d = math.hypot(cx - px, cy - py)
            if d < 0.8:
                pairs.append((d, i, j))
    pairs.sort()
    used_t, used_c = set(), set()
    for _d, i, j in pairs:
        if i in used_t or j in used_c:
            continue
        used_t.add(i)
        used_c.add(j)
        tr = tracks[i]
        cx, cy, r = clusters[j]
        dt = (step - tr["ref_step"]) / 60.0
        if dt >= 0.08:
            rvx = (cx - tr["ref"][0]) / dt
            rvy = (cy - tr["ref"][1]) / dt
            if tr["has_v"]:
                vx = 0.5 * tr["v"][0] + 0.5 * rvx
                vy = 0.5 * tr["v"][1] + 0.5 * rvy
            else:
                vx, vy = rvx, rvy
            sp = math.hypot(vx, vy)
            if sp > 1.5:
                vx, vy = vx * 1.5 / sp, vy * 1.5 / sp
            tr["v"] = (vx, vy)
            tr["has_v"] = True
            tr["ref"] = (cx, cy)
            tr["ref_step"] = step
        tr["p"] = (cx, cy)
        tr["seen"] = step
        tr["r"] = r
    for j, (cx, cy, r) in enumerate(clusters):
        if j not in used_c:
            tracks.append({"p": (cx, cy), "v": (0.0, 0.0), "has_v": False, "ref": (cx, cy),
                           "ref_step": step, "seen": step, "r": r})


def _track_movers(pose):
    """앞·뒤 2D 라이다에서 정적 지도(WALLS 띠·바깥, 0.2 m 부풀린 boxes/FIXED_OBSTACLES)를 뺀 6 m 안의 점을
    월드 좌표 덩어리로 나눠 추적합니다. 새 스캔이 들어온 호출에서만 갱신합니다.
    반환: (트랙 목록, 3 m 안의 전체 점(로봇 기준), 3 m 안의 정적 점(월드), 스캔 수신 여부)."""
    S = _TRK
    S["step"] += 1
    step = S["step"]
    f = lidar2d_points("front")
    b = lidar2d_points("back")
    f = f if (f is not None and len(f) > 0) else None
    b = b if (b is not None and len(b) > 0) else None
    if f is None and b is None:
        return S["tracks"], S["raw"], S["static"], False
    if (f is not None and f is not S["f"]) or (b is not None and b is not S["b"]):
        if f is not None:
            S["f"] = f
        if b is not None:
            S["b"] = b
        rob = np.concatenate([s for s in (f, b) if s is not None], axis=0)
        rr = np.hypot(rob[:, 0], rob[:, 1])
        sel = rr < 6.0
        rob, rr = rob[sel], rr[sel]
        near = rr < 3.0
        S["raw"] = rob[near]
        clusters = []
        stat = np.zeros((0, 2))
        if rob.shape[0] > 0:
            wp = to_world(rob, pose)
            wx, wy = wp[:, 0], wp[:, 1]
            keep = ((wx > WALLS[0] + 0.2) & (wx < WALLS[1] - 0.2) &
                    (wy > WALLS[2] + 0.2) & (wy < WALLS[3] - 0.2))
            R = _STATIC_RECTS
            inside = ((wx[:, None] >= R[None, :, 0]) & (wx[:, None] <= R[None, :, 1]) &
                      (wy[:, None] >= R[None, :, 2]) & (wy[:, None] <= R[None, :, 3])).any(axis=1)
            keep &= ~inside
            stat = wp[(~keep) & near]
            if stat.shape[0] > 800:
                stat = stat[::int(math.ceil(stat.shape[0] / 800.0))]
            m = wp[keep]
            if m.shape[0] > 600:
                m = m[::int(math.ceil(m.shape[0] / 600.0))]
            if m.shape[0] >= 3:
                d2 = ((m[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
                dense = (d2 < 0.09).sum(axis=1) >= 3
                m = m[dense]
                if m.shape[0] >= 3:
                    adj = d2[np.ix_(dense, dense)] < 0.25
                    clusters = _clusters(m, adj)
        S["static"] = stat
        _update_tracks(clusters, step)
    S["tracks"] = [tr for tr in S["tracks"] if step - tr["seen"] <= 60]
    return S["tracks"], S["raw"], S["static"], True


def _pursuit_cmd(x, y, th, idx, vmax):
    """경유점 추종 명령 (v, w, 새 idx). idx >= len(_WAYPOINTS) 면 (0, 0, idx)."""
    n = len(_WAYPOINTS)
    if idx >= n:
        return 0.0, 0.0, idx
    tx, ty = _WAYPOINTS[idx]
    d = math.hypot(tx - x, ty - y)
    if idx < n - 1 and d < 0.35:
        idx += 1
        tx, ty = _WAYPOINTS[idx]
        d = math.hypot(tx - x, ty - y)
    if idx == n - 1 and d < 0.25:
        return 0.0, 0.0, n
    e = _wrap_angle(math.atan2(ty - y, tx - x) - th)
    w = _clamp(7.5 * e, -1.2, 1.2)
    if abs(e) > 0.35:
        return 0.0, w, idx
    return min(vmax, 2.5 * d + 0.5), w, idx


def _clear_at(x, y, th, movers, tau):
    """자세 (x, y, th) 의 몸체 사각형과 시각 tau 의 예측 물체 원 사이 (간격 - 요구 간격) 의 최솟값."""
    c, s = math.cos(th), math.sin(th)
    best = 99.0
    for mx, my, vx, vy, r, thr in movers:
        px = mx + vx * tau - x
        py = my + vy * tau - y
        lx = c * px + s * py
        ly = -s * px + c * py
        dx = max(_BODY[0] - lx, lx - _BODY[1], 0.0)
        dy = max(_BODY[2] - ly, ly - _BODY[3], 0.0)
        d = math.hypot(dx, dy) - r - thr
        if d < best:
            best = d
    return best


def _rollout(pose, cand, idx, movers):
    """후보 (종류, v, w) 로 0.1 s 간격 2.5 s 를 굴려 (최소 여유, 궤적) 을 돌려줍니다."""
    kind, v, w = cand
    x, y, th = pose
    traj = []
    cmin = 99.0
    for k in range(1, 26):
        if kind == "path":
            vv, ww, idx = _pursuit_cmd(x, y, th, idx, v)
        elif kind == "align":
            vv, ww = 0.0, _clamp(1.5 * _wrap_angle(GOAL[2] - th), -0.6, 0.6)
        else:
            vv, ww = v, w
        thm = th + 0.05 * ww
        x += vv * math.cos(thm) * 0.1
        y += vv * math.sin(thm) * 0.1
        th += ww * 0.1
        traj.append((x, y, th))
        if movers:
            cmin = min(cmin, _clear_at(x, y, th, movers, 0.1 * k))
    return cmin, traj


def _static_ok(traj, stat):
    """고정 (v, w) 후보의 첫 1 s 궤적이 정적 라이다 점·정적 지도·벽에 닿지 않는지 검사합니다."""
    R = _STATIC_RECTS
    bx0, bx1, by0, by1 = _BODY
    bmx = 0.5 * (bx0 + bx1)
    pts = [(bx0, by0), (bx0, 0.0), (bx0, by1), (bmx, by0), (bmx, by1), (bx1, by0), (bx1, 0.0), (bx1, by1)]
    for x, y, th in traj[:10]:
        c, s = math.cos(th), math.sin(th)
        if stat.shape[0] > 0:
            px = stat[:, 0] - x
            py = stat[:, 1] - y
            lx = c * px + s * py
            ly = -s * px + c * py
            dx = np.maximum(np.maximum(bx0 - lx, lx - bx1), 0.0)
            dy = np.maximum(np.maximum(by0 - ly, ly - by1), 0.0)
            if float(np.hypot(dx, dy).min()) < 0.12:
                return False
        for qx, qy in pts:
            wx = x + c * qx - s * qy
            wy = y + s * qx + c * qy
            if not (WALLS[0] + 0.15 < wx < WALLS[1] - 0.15 and WALLS[2] + 0.15 < wy < WALLS[3] - 0.15):
                return False
            if ((wx >= R[:, 0]) & (wx <= R[:, 1]) & (wy >= R[:, 2]) & (wy <= R[:, 3])).any():
                return False
    return True




_MOVER_R = 0.45     # 움직이는 물체 반경 [m] (0.6 x 0.6 상자의 대각 반경 0.42 m 를 덮음)


_MOVER_GAP = 0.35   # 모든 트랙에 똑같이 요구하는 몸체-물체 간격 [m]


# ========================= [END EDIT REGION] ===============================
