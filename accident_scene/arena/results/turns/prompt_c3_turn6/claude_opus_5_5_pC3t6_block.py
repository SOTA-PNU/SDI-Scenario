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
    """경유점 추종(최고 1.2 m/s) + 물체 추적·관측 왕복 구간 반사 예측 rollout(가감속 1.0 m/s² 반영) 기반 후보 선택.
    래치 없음: 매 호출마다 모든 후보를 다시 평가합니다.
    선택: 2.5 s 동안 위반(간격<0)이 없는 후보 중 가장 빠른 것. 없으면 첫 위반 시각 t_hit 가 가장 늦은 것,
    같으면 cmin 이 큰 것, 그래도 같으면 느린 것. 후진은 정지의 1.0 s cmin<0 일 때만 씁니다.
    미성숙 트랙(생긴 지 1.0 s 미만 또는 속도 갱신 3회 미만)은 속도 0, 반경이 시간에 따라 커지는 원으로 봅니다.
    GOAL 0.5 m 안이면 정렬만 하고 |yaw 오차|<0.1 이면 done."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.ready = 0
        controller.prev = None
        controller.vest = 0.0
        controller.wmeas = 0.0

    x, y, yaw = pose
    n = len(_WAYPOINTS)

    # ---- 현재 전진 속도(지수평균)와 회전 속도(직전 pose 차이) 추정 ----
    wmeas = 0.0
    if controller.prev is not None:
        px0, py0, pyaw0 = controller.prev
        vm = ((x - px0) * math.cos(yaw) + (y - py0) * math.sin(yaw)) * 60.0
        controller.vest = 0.6 * controller.vest + 0.4 * _clamp(vm, -1.0, 2.0)
        wmeas = abs(_wrap_angle(yaw - pyaw0)) * 60.0
    controller.prev = (x, y, yaw)
    controller.wmeas = wmeas
    v0 = controller.vest

    tracks, raw, stat, ok = _track_movers(pose, wmeas)
    controller.ready = controller.ready + 1 if ok else 0
    if controller.ready < 12:          # 라이다가 들어오고 트랙이 잡힐 때까지 대기
        return 0.0, 0.0, False

    # ---- 물체: 성숙 트랙은 등속 + 관측 왕복 구간 끝에서 반사, 미성숙 트랙은 정지 + 커지는 원 ----
    step = _TRK["step"]
    movers = []
    for tr in tracks:
        age_s = (step - tr.get("birth", step)) / 60.0
        nupd = tr.get("nupd", 0)
        if age_s < _MATURE_S or nupd < _MATURE_UPD:
            movers.append((tr["p"][0], tr["p"][1], 0.0, 0.0, _MOVER_R, _MOVER_GAP, None, None,
                           _GROW_RATE, age_s, nupd))
            continue
        age = (step - tr["seen"]) / 60.0
        vx, vy = tr["v"]
        mx = tr["p"][0] + vx * age
        my = tr["p"][1] + vy * age
        ext = _sweep_extent(tr)
        e2, rng = None, None
        if ext is not None:
            ux, uy, lo, hi, smin, smax = ext
            s0 = mx * ux + my * uy
            if lo is not None and s0 < lo:     # 관측 구간 밖으로 나감: 그 끝은 되돌아오는 점이 아님
                lo = None
            if hi is not None and s0 > hi:
                hi = None
            if lo is not None and hi is not None and hi - lo < 0.3:
                lo = hi = None
            e2, rng = (ux, uy, lo, hi), (smin, smax)
        movers.append((mx, my, vx, vy, _MOVER_R, _MOVER_GAP, e2, rng, 0.0, age_s, nupd))

    # ---- 경유점 진행: GOAL 0.5 m 안이면 정렬만, 0.6 m 밖으로 밀려나면 GOAL 추종으로 복귀 ----
    d_goal = math.hypot(GOAL[0] - x, GOAL[1] - y)
    if d_goal < 0.5:
        controller.idx = n
    elif controller.idx >= n and d_goal > 0.6:
        controller.idx = n - 1
    if controller.idx < n:
        _, _, controller.idx = _pursuit_cmd(x, y, yaw, controller.idx, _VMAX)
    align = controller.idx >= n

    if align and abs(_wrap_angle(GOAL[2] - yaw)) < 0.1:
        return 0.0, 0.0, True

    rear_blocked = raw.shape[0] > 0 and bool(
        ((raw[:, 0] < _BODY[0]) & (raw[:, 0] > _BODY[0] - 0.45) & (np.abs(raw[:, 1]) < 0.45)).any())

    FIX0 = ("fix", 0.0, 0.0)
    if align:
        prim = [("align", 0.0, 0.0), FIX0, ("fix", -0.3, 0.0)]
    else:
        prim = [("path", 1.2, 0.0), ("path", 0.8, 0.0), ("path", 0.5, 0.0), ("path", 0.3, 0.0),
                FIX0, ("fix", -0.3, 0.0)]

    cache = {}
    memo = {}

    def ev(cand):
        if cand not in cache:
            cache[cand] = _rollout(pose, cand, controller.idx, movers, v0)
        return cache[cand]

    def stop1():
        if "s1" not in memo:
            memo["s1"] = _cmin_upto(ev(FIX0)[1], movers, 10)
        return memo["s1"]

    def allowed(cand, gate):
        if cand[1] < 0.0:
            if rear_blocked:
                return False
            if gate and stop1() >= 0.0:      # 정지가 1.0 s 안에 안전하면 후진하지 않음
                return False
            if not _static_ok(ev(cand)[1], stat):
                return False
        return True

    def better(a, b):
        """a, b = (t_hit, cmin, |v|). 늦은 t_hit → 큰 cmin → 느린 후보 순으로 a 가 낫으면 True."""
        if abs(a[0] - b[0]) > 1e-6:
            return a[0] > b[0]
        if abs(a[1] - b[1]) > 1e-6:
            return a[1] > b[1]
        return a[2] < b[2] - 1e-9

    def best_of(cands, gate):
        best = None
        for cand in cands:
            if not allowed(cand, gate):
                continue
            cm, _tj, th = ev(cand)
            k = (th, cm, abs(cand[1]))
            if best is None or better(k, best[0]):
                best = (k, cand)
        return best[1] if best is not None else FIX0

    chosen = None
    if align:
        a_cand = prim[0]
        a_cmin, a_traj, _a_th = ev(a_cand)
        if a_cmin >= 0.0 or _cmin_upto(a_traj, movers, 10) >= 0.0:
            chosen = a_cand
        else:
            chosen = best_of(prim, True)
    else:
        # ---- 2.5 s 무위반 후보 중 가장 빠른 것 (prim 은 빠른 순) ----
        for cand in prim:
            if not allowed(cand, True):
                continue
            if ev(cand)[2] >= _NO_HIT:
                chosen = cand
                break
        if chosen is None:
            chosen = best_of(prim, True)

    # ---- 진단용 출력: 0.5 s 마다 후보별 (t_hit, cmin), 고른 후보, 가장 가까운 트랙의 나이·갱신 횟수·구간 ----
    if step % 30 == 0:
        parts = []
        for c in prim:
            cm, _tj, th = ev(c)
            parts.append("%s(%.1f)=(%s,%.2f)" % (c[0], c[1], "-" if th >= _NO_HIT else "%.1f" % th, cm))
        near = "none"
        if movers:
            mb = min(movers, key=lambda m: math.hypot(m[0] - x, m[1] - y))
            ext_s = "none"
            if mb[6] is not None and mb[7] is not None:
                e = mb[6]
                ext_s = "axis=(%.2f,%.2f) obs=[%.2f,%.2f] lo=%s hi=%s" % (
                    e[0], e[1], mb[7][0], mb[7][1],
                    "-" if e[2] is None else "%.2f" % e[2],
                    "-" if e[3] is None else "%.2f" % e[3])
            near = "p=(%.2f,%.2f) v=(%.2f,%.2f) d=%.2f age=%.2f nupd=%d mature=%d ext=%s" % (
                mb[0], mb[1], mb[2], mb[3], math.hypot(mb[0] - x, mb[1] - y), mb[9], mb[10],
                int(mb[8] == 0.0), ext_s)
        print("DBG t=%.2f dist=%.2f v0=%.2f w=%.2f chosen=%s(%.1f) cand=[%s] nearest=%s" % (
            t, d_goal, v0, wmeas, chosen[0], chosen[1], " ".join(parts), near), flush=True)

    kind, v, w = chosen
    if kind == "path":
        v, w, _ = _pursuit_cmd(x, y, yaw, controller.idx, v)
    elif kind == "align":
        v, w = 0.0, _align_w(yaw)
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


def _update_tracks(clusters, step, rot_ok=True):
    """덩어리 중심을 직전 트랙(예측 위치) 중 0.8 m 안 최근접과 짝짓고 속도를 갱신합니다.
    트랙마다 생성 시각 birth 와 속도 갱신 횟수 nupd 를 기록합니다.
    rot_ok=False(로봇 회전 속도 > 0.6 rad/s 인 스캔)면 위치(p, seen)만 갱신하고 속도·위치 기록(hist)은 갱신하지 않으며,
    속도 기준점을 버려 회전이 끝난 뒤 첫 스캔에서 다시 잡습니다. hist 는 최근 15 s 의 [(step, x, y)] 입니다."""
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
        if not rot_ok:
            tr["ref_step"] = None                  # 회전 중 위치는 속도 기준점으로 쓰지 않음
        elif tr.get("ref_step") is None:
            tr["ref"] = (cx, cy)                   # 회전이 끝난 뒤 기준점을 다시 잡음
            tr["ref_step"] = step
        else:
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
                tr["nupd"] = tr.get("nupd", 0) + 1
                tr["ref"] = (cx, cy)
                tr["ref_step"] = step
        tr["p"] = (cx, cy)
        tr["seen"] = step
        tr["r"] = r
        if rot_ok:
            h = tr.setdefault("hist", [])
            h.append((step, cx, cy))
            while h and step - h[0][0] > _HIST_STEPS:
                h.pop(0)
    for j, (cx, cy, r) in enumerate(clusters):
        if j not in used_c:
            tracks.append({"p": (cx, cy), "v": (0.0, 0.0), "has_v": False, "ref": (cx, cy),
                           "ref_step": step if rot_ok else None, "seen": step, "r": r,
                           "hist": [(step, cx, cy)] if rot_ok else [], "ext": None,
                           "birth": step, "nupd": 0})


def _track_movers(pose, wrate=0.0):
    """앞·뒤 2D 라이다에서 정적 지도(WALLS 띠·바깥, 0.2 m 부풀린 boxes/FIXED_OBSTACLES)를 뺀 6 m 안의 점을
    월드 좌표 덩어리로 나눠 추적합니다. 새 스캔이 들어온 호출에서만 갱신합니다.
    wrate 는 직전 pose 차이로 잰 회전 속도 [rad/s] 이고, 0.6 rad/s 를 넘으면 트랙 속도를 갱신하지 않습니다.
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
        _update_tracks(clusters, step, wrate <= _ROT_LIMIT)
    S["tracks"] = [tr for tr in S["tracks"] if step - tr["seen"] <= 60]
    return S["tracks"], S["raw"], S["static"], True


def _pursuit_cmd(x, y, th, idx, vmax):
    """경유점 추종 명령 (v, w, 새 idx). idx >= len(_WAYPOINTS) 면 (0, 0, idx).
    회전 한계 ±1.5 rad/s. 방향 오차 > 1.0 rad 면 제자리 회전, 0.35–1.0 rad 면 v=vcap*cos(e) 로 돌며 달림.
    마지막 경유점(GOAL)으로 갈 때는 vcap=min(vmax, 0.6*d+0.15) 로 감속하고, 0.5 m 안이면 도착으로 봅니다."""
    n = len(_WAYPOINTS)
    if idx >= n:
        return 0.0, 0.0, idx
    tx, ty = _WAYPOINTS[idx]
    d = math.hypot(tx - x, ty - y)
    if idx < n - 1 and d < 0.35:
        idx += 1
        tx, ty = _WAYPOINTS[idx]
        d = math.hypot(tx - x, ty - y)
    if idx == n - 1 and d < 0.5:
        return 0.0, 0.0, n
    e = _wrap_angle(math.atan2(ty - y, tx - x) - th)
    w = _clamp(7.5 * e, -_WMAX, _WMAX)
    if abs(e) > 1.0:
        return 0.0, w, idx
    if idx == n - 1:
        vc = min(vmax, 0.6 * d + 0.15)
    else:
        vc = min(vmax, 2.5 * d + 0.5)
    if abs(e) > 0.35:
        vc *= math.cos(e)
    return vc, w, idx


def _clear_at(x, y, th, movers, tau):
    """자세 (x, y, th) 의 몸체 사각형과 시각 tau 의 예측 물체 원 사이 (부호 간격 - 반경 - 요구 간격) 의 최솟값.
    부호 간격: 물체 중심이 몸체 밖이면 몸체 바깥면까지 거리, 몸체 안이면 가장 가까운 면까지 깊이의 음수
    (몸체 중심 쪽으로 깊이 들어갈수록 더 작아져 0 에서 잘리지 않음).
    미성숙 트랙(m[8] > 0)은 반경을 _MOVER_R + m[8]·tau (증가분 최대 _GROW_MAX) 로 키웁니다."""
    c, s = math.cos(th), math.sin(th)
    bx0, bx1, by0, by1 = _BODY
    best = 99.0
    for m in movers:
        qx, qy = _mover_pos(m, tau)
        px = qx - x
        py = qy - y
        lx = c * px + s * py
        ly = -s * px + c * py
        if bx0 <= lx <= bx1 and by0 <= ly <= by1:
            d0 = -min(lx - bx0, bx1 - lx, ly - by0, by1 - ly)
        else:
            dx = max(bx0 - lx, lx - bx1, 0.0)
            dy = max(by0 - ly, ly - by1, 0.0)
            d0 = math.hypot(dx, dy)
        r = m[4]
        if len(m) > 8 and m[8] > 0.0:
            r += min(m[8] * tau, _GROW_MAX)
        d = d0 - r - m[5]
        if d < best:
            best = d
    return best


def _rollout(pose, cand, idx, movers, v0=0.0):
    """후보 (종류, v, w) 로 0.1 s 간격 2.5 s 를 굴립니다. 현재 추정 속도 v0 에서 시작해 가감속을 _ACC 로 제한합니다.
    반환: (최소 여유 cmin, 궤적, 여유가 처음 0 미만이 되는 시각 t_hit [s], 없으면 _NO_HIT)."""
    kind, v, w = cand
    x, y, th = pose
    vc = v0
    dv = _ACC * 0.1
    traj = []
    cmin = 99.0
    thit = _NO_HIT
    for k in range(1, 26):
        if kind == "path":
            vv, ww, idx = _pursuit_cmd(x, y, th, idx, v)
        elif kind == "align":
            vv, ww = 0.0, _align_w(th)
        else:
            vv, ww = v, w
        vn = vc + _clamp(vv - vc, -dv, dv)
        vm = 0.5 * (vc + vn)
        vc = vn
        thm = th + 0.05 * ww
        x += vm * math.cos(thm) * 0.1
        y += vm * math.sin(thm) * 0.1
        th += ww * 0.1
        traj.append((x, y, th))
        if movers:
            a = _clear_at(x, y, th, movers, 0.1 * k)
            if a < cmin:
                cmin = a
            if a < 0.0 and thit >= _NO_HIT:
                thit = 0.1 * k
    return cmin, traj, thit


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




def _cmin_upto(traj, movers, kmax):
    """rollout 궤적 traj 의 처음 kmax 단계(0.1 s 간격)만 본 최소 여유. 물체가 없으면 99."""
    cmin = 99.0
    if not movers:
        return cmin
    for k, (px, py, pth) in enumerate(traj[:kmax], start=1):
        cmin = min(cmin, _clear_at(px, py, pth, movers, 0.1 * k))
    return cmin




_VMAX = 1.2          # 경유점 추종 최고 속도 [m/s]


_WMAX = 1.5          # 경유점 추종 회전 한계 [rad/s]


_ACC = 1.0           # rollout 가감속 한계 [m/s^2]


_REV_MIN_SPEED = 0.2  # 되돌아옴 가설을 만드는 최소 트랙 속력 [m/s]


_REV_HORIZON = 1.5   # 되돌아옴 가설 검사 구간 [s]


def _align_w(th):
    """GOAL 방향 정렬용 제자리 회전 속도."""
    e = _wrap_angle(GOAL[2] - th)
    return 0.0 if abs(e) < 0.05 else _clamp(2.0 * e, -1.0, 1.0)


def _in_crossing_lane(pose, movers):
    """등속 가설(tag 0) 트랙 중 속력 0.2 m/s 이상이고 이동선이 로봇 진행 방향(yaw)과 45° 넘게 엇갈리는 트랙에 대해,
    그 이동선(방향 무관)에서 로봇 몸체 네 모서리가 수직 1.0 m 안에 있으면(또는 선을 걸치면) True."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(x + c * bx - s * by, y + s * bx + c * by)
               for bx in (_BODY[0], _BODY[1]) for by in (_BODY[2], _BODY[3])]
    cos45 = math.cos(math.pi / 4.0)
    for m in movers:
        if m[7] != 0:
            continue
        mx, my, vx, vy = m[0], m[1], m[2], m[3]
        sp = math.hypot(vx, vy)
        if sp < _REV_MIN_SPEED:
            continue
        ux, uy = vx / sp, vy / sp
        if abs(ux * c + uy * s) > cos45:
            continue
        ds = [(px - mx) * (-uy) + (py - my) * ux for px, py in corners]
        if min(ds) <= 1.0 and max(ds) >= -1.0:
            return True
    return False




_MOVE_MIN_SPEED = 0.2   # 이동선·왕복 구간을 만드는 최소 트랙 속력 [m/s]


_HIST_STEPS = 900       # 트랙 위치 기록 길이 [호출 수] (15 s)


_SWEEP_MARGIN = 0.15    # 관측 왕복 구간 끝을 바깥으로 넓히는 여유 [m]


def _sweep_extent(tr):
    """트랙 위치 기록으로 관측 왕복 구간을 구합니다. 반환: (ux, uy, lo, hi, smin, smax) 또는 None.
    기록이 2 s 이상이고 속력이 0.2 m/s 이상일 때만 새로 구하고, 아니면 직전에 구한 값을 그대로 씁니다.
    축 u 는 속도 방향이고, s = 위치·u 의 최소·최대가 smin, smax 입니다. lo/hi 는 그 끝에서 실제로 되돌아온 것이
    관측됐을 때만(끝의 앞뒤 기록 모두에 0.3 m 이상 안쪽 점이 있을 때) 값을 갖고, 0.15 m 바깥으로 넓힙니다."""
    h = tr.get("hist", [])
    if len(h) < 3 or not tr["has_v"]:
        return tr.get("ext")
    if (h[-1][0] - h[0][0]) / 60.0 < 2.0:
        return tr.get("ext")
    vx, vy = tr["v"]
    sp = math.hypot(vx, vy)
    if sp < _MOVE_MIN_SPEED:
        return tr.get("ext")
    ux, uy = vx / sp, vy / sp
    a = np.asarray(h, dtype=np.float64)
    s = a[:, 1] * ux + a[:, 2] * uy
    i0, i1 = int(np.argmin(s)), int(np.argmax(s))
    smin, smax = float(s[i0]), float(s[i1])
    lo = hi = None
    if bool((s[:i0] >= smin + 0.3).any()) and bool((s[i0 + 1:] >= smin + 0.3).any()):
        lo = smin - _SWEEP_MARGIN
    if bool((s[:i1] <= smax - 0.3).any()) and bool((s[i1 + 1:] <= smax - 0.3).any()):
        hi = smax + _SWEEP_MARGIN
    ext = (ux, uy, lo, hi, smin, smax)
    tr["ext"] = ext
    return ext


def _mover_pos(m, tau):
    """물체 m = (x, y, vx, vy, r, gap, ext, rng) 의 시각 tau 예측 위치.
    등속으로 움직이되, ext=(ux, uy, lo, hi) 의 관측 구간 끝(lo/hi 가 있는 쪽)을 넘으면 그 축에서 반사합니다."""
    mx, my, vx, vy = m[0], m[1], m[2], m[3]
    ext = m[6]
    if ext is None or (ext[2] is None and ext[3] is None):
        return mx + vx * tau, my + vy * tau
    ux, uy, lo, hi = ext
    s0 = mx * ux + my * uy
    va = vx * ux + vy * uy
    s = s0 + va * tau
    if lo is not None and hi is not None:
        L = hi - lo
        z = (s - lo) % (2.0 * L)
        if z > L:
            z = 2.0 * L - z
        s = lo + z
    elif lo is not None and s < lo:
        s = 2.0 * lo - s
    elif hi is not None and s > hi:
        s = 2.0 * hi - s
    ds = s - s0
    return (mx + ux * ds + (vx - va * ux) * tau,
            my + uy * ds + (vy - va * uy) * tau)


def _crossing_line(pose, traj, movers):
    """통과 확정 래치를 지웠습니다(시도 6). 이 함수를 부르는 곳은 없습니다."""
    return None


def _latch_done(pose, line):
    """통과 확정 래치를 지웠습니다(시도 6). 이 함수를 부르는 곳은 없습니다."""
    return True




_NO_HIT = 99.0       # rollout 2.5 s 동안 위반이 없을 때의 t_hit 값


_MATURE_S = 1.0      # 속도를 믿는 최소 트랙 나이 [s]


_MATURE_UPD = 3      # 속도를 믿는 최소 속도 갱신 횟수


_GROW_RATE = 0.7     # 미성숙 트랙 반경 증가 속도 [m/s]


_GROW_MAX = 1.2      # 미성숙 트랙 반경 증가 상한 [m]


_ROT_LIMIT = 0.6     # 이 회전 속도 [rad/s] 를 넘는 스캔에서는 트랙 속도를 갱신하지 않음


# ========================= [END EDIT REGION] ===============================
