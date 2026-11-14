# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# 작성: Claude (Anthropic), 2026-10-29.
# 정적 지도(벽, MISSION 상자, 장면의 고정물)로 첫 호출에 경로를 계산하고 그 경로를 따라간다.
# 지도에 없는 물체는 앞뒤 2D 라이다로만 감지한다. 움직이는 물체의 위치, 경로, 시각은 코드에 없다.
# 매 호출: 라이다 점 -> 지도에 없는 점 -> 묶음 -> 추적과 속도 추정 -> 후보 명령마다 몇 초 앞을 굴려
# 몸체와 예측한 점의 간격을 보고, 안전한 것 가운데 가장 앞으로 가는 명령을 고른다.
import heapq

_WALLS = (-10.30, 9.35, -12.15, 17.95)        # 벽 안쪽 면: x 최소, x 최대, y 최소, y 최대
_FIXED = np.array([                           # 고정물 43개의 경계 상자: x 최소, x 최대, y 최소, y 최대
    (-10.45, -10.00, -9.25, -8.80),
    (9.10, 9.55, -9.20, -8.75),
    (-9.95, -8.70, -3.80, -3.40),
    (-9.85, -8.80, -3.80, -3.60),
    (-9.90, -8.80, -3.45, -2.20),
    (-9.60, -6.05, -3.45, -2.25),
    (1.85, 5.30, -3.45, -0.70),
    (-9.60, -8.85, -3.35, -1.05),
    (-10.45, -10.00, -3.25, -2.80),
    (9.10, 9.55, -3.20, -2.75),
    (-9.85, -8.80, -2.25, -0.95),
    (-9.90, -8.80, -1.00, 0.25),
    (-9.60, -8.85, -0.90, 0.20),
    (-9.85, -8.80, 0.20, 0.40),
    (-9.85, -8.80, 0.45, 1.70),
    (-9.85, -8.75, 0.60, 4.05),
    (-9.85, -8.80, 1.65, 2.95),
    (-10.45, -10.00, 2.75, 3.20),
    (9.10, 9.55, 2.80, 3.25),
    (-9.85, -8.80, 2.90, 4.15),
    (-9.85, -8.80, 4.20, 4.40),
    (-9.75, -8.70, 4.45, 5.70),
    (-9.35, -8.70, 4.75, 5.40),
    (-9.75, -8.70, 5.65, 6.95),
    (-9.70, -8.75, 5.65, 8.25),
    (-9.75, -8.70, 6.90, 8.15),
    (-9.85, -8.80, 8.20, 8.40),
    (-9.75, -8.70, 8.45, 9.70),
    (-9.75, -9.00, 8.70, 9.65),
    (-10.45, -10.00, 8.75, 9.20),
    (9.10, 9.55, 8.80, 9.25),
    (-9.75, -8.70, 9.65, 10.90),
    (-9.75, -9.00, 9.85, 10.80),
    (-9.75, -8.70, 10.85, 12.15),
    (-9.75, -9.00, 11.15, 12.10),
    (-9.95, -8.70, 12.10, 12.50),
    (-9.85, -8.80, 12.20, 12.40),
    (-7.05, -5.20, 12.90, 14.35),
    (-6.85, -5.35, 13.00, 14.40),
    (-6.60, -5.50, 13.20, 14.15),
    (-10.45, -10.00, 14.75, 15.20),
    (9.10, 9.55, 14.80, 15.25),
    (-4.15, -4.05, 17.75, 18.00),
], dtype=np.float64)
_BODY_C = np.array([[-0.05, 0.0], [-0.41, 0.0]])   # 몸체(x -0.59..0.13, 바퀴 y +-0.25)를 덮는 두 원의 중심
_BODY_R = 0.32
_CFG = dict(res=0.1, c_min=0.75, c_pref=1.3, c_los=1.0,                       # 경로 계산
            near=8.0, margin=0.25, voxel=0.1, gap=0.45, gate=1.0, win=1.0, lost=0.6,  # 감지와 추적
            T=3.0, T_resume=4.0, dt=0.1, d_safe=0.30, grow=0.10)                 # 예측과 고르기
_ST = {"rects": None, "path": None, "wp": 1, "align": False, "tracks": [], "nid": 0,
       "scans": (None, None), "mode": "go", "dbg_t": -9.0}


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clear(xs, ys):
    """점(배열)에서 가장 가까운 지도 물체나 벽 안쪽 면까지의 거리 [m]. 물체 안이면 0."""
    r = _ST["rects"]
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    dx = np.maximum(np.maximum(r[:, 0] - xs[..., None], xs[..., None] - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - ys[..., None], ys[..., None] - r[:, 3]), 0.0)
    d = np.hypot(dx, dy).min(axis=-1)
    x0, x1, y0, y1 = _WALLS
    return np.maximum(0.0, np.minimum(d, np.minimum(np.minimum(xs - x0, x1 - xs), np.minimum(ys - y0, y1 - ys))))


def _seg_clear(p, q):
    n = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.05)))
    k = np.linspace(0.0, 1.0, n + 1)
    return float(_clear(p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k).min())


def _plan(start, goal):
    """격자 A*. 여유 c_min 미만인 칸은 막고, c_pref 미만이면 비용을 더한다. 그 뒤 여유 c_los 이상인 직선으로 줄인다."""
    res = _CFG["res"]
    x0, x1, y0, y1 = _WALLS
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    C = np.stack([_clear(np.full(ys.size, x), ys) for x in xs])        # (nx, ny)
    nx, ny = C.shape

    def cell(p):
        return (min(nx - 1, max(0, int((p[0] - x0) / res))), min(ny - 1, max(0, int((p[1] - y0) / res))))

    s, g = cell(start), cell(goal)
    free = C >= _CFG["c_min"]
    free[s] = free[g] = True
    pen = 1.0 + 4.0 * np.maximum(0.0, _CFG["c_pref"] - C)
    dist, prev, done = {s: 0.0}, {}, set()
    heap = [(0.0, s)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while heap:
        _, u = heapq.heappop(heap)
        if u in done:
            continue
        if u == g:
            break
        done.add(u)
        for dx, dy, length in steps:
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < nx and 0 <= v[1] < ny and free[v]:
                nd = dist[u] + length * res * 0.5 * (pen[u] + pen[v])
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd + res * math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g != s and g not in prev:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    pts = [(float(xs[i]), float(ys[j])) for i, j in reversed(cells)]
    pts[0], pts[-1] = (start[0], start[1]), (goal[0], goal[1])
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and _seg_clear(pts[i], pts[j + 1]) >= _CFG["c_los"]:
            j += 1
        out.append(pts[j])
        i = j
    return out


def _cluster(P):
    """가까운 점끼리(gap 이하) 묶는다."""
    n = len(P)
    if n == 0:
        return []
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) <= _CFG["gap"]
    label = -np.ones(n, dtype=int)
    groups = []
    for i in range(n):
        if label[i] >= 0:
            continue
        label[i] = len(groups)
        stack, members = [i], []
        while stack:
            k = stack.pop()
            members.append(k)
            for m in np.nonzero(d[k] & (label < 0))[0]:
                label[m] = label[i]
                stack.append(m)
        groups.append(P[members])
    return groups


def _fit_vel(hist):
    """추적 묶음 중심의 최근 win 초를 직선으로 맞춘 속도. 자료가 모자라면 0."""
    h = np.array(hist)
    if len(h) < 4 or h[-1, 0] - h[0, 0] < 0.4:
        return np.zeros(2)
    tt = h[:, 0] - h[:, 0].mean()
    den = float((tt * tt).sum())
    v = np.array([(tt * (h[:, 1] - h[:, 1].mean())).sum() / den, (tt * (h[:, 2] - h[:, 2].mean())).sum() / den])
    sp = float(np.hypot(*v))
    if sp < 0.08:
        return np.zeros(2)
    return v * min(1.0, 1.5 / sp)


def _sense(t, pose):
    """새 스캔이 왔을 때만 지도에 없는 점을 묶어 추적을 갱신한다."""
    front, back = lidar2d_points("front"), lidar2d_points("back")
    if front is _ST["scans"][0] and back is _ST["scans"][1]:
        return
    _ST["scans"] = (front, back)
    parts = [to_world(p, pose)[:, :2] for p in (front, back) if len(p)]
    P = np.concatenate(parts) if parts else np.zeros((0, 2))
    if len(P):
        P = P[np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]) < _CFG["near"]]
    if len(P):
        P = P[_clear(P[:, 0], P[:, 1]) > _CFG["margin"]]
    if len(P):
        _, keep = np.unique(np.floor(P / _CFG["voxel"]).astype(np.int64), axis=0, return_index=True)
        P = P[keep]
    groups = _cluster(P)
    tracks = _ST["tracks"]
    for tr in tracks:
        tr["hit"] = False
    for grp in groups:
        c = grp.mean(axis=0)
        best, bd = None, _CFG["gate"]
        for tr in tracks:
            if tr["hit"]:
                continue
            last_t, lx, ly = tr["hist"][-1]
            dd = math.hypot(c[0] - (lx + tr["vel"][0] * (t - last_t)), c[1] - (ly + tr["vel"][1] * (t - last_t)))
            if dd < bd:
                best, bd = tr, dd
        if best is None:
            best = {"id": _ST["nid"], "hist": [], "vel": np.zeros(2)}
            _ST["nid"] += 1
            tracks.append(best)
        best["hit"], best["seen"], best["pts"] = True, t, grp
        best["hist"] = [h for h in best["hist"] if t - h[0] <= _CFG["win"]] + [(t, float(c[0]), float(c[1]))]
        best["vel"] = _fit_vel(best["hist"])
    _ST["tracks"] = [tr for tr in tracks if t - tr["seen"] <= _CFG["lost"]]


def _follow(pose):
    """경로 추종(시험으로 통과를 확인한 방법): (v, w, done)."""
    x, y, yaw = pose
    path = _ST["path"]
    if _ST["align"]:
        e = _wrap(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False
    i = _ST["wp"]
    tx, ty = path[i]
    dist = math.hypot(tx - x, ty - y)
    if i < len(path) - 1 and dist < 0.35:
        _ST["wp"] = i = i + 1
        tx, ty = path[i]
        dist = math.hypot(tx - x, ty - y)
    if i == len(path) - 1 and dist < 0.25:
        _ST["align"] = True
        return _follow(pose)
    e = _wrap(math.atan2(ty - y, tx - x) - yaw)
    w = _clamp(1.5 * e, -0.8, 0.8)
    if abs(e) > 0.35:
        return 0.0, w, False
    return min(0.6, 0.5 * dist + 0.1), w, False


def _margin(pose, v, w, T, P0, V, static):
    """(v, w) 로 T 초 굴렸을 때 몸체 두 원과 예측한 점 사이의 가장 작은 여유(안전 거리를 뺀 값)와 지도 여유."""
    dt = _CFG["dt"]
    K = int(round(T / dt))
    x, y, th = pose
    poses = [(x, y, th)]
    for _ in range(K):
        th += w * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        poses.append((x, y, th))
    Q = np.array(poses)
    c, s = np.cos(Q[:, 2]), np.sin(Q[:, 2])
    cx = Q[:, 0:1] + c[:, None] * _BODY_C[None, :, 0] - s[:, None] * _BODY_C[None, :, 1]     # (K+1, 2)
    cy = Q[:, 1:2] + s[:, None] * _BODY_C[None, :, 0] + c[:, None] * _BODY_C[None, :, 1]
    worst = float("inf")
    if len(P0):
        tk = np.arange(K + 1) * dt
        px = P0[None, :, 0] + V[None, :, 0] * tk[:, None]                                    # (K+1, M)
        py = P0[None, :, 1] + V[None, :, 1] * tk[:, None]
        d = np.hypot(px[:, None, :] - cx[:, :, None], py[:, None, :] - cy[:, :, None]).min(axis=(1, 2))
        worst = float((d - _BODY_R - _CFG["d_safe"] - _CFG["grow"] * tk).min())
    st = float(_clear(cx, cy).min()) - _BODY_R if static else float("inf")
    return worst, st


def _choose(pose, vn, wn):
    """후보 명령 가운데 안전한 것을 고른다. 반환: (이름, v, w, 여유)."""
    tracks = _ST["tracks"]
    if not tracks:
        return "go", vn, wn, float("inf")
    P0 = np.concatenate([tr["pts"] for tr in tracks])
    V = np.concatenate([np.repeat(tr["vel"][None, :], len(tr["pts"]), axis=0) for tr in tracks])
    if float(np.hypot(P0[:, 0] - pose[0], P0[:, 1] - pose[1]).min()) > 6.0:
        return "go", vn, wn, float("inf")
    on_path = [("go", vn, wn)] + ([("slow", 0.6 * vn, wn), ("creep", 0.3 * vn, wn)] if vn > 0 else []) + [("hold", 0.0, 0.0)]
    escape = [("back", -0.25, 0.0), ("backL", -0.2, 0.6), ("backR", -0.2, -0.6), ("arcL", 0.3, 0.7), ("arcR", 0.3, -0.7)]
    scored = []
    for name, v, w in on_path:
        T = _CFG["T_resume"] if (name == "go" and _ST["mode"] != "go") else _CFG["T"]
        m, _ = _margin(pose, v, w, T, P0, V, False)
        if m >= 0.0:
            return name, v, w, m
        scored.append((m, name, v, w))
    for name, v, w in escape:
        m, st = _margin(pose, v, w, _CFG["T"], P0, V, True)
        if st >= 0.05:
            scored.append((m, name, v, w))
    m, name, v, w = max(scored)
    return name, v, w, m


def controller(t, pose, env):
    if _ST["rects"] is None:
        boxes = [(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0, b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0)
                 for b in MISSION["boxes"]]
        _ST["rects"] = np.concatenate([np.array(boxes, dtype=np.float64), _FIXED])
        _ST["path"] = _plan((pose[0], pose[1]), (GOAL[0], GOAL[1]))
        p = _ST["path"]
        print("DBG plan %d pts, length %.2f m, min segment clearance %.2f m" % (
            len(p), sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])),
            min(_seg_clear(a, b) for a, b in zip(p, p[1:]))), [(round(a, 2), round(b, 2)) for a, b in p])
    _sense(t, pose)
    vn, wn, done = _follow(pose)
    if done:
        print("DBG done t=%.2f pose=(%.2f, %.2f, %.2f)" % (t, pose[0], pose[1], pose[2]))
        return 0.0, 0.0, True
    name, v, w, m = _choose(pose, vn, wn)
    if name != _ST["mode"] or t - _ST["dbg_t"] >= 2.0:
        near = min(((math.hypot(tr["pts"][:, 0].mean() - pose[0], tr["pts"][:, 1].mean() - pose[1]), tr) for tr in _ST["tracks"]),
                   key=lambda z: z[0], default=(None, None))
        print("DBG t=%.2f pose=(%.2f, %.2f, %.2f) wp=%d mode=%s v=%.2f w=%.2f margin=%s tracks=%d near=%s" % (
            t, pose[0], pose[1], pose[2], _ST["wp"], name, v, w, "inf" if m == float("inf") else "%.2f" % m,
            len(_ST["tracks"]), "-" if near[1] is None else "%.2f m v=(%.2f, %.2f)" % (near[0], near[1]["vel"][0], near[1]["vel"][1])))
        _ST["dbg_t"] = t
    _ST["mode"] = name
    return v, w, False


