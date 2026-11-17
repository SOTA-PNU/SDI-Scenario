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
    """Waypoint following + lidar mover detection: 1.0 s stop-and-wait (pause_until, v=w=0),
    predictive slow-down, evasion only when not paused, forced resume 10 s after wait_start."""
    if not hasattr(controller, "st"):
        controller.st = {"idx": 0, "pause_until": -1.0, "last_seen": -1.0, "wait_start": None,
                         "force_until": -1.0, "free_since": None, "near_hist": [], "near": float("inf"),
                         "evade_until": -1.0, "evade_v": 0.0, "evade_w": 0.0,
                         "prev": None, "vm": 0.0, "wz": 0.0, "tracks": [],
                         "last_f": None, "last_b": None, "raw": np.zeros((0, 2))}
    st = controller.st
    x, y, yaw = pose
    _update_motion(st, t, pose)
    _perceive(st, t, pose)
    M = _track_array(st, t)
    if len(M):
        st["last_seen"] = t
    n = len(_WAYPOINTS)

    # ---- target selection (waypoints / goal alignment) ----
    if st["idx"] >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.45:
        st["idx"] = n - 1                       # pushed off the goal: go back
    aligning = st["idx"] >= n
    e_goal = _wrap_angle(GOAL[2] - yaw)
    if aligning:
        v_nom, w_nom = 0.0, _clamp(1.5 * e_goal, -0.6, 0.6)
    else:
        wp = _WAYPOINTS[st["idx"]]
        if st["idx"] < n - 1 and math.hypot(wp[0] - x, wp[1] - y) < 0.35:
            st["idx"] += 1
            wp = _WAYPOINTS[st["idx"]]
        v_nom, w_nom, dist = _follow(pose, wp)
        if st["idx"] == n - 1 and dist < 0.25:
            st["idx"] = n                       # alignment starts on the next call

    # ---- forced resume: 10 s after wait_start, ignore pause/_margin/_lidar_clear ----
    if t >= st["force_until"] and st["wait_start"] is not None and t - st["wait_start"] > 10.0:
        st["force_until"] = t + 2.5             # no pause re-entry during this window
        st["wait_start"] = None
        st["free_since"] = None
        st["pause_until"] = -1.0
        st["evade_until"] = -1.0
    if t < st["force_until"]:
        return v_nom, w_nom, False

    # ---- stop-and-wait: while paused, stand still ----
    if t < st["pause_until"]:
        st["free_since"] = None
        return 0.0, 0.0, False

    # ---- evasion (only when not paused): a mover would hit us even if we stopped now ----
    if t < st["evade_until"]:
        if _lidar_clear(st["raw"], st["evade_v"]):
            return st["evade_v"], st["evade_w"], False
        st["evade_until"] = -1.0
    hold, _ = _margin(pose, st["vm"], 0.0, 0.0, M, 2.5, 0.1)
    if hold < 0.0:
        best = None
        for cv, cw in ((v_nom, w_nom), (1.0, 0.0), (-0.8, 0.0), (0.0, 0.0)):
            if not _lidar_clear(st["raw"], cv):
                continue
            m, _ = _margin(pose, st["vm"], cv, cw, M, 2.5, 0.1)
            if best is None or m > best[0] + 0.02:
                best = (m, cv, cw)
        if best is not None and (best[1] != 0.0 or best[2] != 0.0):
            st["evade_v"], st["evade_w"] = best[1], best[2]
            st["evade_until"] = t + 0.4
            return best[1], best[2], False
        return 0.0, 0.0, False

    # ---- goal alignment: finish only when nothing is about to reach us ----
    if aligning:
        if abs(e_goal) < 0.05:
            if _margin(pose, st["vm"], 0.0, 0.0, M, 4.0, 0.15)[0] >= 0.0:
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        _mark_free(st, t)
        return 0.0, w_nom, False

    # ---- driving: rotate in place, or drive at the largest predicted-safe speed ----
    if v_nom <= 0.0:
        _mark_free(st, t)
        return 0.0, w_nom, False
    for fac in (1.0, 0.6, 0.3):
        m, _ = _margin(pose, st["vm"], v_nom * fac, w_nom, M, 3.0, 0.2)
        if m >= 0.0 and _lidar_clear(st["raw"], v_nom * fac, 0.35):
            _mark_free(st, t)
            return v_nom * fac, w_nom, False

    # blocked: stop and wait 1.0 s
    st["pause_until"] = t + 1.0
    st["free_since"] = None
    if st["wait_start"] is None:
        st["wait_start"] = t
    return 0.0, 0.0, False




_R_ROBOT = 0.42


_R_MOVER = 0.43


_ROBOT_CIRCLES = (0.01, -0.46)


_STATIC_RECTS = np.array(
    [(b["x"] - 0.5 * b["width"], b["x"] + 0.5 * b["width"],
      b["y"] - 0.5 * b["depth"], b["y"] + 0.5 * b["depth"]) for b in MISSION["boxes"]]
    + [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES], dtype=np.float64)


def _is_static_world(W, mg):
    """True where a world point (N,2+) lies on a wall or a mapped static obstacle (margin mg: scalar or (N,))."""
    x, y = W[:, 0], W[:, 1]
    m = (x <= WALLS[0] + mg) | (x >= WALLS[1] - mg) | (y <= WALLS[2] + mg) | (y >= WALLS[3] - mg)
    R = _STATIC_RECTS
    mgc = np.asarray(mg, dtype=np.float64).reshape(-1, 1) if np.ndim(mg) else mg
    inr = ((x[:, None] >= R[None, :, 0] - mgc) & (x[:, None] <= R[None, :, 1] + mgc)
           & (y[:, None] >= R[None, :, 2] - mgc) & (y[:, None] <= R[None, :, 3] + mgc)).any(axis=1)
    return m | inr


def _map_free(x, y, r):
    """True if a disc of radius r at (x, y) clears the walls and every static rectangle."""
    if x < WALLS[0] + r or x > WALLS[1] - r or y < WALLS[2] + r or y > WALLS[3] - r:
        return False
    R = _STATIC_RECTS
    dx = np.maximum(np.maximum(R[:, 0] - x, x - R[:, 1]), 0.0)
    dy = np.maximum(np.maximum(R[:, 2] - y, y - R[:, 3]), 0.0)
    return bool(np.all(np.hypot(dx, dy) > r))


def _segment_free(p, q, r):
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.2) + 1)
    for k in range(n + 1):
        s = k / n
        if not _map_free(p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s, r):
            return False
    return True


def _follow(pose, tgt):
    """The original verified tracking law toward tgt; returns (v, w, dist)."""
    x, y, yaw = pose
    dx, dy = tgt[0] - x, tgt[1] - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)
    w = _clamp(7.5 * e, -2.0, 2.0)
    if abs(e) > 0.35:
        return 0.0, w, dist
    return min(1.5, 2.5 * dist + 0.5), w, dist


def _update_motion(st, t, pose):
    """Measured forward speed and yaw rate from successive poses."""
    if st["prev"] is not None:
        pt, (px, py, ph) = st["prev"]
        dt = t - pt
        if dt > 1e-4:
            vm = ((pose[0] - px) * math.cos(pose[2]) + (pose[1] - py) * math.sin(pose[2])) / dt
            wz = _wrap_angle(pose[2] - ph) / dt
            st["vm"] = 0.7 * st["vm"] + 0.3 * vm
            st["wz"] = 0.7 * st["wz"] + 0.3 * wz
    st["prev"] = (t, (pose[0], pose[1], pose[2]))


def _perceive(st, t, pose):
    """Front+back 2D lidar -> drop static-map points -> record nearest non-map distance (0.5 s history,
    pause 1.0 s if <= 1.0 m and decreasing) -> cluster -> track movers (position, velocity)."""
    f = lidar2d_points("front")
    b = lidar2d_points("back")
    if f is st["last_f"] and b is st["last_b"]:
        return
    st["last_f"], st["last_b"] = f, b
    P = np.concatenate([np.asarray(f).reshape(-1, 2), np.asarray(b).reshape(-1, 2)])
    st["raw"] = P
    D = np.zeros((0, 2))
    near = float("inf")
    if len(P):
        r = np.hypot(P[:, 0], P[:, 1])
        keep = r < 8.0
        P, r = P[keep], r[keep]
        if len(P):
            W = to_world(P, pose)
            mg = 0.25 + 0.12 * r * abs(st["wz"]) + 0.05 * abs(st["vm"])
            dyn = ~_is_static_world(W, mg)
            D, Dr = W[dyn][:, :2], r[dyn]
            if len(Dr):
                near = float(Dr.min())
            D = D[np.argsort(Dr)][:1500]

    # nearest non-map distance, 0.5 s history, pause rule
    hist = [h for h in st["near_hist"] if t - h[0] <= 0.5]
    old = hist[0][1] if hist else None
    hist.append((t, near))
    st["near_hist"] = hist
    st["near"] = near
    if (t >= st["force_until"] and near <= 1.0 and old is not None and near < old - 0.02):
        st["pause_until"] = t + 1.0
        st["free_since"] = None
        if st["wait_start"] is None:
            st["wait_start"] = t

    cents = []
    it = 0
    while len(D) and len(cents) < 10 and it < 30:
        it += 1
        sd = D[0]
        m = np.hypot(D[:, 0] - sd[0], D[:, 1] - sd[1]) < 0.9
        if int(m.sum()) >= 4:
            c = D[m].mean(axis=0)
            dx, dy = c[0] - pose[0], c[1] - pose[1]
            L = math.hypot(dx, dy)
            if L > 1e-6:
                c = (c[0] + 0.2 * dx / L, c[1] + 0.2 * dy / L)   # visible face -> box centre
            cents.append((float(c[0]), float(c[1])))
        D = D[~m]
    tracks = st["tracks"]
    used = set()
    new = []
    for c in cents:
        bi, bd = -1, 0.9
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            age = t - tr["t"]
            d = math.hypot(c[0] - tr["x"] - tr["vx"] * age, c[1] - tr["y"] - tr["vy"] * age)
            if d < bd:
                bi, bd = i, d
        if bi < 0:
            new.append({"x": c[0], "y": c[1], "vx": 0.0, "vy": 0.0, "t": t, "t0": t, "n": 1,
                        "hist": [(t, c[0], c[1])]})
            continue
        used.add(bi)
        tr = tracks[bi]
        tr["hist"] = [h for h in tr["hist"] if t - h[0] <= 0.6] + [(t, c[0], c[1])]
        t0, x0, y0 = tr["hist"][0]
        if t - t0 >= 0.25:
            vx, vy = (c[0] - x0) / (t - t0), (c[1] - y0) / (t - t0)
            sp = math.hypot(vx, vy)
            if sp > 2.0:
                vx, vy = vx * 2.0 / sp, vy * 2.0 / sp
            tr["vx"], tr["vy"] = vx, vy
        tr["x"], tr["y"], tr["t"] = c[0], c[1], t
        tr["n"] += 1
    st["tracks"] = [tr for i, tr in enumerate(tracks) if i in used or t - tr["t"] < 1.0] + new


def _track_array(st, t):
    """Confirmed tracks as rows (x, y, vx, vy, age, uncertainty growth per s)."""
    rows = []
    for tr in st["tracks"]:
        if tr["n"] < 2:
            continue
        sp = math.hypot(tr["vx"], tr["vy"])
        g = 0.3 if t - tr["t0"] < 0.3 else (0.15 if sp > 0.1 else 0.05)
        rows.append((tr["x"], tr["y"], tr["vx"], tr["vy"], t - tr["t"], g))
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def _margin(pose, vm, v, w, M, T, base):
    """Predict robot (command v, w with accel limits) and movers (constant velocity) over T s.
    Returns (min over time/tracks of separation - required, index of worst track); >= 0 means safe."""
    if len(M) == 0:
        return 9.0, -1
    n = int(round(T / 0.1)) + 1
    ts = np.arange(n) * 0.1
    xs, ys, hs = np.empty(n), np.empty(n), np.empty(n)
    x, y, h = pose
    s = vm
    for k in range(n):
        xs[k], ys[k], hs[k] = x, y, h
        a = 3.0 if (abs(v) > abs(s) and v * s >= 0.0) else 1.2
        s += max(-a * 0.1, min(a * 0.1, v - s))
        x += s * math.cos(h) * 0.1
        y += s * math.sin(h) * 0.1
        h += w * 0.1
    tt = ts[:, None] + M[None, :, 4]
    mx = M[None, :, 0] + M[None, :, 2] * tt
    my = M[None, :, 1] + M[None, :, 3] * tt
    req = base + _R_ROBOT + _R_MOVER + ts[:, None] * M[None, :, 5]
    worst = np.full(len(M), 9.0)
    for off in _ROBOT_CIRCLES:
        cx = (xs + off * np.cos(hs))[:, None]
        cy = (ys + off * np.sin(hs))[:, None]
        worst = np.minimum(worst, (np.hypot(mx - cx, my - cy) - req).min(axis=0))
    j = int(np.argmin(worst))
    return float(worst[j]), j


def _lidar_clear(P, v, reach=0.6):
    """No lidar point (static or moving) in the body-wide strip just ahead (v>0) / behind (v<0)."""
    if len(P) == 0 or abs(v) < 1e-3:
        return True
    side = np.abs(P[:, 1]) < 0.45
    if v > 0:
        m = side & (P[:, 0] > _BODY[1]) & (P[:, 0] < _BODY[1] + reach)
    else:
        m = side & (P[:, 0] < _BODY[0]) & (P[:, 0] > _BODY[0] - reach)
    return not bool(m.any())


def _plan_detour(pose, ux, uy, side_pref, fwds, target):
    """Temporary side waypoint (path frame u, left normal n) that is clear of the static map and
    from which the straight line to target is clear too."""
    x, y = pose[0], pose[1]
    nx, ny = -uy, ux
    for side in (side_pref, -side_pref):
        for lat in (1.6, 1.2):
            for fw in fwds:
                px = x + ux * fw + nx * side * lat
                py = y + uy * fw + ny * side * lat
                if (_map_free(px, py, 0.8) and _segment_free((x, y), (px, py), 0.5)
                        and _segment_free((px, py), target, 0.5)):
                    return (px, py)
    return None




def _mark_free(st, t):
    """Called on normal (unpaused) motion: clear wait_start only after 2 s of uninterrupted motion,
    so short drive/pause cycles still reach the 10 s forced resume."""
    if st["free_since"] is None:
        st["free_since"] = t
    elif t - st["free_since"] > 2.0:
        st["wait_start"] = None


# ========================= [END EDIT REGION] ===============================
