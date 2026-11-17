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
    """Waypoint following + lidar mover tracking.  Priority: (1) VO local planner (_vo_planner) whenever standing
    still would be hit within _HORIZON or a mover is within _SAFE_DIST; _plan_escape only when VO finds no safe
    (v, w), (2) goal alignment, (3) driving at the largest predicted-safe speed, pause / stuck detour when blocked.
    In-place turns and fallback commands pass through _guard (swept-body lidar check)."""
    if not hasattr(controller, "st"):
        controller.st = {"idx": 0, "pause_until": -1.0, "last_seen": -1.0, "wait_start": None,
                         "free_since": None, "prev": None, "vm": 0.0, "wz": 0.0, "tracks": [],
                         "last_f": None, "last_b": None, "raw": np.zeros((0, 2)), "ttc": float("inf"),
                         "esc": None, "esc_t": -1.0, "esc_until": -1.0, "detour": None,
                         "av": None, "av_t": -1.0, "vo_prev": None, "vo_on": False}
    st = controller.st
    x, y, yaw = pose
    _update_motion(st, t, pose)
    _perceive(st, t, pose)
    M = _track_array(st, t)
    if len(M):
        st["last_seen"] = t
    n = len(_WAYPOINTS)

    # ---- target selection (waypoints / temporary detour / goal alignment) ----
    if st["idx"] >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.45:
        st["idx"] = n - 1                       # pushed off the goal: go back
    aligning = st["idx"] >= n
    e_goal = _wrap_angle(GOAL[2] - yaw)
    if st["detour"] is not None and math.hypot(st["detour"][0] - x, st["detour"][1] - y) < 0.4:
        st["detour"] = None                     # detour reached: resume the waypoint path
    wp = (GOAL[0], GOAL[1])
    tgt = wp
    if aligning:
        v_nom, w_nom = 0.0, _clamp(1.5 * e_goal, -0.6, 0.6)
    else:
        wp = _WAYPOINTS[st["idx"]]
        if st["idx"] < n - 1 and math.hypot(wp[0] - x, wp[1] - y) < 0.35:
            st["idx"] += 1
            wp = _WAYPOINTS[st["idx"]]
        tgt = st["detour"] if st["detour"] is not None else wp
        v_nom, w_nom, dist = _follow(pose, tgt)
        if st["detour"] is None and st["idx"] == n - 1 and dist < 0.25:
            st["idx"] = n                       # alignment starts on the next call

    # ---- movers: VO local planner (evasion and near movers); _plan_escape only as fallback ----
    hold, _ = _margin(pose, st["vm"], 0.0, 0.0, M, _HORIZON, 0.15)
    gap = _min_gap(pose, M)
    if hold < 0.0 or (gap <= _SAFE_DIST and not aligning):
        st["pause_until"] = -1.0
        st["vo_on"] = True
        cmd = _vo_planner(st, pose, tgt, M)
        if cmd is not None:
            st["esc"] = None
            if cmd[0] > 0.05:
                _mark_free(st, t)
            else:
                st["free_since"] = None
                if st["wait_start"] is None:
                    st["wait_start"] = t
            return cmd[0], cmd[1], False
        st["vo_prev"] = None
        st["free_since"] = None
        if st["wait_start"] is None:
            st["wait_start"] = t
        if st["esc"] is None or t - st["esc_t"] >= 0.2:
            st["esc"] = _plan_escape(st, pose, M, wp)
            st["esc_t"] = t
        v, w, _ = _escape_cmd(st, pose)         # includes the _lidar_clear strip check
        v, w = _guard(st, pose, M, v, w)
        return v, w, False
    if st["vo_on"]:                             # left VO / evasion: rejoin the path safely
        st["vo_on"] = False
        st["esc"] = None
        st["vo_prev"] = None
        if not aligning:
            _set_return(st, pose)

    # ---- dynamic pause: only kept while a mover is within _SAFE_DIST ----
    if t < st["pause_until"]:
        if _should_pause(st, pose, M):
            st["free_since"] = None
            return 0.0, 0.0, False
        st["pause_until"] = -1.0

    # ---- goal alignment: finish only when nothing is about to reach us ----
    if aligning:
        if abs(e_goal) < 0.05:
            if _margin(pose, st["vm"], 0.0, 0.0, M, 4.0, 0.15)[0] >= 0.0:
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        _mark_free(st, t)
        v, w = _guard(st, pose, M, 0.0, w_nom)
        return v, w, False

    # ---- driving: rotate in place (swept-body guarded), or drive at the largest predicted-safe speed ----
    if v_nom <= 0.0:
        _mark_free(st, t)
        v, w = _guard(st, pose, M, 0.0, w_nom)
        return v, w, False
    vc = _safe_speed(st, pose, tgt, v_nom, M)
    if vc is None:
        vc = _predictive_speed(st, pose, tgt, v_nom, M)
    if vc is not None:
        _mark_free(st, t)
        return vc, w_nom, False

    # ---- blocked by a farther mover: TTC-based pause; if blocked for long, go around the blocker ----
    pause, _ = _dynamic_pause(st, pose, M)
    st["pause_until"] = t + (pause if pause is not None else 0.5)
    st["free_since"] = None
    if st["wait_start"] is None:
        st["wait_start"] = t
    elif t - st["wait_start"] > 3.0 and st["detour"] is None:
        d = _stuck_detour(st, pose, M, wp)
        if d is not None:
            st["detour"] = d
            st["wait_start"] = None
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
    """Front+back 2D lidar -> drop static-map points -> cluster -> track movers (position, velocity)
    -> TTC-based dynamic pause (only while a mover is within _SAFE_DIST)."""
    f = lidar2d_points("front")
    b = lidar2d_points("back")
    if f is st["last_f"] and b is st["last_b"]:
        return
    st["last_f"], st["last_b"] = f, b
    P = np.concatenate([np.asarray(f).reshape(-1, 2), np.asarray(b).reshape(-1, 2)])
    st["raw"] = P
    D = np.zeros((0, 2))
    if len(P):
        r = np.hypot(P[:, 0], P[:, 1])
        keep = r < 8.0
        P, r = P[keep], r[keep]
        if len(P):
            W = to_world(P, pose)
            mg = 0.25 + 0.12 * r * abs(st["wz"]) + 0.05 * abs(st["vm"])
            dyn = ~_is_static_world(W, mg)
            D, Dr = W[dyn][:, :2], r[dyn]
            D = D[np.argsort(Dr)][:1500]

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

    # dynamic pause from time-to-collision (replaces the fixed 1.0 m / 1.0 s rule)
    Mt = _track_array(st, t)
    pause, ttc = _dynamic_pause(st, pose, Mt)
    st["ttc"] = ttc
    if pause is not None and _should_pause(st, pose, Mt):
        st["pause_until"] = max(st["pause_until"], t + pause)
        st["free_since"] = None
        if st["wait_start"] is None:
            st["wait_start"] = t


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
    """Predict robot (constant command v, w with accel limits) and movers (constant velocity) over T s.
    Returns (min over time/tracks of separation - required, index of worst track); >= 0 means safe."""
    if len(M) == 0:
        return 9.0, -1
    xs, ys, hs = _rollout(pose, vm, T, v=v, w=w)
    return _traj_margin(xs, ys, hs, M, base)


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




_HORIZON = 3.5


_SAFE_DIST = 1.7


def _rollout(pose, vm, T, v=0.0, w=0.0, tgt=None, vmax=1.5):
    """Robot poses every 0.1 s over T s: constant command (v, w), or the _follow law toward tgt
    (speed capped at vmax, stopping at tgt); accel 3.0 m/s^2, decel 1.2 m/s^2."""
    n = int(round(T / 0.1)) + 1
    xs, ys, hs = np.empty(n), np.empty(n), np.empty(n)
    x, y, h = pose
    s = vm
    for k in range(n):
        xs[k], ys[k], hs[k] = x, y, h
        if tgt is not None:
            vv, ww, d = _follow((x, y, h), tgt)
            if d < 0.15:
                vv, ww = 0.0, 0.0
            vv = min(vv, vmax)
        else:
            vv, ww = v, w
        a = 3.0 if (abs(vv) > abs(s) and vv * s >= 0.0) else 1.2
        s += max(-a * 0.1, min(a * 0.1, vv - s))
        x += s * math.cos(h) * 0.1
        y += s * math.sin(h) * 0.1
        h += ww * 0.1
    return xs, ys, hs


def _traj_margin(xs, ys, hs, M, base):
    """Min over time/tracks of (robot-circle to mover distance - required) for a predicted robot trajectory."""
    if len(M) == 0:
        return 9.0, -1
    ts = np.arange(len(xs)) * 0.1
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


def _traj_static_ok(xs, ys, hs, need=0.03):
    """True if the predicted trajectory keeps both robot circles (radius _R_ROBOT, which also covers the body
    corners) at least `need` from the walls and mapped obstacles - or, when the start pose is already closer,
    never more than 0.02 m closer than at the start (so an in-place turn cannot swing the body into a box)."""
    c = _static_clear(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64),
                      np.asarray(hs, dtype=np.float64))
    if len(c) < 2:
        return True
    return bool(c[1:].min() >= min(need, c[0] - 0.02))


def _min_gap(pose, M):
    """Smallest current clearance (centre distance - radii) between robot circles and any tracked mover."""
    if len(M) == 0:
        return float("inf")
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    mx = M[:, 0] + M[:, 2] * M[:, 4]
    my = M[:, 1] + M[:, 3] * M[:, 4]
    g = float("inf")
    for off in _ROBOT_CIRCLES:
        d = np.hypot(mx - (x + off * c), my - (y + off * s)) - _R_ROBOT - _R_MOVER
        g = min(g, float(d.min()))
    return g


def _dynamic_pause(st, pose, M):
    """Time to collision t_tc per track (robot moving at measured vm along heading, mover at constant velocity,
    contact radius R_ROBOT+R_MOVER+0.25) -> pause = max(0, t_tc - t_stop) + 0.5 with t_stop = |vm| / 1.0,
    clamped to [0.5, 3.0] s.  Returns (pause or None when t_tc > _HORIZON, t_tc)."""
    if len(M) == 0:
        return None, float("inf")
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    vm = st["vm"]
    rvx, rvy = vm * c, vm * s
    R = _R_ROBOT + _R_MOVER + 0.25
    ttc = float("inf")
    for row in M:
        mx = row[0] + row[2] * row[4]
        my = row[1] + row[3] * row[4]
        rx, ry = row[2] - rvx, row[3] - rvy
        for off in _ROBOT_CIRCLES:
            px = mx - (x + off * c)
            py = my - (y + off * s)
            a = rx * rx + ry * ry
            b = 2.0 * (px * rx + py * ry)
            cc = px * px + py * py - R * R
            if a < 1e-6 or b >= 0.0:
                continue                       # not closing
            if cc <= 0.0:
                tc = 0.0
            else:
                disc = b * b - 4.0 * a * cc
                if disc < 0.0:
                    continue
                tc = (-b - math.sqrt(disc)) / (2.0 * a)
            ttc = min(ttc, tc)
    if ttc > _HORIZON:
        return None, ttc
    t_stop = abs(vm) / 1.0
    return _clamp(max(0.0, ttc - t_stop) + 0.5, 0.5, 3.0), ttc


def _should_pause(st, pose, M):
    """Keep pausing only while some tracked mover is within _SAFE_DIST; farther away, resume normal driving."""
    return _min_gap(pose, M) <= _SAFE_DIST


def _unit(ux, uy):
    L = math.hypot(ux, uy)
    return (ux / L, uy / L) if L > 1e-6 else (1.0, 0.0)


def _plan_escape(st, pose, M, target):
    """Choose the evasive action with the largest predicted clearance over _HORIZON: side points
    perpendicular to / away from the worst mover, a _plan_detour side waypoint, backing up, driving on, stopping.
    Candidates must keep clear of the static map and have a lidar-free strip in the driving direction."""
    x, y, yaw = pose
    vm = st["vm"]
    _, j = _margin(pose, vm, 0.0, 0.0, M, _HORIZON, 0.15)
    j = max(j, 0)
    age = M[j, 4]
    mx, my = M[j, 0] + M[j, 2] * age, M[j, 1] + M[j, 3] * age
    ax, ay = x - mx, y - my
    if math.hypot(ax, ay) > 1e-6:
        ax, ay = _unit(ax, ay)
    else:
        ax, ay = -math.cos(yaw), -math.sin(yaw)
    sp = math.hypot(M[j, 2], M[j, 3])
    if sp > 0.1:
        px, py = -M[j, 3] / sp, M[j, 2] / sp
        if px * ax + py * ay < 0.0:
            px, py = -px, -py
    else:
        px, py = -ay, ax
    dirs = [(px, py), (ax, ay), _unit(px + ax, py + ay), (-px, -py)]
    cands = [("cmd", (0.0, 0.0)), ("cmd", (-0.8, 0.0)), ("cmd", (1.0, 0.0)),
             ("cmd", (-0.8, 0.6)), ("cmd", (-0.8, -0.6))]
    for dist in (1.6, 2.6):
        for ux, uy in dirs:
            tx, ty = x + ux * dist, y + uy * dist
            if _map_free(tx, ty, 0.55):
                cands.append(("goto", (tx, ty)))
    tvx, tvy = target[0] - x, target[1] - y
    if math.hypot(tvx, tvy) > 0.5:
        ux, uy = _unit(tvx, tvy)
        side = 1 if (-uy * px + ux * py) >= 0.0 else -1
        d = _plan_detour(pose, ux, uy, side, (0.0, 1.0, -1.0), target)
        if d is not None:
            cands.append(("goto", (float(d[0]), float(d[1]))))
    if st["esc"] is not None and st["esc"] not in cands:
        cands.append(st["esc"])
    best, best_s = None, -1e9
    for cand in cands:
        kind, val = cand
        if kind == "cmd":
            xs, ys, hs = _rollout(pose, vm, _HORIZON, v=val[0], w=val[1])
            v0 = val[0]
        else:
            xs, ys, hs = _rollout(pose, vm, _HORIZON, tgt=val)
            v0 = _follow(pose, val)[0]
        if abs(v0) > 1e-3 and not _lidar_clear(st["raw"], v0, 0.5):
            continue
        if not _traj_static_ok(xs, ys, hs):
            continue
        m, _ = _traj_margin(xs, ys, hs, M, 0.15)
        score = m + (0.08 if cand == st["esc"] else 0.0)
        if score > best_s:
            best, best_s = cand, score
    return best if best is not None else ("cmd", (0.0, 0.0))


def _escape_cmd(st, pose):
    """Command for the current evasive action; forward/backward motion is cut if the lidar strip is blocked."""
    kind, val = st["esc"]
    if kind == "cmd":
        v, w = val
    else:
        v, w, d = _follow(pose, val)
        if d < 0.2:
            v, w = 0.0, 0.0
    if abs(v) > 1e-3 and not _lidar_clear(st["raw"], v, 0.4):
        v = 0.0
    return v, w, False


def _set_return(st, pose):
    """After an evasion: if the straight line back to the current waypoint crosses the static map,
    go first to a point on the current path leg."""
    n = len(_WAYPOINTS)
    i = min(st["idx"], n - 1)
    wp = _WAYPOINTS[i]
    prev = _WAYPOINTS[i - 1] if i > 0 else (START[0], START[1])
    x, y = pose[0], pose[1]
    if _segment_free((x, y), wp, 0.45):
        st["detour"] = None
        return
    lx, ly = wp[0] - prev[0], wp[1] - prev[1]
    L = math.hypot(lx, ly)
    if L < 1e-6:
        st["detour"] = None
        return
    ux, uy = lx / L, ly / L
    s = _clamp((x - prev[0]) * ux + (y - prev[1]) * uy, 0.0, L)
    s = min(L, s + 0.5)
    rp = (prev[0] + ux * s, prev[1] + uy * s)
    st["detour"] = rp if _segment_free((x, y), rp, 0.45) else None


def _stuck_detour(st, pose, M, target):
    """Long blockage: side waypoint (via _plan_detour) on the side away from the nearest mover."""
    x, y = pose[0], pose[1]
    tvx, tvy = target[0] - x, target[1] - y
    if math.hypot(tvx, tvy) < 0.5:
        return None
    ux, uy = _unit(tvx, tvy)
    side = 1
    if len(M):
        k = int(np.argmin(np.hypot(M[:, 0] - x, M[:, 1] - y)))
        dx, dy = M[k, 0] - x, M[k, 1] - y
        side = -1 if (-uy * dx + ux * dy) > 0.0 else 1
    return _plan_detour(pose, ux, uy, side, (1.5, 1.0, 0.5), target)




def _safe_speed(st, pose, tgt, v_nom, M):
    """Largest of v_nom * (1.0, 0.6, 0.3) whose predicted path toward tgt keeps margin >= 0 from every
    tracked mover over _HORIZON and whose lidar strip ahead is clear; None if none is safe."""
    for fac in (1.0, 0.6, 0.3):
        vc = v_nom * fac
        xs, ys, hs = _rollout(pose, st["vm"], _HORIZON, tgt=tgt, vmax=vc)
        m, _ = _traj_margin(xs, ys, hs, M, 0.3)
        if m >= 0.0 and _lidar_clear(st["raw"], vc, 0.35):
            return vc
    return None


def _avoid_detour(st, pose, M, target):
    """Side detour point around the nearest mover.  Candidates: side points left/right of the line to target
    (lateral 1.6 / 1.2 / 0.8 m x forward 1.5 / 0.8 / 0.0 m, the side away from the nearest mover preferred)
    plus the current detour (hysteresis).  Every candidate must clear the static map, have a clear straight
    line on to target, a free lidar strip, a statically clear predicted trajectory and margin >= 0 from every
    tracked mover over _HORIZON.  The 0.1 * forward bonus is added only to side points (compared among
    themselves); the kept detour competes on margin + hysteresis bonus.  Returns the best (x, y) or None."""
    x, y = pose[0], pose[1]
    tvx, tvy = target[0] - x, target[1] - y
    tdist = math.hypot(tvx, tvy)
    if tdist < 1e-3:
        return None
    ux, uy = _unit(tvx, tvy)
    nx, ny = -uy, ux
    side_pref = 1
    if len(M):
        mx = M[:, 0] + M[:, 2] * M[:, 4]
        my = M[:, 1] + M[:, 3] * M[:, 4]
        k = int(np.argmin(np.hypot(mx - x, my - y)))
        side_pref = -1 if (nx * (mx[k] - x) + ny * (my[k] - y)) > 0.0 else 1
    cands = []                                                  # (x, y, bonus, forward bonus)
    for side in (side_pref, -side_pref):
        for lat in (1.6, 1.2, 0.8):
            for fw in (1.5, 0.8, 0.0):
                cands.append((x + ux * fw + nx * side * lat, y + uy * fw + ny * side * lat,
                              0.1 if side == side_pref else 0.0, 0.1 * fw))
    if st["av"] is not None:
        cands.append((st["av"][0], st["av"][1], 0.15, 0.0))     # hysteresis: keep the current detour
    best, best_s = None, -1e9
    for px, py, bonus, fwb in cands:
        if math.hypot(px - x, py - y) < 0.3:
            continue
        if not (_map_free(px, py, 0.55) and _segment_free((x, y), (px, py), 0.4)
                and _segment_free((px, py), target, 0.45)):
            continue
        xs, ys, hs = _rollout(pose, st["vm"], _HORIZON, tgt=(px, py))
        if not _traj_static_ok(xs, ys, hs):
            continue
        v0 = _follow(pose, (px, py))[0]
        if v0 > 1e-3 and not _lidar_clear(st["raw"], v0, 0.5):
            continue
        m, _ = _traj_margin(xs, ys, hs, M, 0.15)
        if m < 0.0:
            continue
        s = min(m, 1.0) + bonus + fwb
        if s > best_s:
            best, best_s = (float(px), float(py)), s
    return best




_TTC_SLOW = 2.0


_SHORT_T = 1.5


def _predictive_speed(st, pose, tgt, v_nom, M):
    """TTC-based continuous speed scaling.  ratio = clamp(ttc / _HORIZON, 0.2, 1) (1 when ttc >= _HORIZON);
    candidates v_nom * ratio * (1.0, 0.6, 0.3) (at least 0.15 m/s).  Returns the fastest candidate whose rollout
    toward tgt keeps margin >= 0 from every mover over _HORIZON.  If none does but ttc >= _TTC_SLOW, the slowest
    candidate that is safe over _SHORT_T and no much worse than standing still over _HORIZON is returned
    (re-checked every step).  None when nothing qualifies or the lidar strip ahead is blocked."""
    if v_nom <= 0.0:
        return None
    if not _lidar_clear(st["raw"], v_nom, 0.35):
        return None
    if len(M) == 0:
        return v_nom
    _, ttc = _dynamic_pause(st, pose, M)
    ratio = 1.0 if ttc >= _HORIZON else _clamp(ttc / _HORIZON, 0.2, 1.0)
    mh, _ = _margin(pose, st["vm"], 0.0, 0.0, M, _HORIZON, 0.15)
    ns = int(round(_SHORT_T / 0.1)) + 1
    slow = None
    for fac in (1.0, 0.6, 0.3):
        vc = _clamp(v_nom * ratio * fac, min(0.15, v_nom), v_nom)
        xs, ys, hs = _rollout(pose, st["vm"], _HORIZON, tgt=tgt, vmax=vc)
        m, _ = _traj_margin(xs, ys, hs, M, 0.15)
        if m >= 0.0:
            return vc
        if ttc >= _TTC_SLOW and m >= mh - 0.3:
            ms, _ = _traj_margin(xs[:ns], ys[:ns], hs[:ns], M, 0.1)
            if ms >= 0.0:
                slow = vc
    return slow




_VO_HORIZON = 2.0


_VO_SPEED_STEPS = 5


_VO_MARGIN = 0.15


_VO_DRIVE_T = 1.5


_VO_DELTAS = (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.3, -1.3)


def _static_dist(px, py):
    """Distance from world points to the nearest wall face or mapped static rectangle (0 inside a rectangle)."""
    px = np.asarray(px, dtype=np.float64).reshape(-1)
    py = np.asarray(py, dtype=np.float64).reshape(-1)
    R = _STATIC_RECTS
    dx = np.maximum(np.maximum(R[None, :, 0] - px[:, None], px[:, None] - R[None, :, 1]), 0.0)
    dy = np.maximum(np.maximum(R[None, :, 2] - py[:, None], py[:, None] - R[None, :, 3]), 0.0)
    d = np.hypot(dx, dy).min(axis=1)
    wall = np.minimum(np.minimum(px - WALLS[0], WALLS[1] - px), np.minimum(py - WALLS[2], WALLS[3] - py))
    return np.minimum(d, wall)


def _static_clear(xs, ys, hs):
    """Per pose: min over the two robot circles of (static distance - _R_ROBOT)."""
    out = None
    for off in _ROBOT_CIRCLES:
        d = _static_dist(xs + off * np.cos(hs), ys + off * np.sin(hs)) - _R_ROBOT
        out = d if out is None else np.minimum(out, d)
    return out


def _sweep_clear(P, v, w, T=0.5, turn=None):
    """Swept-body check on raw lidar points (robot frame, static and moving): False if any point outside the
    body (+0.05 m) would enter it while executing (v, w) for T s (heading change capped at |turn| if given)."""
    if len(P) == 0 or (abs(v) < 1e-3 and abs(w) < 1e-3):
        return True
    r = np.hypot(P[:, 0], P[:, 1])
    Q = P[r < 0.9 + abs(v) * T]
    if len(Q) == 0:
        return True
    mg = 0.05

    def inside(qx, qy):
        return (qx >= _BODY[0] - mg) & (qx <= _BODY[1] + mg) & (qy >= _BODY[2] - mg) & (qy <= _BODY[3] + mg)

    Q = Q[~inside(Q[:, 0], Q[:, 1])]
    if len(Q) == 0:
        return True
    lim = float("inf") if turn is None else abs(turn)
    xr = yr = h = 0.0
    for _ in range(max(1, int(round(T / 0.05)))):
        xr += v * math.cos(h) * 0.05
        yr += v * math.sin(h) * 0.05
        h = _clamp(h + w * 0.05, -lim, lim)
        c, s = math.cos(h), math.sin(h)
        dx, dy = Q[:, 0] - xr, Q[:, 1] - yr
        if inside(c * dx + s * dy, -s * dx + c * dy).any():
            return False
    return True


def _guard(st, pose, M, v, w):
    """Final check of a command: keep (v, w) if its swept body and lidar strip are clear; otherwise creep
    forward/back 0.3 m/s or stop, choosing the option with the best mover margin + static clearance."""
    P = st["raw"]
    if _sweep_clear(P, v, w, 0.5) and (abs(v) < 1e-3 or _lidar_clear(P, v, 0.3)):
        return v, w
    best, best_s = (0.0, 0.0), -1e9
    for cv in (0.3, -0.3, 0.0):
        if abs(cv) > 1e-3 and not (_sweep_clear(P, cv, 0.0, 0.5) and _lidar_clear(P, cv, 0.35)):
            continue
        xs, ys, hs = _rollout(pose, st["vm"], 1.5, v=cv, w=0.0)
        if not _traj_static_ok(xs, ys, hs):
            continue
        m = _traj_margin(xs, ys, hs, M, 0.05)[0]
        sc = float(_static_clear(xs[5:6], ys[5:6], hs[5:6])[0])
        s = min(m, 0.5) + min(sc, 0.3)
        if s > best_s:
            best, best_s = (cv, 0.0), s
    return best


def _compute_vo(pose, M, horizon):
    """Velocity obstacles of the tracked movers, one row per (track, robot circle):
    (px, py, mvx, mvy, R, g) = mover centre relative to the robot circle (extrapolated by track age), mover
    velocity, combined radius _R_ROBOT + _R_MOVER + _VO_MARGIN, uncertainty growth.  A robot velocity V is in the
    VO of a row when p - (V - mv) * tau comes within R + 0.5 g tau for some tau in (0, horizon].
    Returns (rows, horizon)."""
    if len(M) == 0:
        return np.zeros((0, 6)), horizon
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    mx = M[:, 0] + M[:, 2] * M[:, 4]
    my = M[:, 1] + M[:, 3] * M[:, 4]
    R = np.full(len(M), _R_ROBOT + _R_MOVER + _VO_MARGIN)
    rows = [np.stack([mx - (x + off * c), my - (y + off * s), M[:, 2], M[:, 3], R, M[:, 5]], axis=1)
            for off in _ROBOT_CIRCLES]
    return np.concatenate(rows), horizon


def _vo_ttc(vo, vx, vy):
    """First time (s) the robot velocity (vx, vy) enters a velocity obstacle; 0 if already overlapping and
    closing; inf if no collision within the VO horizon."""
    rows, H = vo
    if len(rows) == 0:
        return float("inf")
    px, py, mvx, mvy, R, g = rows.T
    ux, uy = vx - mvx, vy - mvy
    d0 = np.hypot(px, py)
    if np.any((d0 < R) & (px * ux + py * uy > 0.0)):
        return 0.0
    ts = np.arange(1, int(round(H / 0.1)) + 1) * 0.1
    dx = px[:, None] - ux[:, None] * ts[None, :]
    dy = py[:, None] - uy[:, None] * ts[None, :]
    hit = (np.hypot(dx, dy) < R[:, None] + 0.5 * g[:, None] * ts[None, :]) & (d0 >= R)[:, None]
    if not hit.any():
        return float("inf")
    return float(ts[int(np.argmax(hit.any(axis=0)))])


def _vo_rollout(pose, vm, th, v, T, stop_xy=None, t_drive=None):
    """Robot poses every 0.1 s over T s when holding heading th (same 7.5 gain as _follow) and driving at v for
    t_drive s (forward only once |heading error| <= 0.6; stops near stop_xy), then stopping;
    accel 3.0 m/s^2, decel 1.2 m/s^2."""
    if t_drive is None:
        t_drive = _VO_DRIVE_T
    n = int(round(T / 0.1)) + 1
    xs, ys, hs = np.empty(n), np.empty(n), np.empty(n)
    x, y, h = pose
    s = vm
    for k in range(n):
        xs[k], ys[k], hs[k] = x, y, h
        e = _wrap_angle(th - h)
        ww = _clamp(7.5 * e, -2.0, 2.0)
        vv = v if (k * 0.1 < t_drive and (v < 0.0 or abs(e) <= 0.6)) else 0.0
        if stop_xy is not None and math.hypot(stop_xy[0] - x, stop_xy[1] - y) < 0.3:
            vv = 0.0
        a = 3.0 if (abs(vv) > abs(s) and vv * s >= 0.0) else 1.2
        s += max(-a * 0.1, min(a * 0.1, vv - s))
        x += s * math.cos(h) * 0.1
        y += s * math.sin(h) * 0.1
        h += ww * 0.1
    return xs, ys, hs


def _vo_planner(st, pose, tgt, M):
    """VO local planner.  Candidates: headings bearing(tgt) + _VO_DELTAS x _VO_SPEED_STEPS speeds in [0, vmax]
    (forward only if the heading is within 0.6 rad), plus stop and a 0.5 m/s back-up.  A candidate must be
    outside every velocity obstacle over _VO_HORIZON, have a clear lidar strip and swept body, keep clear of the
    static map, and its predicted motion (drive _VO_DRIVE_T s, then stop) must keep margin >= 0 from every mover
    over _HORIZON or beat standing still by 0.05.  Cheapest by progress, deviation, turn, margin and hysteresis.
    Returns (v, w) or None."""
    x, y, yaw = pose
    dx, dy = tgt[0] - x, tgt[1] - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) if dist > 1e-3 else yaw
    vmax = min(1.5, 2.5 * dist + 0.5)
    vo = _compute_vo(pose, M, _VO_HORIZON)
    P = st["raw"]
    vm = st["vm"]
    xs, ys, hs = _vo_rollout(pose, vm, yaw, 0.0, _HORIZON)
    hold_c = _traj_margin(xs, ys, hs, M, 0.05)[0]
    cands = [("stop", yaw, 0.0, 0.0), ("rev", yaw, -0.5, 0.0)]
    for d in _VO_DELTAS:
        th = _wrap_angle(bearing + d)
        e = _wrap_angle(th - yaw)
        w = _clamp(7.5 * e, -2.0, 2.0)
        for k in range(_VO_SPEED_STEPS):
            v = vmax * k / (_VO_SPEED_STEPS - 1)
            if v > 0.0 and abs(e) > 0.6:
                break
            cands.append((d, th, v, w))
    prev = st.get("vo_prev")
    best, best_c = None, float("inf")
    for d, th, v, w in cands:
        if _vo_ttc(vo, v * math.cos(th), v * math.sin(th)) <= vo[1]:
            continue
        if abs(v) > 1e-3 and not _lidar_clear(P, v, 0.3 + 0.4 * abs(v)):
            continue
        e = _wrap_angle(th - yaw)
        if not _sweep_clear(P, v, w, 0.5, abs(e)):
            continue
        xs, ys, hs = _vo_rollout(pose, vm, th, v, _HORIZON, stop_xy=(tgt if d == 0.0 else None))
        if not _traj_static_ok(xs, ys, hs):
            continue
        m = _traj_margin(xs, ys, hs, M, 0.05)[0]
        if m < 0.0 and m < hold_c + 0.05:
            continue
        if d == "stop":
            cost = 1.5
        else:
            prog = v * math.cos(_wrap_angle(th - bearing))
            cost = (vmax - prog) / max(vmax, 0.1) + 0.15 * abs(e)
            if d == "rev":
                cost += 0.6
            else:
                cost += 0.35 * abs(d)
        if prev is not None and d != prev:
            cost += 0.15
        cost -= 0.5 * _clamp(m, -0.5, 0.5)
        if cost < best_c:
            best, best_c = (d, v, w), cost
    if best is None:
        return None
    st["vo_prev"] = best[0]
    return best[1], best[2]


# ========================= [END EDIT REGION] ===============================
