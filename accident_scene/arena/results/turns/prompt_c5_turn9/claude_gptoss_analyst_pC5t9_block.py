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
    """Waypoint following + fused 2D/3D lidar mover tracking.  Priority: (1) evasion with the VO planner
    (_plan_escape fallback) when standing still would be hit within _HORIZON, (1a) raw-lidar proximity stop /
    back-up (_proximity_stop_and_backup, no tracks needed), (1b) raw-lidar TTC alarm (_ttc_stop_check): call the
    VO planner (strict); use its command when it moves the robot (pause_until reset), otherwise stop and extend
    pause_until; a TTC stop longer than _TTC_MAX_WAIT with no closing tracked mover disables the TTC alarm for
    _TTC_GRACE s, (2) immediate stop while a mover closing on the robot is within _STOP_DIST, (3) goal alignment,
    (4) VO cruise while a mover is within _SAFE_DIST, (5) normal driving at the largest predicted-safe speed;
    stop / stuck detour when blocked."""
    if not hasattr(controller, "st"):
        controller.st = {"idx": 0, "pause_until": -1.0, "last_seen": -1.0, "wait_start": None,
                         "free_since": None, "prev": None, "vm": 0.0, "wz": 0.0, "tracks": [],
                         "last_f": None, "last_b": None, "raw": np.zeros((0, 2)), "ttc": float("inf"),
                         "esc": None, "esc_t": -1.0, "esc_until": -1.0, "detour": None,
                         "av": None, "av_t": -1.0, "vo_prev": None, "vo_on": False,
                         "c3": [], "c3_seen": [], "p_t": -1.0, "stop_t0": None,
                         "um_src": None, "um": np.zeros((0, 2)),
                         "ttc_t0": None, "ttc_skip_until": -1.0}
    st = controller.st
    st.setdefault("ttc_t0", None)
    st.setdefault("ttc_skip_until", -1.0)
    for key, val0 in (("px_on", False), ("px_t0", -1.0), ("px_last", -1e9), ("px_skip_until", -1.0),
                      ("px_back_until", -1.0), ("px_back_v", 0.0), ("px_hist", [])):
        st.setdefault(key, val0)
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

    hold, _ = _margin(pose, st["vm"], 0.0, 0.0, M, _HORIZON, 0.15)
    gap = _min_gap(pose, M)

    # ---- (1) evasion: standing still would be hit -> VO planner, _plan_escape only as fallback ----
    if hold < 0.0:
        st["pause_until"] = -1.0
        st["stop_t0"] = None
        st["ttc_t0"] = None
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

    # ---- (1a) proximity stop / back-up on raw unmapped lidar points (works without tracks) ----
    px = _proximity_stop_and_backup(st, t, pose, M)
    if px is not None:
        st["stop_t0"] = None
        st["ttc_t0"] = None
        return px[0], px[1], False

    # ---- (1b) raw-lidar TTC alarm -> VO planner (strict); stop + pause_until only when it fails ----
    ttc_hit = False if t < st["ttc_skip_until"] else _ttc_stop_check(st, t, pose)
    if ttc_hit:
        st["vo_on"] = True
        cmd = _vo_planner(st, pose, tgt, M, base=_CRUISE_BASE, horizon=_VO_HORIZON_CRUISE, strict=True)
        if cmd is not None:
            v, w = cmd
            if v > 0.0:
                v = min(v, _ttc_speed_cap(st, pose))
            if abs(v) > 0.05 or abs(w) > 0.05:
                st["pause_until"] = -1.0        # evasion succeeded: drop the pause
                st["stop_t0"] = None
                st["ttc_t0"] = None
                if abs(v) > 0.05:
                    _mark_free(st, t)
                else:
                    st["free_since"] = None
                    if st["wait_start"] is None:
                        st["wait_start"] = t
                return v, w, False
        st["vo_prev"] = None
        st["pause_until"] = max(st["pause_until"], t + 0.5)   # VO failed: extend the pause and stop
        if st["ttc_t0"] is None:
            st["ttc_t0"] = t
        if t - st["ttc_t0"] < _TTC_MAX_WAIT or _should_pause(st, pose, M):
            st["free_since"] = None
            if st["wait_start"] is None:
                st["wait_start"] = t
            return 0.0, 0.0, False
        # long TTC stop and no closing tracked mover: the unmapped points are not coming at us -> grace period
        st["ttc_skip_until"] = t + _TTC_GRACE
        st["ttc_t0"] = None
        st["pause_until"] = -1.0
        st["stop_t0"] = None
    else:
        st["ttc_t0"] = None

    # ---- (2) immediate stop: a mover closing on the robot (relative velocity) is within _STOP_DIST ----
    closing = _should_pause(st, pose, M)
    if closing:
        st["pause_until"] = max(st["pause_until"], t + 0.3)
    if t < st["pause_until"]:
        if st["stop_t0"] is None:
            st["stop_t0"] = t
        if closing or t - st["stop_t0"] < _STOP_MAX_WAIT:
            st["free_since"] = None             # keep stopping while a closing mover is within 2 m
            if st["wait_start"] is None:
                st["wait_start"] = t
            return 0.0, 0.0, False
        st["pause_until"] = -1.0                # long stop and no closing mover within 2 m: drop the tail
    st["stop_t0"] = None

    # ---- (3) goal alignment: finish only when nothing is about to reach us ----
    if aligning:
        if abs(e_goal) < 0.05:
            if _margin(pose, st["vm"], 0.0, 0.0, M, 4.0, 0.15)[0] >= 0.0:
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        _mark_free(st, t)
        v, w = _guard(st, pose, M, 0.0, w_nom)
        return v, w, False

    if gap <= _SAFE_DIST:
        # ---- (4) mover within 3 m: VO cruise, longer horizon, larger margin, only margin >= 0 candidates ----
        st["vo_on"] = True
        cmd = _vo_planner(st, pose, tgt, M, base=_CRUISE_BASE, horizon=_VO_HORIZON_CRUISE, strict=True)
        if cmd is not None:
            if cmd[0] > 0.05:
                _mark_free(st, t)
            else:
                st["free_since"] = None
                if st["wait_start"] is None:
                    st["wait_start"] = t
            return cmd[0], cmd[1], False
        st["vo_prev"] = None
    else:
        # ---- (5) normal driving; after VO / evasion rejoin the path safely ----
        if st["vo_on"]:
            st["vo_on"] = False
            st["esc"] = None
            st["vo_prev"] = None
            _set_return(st, pose)
            if st["detour"] is not None:
                tgt = st["detour"]
                v_nom, w_nom, _ = _follow(pose, tgt)
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

    # ---- blocked: stop; if blocked for long, go around the blocker ----
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
    """Front+back 2D lidar + 3D lidar (obstacle-height points) -> one point set P (robot frame, kept in st["raw"]
    for the lidar strip / swept-body checks) -> drop static-map points (_is_static_world) -> 0.08 m grid
    -> cluster -> track movers (position, velocity).  Runs on every new 2D scan (or every 0.1 s)."""
    f = lidar2d_points("front")
    b = lidar2d_points("back")
    W3 = _cloud3d(st, t, pose)
    if f is st["last_f"] and b is st["last_b"] and t - st["p_t"] < 0.1:
        return
    st["last_f"], st["last_b"], st["p_t"] = f, b, t
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    P2 = np.concatenate([np.asarray(f, dtype=np.float64).reshape(-1, 2),
                         np.asarray(b, dtype=np.float64).reshape(-1, 2)])
    if len(W3):
        dx, dy = W3[:, 0] - x, W3[:, 1] - y
        P3 = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
    else:
        P3 = np.zeros((0, 2))
    st["raw"] = np.concatenate([P2, P3])

    parts = []
    if len(P2):
        r2 = np.hypot(P2[:, 0], P2[:, 1])
        k = r2 < 8.0
        if k.any():
            W2 = to_world(P2[k], pose)[:, :2]
            mg = 0.25 + 0.12 * r2[k] * abs(st["wz"]) + 0.05 * abs(st["vm"])
            parts.append(W2[~_is_static_world(W2, mg)])
    if len(W3):
        mg3 = 0.25 + 0.05 * abs(st["vm"])
        parts.append(W3[~_is_static_world(W3, mg3)])
    D = np.concatenate(parts) if parts else np.zeros((0, 2))
    if len(D):
        key = np.floor(D / 0.08).astype(np.int64)
        _, ui = np.unique((key[:, 0] + 2000) * 5000 + (key[:, 1] + 2000), return_index=True)
        D = D[ui]
        dr = np.hypot(D[:, 0] - x, D[:, 1] - y)
        D = D[np.argsort(dr)][:3000]

    cents = []
    it = 0
    while len(D) and len(cents) < 10 and it < 30:
        it += 1
        sd = D[0]
        m = np.hypot(D[:, 0] - sd[0], D[:, 1] - sd[1]) < 0.9
        if int(m.sum()) >= 4:
            cc = D[m].mean(axis=0)
            ddx, ddy = cc[0] - x, cc[1] - y
            L = math.hypot(ddx, ddy)
            if L > 1e-6:
                cc = (cc[0] + 0.2 * ddx / L, cc[1] + 0.2 * ddy / L)   # visible face -> box centre
            cents.append((float(cc[0]), float(cc[1])))
        D = D[~m]
    tracks = st["tracks"]
    used = set()
    new = []
    for cpt in cents:
        bi, bd = -1, 0.9
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            age = t - tr["t"]
            d = math.hypot(cpt[0] - tr["x"] - tr["vx"] * age, cpt[1] - tr["y"] - tr["vy"] * age)
            if d < bd:
                bi, bd = i, d
        if bi < 0:
            new.append({"x": cpt[0], "y": cpt[1], "vx": 0.0, "vy": 0.0, "t": t, "t0": t, "n": 1,
                        "hist": [(t, cpt[0], cpt[1])]})
            continue
        used.add(bi)
        tr = tracks[bi]
        tr["hist"] = [h for h in tr["hist"] if t - h[0] <= 0.6] + [(t, cpt[0], cpt[1])]
        t0, x0, y0 = tr["hist"][0]
        if t - t0 >= 0.25:
            vx, vy = (cpt[0] - x0) / (t - t0), (cpt[1] - y0) / (t - t0)
            sp = math.hypot(vx, vy)
            if sp > 2.0:
                vx, vy = vx * 2.0 / sp, vy * 2.0 / sp
            tr["vx"], tr["vy"] = vx, vy
        tr["x"], tr["y"], tr["t"] = cpt[0], cpt[1], t
        tr["n"] += 1
    st["tracks"] = [tr for i, tr in enumerate(tracks) if i in used or t - tr["t"] < 1.0] + new

    Mt = _track_array(st, t)
    _, st["ttc"] = _dynamic_pause(st, pose, Mt)


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


_SAFE_DIST = 3.0


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
    """Stop immediately while a mover closing on the robot (relative velocity, robot motion included) is
    within _STOP_DIST (2 m) of it."""
    return _approach_gap(pose, M, st["vm"]) <= _STOP_DIST


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


def _vo_planner(st, pose, tgt, M, base=0.05, horizon=None, strict=False):
    """VO local planner.  Candidates: headings bearing(tgt) + _VO_DELTAS x _VO_SPEED_STEPS speeds in [0, vmax]
    (forward only if the heading is within 0.6 rad), plus stop and a 0.5 m/s back-up.  A candidate must be
    outside every velocity obstacle over horizon (default _VO_HORIZON), have a clear lidar strip and swept body
    (fused 2D + 3D points), keep clear of the static map, and its predicted motion (drive _VO_DRIVE_T s, then stop)
    must keep margin >= 0 (clearance base) from every mover over _HORIZON - or, unless strict, beat standing still
    by 0.05.  Cheapest by progress, deviation, turn, margin and hysteresis.  Returns (v, w) or None."""
    x, y, yaw = pose
    dx, dy = tgt[0] - x, tgt[1] - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) if dist > 1e-3 else yaw
    vmax = min(1.5, 2.5 * dist + 0.5)
    vo = _compute_vo(pose, M, _VO_HORIZON if horizon is None else horizon)
    P = st["raw"]
    vm = st["vm"]
    xs, ys, hs = _vo_rollout(pose, vm, yaw, 0.0, _HORIZON)
    hold_c = _traj_margin(xs, ys, hs, M, base)[0]
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
        m = _traj_margin(xs, ys, hs, M, base)[0]
        if m < 0.0 and (strict or m < hold_c + 0.05):
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




_STOP_DIST = 2.0


_APPROACH_V = 0.2


_STOP_MAX_WAIT = 6.0


_CRUISE_BASE = 0.25


_VO_HORIZON_CRUISE = 2.5


_Z3D = (0.2, 1.1)


_R3D = 8.0


def _cloud3d(st, t, pose):
    """3D lidar: each new ~60 deg slice (base buffer _S["cloud"]) is cut to obstacle height (_Z3D, drops the floor
    and the ceiling), limited to _R3D, projected to x-y and converted to world with the pose at its arrival, so the
    0.1 s sweep stays consistent while the robot moves.  Returns the world points (N, 2) of the last ~0.12 s."""
    lidar3d_points()                            # subscribes and spins (base sensor code)
    seen = st["c3_seen"]
    for arr in list(_S["cloud"]):
        if any(arr is a for a in seen):
            continue
        seen.append(arr)
        A = np.asarray(arr, dtype=np.float64)
        if A.ndim != 2 or len(A) == 0 or A.shape[1] < 3:
            continue
        r = np.hypot(A[:, 0], A[:, 1])
        k = (A[:, 2] > _Z3D[0]) & (A[:, 2] < _Z3D[1]) & (r < _R3D)
        Q = A[k, :2]
        if len(Q) > 1500:
            Q = Q[::int(math.ceil(len(Q) / 1500.0))]
        if len(Q):
            st["c3"].append((t, to_world(Q, pose)[:, :2]))
    st["c3_seen"] = seen[-16:]
    st["c3"] = [e for e in st["c3"] if t - e[0] <= 0.12][-8:]
    if not st["c3"]:
        return np.zeros((0, 2))
    return np.concatenate([e[1] for e in st["c3"]])


def _approach_gap(pose, M, vm=0.0):
    """Smallest current clearance to a tracked mover that is closing on the robot: the radial component of the
    relative velocity (mover velocity minus robot velocity vm along the heading) toward the robot circle exceeds
    _APPROACH_V.  Covers the robot driving toward a mover as well as a mover coming at the robot; inf if none."""
    if len(M) == 0:
        return float("inf")
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rvx, rvy = vm * c, vm * s
    mx = M[:, 0] + M[:, 2] * M[:, 4]
    my = M[:, 1] + M[:, 3] * M[:, 4]
    ux, uy = M[:, 2] - rvx, M[:, 3] - rvy
    g = float("inf")
    for off in _ROBOT_CIRCLES:
        px = mx - (x + off * c)
        py = my - (y + off * s)
        d = np.hypot(px, py)
        ap = -(px * ux + py * uy) / np.maximum(d, 1e-6)
        sel = ap > _APPROACH_V
        if sel.any():
            g = min(g, float((d[sel] - _R_ROBOT - _R_MOVER).min()))
    return g




_MAX_OBJECT_SPEED = 2.0


_TTC_THRESHOLD = 1.0


_TTC_MIN_PTS = 3


_TTC_RANGE = 5.0


_TTC_MAX_WAIT = 3.0


_TTC_GRACE = 2.0


def _unmapped_raw(st, pose):
    """Raw fused lidar points (robot frame, st["raw"]) within _TTC_RANGE that are NOT on the static map
    (walls, MISSION boxes, FIXED_OBSTACLES; same margins as _perceive).  Cached per new st["raw"]."""
    P = st["raw"]
    if st.get("um_src") is P:
        return st["um"]
    out = np.zeros((0, 2))
    if len(P):
        r = np.hypot(P[:, 0], P[:, 1])
        k = r < _TTC_RANGE
        if k.any():
            Q = P[k]
            W = to_world(Q, pose)[:, :2]
            mg = 0.25 + 0.12 * r[k] * abs(st["wz"]) + 0.05 * abs(st["vm"])
            out = Q[~_is_static_world(W, mg)]
    st["um_src"] = P
    st["um"] = out
    return out


def _ttc_stop_check(st, t, pose):
    """Raw-lidar TTC check (decision only, no stop command): for every unmapped lidar point (static map removed by
    _unmapped_raw), TTC = (distance to the robot body) / (max(robot closing speed toward the point, 1e-3)
    + _MAX_OBJECT_SPEED), i.e. the mover is assumed to come at the worst-case 2 m/s.  Returns True when the
    minimum TTC is below _TTC_THRESHOLD; pause_until is managed by controller() only when evasion fails."""
    U = _unmapped_raw(st, pose)
    if len(U) == 0:
        return False
    dx = np.maximum(np.maximum(_BODY[0] - U[:, 0], U[:, 0] - _BODY[1]), 0.0)
    dy = np.maximum(np.maximum(_BODY[2] - U[:, 1], U[:, 1] - _BODY[3]), 0.0)
    d = np.hypot(dx, dy)
    rc = np.maximum(np.hypot(U[:, 0], U[:, 1]), 1e-6)
    closing = np.maximum(st["vm"] * U[:, 0] / rc, 1e-3)
    ttc = d / (closing + _MAX_OBJECT_SPEED)
    return float(ttc.min()) < _TTC_THRESHOLD




def _ttc_speed_cap(st, pose):
    """Forward speed limit while evading on a TTC alarm: slow down with the distance from the front face to the
    nearest unmapped lidar point in a 1.6 m wide strip ahead (untracked movers are not in the VO)."""
    U = _unmapped_raw(st, pose)
    if len(U) == 0:
        return 1.5
    front = (U[:, 0] > _BODY[1]) & (np.abs(U[:, 1]) < 0.8)
    if not front.any():
        return 1.5
    d = float(U[front, 0].min()) - _BODY[1]
    return _clamp(0.6 * (d - 0.3), 0.0, 1.5)




_PX_STOP_D = 0.8


_PX_BACK_D = 0.5


_PX_BACK_V = 0.4


_PX_BACK_T = 0.5


_PX_WAIT = 2.0


_PX_MAX_WAIT = 4.0


_PX_GRACE = 2.5


_PX_SELF = 0.05


def _px_dist(st, pose):
    """Distance from the robot body box to the unmapped raw lidar points (_unmapped_raw: static map removed).
    Points within _PX_SELF of the body are treated as the robot itself (same margin as _sweep_clear).
    Returns (the _TTC_MIN_PTS-th smallest distance, robot-frame (x, y) of the nearest point), or (inf, None)
    when fewer than _TTC_MIN_PTS points remain."""
    U = _unmapped_raw(st, pose)
    if len(U) == 0:
        return float("inf"), None
    dx = np.maximum(np.maximum(_BODY[0] - U[:, 0], U[:, 0] - _BODY[1]), 0.0)
    dy = np.maximum(np.maximum(_BODY[2] - U[:, 1], U[:, 1] - _BODY[3]), 0.0)
    d = np.hypot(dx, dy)
    k = d > _PX_SELF
    if int(k.sum()) < _TTC_MIN_PTS:
        return float("inf"), None
    d, U = d[k], U[k]
    o = np.argsort(d)
    return float(d[o[_TTC_MIN_PTS - 1]]), (float(U[o[0], 0]), float(U[o[0], 1]))


def _px_hold(st, t):
    """Bookkeeping while the proximity stop / back-up is active (normal-motion timers are interrupted)."""
    st["px_last"] = t
    st["free_since"] = None
    if st["wait_start"] is None:
        st["wait_start"] = t


def _px_backup_cmd(st, pose, M, near):
    """Back-up speed for the proximity stop: away from the nearest unmapped point (point on the front half ->
    reverse -_PX_BACK_V, on the rear half -> forward +_PX_BACK_V; the opposite direction is tried only for a point
    beside the body).  A direction is used only if its lidar strip and swept body are clear, the predicted motion
    (drive _PX_BACK_T s, then stop) keeps clear of the static map, and its tracked-mover margin is not worse than
    standing still.  Returns the speed or None."""
    P = st["raw"]
    vm = st["vm"]
    yaw = pose[2]
    mid = 0.5 * (_BODY[0] + _BODY[1])
    pref = -_PX_BACK_V if near[0] >= mid else _PX_BACK_V
    opts = [pref]
    if _BODY[0] <= near[0] <= _BODY[1]:
        opts.append(-pref)
    xs0, ys0, hs0 = _vo_rollout(pose, vm, yaw, 0.0, 1.5)
    m0 = _traj_margin(xs0, ys0, hs0, M, 0.05)[0]
    for v in opts:
        if not (_lidar_clear(P, v, 0.5) and _sweep_clear(P, v, 0.0, _PX_BACK_T)):
            continue
        xs, ys, hs = _vo_rollout(pose, vm, yaw, v, 1.5, t_drive=_PX_BACK_T)
        if not _traj_static_ok(xs, ys, hs):
            continue
        m = _traj_margin(xs, ys, hs, M, 0.05)[0]
        if m < min(0.0, m0):
            continue
        return v
    return None


def _proximity_stop_and_backup(st, t, pose, M):
    """Pure proximity stop / back-up on raw unmapped lidar points (no tracks needed).
    (a) Stopped back-up, independent of px_on / closing and checked before the grace period: while the robot is
    (nearly) stopped (|vm| < 0.1) with an unmapped point within _PX_BACK_D of the body -> px_on, pause_until =
    t + _PX_WAIT, move _PX_BACK_T s away from it at _PX_BACK_V (checked by _px_backup_cmd); stop when no safe
    direction exists.  This also covers a stop set by another layer and a mover closing at < 0.15 m/s.
    (b) Moving trigger: an unmapped point within _PX_STOP_D while the robot drives forward, or while that distance
    shrinks faster than the robot's own speed + 0.15 m/s -> stop at once and pause_until = t + _PX_WAIT.
    A stop longer than _PX_MAX_WAIT with nothing closing hands control back for _PX_GRACE s, but only when no
    back-up is being attempted (d >= _PX_BACK_D).  Returns (v, w) while active, None otherwise."""
    d, near = _px_dist(st, pose)
    vm = st["vm"]
    hist = [e for e in st["px_hist"] if t - e[0] <= 0.8]
    old = [e for e in hist if t - e[0] >= 0.4]
    cm = False
    if old and d < _PX_STOP_D + 0.5:
        rate = (old[-1][1] - d) / max(t - old[-1][0], 1e-3)
        cm = rate > abs(vm) + 0.15
    if d < float("inf"):
        hist.append((t, d))
    st["px_hist"] = hist

    # back-up in progress (cancelled if another layer dropped the pause or the strip gets blocked)
    if st["px_back_until"] > t and st["pause_until"] >= st["px_back_until"]:
        v = st["px_back_v"]
        if _lidar_clear(st["raw"], v, 0.4) and _sweep_clear(st["raw"], v, 0.0, 0.3):
            _px_hold(st, t)
            return v, 0.0
    st["px_back_until"] = -1.0

    # (a) stopped with an unmapped point within _PX_BACK_D: back up (independent of px_on / cm / grace)
    if abs(vm) < 0.1 and d < _PX_BACK_D and near is not None:
        if t - st["px_last"] > 1.5:
            st["px_t0"] = t
        st["px_on"] = True
        st["pause_until"] = max(st["pause_until"], t + _PX_WAIT)
        v = _px_backup_cmd(st, pose, M, near)
        if v is not None:
            st["px_back_v"] = v
            st["px_back_until"] = t + _PX_BACK_T
            st["pause_until"] = max(st["pause_until"], t + _PX_BACK_T + _PX_WAIT)
            _px_hold(st, t)
            return v, 0.0
        _px_hold(st, t)
        return 0.0, 0.0

    # grace after a long stop: only a point actually coming at us re-arms the stop
    if t < st["px_skip_until"] and not cm:
        st["px_on"] = False
        return None

    # (b) moving trigger
    if d < _PX_STOP_D and (vm > 0.05 or cm):
        if t - st["px_last"] > 1.5:
            st["px_t0"] = t
        st["px_on"] = True
        st["pause_until"] = max(st["pause_until"], t + _PX_WAIT)
    if not (st["px_on"] and t < st["pause_until"]):
        st["px_on"] = False
        return None

    # long stop release: only when no back-up is attempted (d >= _PX_BACK_D) and nothing is closing
    if t - st["px_t0"] > _PX_MAX_WAIT and not cm and d >= _PX_BACK_D:
        st["px_on"] = False
        st["pause_until"] = -1.0
        st["px_skip_until"] = t + _PX_GRACE
        return None

    _px_hold(st, t)
    return 0.0, 0.0


# ========================= [END EDIT REGION] ===============================
