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
    """Waypoint following with the VO planner as the primary mover avoidance (diagnosis: VO primary, A* fallback).
    (1) perception + mover tracking; (2) route target via _route_target (goal alignment once the route is done);
    (3) when a tracked mover is within _SAFE_DIST (held _NEAR_HOLD s): _vo_planner every step, result checked by
    _guard; pause_until is cleared and A* is skipped.  Only when VO finds no command for more than _VO_FAIL_T s
    the A* replanner (_replan_route, every _REPLAN_DT s) replaces the route, and the robot drives on it with
    _plan_speed + _guard; (4) with no mover near: TTC pause, _predictive_speed, _guard."""
    if not hasattr(controller, "st"):
        controller.st = {"idx": 0, "prev": None, "vm": 0.0, "wz": 0.0, "tracks": [],
                         "last_f": None, "last_b": None, "raw": np.zeros((0, 2)), "ttc": float("inf"),
                         "c3": [], "c3_seen": [], "p_t": -1.0,
                         "um_src": None, "um": np.zeros((0, 2)),
                         "pause_until": -1.0, "detour": None, "det_until": -1.0,
                         "wps": [(float(p[0]), float(p[1]), k) for k, p in enumerate(_WAYPOINTS)],
                         "orig_next": 0, "plan_t": -1e9, "plan_until": -1.0, "plan_pts": None,
                         "vo_until": -1.0, "vo_none_since": None, "vo_prev": None}
    st = controller.st
    x, y, yaw = pose

    # ---- 1) perception and track update ----
    _update_motion(st, t, pose)
    _perceive(st, t, pose)
    M = _track_array(st, t)

    # ---- 2) route target / goal alignment ----
    wp = _route_target(st, pose)
    if wp is None:
        st["vo_none_since"] = None
        e_goal = _wrap_angle(GOAL[2] - yaw)
        if abs(e_goal) < 0.05:
            if _margin(pose, st["vm"], 0.0, 0.0, M, 4.0, 0.15)[0] >= 0.0:
                return 0.0, 0.0, True
            return 0.0, 0.0, False
        v, w = _guard(st, pose, M, 0.0, _clamp(1.5 * e_goal, -0.6, 0.6))
        return v, w, False

    # ---- 3) mover near: VO planner first, A* only after VO has failed for > _VO_FAIL_T s ----
    if len(M) and _min_gap(pose, M) <= _SAFE_DIST:
        st["vo_until"] = t + _NEAR_HOLD
    if t < st["vo_until"]:
        st["pause_until"] = -1.0
        st["plan_until"] = -1.0
        cmd = _vo_planner(st, pose, wp, M, strict=False)
        if cmd is not None:
            st["vo_none_since"] = None
            v, w = _guard(st, pose, M, cmd[0], cmd[1])
            return v, w, False
        if st["vo_none_since"] is None:
            st["vo_none_since"] = t
        if (t - st["vo_none_since"] > _VO_FAIL_T and len(M)
                and t - st["plan_t"] >= _REPLAN_DT):
            new = _replan_route(st, t, pose, M)
            st["plan_t"] = t
            if new is not None:
                st["wps"] = new
                st["idx"] = 0
                st["detour"] = None
                wp2 = _route_target(st, pose)
                if wp2 is not None:
                    wp = wp2
        v_nom, w_nom, _ = _follow(pose, wp)
        vc = _plan_speed(st, pose, wp, min(v_nom, 1.0), M) if v_nom > 0.0 else 0.0
        v, w = _guard(st, pose, M, vc, w_nom)
        return v, w, False

    # ---- 4) no mover near: TTC pause, predictive speed, final check ----
    st["vo_none_since"] = None
    pause, _ttc = _dynamic_pause(st, pose, M)
    if pause is not None and pause > 0.0:
        st["pause_until"] = t + pause
    if t < st["pause_until"]:
        return 0.0, 0.0, False
    v_nom, w_nom, _ = _follow(pose, wp)
    vc = _predictive_speed(st, pose, wp, v_nom, M) if v_nom > 0.0 else 0.0
    if vc is None:
        vc = 0.0
    v, w = _guard(st, pose, M, vc, w_nom)
    return v, w, False




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




_PX_STOP_D = 0.9


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




_EARLY_STOP_DIST = 1.5


_EARLY_STOP_TTC = 0.8


_EARLY_MAX_WAIT = 3.0


_EARLY_GRACE = 2.0


def _early_raw_stop(st, t, pose, M):
    """Early raw-point emergency stop (no tracks needed).  d = distance from the robot body to the unmapped raw
    lidar points (_px_dist on _unmapped_raw, static map removed); the relative closing speed is estimated from
    the change of d against a sample >= 0.25 s old.  TTC = d / (max(rel, 0.1) + _MAX_OBJECT_SPEED).
    d < _EARLY_STOP_DIST and TTC < _EARLY_STOP_TTC -> stop at once, pause_until = t + 1.0 (timers via _px_hold).
    While active and (nearly) stopped, a point that is coming at the robot (distance shrinking faster than the
    robot's own motion explains, + 0.15 m/s) or one within _PX_BACK_D triggers a back-up away from it
    (_px_backup_cmd, held _PX_BACK_T s while the lidar strip / swept body stay clear).  A stop longer than
    _EARLY_MAX_WAIT with nothing coming at the robot hands control back for _EARLY_GRACE s; during the grace
    only an approaching point re-arms the stop.  Returns (v, w) while active, None otherwise."""
    for key, val0 in (("es_on", False), ("es_t0", -1.0), ("es_skip_until", -1.0), ("es_hist", []),
                      ("es_back_until", -1.0), ("es_back_v", 0.0)):
        st.setdefault(key, val0)
    d, near = _px_dist(st, pose)
    vm = st["vm"]
    hist = [e for e in st["es_hist"] if t - e[0] <= 1.0]
    old = [e for e in hist if t - e[0] >= 0.25]
    rate = 0.0
    if old and d < float("inf"):
        rate = (old[-1][1] - d) / max(t - old[-1][0], 1e-3)     # > 0: distance shrinking
    if d < float("inf"):
        hist.append((t, d))
    st["es_hist"] = hist

    own = 0.0
    if near is not None:
        rn = math.hypot(near[0], near[1])
        if rn > 1e-6:
            own = vm * near[0] / rn                              # closing explained by the robot's own motion
    cm = d < _EARLY_STOP_DIST and (rate - own) > 0.15            # the point itself is coming at the robot
    ttc = d / (max(rate, 0.1) + _MAX_OBJECT_SPEED)
    trigger = d < _EARLY_STOP_DIST and ttc < _EARLY_STOP_TTC

    # grace after a long stop: only a point actually coming at the robot re-arms the stop
    if t < st["es_skip_until"] and not cm:
        st["es_on"] = False
        st["es_back_until"] = -1.0
        return None

    if trigger:
        if not st["es_on"]:
            st["es_t0"] = t
        st["es_on"] = True
        st["pause_until"] = max(st["pause_until"], t + 1.0)
    if not (st["es_on"] and t < st["pause_until"]):
        st["es_on"] = False
        st["es_back_until"] = -1.0
        return None

    # back-up in progress
    if st["es_back_until"] > t:
        v = st["es_back_v"]
        if _lidar_clear(st["raw"], v, 0.4) and _sweep_clear(st["raw"], v, 0.0, 0.3):
            _px_hold(st, t)
            return v, 0.0
    st["es_back_until"] = -1.0

    # stopped (or already backing) and the point is coming at us / very close: move away from it
    if near is not None and vm < 0.15 and (cm or d < _PX_BACK_D):
        v = _px_backup_cmd(st, pose, M, near)
        if v is not None:
            st["es_back_v"] = v
            st["es_back_until"] = t + _PX_BACK_T
            st["pause_until"] = max(st["pause_until"], t + _PX_BACK_T + 1.0)
            _px_hold(st, t)
            return v, 0.0

    # long stop with nothing coming at the robot: hand control back for a grace period
    if t - st["es_t0"] > _EARLY_MAX_WAIT and not cm and d >= _PX_BACK_D:
        st["es_on"] = False
        st["es_back_until"] = -1.0
        st["pause_until"] = -1.0
        st["es_skip_until"] = t + _EARLY_GRACE
        return None

    _px_hold(st, t)
    return 0.0, 0.0




_PF_RANGE = 2.0


_PF_REP_GAIN = 1.0


_PF_ATT_GAIN = 0.5


_PF_STOP_D = 0.3


_PF_DEBOUNCE = 0.3


_PF_REV_V = 0.4


_PF_W_MAX = 0.6


def _potential_field_avoidance(pose, raw, wp, M=None, prev_side=0.0):
    """Potential field on unmapped lidar points (raw: robot frame, static map already removed).
    Attraction _PF_ATT_GAIN toward wp; repulsion from points within _PF_RANGE of the body, direction = sum of
    (1/d - 1/_PF_RANGE)-weighted unit vectors away from the points, magnitude _PF_REP_GAIN * the largest weight.
    When the repulsion opposes the attraction a tangential term is added (side: pass behind the nearest moving
    track, else toward the goal side, kept from prev_side when ambiguous, flipped if that side is mapped-blocked)
    so the field has no dead stop in front of an object.  Speed factor = clamp((d_front - _PF_STOP_D) /
    (_PF_RANGE - _PF_STOP_D), 0, 1) with d_front the nearest point within +-70 deg of the field direction.
    rev = True when the field points backward and a point is within 0.8 m of the front face.
    Returns (heading, speed factor, nearest point distance, side, rev)."""
    x, y, yaw = pose
    ax, ay = _unit(wp[0] - x, wp[1] - y)
    U = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    d = np.zeros(0)
    if len(U):
        dx = np.maximum(np.maximum(_BODY[0] - U[:, 0], U[:, 0] - _BODY[1]), 0.0)
        dy = np.maximum(np.maximum(_BODY[2] - U[:, 1], U[:, 1] - _BODY[3]), 0.0)
        d = np.hypot(dx, dy)
        k = (d > _PX_SELF) & (d < _PF_RANGE)
        U, d = U[k], d[k]
    if len(U) < _TTC_MIN_PTS:
        return math.atan2(ay, ax), 1.0, float("inf"), prev_side, False
    cxb = 0.5 * (_BODY[0] + _BODY[1])
    rx, ry = cxb - U[:, 0], -U[:, 1]
    L = np.maximum(np.hypot(rx, ry), 1e-6)
    wgt = 1.0 / np.maximum(d, 0.05) - 1.0 / _PF_RANGE
    sx = float(np.sum(wgt * rx / L))
    sy = float(np.sum(wgt * ry / L))
    c, s = math.cos(yaw), math.sin(yaw)
    rux, ruy = _unit(c * sx - s * sy, s * sx + c * sy)
    mag = _PF_REP_GAIN * float(wgt.max())
    fx = _PF_ATT_GAIN * ax + mag * rux
    fy = _PF_ATT_GAIN * ay + mag * ruy
    side = prev_side
    opp = -(rux * ax + ruy * ay)
    if opp > -0.2:
        px, py = -ruy, rux
        sgn = px * ax + py * ay
        if M is not None and len(M):
            mx = M[:, 0] + M[:, 2] * M[:, 4]
            my = M[:, 1] + M[:, 3] * M[:, 4]
            j = int(np.argmin(np.hypot(mx - x, my - y)))
            sp = math.hypot(M[j, 2], M[j, 3])
            if math.hypot(mx[j] - x, my[j] - y) < 3.5 and sp > 0.2:
                vd = (px * M[j, 2] + py * M[j, 3]) / sp
                if abs(vd) > 0.3:
                    sgn = -vd                   # pass behind the moving object
        if abs(sgn) < 0.3 and prev_side != 0.0:
            sgn = prev_side
        side = 1.0 if sgn >= 0.0 else -1.0
        if (not _map_free(x + side * px, y + side * py, 0.5)
                and _map_free(x - side * px, y - side * py, 0.5)):
            side = -side
        kt = mag * (opp + 0.2)
        fx += kt * side * px
        fy += kt * side * py
    th = math.atan2(fy, fx)
    er = _wrap_angle(th - yaw)
    b = np.arctan2(U[:, 1], U[:, 0] - cxb)
    front = np.cos(b - er) > 0.34
    d_f = float(d[front].min()) if front.any() else float("inf")
    fac = _clamp((d_f - _PF_STOP_D) / (_PF_RANGE - _PF_STOP_D), 0.0, 1.0)
    ahead = (U[:, 0] > _BODY[1] - 0.05) & (np.abs(U[:, 1]) < 0.6) & (d < 0.8)
    rev = bool(math.cos(er) < -0.5 and ahead.any())
    return th, fac, float(d.min()), side, rev


def _pf_wait(st, t, pose, M, wp):
    """Bookkeeping while potential-field mode is not moving: wait timers, and after 3 s a stuck detour."""
    st["free_since"] = None
    if st["wait_start"] is None:
        st["wait_start"] = t
    elif t - st["wait_start"] > 3.0 and st["detour"] is None:
        d = _stuck_detour(st, pose, M, wp)
        if d is not None:
            st["detour"] = d
            st["wait_start"] = None




_DWA_HORIZON = 2.0


_DWA_DT = 0.1


_DWA_V_STEPS = 5


_DWA_W_STEPS = 7


_DWA_COST_WEIGHT = {"margin": 1.0, "speed": 0.2, "turn": 0.1, "heading": 0.5}


_DWA_W_MAX = 1.2


_DWA_WINDOW_T = 0.5


_DWA_MARGIN_CAP = 0.5


_DWA_DEBOUNCE = 0.3


_DWA_UM_T = 1.0


def _dwa_candidates(pose, vm, w_aim=None):
    """Dynamic window: forward speeds reachable within _DWA_WINDOW_T s (accel 3.0, decel 1.2 m/s^2, capped
    to [0, 1.5]) in _DWA_V_STEPS steps plus stop (0) and a slow back-up (-0.3); yaw rates in _DWA_W_STEPS steps over
    [-_DWA_W_MAX, _DWA_W_MAX] plus w_aim (turn onto the target bearing in 0.5 s).  Returns [(v, w), ...]."""
    lo = max(0.0, vm - 1.2 * _DWA_WINDOW_T)
    hi = min(1.5, max(vm, 0.0) + 3.0 * _DWA_WINDOW_T)
    if hi < lo:
        hi = lo
    vs = sorted(set([round(float(v), 3) for v in np.linspace(lo, hi, _DWA_V_STEPS)] + [0.0, -0.3]))
    ws = [float(w) for w in np.linspace(-_DWA_W_MAX, _DWA_W_MAX, _DWA_W_STEPS)]
    if w_aim is not None and all(abs(w_aim - w) > 0.05 for w in ws):
        ws.append(float(w_aim))
    return [(v, w) for v in vs for w in ws]


def _dwa_unmapped_world(st, pose):
    """Unmapped raw lidar points (static map removed) within 3 m, subsampled to <= 400, in world frame."""
    U = _unmapped_raw(st, pose)
    if len(U) == 0:
        return np.zeros((0, 2))
    r = np.hypot(U[:, 0], U[:, 1])
    U = U[r < 3.0]
    if len(U) == 0:
        return np.zeros((0, 2))
    if len(U) > 400:
        U = U[::int(math.ceil(len(U) / 400.0))]
    return to_world(U, pose)[:, :2]


def _dwa_point_clear(xs, ys, hs, Wp):
    """Per pose: min over the two robot circles of (distance to the nearest point of Wp - _R_ROBOT)."""
    out = None
    for off in _ROBOT_CIRCLES:
        cx = xs + off * np.cos(hs)
        cy = ys + off * np.sin(hs)
        d = np.hypot(Wp[None, :, 0] - cx[:, None], Wp[None, :, 1] - cy[:, None]).min(axis=1) - _R_ROBOT
        out = d if out is None else np.minimum(out, d)
    return out


def _dwa_select(st, pose, tgt, M):
    """DWA: every (v, w) from _dwa_candidates is rolled out with _rollout over _DWA_HORIZON s.  A candidate must
    have a clear lidar strip and swept body, keep clear of the static map (checked up to where the path passes
    closest to tgt, since the robot turns there), not move toward unmapped (untracked) lidar points over
    _DWA_UM_T s, and keep mover margin >= 0 (_traj_margin, clearance 0.15).  Cost = -margin (min of mover margin
    and static clearance + 0.2, capped at _DWA_MARGIN_CAP) + speed * (v_nom - v)^2 + turn * |w| + heading *
    |bearing error after 0.5 s| + small smoothing.  Returns the cheapest (v, w) or None."""
    x, y, yaw = pose
    dx, dy = tgt[0] - x, tgt[1] - y
    dist = math.hypot(dx, dy)
    v_nom = min(1.5, 2.5 * dist + 0.5)
    e0 = _wrap_angle(math.atan2(dy, dx) - yaw) if dist > 1e-3 else 0.0
    w_aim = _clamp(e0 / 0.5, -1.5, 1.5)
    P = st["raw"]
    vm = st["vm"]
    Wp = _dwa_unmapped_world(st, pose)
    ku = int(round(_DWA_UM_T / _DWA_DT)) + 1
    kh = int(round(0.5 / _DWA_DT))
    Wc = _DWA_COST_WEIGHT
    prev = st.get("dwa_prev")
    best, best_c = None, float("inf")
    for v, w in _dwa_candidates(pose, vm, w_aim):
        if abs(v) > 1e-3 and not _lidar_clear(P, v, 0.3 + 0.4 * abs(v)):
            continue
        if not _sweep_clear(P, v, w, 0.5):
            continue
        xs, ys, hs = _rollout(pose, vm, _DWA_HORIZON, v=v, w=w)
        dt_ = np.hypot(xs - tgt[0], ys - tgt[1])
        kc = int(np.argmin(dt_))
        ks = len(xs) if dt_[kc] > 0.5 else max(kc + 1, 2)
        if not _traj_static_ok(xs[:ks], ys[:ks], hs[:ks]):
            continue
        if len(Wp):
            pc = _dwa_point_clear(xs[:ku], ys[:ku], hs[:ku], Wp)
            if float(pc[1:].min()) < min(0.05, float(pc[0]) - 0.02):
                continue
        m_mov = _traj_margin(xs, ys, hs, M, 0.15)[0]
        if m_mov < 0.0:
            continue
        sc = float(_static_clear(xs[:ks], ys[:ks], hs[:ks]).min())
        m = min(m_mov, sc + 0.2, _DWA_MARGIN_CAP)
        hx, hy, hh = xs[kh], ys[kh], hs[kh]
        if math.hypot(tgt[0] - hx, tgt[1] - hy) > 0.3:
            eh = abs(_wrap_angle(math.atan2(tgt[1] - hy, tgt[0] - hx) - hh))
        else:
            eh = 0.0
        cost = (-Wc["margin"] * m + Wc["speed"] * (v_nom - v) ** 2 + Wc["turn"] * abs(w)
                + Wc["heading"] * eh)
        if prev is not None:
            cost += 0.05 * (abs(v - prev[0]) + abs(w - prev[1]))
        if cost < best_c:
            best, best_c = (v, w), cost
    if best is None:
        return None
    st["dwa_prev"] = best
    return best




_VFH_N = 72


_VFH_BIN = 2.0 * math.pi / _VFH_N


_VFH_ANG = -math.pi + (np.arange(_VFH_N) + 0.5) * _VFH_BIN


_VFH_CX = 0.5 * (_BODY[0] + _BODY[1])


_VFH_HALF_L = 0.5 * (_BODY[1] - _BODY[0])


_VFH_HALF_W = 0.49


_VFH_RANGE = 3.0


_VFH_TRIGGER = 2.5


_VFH_MIN_PTS = 2


_VFH_ENLARGE = 0.78


_VFH_MIN_WIDTH = 3


_VFH_HOLD = 1.0


_VFH_PRED_T = 1.2


_VFH_LOOK = (0.0, 0.3, 0.6, 0.9, 1.2)


_VFH_STALL_T = 1.5


def _vfh_wrap(a):
    """Wrap an angle array to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _vfh_histogram(st, pose, M):
    """Polar histogram around the body centre (robot frame, _VFH_N bins of 5 deg): unmapped raw lidar points
    (_unmapped_raw, static map removed, own body dropped) within _VFH_RANGE are counted per bin; a bin with
    >= _VFH_MIN_PTS points is an obstacle at its nearest distance.  Tracked movers add their predicted centres
    over _VFH_PRED_T s (radius _R_MOVER).  Every obstacle is enlarged by _VFH_ENLARGE (body corner + margin):
    half angle asin(r / d), or a full half-plane when it already overlaps.  Returns the blocked bins and the
    obstacle points kept for the free-distance check."""
    x, y, yaw = pose
    cnt = np.zeros(_VFH_N)
    dmin = np.full(_VFH_N, np.inf)
    rx = ry = np.zeros(0)
    U = _unmapped_raw(st, pose)
    if len(U):
        bx = np.maximum(np.maximum(_BODY[0] - U[:, 0], U[:, 0] - _BODY[1]), 0.0)
        by = np.maximum(np.maximum(_BODY[2] - U[:, 1], U[:, 1] - _BODY[3]), 0.0)
        qx = U[:, 0] - _VFH_CX
        qy = U[:, 1]
        d = np.hypot(qx, qy)
        k = (np.hypot(bx, by) > _PX_SELF) & (d < _VFH_RANGE)
        qx, qy, d = qx[k], qy[k], d[k]
        if len(qx):
            idx = np.floor((np.arctan2(qy, qx) + math.pi) / _VFH_BIN).astype(np.int64) % _VFH_N
            np.add.at(cnt, idx, 1.0)
            np.minimum.at(dmin, idx, d)
            keep = cnt[idx] >= _VFH_MIN_PTS
            rx, ry = qx[keep], qy[keep]
            if len(rx) > 400:
                step = int(math.ceil(len(rx) / 400.0))
                rx, ry = rx[::step], ry[::step]
    occ = cnt >= _VFH_MIN_PTS
    phi = [_VFH_ANG[occ]]
    dd = [dmin[occ]]
    ext = [np.zeros(int(occ.sum()))]
    mx = my = np.zeros(0)
    if len(M):
        c, s = math.cos(yaw), math.sin(yaw)
        taus = np.arange(0.0, _VFH_PRED_T + 1e-9, 0.3)
        sp = np.hypot(M[:, 2], M[:, 3])
        tt = M[:, 4][:, None] + np.where(sp[:, None] > 0.1, taus[None, :], 0.0)
        dx = (M[:, 0][:, None] + M[:, 2][:, None] * tt - x).ravel()
        dy = (M[:, 1][:, None] + M[:, 3][:, None] * tt - y).ravel()
        px = c * dx + s * dy - _VFH_CX
        py = -s * dx + c * dy
        dm = np.hypot(px, py)
        k = (dm - _R_MOVER) < _VFH_RANGE
        mx, my = px[k], py[k]
        phi.append(np.arctan2(my, mx))
        dd.append(dm[k])
        ext.append(np.full(int(k.sum()), _R_MOVER))
    PHI = np.concatenate(phi)
    DD = np.concatenate(dd)
    EXT = np.concatenate(ext)
    blocked = np.zeros(_VFH_N, dtype=bool)
    if len(PHI):
        rr = _VFH_ENLARGE + EXT
        half = np.where(DD > rr, np.arcsin(np.clip(rr / np.maximum(DD, 1e-6), 0.0, 1.0)), 0.5 * math.pi)
        diff = np.abs(_vfh_wrap(_VFH_ANG[None, :] - PHI[:, None]))
        blocked = (diff <= half[:, None] + 0.5 * _VFH_BIN).any(axis=0)
    return {"cnt": cnt, "blocked": blocked, "rx": rx, "ry": ry, "mx": mx, "my": my}


def _vfh_static_free(pose):
    """Per bin: True if driving straight along that bearing for up to 1.2 m keeps both robot circles clear of the
    walls and mapped static obstacles (same rule as _traj_static_ok)."""
    x, y, yaw = pose
    h = yaw + _VFH_ANG
    s = np.array(_VFH_LOOK)
    xs = x + np.cos(h)[:, None] * s[None, :]
    ys = y + np.sin(h)[:, None] * s[None, :]
    hs = np.repeat(h[:, None], len(s), axis=1)
    c = _static_clear(xs.ravel(), ys.ravel(), hs.ravel()).reshape(_VFH_N, len(s))
    return c[:, 1:].min(axis=1) >= np.minimum(0.05, c[:, 0] - 0.02)


def _vfh_gap_dist(H, ang):
    """Free distance from the body centre along each bearing in ang (robot frame): nearest obstacle point inside a
    body-wide corridor (_VFH_HALF_W, + _R_MOVER for predicted mover centres).  inf if the corridor is empty."""
    ang = np.asarray(ang, dtype=np.float64).reshape(-1)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = np.full(len(ang), np.inf)
    for px, py, e in ((H["rx"], H["ry"], 0.0), (H["mx"], H["my"], _R_MOVER)):
        if len(px) == 0:
            continue
        al = ca * px[None, :] + sa * py[None, :]
        la = -sa * px[None, :] + ca * py[None, :]
        ins = (al > 0.0) & (np.abs(la) < _VFH_HALF_W + e)
        out = np.minimum(out, np.where(ins, al - e, np.inf).min(axis=1))
    return out


def _vfh_select_gap(free, eg, prev_r, dg):
    """Gap selection (diagnosis step 2): circular runs (valleys) of consecutive free bins, each with its width
    (bin count) and centre bearing (circular mean of its bin angles, wrap-safe).  Among valleys of at least
    _VFH_MIN_WIDTH bins the widest wins; on equal width the one whose centre is closest to the goal bearing eg.
    Returns the chosen valley's centre bearing.  eg when every bin is free, None when no valley is wide enough.
    prev_r and dg are kept for the _vfh_cmd call signature and are not used."""
    n = _VFH_N
    free = np.asarray(free, dtype=bool).reshape(-1)
    if not free.any():
        return None
    if free.all():
        return float(eg)
    start = int(np.argmin(free))                # a blocked bin: no run wraps past it
    valleys = []
    run = []
    for j in list(range(start + 1, n)) + list(range(0, start + 1)):
        if free[j]:
            run.append(j)
        elif run:
            valleys.append(run)
            run = []
    best = None
    for run in valleys:
        wid = len(run)
        if wid < _VFH_MIN_WIDTH:
            continue
        a = _VFH_ANG[np.array(run, dtype=np.int64)]
        centre = math.atan2(float(np.sin(a).sum()), float(np.cos(a).sum()))
        de = abs(_wrap_angle(centre - eg))
        key = (-wid, de)
        if best is None or key < best[0]:
            best = (key, wid, centre)
    if best is None:
        return None
    return float(best[2])


def _vfh_cmd(st, pose, tgt, M):
    """VFH command toward tgt: histogram -> gap bearing th (robot frame) -> w = clamp(7.5 th, +-_DWA_W_MAX);
    forward only when |th| <= 0.35, at v = min(v_nom from _follow's distance, 0.9 * (free distance along th -
    half body length - 0.2)), cut by the _lidar_clear strip and reduced (1.0 / 0.5 / 0) if its 1.5 s rollout comes
    closer to a tracked mover than standing still.  Returns (v, w) or None when no gap exists."""
    x, y, yaw = pose
    _, _, dist = _follow(pose, tgt)
    v_nom = min(1.5, 2.5 * dist + 0.5)
    eg = _wrap_angle(math.atan2(tgt[1] - y, tgt[0] - x) - yaw) if dist > 1e-3 else 0.0
    H = _vfh_histogram(st, pose, M)
    free = (~H["blocked"]) & _vfh_static_free(pose)
    dg = _vfh_gap_dist(H, _VFH_ANG)
    prev = st.get("vfh_dir")
    prev_r = _wrap_angle(prev - yaw) if prev is not None else None
    th = _vfh_select_gap(free, eg, prev_r, dg)
    if th is None:
        return None
    st["vfh_dir"] = _wrap_angle(yaw + th)
    w = _clamp(7.5 * th, -_DWA_W_MAX, _DWA_W_MAX)
    if abs(th) > 0.35:
        return 0.0, w
    d_th = float(_vfh_gap_dist(H, [th])[0])
    v = min(v_nom, _clamp(0.9 * (d_th - _VFH_HALF_L - 0.2), 0.0, 1.5))
    if v > 1e-3 and not _lidar_clear(st["raw"], v, 0.3 + 0.4 * v):
        v = 0.0
    if v > 1e-3 and len(M):
        xs, ys, hs = _rollout(pose, st["vm"], 1.5)
        m0 = _traj_margin(xs, ys, hs, M, 0.05)[0]
        for fac in (1.0, 0.5):
            xs, ys, hs = _rollout(pose, st["vm"], 1.5, v=v * fac, w=w)
            if _traj_margin(xs, ys, hs, M, 0.05)[0] >= min(0.0, m0):
                v = v * fac
                break
        else:
            v = 0.0
    return v, w




_DET_LAT = (0.8, 1.2)


_DET_V = 1.0


_DET_MAX_T = 4.0


_DET_REACH = 0.25


def _side_detour(st, pose, M, target):
    """Side evasion point while paused and stopped: candidates 0.8 / 1.2 m from the robot along (a) the direction
    perpendicular to the nearest mover's velocity on the side away from it, (b) away from the mover, (c) their
    bisector, (d) robot left / right.  A candidate must clear the static map (_map_free, straight segment from the
    robot and on to target), have a clear lidar strip when it starts forward, a statically clear predicted motion,
    and a predicted mover margin over _HORIZON better than standing still by 0.1.  Returns the best (x, y) or None."""
    if len(M) == 0:
        return None
    x, y, yaw = pose
    vm = st["vm"]
    mx = M[:, 0] + M[:, 2] * M[:, 4]
    my = M[:, 1] + M[:, 3] * M[:, 4]
    k = int(np.argmin(np.hypot(mx - x, my - y)))
    ax, ay = x - mx[k], y - my[k]
    if math.hypot(ax, ay) > 1e-6:
        ax, ay = _unit(ax, ay)
    else:
        ax, ay = -math.cos(yaw), -math.sin(yaw)
    sp = math.hypot(M[k, 2], M[k, 3])
    if sp > 0.1:
        px, py = -M[k, 3] / sp, M[k, 2] / sp
        if px * ax + py * ay < 0.0:
            px, py = -px, -py
    else:
        px, py = ax, ay
    lx, ly = -math.sin(yaw), math.cos(yaw)
    dirs = [((px, py), 0.10), (_unit(px + ax, py + ay), 0.08), ((ax, ay), 0.05),
            ((lx, ly), 0.0), ((-lx, -ly), 0.0)]
    m_hold = _margin(pose, vm, 0.0, 0.0, M, _HORIZON, 0.15)[0]
    best, best_s = None, -1e9
    for (ux, uy), bonus in dirs:
        for lat in _DET_LAT:
            qx, qy = x + ux * lat, y + uy * lat
            if not (_map_free(qx, qy, 0.55) and _segment_free((x, y), (qx, qy), 0.35)
                    and _segment_free((qx, qy), target, 0.45)):
                continue
            v0 = min(_follow(pose, (qx, qy))[0], _DET_V)
            if v0 > 1e-3 and not _lidar_clear(st["raw"], v0, 0.5):
                continue
            xs, ys, hs = _rollout(pose, vm, _HORIZON, tgt=(qx, qy), vmax=_DET_V)
            if not _traj_static_ok(xs, ys, hs):
                continue
            m, _ = _traj_margin(xs, ys, hs, M, 0.15)
            if m < m_hold + 0.1:
                continue
            s = min(m, 1.0) + bonus - 0.05 * lat
            if s > best_s:
                best, best_s = (float(qx), float(qy)), s
    return best


def _detour_cmd(st, t, pose, M):
    """Command toward the active side detour (_follow, speed capped at _DET_V, final check with _guard).
    Ends the detour (st["detour"] = None, returns None) when it is reached, after _DET_MAX_T s, or when driving on
    toward it is predicted to be unsafe and worse than standing still."""
    det = st["detour"]
    x, y = pose[0], pose[1]
    if math.hypot(det[0] - x, det[1] - y) < _DET_REACH or t > st["det_until"]:
        st["detour"] = None
        return None
    if len(M):
        xs, ys, hs = _rollout(pose, st["vm"], _HORIZON, tgt=det, vmax=_DET_V)
        m = _traj_margin(xs, ys, hs, M, 0.15)[0]
        if m < 0.0:
            m0 = _margin(pose, st["vm"], 0.0, 0.0, M, _HORIZON, 0.15)[0]
            if m < m0 - 0.05:
                st["detour"] = None
                return None
    v, w, _ = _follow(pose, det)
    v = min(v, _DET_V)
    return _guard(st, pose, M, v, w)




_PLAN_HORIZON = 3.0


_GRID_RES = 0.2


_GNX = int(math.ceil((WALLS[1] - WALLS[0]) / _GRID_RES))


_GNY = int(math.ceil((WALLS[3] - WALLS[2]) / _GRID_RES))


_STATIC_BLOCK = 0.6


_STATIC_SOFT = 1.2


_DYN_PEN = 4.0


_DYN_CORE_PEN = 12.0


_PLAN_V = 1.0


_PLAN_TWIN = 1.0


_REPLAN_DT = 0.5


_NEAR_HOLD = 1.5


_LOS_MAX = 3.0


_ASTAR_MAX_EXP = 30000


_GRID_CACHE = {}


def _static_grid():
    """Cell centres of the _GRID_RES grid over the WALLS rectangle and their static clearance (computed once)."""
    if "sd" not in _GRID_CACHE:
        gx = WALLS[0] + (np.arange(_GNX) + 0.5) * _GRID_RES
        gy = WALLS[2] + (np.arange(_GNY) + 0.5) * _GRID_RES
        XC, YC = np.meshgrid(gx, gy)
        sd = _static_dist(XC.ravel(), YC.ravel()).reshape(_GNY, _GNX)
        _GRID_CACHE.update(gx=gx, gy=gy, XC=XC, YC=YC, sd=sd)
    return _GRID_CACHE


def _cell(x, y):
    """Grid cell (ix, iy) containing world point (x, y), clamped to the grid."""
    ix = int(_clamp(math.floor((x - WALLS[0]) / _GRID_RES), 0, _GNX - 1))
    iy = int(_clamp(math.floor((y - WALLS[2]) / _GRID_RES), 0, _GNY - 1))
    return ix, iy


def _build_dynamic_grid(st, pose, M):
    """Time-augmented occupancy grid.  Static: cells with clearance < _STATIC_BLOCK are blocked (0.45 within
    0.6 m of the robot), soft cost up to 0.5 below _STATIC_SOFT.  Dynamic: every track is extrapolated over
    _PLAN_HORIZON s (0.25 s steps, radius _R_ROBOT + _R_MOVER + 0.25 + growth); a moving track only marks cells
    the robot reaches (straight line at _PLAN_V) within _PLAN_TWIN s of that time, a stopped track marks its cells
    always.  Penalty _DYN_PEN, _DYN_CORE_PEN inside the contact radius."""
    G = _static_grid()
    XC, YC, sd = G["XC"], G["YC"], G["sd"]
    x, y = pose[0], pose[1]
    dr = np.hypot(XC - x, YC - y)
    blocked = sd < np.where(dr < 0.6, 0.45, _STATIC_BLOCK)
    soft = 0.5 * np.clip((_STATIC_SOFT - sd) / (_STATIC_SOFT - _STATIC_BLOCK), 0.0, 1.0)
    dyn = np.zeros_like(sd)
    trob = dr / _PLAN_V
    taus = np.arange(0.0, _PLAN_HORIZON + 1e-9, 0.25)
    for row in np.asarray(M, dtype=np.float64).reshape(-1, 6):
        mx0, my0, vx, vy, age, g = (float(v) for v in row)
        moving = math.hypot(vx, vy) > 0.1
        for tau in taus:
            if not moving and tau > 0.0:
                break
            cx = mx0 + vx * (age + tau)
            cy = my0 + vy * (age + tau)
            R = _R_ROBOT + _R_MOVER + 0.25 + g * tau
            j0 = max(0, int(math.floor((cx - R - WALLS[0]) / _GRID_RES)))
            j1 = min(_GNX, int(math.floor((cx + R - WALLS[0]) / _GRID_RES)) + 2)
            i0 = max(0, int(math.floor((cy - R - WALLS[2]) / _GRID_RES)))
            i1 = min(_GNY, int(math.floor((cy + R - WALLS[2]) / _GRID_RES)) + 2)
            if j0 >= j1 or i0 >= i1:
                continue
            d = np.hypot(XC[i0:i1, j0:j1] - cx, YC[i0:i1, j0:j1] - cy)
            hit = d < R
            if moving:
                hit &= np.abs(trob[i0:i1, j0:j1] - tau) <= _PLAN_TWIN
            if not hit.any():
                continue
            pen = np.where(d < _R_ROBOT + _R_MOVER, _DYN_CORE_PEN, _DYN_PEN)
            sub = dyn[i0:i1, j0:j1]
            np.maximum(sub, np.where(hit, pen, 0.0), out=sub)
    return {"blocked": blocked, "soft": soft, "dyn": dyn}


def _hpush(h, item):
    """Binary min-heap push (items are (key, index))."""
    h.append(item)
    i = len(h) - 1
    while i > 0:
        p = (i - 1) >> 1
        if h[p][0] <= item[0]:
            break
        h[i] = h[p]
        i = p
    h[i] = item


def _hpop(h):
    """Binary min-heap pop of the smallest item."""
    last = h.pop()
    if not h:
        return last
    top = h[0]
    n = len(h)
    i = 0
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n and h[c + 1][0] < h[c][0]:
            c += 1
        if h[c][0] >= last[0]:
            break
        h[i] = h[c]
        i = c
    h[i] = last
    return top


def _los_ok(grid, p, q, cap, start):
    """Straight segment p -> q stays off blocked cells (except within 0.3 m of start) and on cells whose dynamic
    penalty does not exceed cap."""
    B, D = grid["blocked"], grid["dyn"]
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    n = max(1, int(L / 0.1))
    for k in range(1, n + 1):
        s = k / n
        px = p[0] + (q[0] - p[0]) * s
        py = p[1] + (q[1] - p[1]) * s
        ix, iy = _cell(px, py)
        if B[iy, ix] and math.hypot(px - start[0], py - start[1]) > 0.3:
            return False
        if D[iy, ix] > cap + 1e-9:
            return False
    return True


def _replan_path(pose, goal, grid):
    """A* on the time-augmented grid from the robot cell to the goal cell (8-connected, no corner cutting,
    cost = step length * (1 + soft) + dynamic penalty * _GRID_RES, octile heuristic).  The cell path is
    shortened by line-of-sight (segments up to _LOS_MAX m, not entering a higher dynamic penalty than the cells
    they replace).  Returns the waypoint sequence [(x, y), ...] ending exactly at goal, or None."""
    blocked = grid["blocked"]
    nx = _GNX
    N = _GNX * _GNY
    sx, sy = _cell(pose[0], pose[1])
    gx, gy = _cell(goal[0], goal[1])
    if blocked[gy, gx]:
        return None
    s_i, g_i = sy * nx + sx, gy * nx + gx
    if s_i == g_i:
        return [(float(goal[0]), float(goal[1]))]
    bl = blocked.ravel().tolist()
    bl[s_i] = False
    soft = grid["soft"].ravel().tolist()
    pen = (grid["dyn"] * _GRID_RES).ravel().tolist()
    res = _GRID_RES
    dg = res * math.sqrt(2.0)
    gs = [float("inf")] * N
    par = [-1] * N
    done = bytearray(N)
    gs[s_i] = 0.0
    nbr = ((1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res),
           (1, 1, dg), (1, -1, dg), (-1, 1, dg), (-1, -1, dg))
    h0x, h0y = abs(sx - gx), abs(sy - gy)
    heap = [(res * (max(h0x, h0y) + 0.41421356 * min(h0x, h0y)), s_i)]
    exp = 0
    while heap:
        _, i = _hpop(heap)
        if done[i]:
            continue
        if i == g_i:
            break
        done[i] = 1
        exp += 1
        if exp > _ASTAR_MAX_EXP:
            return None
        iy, ix = divmod(i, nx)
        gi = gs[i]
        for ddx, ddy, L in nbr:
            jx, jy = ix + ddx, iy + ddy
            if jx < 0 or jx >= nx or jy < 0 or jy >= _GNY:
                continue
            j = jy * nx + jx
            if bl[j] or done[j]:
                continue
            if ddx and ddy and (bl[iy * nx + jx] or bl[jy * nx + ix]):
                continue
            ng = gi + L * (1.0 + soft[j]) + pen[j]
            if ng < gs[j]:
                gs[j] = ng
                par[j] = i
                hx, hy = abs(jx - gx), abs(jy - gy)
                _hpush(heap, (ng + res * (max(hx, hy) + 0.41421356 * min(hx, hy)), j))
    if not math.isfinite(gs[g_i]):
        return None
    cells = []
    i = g_i
    while i >= 0:
        cells.append(i)
        if i == s_i:
            break
        i = par[i]
    cells.reverse()
    if cells[0] != s_i:
        return None
    G = _static_grid()
    gxs, gys = G["gx"], G["gy"]
    D = grid["dyn"]
    pts, dv = [], []
    for c in cells:
        cy_, cx_ = divmod(c, nx)
        pts.append((float(gxs[cx_]), float(gys[cy_])))
        dv.append(float(D[cy_, cx_]))
    start = (float(pose[0]), float(pose[1]))
    pts[0] = start
    pts[-1] = (float(goal[0]), float(goal[1]))
    out = []
    i = 0
    last = len(pts) - 1
    kmax = max(1, int(_LOS_MAX / _GRID_RES))
    while i < last:
        j = min(last, i + kmax)
        while j > i + 1:
            if _los_ok(grid, pts[i], pts[j], max(dv[i:j + 1]), start):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _path_near_mask(pts, r):
    """Grid cells within r of the polyline pts (used as hysteresis toward the previous plan)."""
    G = _static_grid()
    XC, YC = G["XC"], G["YC"]
    near = np.zeros(XC.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            s = np.zeros_like(XC)
        else:
            s = np.clip(((XC - ax) * dx + (YC - ay) * dy) / L2, 0.0, 1.0)
        near |= np.hypot(XC - ax - s * dx, YC - ay - s * dy) < r
    return near


def _replan_route(st, t, pose, M):
    """New active route: A* path (_build_dynamic_grid + _replan_path) to the next original waypoint not yet
    reached (skipping ones within 1.2 m), then the rest of _WAYPOINTS.  Cells away from the previous plan cost
    0.3 more per metre (hysteresis against left/right flipping).  Entries are (x, y, original index or -1)."""
    no = len(_WAYPOINTS)
    x, y = pose[0], pose[1]
    k = min(st["orig_next"], no - 1)
    while k < no - 1 and math.hypot(_WAYPOINTS[k][0] - x, _WAYPOINTS[k][1] - y) < 1.2:
        k += 1
    tgt = _WAYPOINTS[k]
    if math.hypot(tgt[0] - x, tgt[1] - y) < 0.6:
        return None
    grid = _build_dynamic_grid(st, pose, M)
    prev = st.get("plan_pts")
    if prev is not None and len(prev) >= 2 and t - st["plan_t"] <= _NEAR_HOLD + _REPLAN_DT:
        grid["soft"] = grid["soft"] + 0.3 * (~_path_near_mask(prev, 0.6))
    pts = _replan_path(pose, tgt, grid)
    if not pts:
        return None
    st["plan_pts"] = [(float(x), float(y))] + [(float(p[0]), float(p[1])) for p in pts]
    route = [(float(p[0]), float(p[1]), -1) for p in pts[:-1]]
    route.append((float(tgt[0]), float(tgt[1]), k))
    route += [(float(p[0]), float(p[1]), j) for j, p in enumerate(_WAYPOINTS) if j > k]
    return route


def _lookahead(wps, idx, pose, dist):
    """First route point from idx on that is at least dist from the robot (else the last one)."""
    x, y = pose[0], pose[1]
    for k in range(min(idx, len(wps) - 1), len(wps)):
        if math.hypot(wps[k][0] - x, wps[k][1] - y) >= dist:
            return (wps[k][0], wps[k][1])
    return (wps[-1][0], wps[-1][1])


def _plan_speed(st, pose, wp, v_nom, M):
    """Speed on the replanned route near movers: _predictive_speed first; if nothing is safe, drive slowly
    (0.8 / 0.5 m/s) along the route only when that is statically clear and its predicted mover margin beats
    standing still by 0.1 (a mover can hit a stopped robot); otherwise stop."""
    if v_nom <= 0.0:
        return 0.0
    vc = _predictive_speed(st, pose, wp, v_nom, M)
    if vc is not None:
        return vc
    if len(M) == 0 or not _lidar_clear(st["raw"], 0.5, 0.35):
        return 0.0
    vm = st["vm"]
    look = _lookahead(st["wps"], st["idx"], pose, 1.5)
    m0 = _margin(pose, vm, 0.0, 0.0, M, _HORIZON, 0.15)[0]
    for vc in (min(v_nom, 0.8), min(v_nom, 0.5)):
        xs, ys, hs = _rollout(pose, vm, _HORIZON, tgt=look, vmax=vc)
        if not _traj_static_ok(xs, ys, hs):
            continue
        if _traj_margin(xs, ys, hs, M, 0.15)[0] >= m0 + 0.1:
            return vc
    return 0.0




_VO_FAIL_T = 1.0


def _route_target(st, pose):
    """Advance along the active route st["wps"] and return the current target (x, y); None once the last point is
    reached (goal alignment).  Pushed off the goal by more than 0.45 m -> back to the last route point."""
    x, y = pose[0], pose[1]
    wps = st["wps"]
    n = len(wps)
    if st["idx"] >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.45:
        st["idx"] = n - 1
    if st["idx"] >= n:
        return None
    e = wps[st["idx"]]
    while st["idx"] < n - 1 and math.hypot(e[0] - x, e[1] - y) < 0.35:
        if e[2] >= 0:
            st["orig_next"] = max(st["orig_next"], e[2] + 1)
        st["idx"] += 1
        e = wps[st["idx"]]
    if st["idx"] == n - 1 and math.hypot(e[0] - x, e[1] - y) < 0.25:
        st["orig_next"] = len(_WAYPOINTS)
        st["idx"] = n
        return None
    return (e[0], e[1])


# ========================= [END EDIT REGION] ===============================
