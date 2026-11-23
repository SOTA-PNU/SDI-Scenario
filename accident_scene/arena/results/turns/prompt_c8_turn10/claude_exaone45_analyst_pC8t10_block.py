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
    """Waypoint law + 2D-lidar mover handling (attempt 10):
    - evasion starts when TTC < _TTC_TRIG (1.0 s) and is held at least _EVADE_HOLD (1.0 s) after the last trigger
    - evasion ends when (a) TTC >= _TTC_TRIG, lidar clearance clr_all >= _RESUME_F * _CLR_GO and nominal static
      clearance >= _RESUME_ST (0.2 m) held for _RESUME_HOLD, or (b) narrow zone (0 <= st_n < 0.15 m) with the mover
      conditions met, or (c) robot stopped, st_n >= 0.15 m and mover conditions met (no hold), or (d) _EVADE_MAX (6 s) cap
    - at evasion start, current static clearance < 0.25 m turns on static-priority scoring in _choose."""
    S = _CS
    if not S:
        S.update(idx=0, tracks=[], mv={"front": np.zeros((0, 2)), "back": np.zeros((0, 2))},
                 mt={"front": -99.0, "back": -99.0}, last={}, v=0.0, w=0.0,
                 slow=False, stopped=False, n_stop=0, stop_t=0.0, stop_sum=0.0,
                 started=False, reported=False,
                 evading=False, evade_until=-1.0, n_evade=0, evade_t=0.0, evade_sum=0.0,
                 evade_on=False, evade_start=0.0, cool_until=-1.0, safe_since=-1.0, st_prio=False)
    x, y, yaw = pose
    n = len(_WAYPOINTS)

    # 1) sensing: both 2D lidars -> world points -> drop static map -> mover clusters/tracks
    fresh = {}
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p.shape[0] > 0 and p is not S["last"].get(which):
            S["last"][which] = p
            S["mv"][which] = _ingest(p, pose, abs(S["w"]))
            S["mt"][which] = t
            fresh[which] = True
    if fresh.get("front") or (fresh.get("back") and t - S["mt"]["front"] > 0.25):
        parts = [S["mv"][k] for k in ("front", "back") if t - S["mt"][k] < 0.25]
        _update_tracks(np.concatenate(parts) if parts else np.zeros((0, 2)), t)

    if not S["started"]:   # look before the first move (a mover may be near START)
        if not ((S["mt"]["front"] > -1.0 and t >= 0.3) or t >= 2.0):
            return 0.0, 0.0, False
        S["started"] = True

    # 2) goal alignment / finish
    if S["idx"] >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.5:
        S["idx"] = n - 1
    if S["idx"] >= n and abs(_wrap_angle(GOAL[2] - yaw)) < 0.05:
        _log_stop(t, pose, False, 9.0)
        _log_evade(t, pose, False, 99.0, 9.0)
        if not S["reported"]:
            S["reported"] = True
            print("[C2 debug] done t=%.2f stop_episodes=%d stopped_total=%.2fs evade_episodes=%d evade_total=%.2fs"
                  % (t, S["n_stop"], S["stop_sum"], S["n_evade"], S["evade_sum"]), flush=True)
        return 0.0, 0.0, True

    # 3) nominal command (verified law) and its predicted safety against ALL movers
    v_n, w_n, S["idx"] = _nominal_cmd(x, y, yaw, S["idx"], 1.0)
    target = _WAYPOINTS[min(S["idx"], n - 1)]
    P, U = _movers_now(t)                    # all-around movers (front + back lidar)

    d_near = _nearest_mover(P, pose)
    ttc = _ttc(P, U, pose, S["v"], _EARLY_R)  # recomputed every frame, also during evasion
    anchors = _mover_anchors(t, pose)         # all movers within _EARLY_R, with velocities

    traj_n = _rollout_nominal(pose, S["idx"], S["v"], 1.0)
    st_n = _static_clearance(traj_n[None, :_KS + 1])[0]
    clr_all = _mover_clearance(traj_n[None], P, U)[0]
    st0 = _pose_static_clearance(pose)        # static clearance of the current pose

    # evasion trigger: TTC < _TTC_TRIG; hold at least _EVADE_HOLD after the last trigger
    if ttc < _TTC_TRIG and (t >= S["cool_until"] or clr_all < _CLR_GO):
        if not S["evade_on"]:
            S["evade_on"] = True
            S["evade_start"] = t
            S["safe_since"] = -1.0
            S["st_prio"] = st0 < _ST_PRIO     # direction 5: near a static obstacle -> static-priority choice
        S["evade_until"] = max(S["evade_until"], t + _EVADE_HOLD)
    if S["evade_on"]:
        if S["st_prio"] and st0 >= _ST_PRIO:
            S["st_prio"] = False
        mover_ok = ttc >= _TTC_TRIG and clr_all >= _CLR_GO * _RESUME_F
        ok = mover_ok and st_n >= _RESUME_ST                       # direction 1: 0.2 m (was 0.3 m)
        if ok:
            if S["safe_since"] < 0.0:
                S["safe_since"] = t
        else:
            S["safe_since"] = -1.0
        sustained = ok and t - S["safe_since"] >= _RESUME_HOLD
        long_ = t - S["evade_start"] > _EVADE_MAX
        # direction 1: narrow zone (0 <= st_n < 0.15 m) - the static requirement cannot be met, end before the cap
        narrow = mover_ok and 0.0 <= st_n < _NARROW_ST and t >= S["evade_until"]
        # direction 6: robot stopped, even with a slowly closing mover (TTC >= trigger) - end at once if st_n >= 0.15 m
        still = mover_ok and abs(S["v"]) < _STILL_V and st_n >= _STOP_RESUME_ST
        if (t >= S["evade_until"] and sustained) or narrow or still \
                or (long_ and st_n >= 0.0 and clr_all >= _CLR_GO):
            if long_:
                S["cool_until"] = t + _EVADE_COOL
            S["evade_on"] = False
            S["safe_since"] = -1.0
            S["st_prio"] = False
    early = S["evade_on"]

    # slow mode on clr_all with hysteresis (enter < _CLR_GO, leave >= _HYST * _CLR_GO)
    if S["slow"]:
        if st_n >= 0.0 and clr_all >= _CLR_GO * _HYST:
            S["slow"] = False
    elif clr_all < _CLR_GO:
        S["slow"] = True

    # 4) decide
    stop = False
    if st_n < 0.0:                                     # off the verified path: safe arc toward the waypoint
        v, w, _ = _choose(pose, P, U, target, anchors, True, st0 < _ST_PRIO)
    elif early:                                        # evasion: relative motion + static clearance (direction 3, 5)
        v, w, _ = _choose(pose, P, U, target, anchors, True, S["st_prio"])
    elif not S["slow"]:
        if clr_all >= _CLR_STOP:
            v, w = v_n, w_n
        else:
            v, w, _ = _choose(pose, P, U, target, anchors)
    else:
        v, w, stop = _slow_cmd(pose, S["idx"], S["v"], clr_all, P, U)
        if stop or S["stopped"]:
            closing = _closing_speed(P, U, pose, S["v"], _CLOSE_R)
            hit_still = False
            if stop:
                still_tr = np.tile(np.array([x, y, yaw], dtype=np.float64), (_KP + 1, 1))[None]
                hit_still = _mover_clearance(still_tr, P, U)[0] < _CLR_STOP
            if hit_still or closing < _CLOSE_V:
                v, w, _ = _choose(pose, P, U, target, anchors)

    _log_evade(t, pose, early, ttc, d_near)
    _log_stop(t, pose, stop, clr_all)
    S["v"], S["w"] = v, w
    return v, w, False




_V_MAX = 1.5            # cruise speed of the verified static run


_W_MAX = 2.0


_STOP_DIST = 2.0        # diagnosis: mover within 2 m ahead -> stop


_STOP_HALF_W = 0.70     # half width of the stop corridor ahead of the robot


_HOLD_S = 0.5           # diagnosis 5: re-sense for 0.5 s after stopping (was 1.0 s)


_DTP = 0.1              # prediction step [s]


_KP = 30                # 3.0 s prediction horizon for movers


_KS = 12                # 1.2 s horizon for the static-map check


_ACC = 2.0              # assumed accel limit used in prediction [m/s^2]


_CLR_GO = 0.35          # min predicted body clearance to keep driving


_CLR_STOP = 0.25        # diagnosis 5: min predicted mover clearance for the scaled path / standing still (was 0.20)


_SENSE_R = 6.0          # only lidar points within this range are used


_ROBOT_R = 0.42         # radius of the 3 body circles used against the static map


_BODY_SAMPLES = (-0.60, -0.225, 0.15)


_CAND = [(cv, cw) for cv in (-0.8, -0.4, 0.0, 0.5, 1.0) for cw in (-1.2, -0.6, 0.0, 0.6, 1.2)]


_STATIC_RECTS = np.array(
    [[b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2]
     for b in MISSION["boxes"]] + [[o[1], o[2], o[3], o[4]] for o in FIXED_OBSTACLES])


_CS = {}


def _nominal_cmd(x, y, yaw, idx, scale):
    """The verified waypoint law as a pure function: returns (v, w, new_idx)."""
    n = len(_WAYPOINTS)
    if idx >= n:
        e = _wrap_angle(GOAL[2] - yaw)
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), idx
    tx, ty = _WAYPOINTS[idx]
    dist = math.hypot(tx - x, ty - y)
    if idx < n - 1 and dist < 0.35:
        idx += 1
        tx, ty = _WAYPOINTS[idx]
        dist = math.hypot(tx - x, ty - y)
    e = _wrap_angle(math.atan2(ty - y, tx - x) - yaw)
    w = _clamp(7.5 * e, -_W_MAX, _W_MAX)
    v = 0.0 if abs(e) > 0.35 else scale * min(_V_MAX, 2.5 * dist + 0.5)
    if idx == n - 1 and dist < 0.25:
        idx += 1
    return v, w, idx


def _rollout_nominal(pose, idx, v0, scale):
    """Predict the robot path (K+1, 3) when it keeps following the waypoint law."""
    x, y, yaw = pose
    va = v0
    traj = np.empty((_KP + 1, 3))
    traj[0] = (x, y, yaw)
    for k in range(_KP):
        v, w, idx = _nominal_cmd(x, y, yaw, idx, scale)
        va += _clamp(v - va, -_ACC * _DTP, _ACC * _DTP)
        x += va * math.cos(yaw) * _DTP
        y += va * math.sin(yaw) * _DTP
        yaw += w * _DTP
        traj[k + 1] = (x, y, yaw)
    return traj


def _rollout_cands(pose, v0):
    """Predict paths (C, K+1, 3) for the constant (v, w) candidates in _CAND."""
    V = np.array([c[0] for c in _CAND])
    W = np.array([c[1] for c in _CAND])
    C = V.size
    x = np.full(C, float(pose[0]))
    y = np.full(C, float(pose[1]))
    yaw = np.full(C, float(pose[2]))
    va = np.full(C, float(v0))
    traj = np.empty((C, _KP + 1, 3))
    traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2] = x, y, yaw
    for k in range(_KP):
        va = va + np.clip(V - va, -_ACC * _DTP, _ACC * _DTP)
        x = x + va * np.cos(yaw) * _DTP
        y = y + va * np.sin(yaw) * _DTP
        yaw = yaw + W * _DTP
        traj[:, k + 1, 0], traj[:, k + 1, 1], traj[:, k + 1, 2] = x, y, yaw
    return traj


def _mover_clearance(traj, P, U):
    """Min distance (C,) from the robot body rectangle to mover points moved with their velocities."""
    if P.shape[0] == 0:
        return np.full(traj.shape[0], 9.0)
    ts = np.arange(traj.shape[1]) * _DTP
    Px = P[:, 0][None, None, :] + U[:, 0][None, None, :] * ts[None, :, None]
    Py = P[:, 1][None, None, :] + U[:, 1][None, None, :] * ts[None, :, None]
    dx = Px - traj[:, :, 0:1]
    dy = Py - traj[:, :, 1:2]
    c = np.cos(traj[:, :, 2:3])
    s = np.sin(traj[:, :, 2:3])
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    ex = np.maximum(np.maximum(_BODY[0] - lx, lx - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - ly, ly - _BODY[3]), 0.0)
    return np.sqrt(ex * ex + ey * ey).min(axis=(1, 2))


def _static_clearance(traj):
    """Min clearance (C,) of 3 body circles to walls, MISSION boxes and FIXED_OBSTACLES (<0 = would touch)."""
    c = np.cos(traj[..., 2])
    s = np.sin(traj[..., 2])
    xs = np.concatenate([traj[..., 0] + o * c for o in _BODY_SAMPLES], axis=1)
    ys = np.concatenate([traj[..., 1] + o * s for o in _BODY_SAMPLES], axis=1)
    R = _STATIC_RECTS
    ex = np.maximum(np.maximum(R[:, 0] - xs[..., None], xs[..., None] - R[:, 1]), 0.0)
    ey = np.maximum(np.maximum(R[:, 2] - ys[..., None], ys[..., None] - R[:, 3]), 0.0)
    d = np.sqrt(ex * ex + ey * ey).min(axis=2)
    xmin, xmax, ymin, ymax = WALLS
    dw = np.minimum(np.minimum(xs - xmin, xmax - xs), np.minimum(ys - ymin, ymax - ys))
    return (np.minimum(d, dw) - _ROBOT_R).min(axis=1)


def _choose(pose, P, U, target, anchor=None, evading=False, static_prio=False):
    """Pick the (v, w) candidate with the best mover clearance, static-map safety and progress.
    anchor: (M, 4) array of moving anchors (x, y, vx, vy), one per mover track within _EARLY_R (or None/empty).
    Relative motion prediction over the first _ANCHOR_K steps (anchor penalty, closing penalty, lateral bonus).
    evading: static clearance as a secondary metric (_ST_W2, direction 3) plus a bonus _ST_W only for candidates
      with stc >= _ST_BONUS_MIN (direction 2); static_prio adds _ST_PRIO_W on static clearance (direction 5).
    Static safety is a hard constraint (direction 2): candidates with stc < 0 are masked out whenever any candidate is
    static-safe. When none is (the robot is already inside the margin), only candidates that do not get closer to the
    static map than the current pose are allowed, preferring the ones that open the clearance.
    Also returns the predicted clearance of standing still (all-around mover points)."""
    trajc = _rollout_cands(pose, _CS["v"])
    clr = _mover_clearance(trajc, P, U)
    stc = _static_clearance(trajc[:, 1:_KS + 1])
    st0 = _pose_static_clearance(pose)
    V = np.array([c[0] for c in _CAND])
    W = np.array([c[1] for c in _CAND])
    end = trajc[:, -1, :2]
    prog = np.hypot(end[:, 0] - target[0], end[:, 1] - target[1])
    score = (10.0 * np.minimum(clr, 1.0) - 0.5 * prog
             - 0.3 * (np.abs(V - _CS["v"]) + 0.3 * np.abs(W - _CS["w"]))
             + 5.0 * np.minimum(stc, 0.0))
    if evading:
        sc = np.clip(stc, 0.0, _ST_SAFE) / _ST_SAFE
        score = score + _ST_W * sc * (stc >= _ST_BONUS_MIN)       # direction 2
        score = score + _ST_W2 * sc                               # direction 3
        if static_prio:
            score = score + _ST_PRIO_W * sc                       # direction 5
    A = None
    if anchor is not None:
        A = np.asarray(anchor, dtype=np.float64).reshape(-1, 4)
        if A.shape[0] == 0:
            A = None
    if A is not None:
        k = max(1, min(_ANCHOR_K, trajc.shape[1] - 1))
        ts = np.arange(k + 1) * _DTP
        mx = A[:, 0][None, :] + A[:, 2][None, :] * ts[:, None]          # (k+1, M)
        my = A[:, 1][None, :] + A[:, 3][None, :] * ts[:, None]
        off = 0.5 * (_BODY[0] + _BODY[1])
        bx = trajc[:, :k + 1, 0] + off * np.cos(trajc[:, :k + 1, 2])     # (C, k+1)
        by = trajc[:, :k + 1, 1] + off * np.sin(trajc[:, :k + 1, 2])
        dd = np.hypot(bx[:, :, None] - mx[None, :, :], by[:, :, None] - my[None, :, :])   # (C, k+1, M)
        da = dd.min(axis=1)                                              # (C, M)
        score = score - 20.0 * np.maximum(_ANCHOR_R - da, 0.0).sum(axis=1)
        vcx = (bx[:, k] - bx[:, k - 1]) / _DTP
        vcy = (by[:, k] - by[:, k - 1]) / _DTP
        sx = mx[k][None, :] - bx[:, k][:, None]
        sy = my[k][None, :] - by[:, k][:, None]
        sd = np.maximum(np.hypot(sx, sy), 1e-3)
        rate = (sx * (A[:, 2][None, :] - vcx[:, None]) + sy * (A[:, 3][None, :] - vcy[:, None])) / sd
        near = sd < _REL_R
        score = score - _REL_W * (np.maximum(-rate, 0.0) * near).max(axis=1)
        cx0, cy0 = float(bx[0, 0]), float(by[0, 0])
        j = int(np.argmin(np.hypot(A[:, 0] - cx0, A[:, 1] - cy0)))
        b = math.atan2(A[j, 1] - cy0, A[j, 0] - cx0)
        dx = bx[:, k] - cx0
        dy = by[:, k] - cy0
        dl = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - b
        dev = np.abs(np.arctan2(np.sin(ang), np.cos(ang)))
        score = score + _LAT_W * 0.5 * (1.0 - np.cos(dev)) * np.minimum(dl / 0.5, 1.0)
    # hard static constraint (direction 2): static-unsafe candidates are never chosen when a safe one exists
    allowed = stc >= 0.0
    if not np.any(allowed):
        allowed = stc >= min(st0, 0.0) - 0.005        # do not get closer to the static map than now
        score = score + 20.0 * (stc - st0)              # prefer opening the static clearance
    if not np.any(allowed):
        i = int(np.argmax(stc))                         # least static-unsafe (escape) candidate
    else:
        i = int(np.argmax(np.where(allowed, score, -np.inf)))
    return float(V[i]), float(W[i]), float(clr[_CAND.index((0.0, 0.0))])


def _static_mask(P, margin):
    """True for world points explained by the static map (walls, MISSION boxes, FIXED_OBSTACLES)."""
    x, y = P[:, 0], P[:, 1]
    xmin, xmax, ymin, ymax = WALLS
    wall = (x <= xmin + margin) | (x >= xmax - margin) | (y <= ymin + margin) | (y >= ymax - margin)
    R = _STATIC_RECTS
    m = margin[:, None]
    inside = ((x[:, None] >= R[:, 0] - m) & (x[:, None] <= R[:, 1] + m)
              & (y[:, None] >= R[:, 2] - m) & (y[:, None] <= R[:, 3] + m)).any(axis=1)
    return wall | inside


def _ingest(pts, pose, wabs):
    """Robot-frame scan -> world points not on the static map (range-limited, lag-aware margin)."""
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = r < _SENSE_R
    if not np.any(keep):
        return np.zeros((0, 2))
    wp = to_world(pts[keep], pose)
    margin = 0.25 + r[keep] * (0.03 + 0.08 * min(wabs, 2.0))
    return wp[~_static_mask(wp, margin)]


def _update_tracks(P, t):
    """Cluster mover points and match clusters to previous tracks to estimate their velocity."""
    old = _CS["tracks"]
    used = set()
    new = []
    rem = P
    while rem.shape[0] > 0 and len(new) < 12:
        d = np.hypot(rem[:, 0] - rem[0, 0], rem[:, 1] - rem[0, 1])
        m = d < 0.9
        cl, rem = rem[m], rem[~m]
        if cl.shape[0] < 4:
            continue
        c = cl.mean(axis=0)
        vel = np.zeros(2)
        best, bd = None, 0.7
        for i, tr in enumerate(old):
            pc = tr["c"] + tr["v"] * (t - tr["t"])
            dd = math.hypot(c[0] - pc[0], c[1] - pc[1])
            if dd < bd:
                best, bd = i, dd
        if best is not None:
            tr = old[best]
            used.add(best)
            dt = t - tr["t"]
            if dt > 1e-3:
                a = min(1.0, dt / 0.25)
                vel = (1.0 - a) * tr["v"] + a * (c - tr["c"]) / dt
                sp = math.hypot(vel[0], vel[1])
                if sp > 2.0:
                    vel = vel * (2.0 / sp)
            else:
                vel = tr["v"].copy()
        k = np.linspace(0, cl.shape[0] - 1, min(cl.shape[0], 30)).astype(int)
        new.append({"c": c, "v": vel, "t": t, "pts": cl[k]})
    for i, tr in enumerate(old):
        if i not in used and t - tr["t"] < 0.4:
            new.append(tr)
    _CS["tracks"] = new


def _movers_now(t):
    """Mover points extrapolated to time t, with per-point velocities: (N, 2), (N, 2)."""
    Ps, Us = [], []
    for tr in _CS["tracks"]:
        age = t - tr["t"]
        if age > 0.6:
            continue
        Ps.append(tr["pts"] + tr["v"] * age)
        Us.append(np.tile(tr["v"], (tr["pts"].shape[0], 1)))
    if not Ps:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(Ps), np.concatenate(Us)


def _to_robot(P, pose):
    """World points (N, 2) -> robot frame (x forward, y left)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx = P[:, 0] - x
    dy = P[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)




_SECTOR_HALF = math.radians(45.0)   # diagnosis 1: stop decision uses mover points within +-45 deg ahead


_EVADE_R = 0.5          # diagnosis 3: side-step while stopped stays within 0.5 m of the stop point


def _front_sector(P, U, pose):
    """Mover points (and velocities) inside the forward +-45 deg sector of the robot (robot-frame apex at origin)."""
    if P.shape[0] == 0:
        return P, U
    L = _to_robot(P, pose)
    m = (L[:, 0] > 0.0) & (np.abs(L[:, 1]) <= L[:, 0] * math.tan(_SECTOR_HALF))
    return P[m], U[m]




_HYST = 1.0             # diagnosis 1: leave slow mode as soon as clr_all >= _CLR_GO (was 1.5)


def _slow_cmd(pose, idx, v0, clr, P, U):
    """Diagnosis 3: follow the nominal path with the speed scaled by s = clr_all / go (0..1), where the threshold
    go = _CLR_GO * f depends on the relative approach speed of movers within _EARLY_R (mover velocity U minus the
    robot velocity v0 along its heading): f = _SLOW_F_MIN when nothing approaches, up to _SLOW_F_MAX when a mover
    closes in at _SLOW_APP_V or faster.
    The scaled path must be static-safe and keep >= _CLR_STOP from ALL mover points (front and back lidar);
    otherwise the robot stops: returns (0.0, 0.0, True)."""
    x, y, yaw = pose
    rel = _closing_speed(P, U, pose, v0, _EARLY_R)
    app = max(-rel, 0.0)
    f = _clamp(_SLOW_F_MIN + (_SLOW_F_MAX - _SLOW_F_MIN) * app / _SLOW_APP_V, _SLOW_F_MIN, _SLOW_F_MAX)
    go = _CLR_GO * f
    s = _clamp(clr / go, 0.0, 1.0)
    v, w, _ = _nominal_cmd(x, y, yaw, idx, s)
    traj = _rollout_nominal(pose, idx, v0, s)
    ok = (_static_clearance(traj[None, :_KS + 1])[0] >= 0.0
          and _mover_clearance(traj[None], P, U)[0] >= _CLR_STOP)
    if not ok:
        return 0.0, 0.0, True
    return v, w, False


def _log_stop(t, pose, stop, clr):
    """Diagnosis 6: count stop episodes and their total duration; print one line on each transition only."""
    S = _CS
    if stop and not S["stopped"]:
        S["n_stop"] += 1
        S["stop_t"] = t
        print("[C2 debug] stop #%d start t=%.2f pos=(%.2f, %.2f) clr=%.2f"
              % (S["n_stop"], t, pose[0], pose[1], clr), flush=True)
    elif not stop and S["stopped"]:
        dur = t - S["stop_t"]
        S["stop_sum"] += dur
        print("[C2 debug] stop #%d end t=%.2f dur=%.2fs total=%.2fs"
              % (S["n_stop"], t, dur, S["stop_sum"]), flush=True)
    S["stopped"] = stop




_CLOSE_R = 2.0          # diagnosis 6: stopped/resume closing-speed check covers movers within 2.0 m of the body (was 1.5)


_CLOSE_V = -0.15        # diagnosis 5: closing speed below this [m/s] (approaching) -> evade immediately


def _closing_speed(P, U, pose, v, radius=_CLOSE_R):
    """Most negative radial (closing) speed [m/s] of mover points within `radius` of the body rectangle,
    relative to the robot moving at v along its heading. 0.0 when no mover point is that near."""
    if P.shape[0] == 0:
        return 0.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    near = np.hypot(ex, ey) < radius
    if not np.any(near):
        return 0.0
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * c, y + off * s
    dx = P[near, 0] - cx
    dy = P[near, 1] - cy
    dist = np.maximum(np.hypot(dx, dy), 1e-3)
    rvx = U[near, 0] - v * c
    rvy = U[near, 1] - v * s
    return float(((dx * rvx + dy * rvy) / dist).min())




_EARLY_R = 3.0          # diagnosis 2: running closing-speed check covers mover points within 3 m of the body


_EVADE_HOLD = 1.0       # direction 2: evasion held at least 1.0 s after the last TTC trigger (was 0.5 s)


def _nearest_mover(P, pose):
    """Diagnosis 2: distance [m] from the robot body rectangle to the nearest mover point (9.0 when none)."""
    if P.shape[0] == 0:
        return 9.0
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    return float(np.hypot(ex, ey).min())


def _log_evade(t, pose, evading, ttc, dist):
    """Count early-evasion episodes and their total duration; print one line on each transition only."""
    S = _CS
    if evading and not S["evading"]:
        S["n_evade"] += 1
        S["evade_t"] = t
        print("[C2 debug] evade #%d start t=%.2f pos=(%.2f, %.2f) ttc=%.2f dist=%.2f"
              % (S["n_evade"], t, pose[0], pose[1], ttc, dist), flush=True)
    elif not evading and S["evading"]:
        dur = t - S["evade_t"]
        S["evade_sum"] += dur
        print("[C2 debug] evade #%d end t=%.2f dur=%.2fs total=%.2fs"
              % (S["n_evade"], t, dur, S["evade_sum"]), flush=True)
    S["evading"] = evading




_TTC_TRIG = 1.0         # direction 1: evasion triggers when TTC < 1.0 s (turn 6: 7/8 pass; 2.0 s: 0/3)


_ANCHOR_R = 1.2         # direction 3: anchor penalty radius reverted to 1.2 m (was 0.8 m)


_ANCHOR_K = 15          # diagnosis 4: anchor penalty looks at the first 1.5 s of each candidate (was 1.0 s)


def _ttc(P, U, pose, v, radius=_EARLY_R):
    """Diagnosis 1: minimum time to collision [s] over mover points within `radius` of the body rectangle.
    TTC = gap to body / max(closing speed, 0.01), closing speed relative to the robot moving at v.
    99.0 when no mover point is near or none is approaching."""
    if P.shape[0] == 0:
        return 99.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    gap = np.hypot(ex, ey)
    near = gap < radius
    if not np.any(near):
        return 99.0
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * c, y + off * s
    dx = P[near, 0] - cx
    dy = P[near, 1] - cy
    dist = np.maximum(np.hypot(dx, dy), 1e-3)
    rvx = U[near, 0] - v * c
    rvy = U[near, 1] - v * s
    rate = (dx * rvx + dy * rvy) / dist
    app = rate < 0.0
    if not np.any(app):
        return 99.0
    ttc = gap[near][app] / np.maximum(-rate[app], 0.01)
    return float(min(ttc.min(), 99.0))


def _nearest_anchor(P, U, pose):
    """Diagnosis 3: (x, y, vx, vy) of the mover point nearest to the body rectangle (within _EARLY_R), else None."""
    if P.shape[0] == 0:
        return None
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    gap = np.hypot(ex, ey)
    i = int(np.argmin(gap))
    if gap[i] >= _EARLY_R:
        return None
    return (float(P[i, 0]), float(P[i, 1]), float(U[i, 0]), float(U[i, 1]))




_SLOW_F_MIN = 0.75      # diagnosis 3: slow threshold factor when no mover is approaching (0.75 * _CLR_GO > _CLR_STOP)


_SLOW_F_MAX = 1.5       # diagnosis 3: slow threshold factor for a mover approaching at _SLOW_APP_V or faster


_SLOW_APP_V = 1.0       # diagnosis 3: relative approach speed [m/s] at which the factor reaches _SLOW_F_MAX




def _mover_anchors(t, pose):
    """(M, 4) array of (x, y, vx, vy): for each live mover track, its point nearest to the body rectangle
    (extrapolated to t with the track velocity), kept only when within _EARLY_R of the body."""
    out = []
    for tr in _CS["tracks"]:
        age = t - tr["t"]
        if age > 0.6:
            continue
        pts = tr["pts"] + tr["v"] * age
        if pts.shape[0] == 0:
            continue
        L = _to_robot(pts, pose)
        ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
        ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
        gap = np.hypot(ex, ey)
        i = int(np.argmin(gap))
        if gap[i] < _EARLY_R:
            out.append((float(pts[i, 0]), float(pts[i, 1]), float(tr["v"][0]), float(tr["v"][1])))
    return np.array(out, dtype=np.float64).reshape(-1, 4)


_RESUME_F = 1.2         # evasion ends only when the nominal path clearance >= _RESUME_F * _CLR_GO (lidar validation)


_EVADE_MAX = 6.0        # direction 4: cap on one continuous evasion [s] (was 8.0 s)


_EVADE_COOL = 2.0       # after a capped evasion, TTC alone cannot re-trigger for this long unless clr_all < _CLR_GO


_REL_R = 2.5            # relative-motion closing penalty applies to movers within this distance at the window end


_REL_W = 2.0            # weight of the closing-speed penalty [score per m/s]


_LAT_W = 2.0            # weight of the angular-deviation (lateral / away) bonus




_RESUME_HOLD = 0.5      # direction 6: resume conditions must hold continuously for 0.5 s before evasion ends


_RESUME_ST = 0.2        # direction 1: nominal-path static clearance required to end evasion [m] (was 0.3)


_ST_SAFE = 0.4          # direction 5: static clearance [m] that earns the full bonus in _choose during evasion


_ST_W = 1.0             # direction 5: weight of the static-clearance bonus during evasion




_NARROW_ST = 0.15       # direction 1: nominal-path static clearance below this (and >= 0) = narrow zone -> end evasion on mover conditions only


_ST_BONUS_MIN = 0.2     # direction 2: static-clearance bonus in _choose only for candidates with stc >= 0.2 m


_ST_W2 = 1.5            # direction 3: weight of the continuous static-clearance term (secondary metric) during evasion


_ST_PRIO = 0.25         # direction 5: current static clearance below this at evasion start -> static-priority choice


_ST_PRIO_W = 3.0        # direction 5: extra static-clearance weight while static priority is on


_STOP_RESUME_ST = 0.15  # direction 6: when stopped, nominal static clearance >= this allows ending evasion at once


_STILL_V = 0.05         # direction 6: |last commanded v| below this counts as stopped


def _pose_static_clearance(pose):
    """Static clearance [m] of the 3 body circles at the current pose only (<0 = inside the safety margin)."""
    tr = np.array([[[float(pose[0]), float(pose[1]), float(pose[2])]]], dtype=np.float64)
    return float(_static_clearance(tr)[0])


# ========================= [END EDIT REGION] ===============================
