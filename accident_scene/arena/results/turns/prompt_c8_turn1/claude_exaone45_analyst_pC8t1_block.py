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
    """Unchanged waypoint law + 2D-lidar mover handling: stop (v=0, w=0) when any mover point is within
    2 m of the body rectangle (all around, also while turning in place), re-sense >= 1 s before resuming,
    slow down on predicted conflicts, evade only if standing still would be hit."""
    S = _CS
    if not S:
        S.update(idx=0, tracks=[], mv={"front": np.zeros((0, 2)), "back": np.zeros((0, 2))},
                 mt={"front": -99.0, "back": -99.0}, last={}, v=0.0, w=0.0,
                 holding=False, hold_until=-1.0, started=False)
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
        return 0.0, 0.0, True

    # 3) nominal command (verified law) and its predicted safety
    v_n, w_n, S["idx"] = _nominal_cmd(x, y, yaw, S["idx"], 1.0)
    target = _WAYPOINTS[min(S["idx"], n - 1)]
    P, U = _movers_now(t)
    traj_n = _rollout_nominal(pose, S["idx"], S["v"], 1.0)
    st_n = _static_clearance(traj_n[None, :_KS + 1])[0]
    clr_n = _mover_clearance(traj_n[None], P, U)[0]

    # diagnosis rule 3-4: min distance from the body rectangle _BODY (x -0.70..0.25, y -0.34..0.34)
    # to every non-static point, all around the robot (covers the area swept while turning in place);
    # below _STOP_DIST -> hold with v=0, w=0. No v_n > 0 condition: also checked while rotating.
    zone = False
    if P.shape[0] > 0:
        L = _to_robot(P, pose)
        ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
        ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
        d_body = np.sqrt(ex * ex + ey * ey)
        zone = bool(d_body.min() < _STOP_DIST)
    if zone and not S["holding"]:
        S["holding"], S["hold_until"] = True, t + _HOLD_S
    if S["holding"] and t >= S["hold_until"] and not zone and clr_n >= _CLR_GO:
        S["holding"] = False

    # 4) decide
    v, w, stop = 0.0, 0.0, False
    if S["holding"]:
        stop = True                                    # v = 0, w = 0
    elif st_n >= 0.0 and clr_n >= _CLR_GO:
        v, w = v_n, w_n
    else:
        traj_s = _rollout_nominal(pose, S["idx"], S["v"], 0.5)
        if (st_n >= 0.0 and _static_clearance(traj_s[None, :_KS + 1])[0] >= 0.0
                and _mover_clearance(traj_s[None], P, U)[0] >= _CLR_GO):
            v, w = 0.5 * v_n, w_n
        elif st_n < 0.0:
            v, w, _ = _choose(pose, P, U, target)      # off the verified path: safe arc toward the waypoint
        else:
            S["holding"], S["hold_until"] = True, t + _HOLD_S
            stop = True
    if stop:
        v, w = 0.0, 0.0
        ev, ew, clr_stop = _choose(pose, P, U, target)
        if clr_stop < _CLR_STOP:                       # exception: standing still would be hit -> step aside
            v, w = ev, ew

    S["v"], S["w"] = v, w
    return v, w, False




_V_MAX = 1.5            # cruise speed of the verified static run


_W_MAX = 2.0


_STOP_DIST = 2.0        # diagnosis: mover within 2 m ahead -> stop


_STOP_HALF_W = 0.70     # half width of the stop corridor ahead of the robot


_HOLD_S = 1.0           # diagnosis: re-sense for 1 s after stopping before going on


_DTP = 0.1              # prediction step [s]


_KP = 30                # 3.0 s prediction horizon for movers


_KS = 12                # 1.2 s horizon for the static-map check


_ACC = 2.0              # assumed accel limit used in prediction [m/s^2]


_CLR_GO = 0.35          # min predicted body clearance to keep driving


_CLR_STOP = 0.20        # if standing still is predicted below this, evade


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


def _choose(pose, P, U, target):
    """Pick the (v, w) candidate with the best mover clearance, static-map safety and progress.
    Also returns the predicted clearance of standing still."""
    trajc = _rollout_cands(pose, _CS["v"])
    clr = _mover_clearance(trajc, P, U)
    stc = _static_clearance(trajc[:, :_KS + 1])
    V = np.array([c[0] for c in _CAND])
    W = np.array([c[1] for c in _CAND])
    end = trajc[:, -1, :2]
    prog = np.hypot(end[:, 0] - target[0], end[:, 1] - target[1])
    score = (10.0 * np.minimum(clr, 1.0) - 0.5 * prog
             - 0.3 * (np.abs(V - _CS["v"]) + 0.3 * np.abs(W - _CS["w"]))
             - 100.0 * (stc < 0.0) + 5.0 * np.minimum(stc, 0.0))
    i = int(np.argmax(score))
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


# ========================= [END EDIT REGION] ===============================
