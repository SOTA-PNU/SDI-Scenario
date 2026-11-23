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
    """Waypoint law + 2D-lidar mover handling (attempt 4):
    - slow mode and the speed scale use clr_all (movers all around, front + back lidar) instead of the
      forward +-45 deg clr_n; _front_sector is no longer called (diagnosis 1, 2, 5)
    - the scaled path must keep >= _CLR_STOP from all movers, otherwise stop (diagnosis 3)
    - while stopped or right after resuming, an approaching mover (closing speed < _CLOSE_V) triggers an
      immediate evasive _choose() (diagnosis 4)
    - slow mode ends when clr_all >= _HYST * _CLR_GO with _HYST = 1.5 (diagnosis 6)."""
    S = _CS
    if not S:
        S.update(idx=0, tracks=[], mv={"front": np.zeros((0, 2)), "back": np.zeros((0, 2))},
                 mt={"front": -99.0, "back": -99.0}, last={}, v=0.0, w=0.0,
                 slow=False, stopped=False, n_stop=0, stop_t=0.0, stop_sum=0.0,
                 started=False, reported=False)
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
        if not S["reported"]:
            S["reported"] = True
            print("[C2 debug] done t=%.2f stop_episodes=%d stopped_total=%.2fs"
                  % (t, S["n_stop"], S["stop_sum"]), flush=True)
        return 0.0, 0.0, True

    # 3) nominal command (verified law) and its predicted safety against ALL movers (no front sector)
    v_n, w_n, S["idx"] = _nominal_cmd(x, y, yaw, S["idx"], 1.0)
    target = _WAYPOINTS[min(S["idx"], n - 1)]
    P, U = _movers_now(t)                    # all-around movers (front + back lidar)
    traj_n = _rollout_nominal(pose, S["idx"], S["v"], 1.0)
    st_n = _static_clearance(traj_n[None, :_KS + 1])[0]
    clr_all = _mover_clearance(traj_n[None], P, U)[0]

    # diagnosis 1, 6: slow mode on clr_all with hysteresis (enter < _CLR_GO, leave >= _HYST * _CLR_GO)
    if S["slow"]:
        if st_n >= 0.0 and clr_all >= _CLR_GO * _HYST:
            S["slow"] = False
    elif clr_all < _CLR_GO:
        S["slow"] = True

    # 4) decide
    stop = False
    if st_n < 0.0:                                     # off the verified path: safe arc toward the waypoint
        v, w, _ = _choose(pose, P, U, target)
    elif not S["slow"]:
        if clr_all >= _CLR_STOP:
            v, w = v_n, w_n
        else:
            v, w, _ = _choose(pose, P, U, target)
    else:
        # diagnosis 2, 3: speed scaled by clr_all / _CLR_GO on the nominal path, all-mover check
        v, w, stop = _slow_cmd(pose, S["idx"], S["v"], clr_all, P, U)
        if stop or S["stopped"]:
            # diagnosis 4: stopped / resuming while a mover closes in -> evade now
            closing = _closing_speed(P, U, pose, S["v"])
            hit_still = False
            if stop:
                still = np.tile(np.array([x, y, yaw], dtype=np.float64), (_KP + 1, 1))[None]
                hit_still = _mover_clearance(still, P, U)[0] < _CLR_STOP
            if hit_still or closing < _CLOSE_V:
                v, w, _ = _choose(pose, P, U, target)

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


def _choose(pose, P, U, target, anchor=None):
    """Pick the (v, w) candidate with the best mover clearance, static-map safety and progress.
    With anchor=(x, y): candidates that end (at 1 s) farther than _EVADE_R from the anchor are penalised.
    Also returns the predicted clearance of standing still (all-around mover points)."""
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
    if anchor is not None:
        k = min(10, trajc.shape[1] - 1)
        da = np.hypot(trajc[:, k, 0] - anchor[0], trajc[:, k, 1] - anchor[1])
        score = score - 20.0 * np.maximum(da - _EVADE_R, 0.0)
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




_SECTOR_HALF = math.radians(45.0)   # diagnosis 1: stop decision uses mover points within +-45 deg ahead


_EVADE_R = 0.5          # diagnosis 3: side-step while stopped stays within 0.5 m of the stop point


def _front_sector(P, U, pose):
    """Mover points (and velocities) inside the forward +-45 deg sector of the robot (robot-frame apex at origin)."""
    if P.shape[0] == 0:
        return P, U
    L = _to_robot(P, pose)
    m = (L[:, 0] > 0.0) & (np.abs(L[:, 1]) <= L[:, 0] * math.tan(_SECTOR_HALF))
    return P[m], U[m]




_HYST = 1.5             # diagnosis 6: leave slow mode only when clr_all >= 1.5 * _CLR_GO


def _slow_cmd(pose, idx, v0, clr, P, U):
    """Diagnosis 2, 3: follow the nominal path with the speed scaled by s = clr_all / _CLR_GO (0..1).
    The scaled path must be static-safe and keep >= _CLR_STOP from ALL mover points (front and back lidar);
    otherwise the robot stops: returns (0.0, 0.0, True)."""
    x, y, yaw = pose
    s = _clamp(clr / _CLR_GO, 0.0, 1.0)
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




_CLOSE_R = 1.0          # diagnosis 4: movers within 1.0 m of the body rectangle are checked for closing speed


_CLOSE_V = -0.10        # diagnosis 4: closing speed below this [m/s] (approaching) -> evade immediately


def _closing_speed(P, U, pose, v):
    """Diagnosis 4: most negative radial (closing) speed [m/s] of mover points within _CLOSE_R of the body,
    relative to the robot moving at v along its heading. 0.0 when no mover point is near."""
    if P.shape[0] == 0:
        return 0.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    near = np.hypot(ex, ey) < _CLOSE_R
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


# ========================= [END EDIT REGION] ===============================
