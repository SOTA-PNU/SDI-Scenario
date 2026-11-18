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
    (-6.0, -9.5),    # detour south of box 00 (diagnosis direction 3)
    (-4.0, 2.0),     # go north between forklift_01 / box 00 and boxes 01, 02
    (-2.8, 7.5),     # pass west of box 03, east of box 09
    (0.5, 11.0),     # pass east of box 10, west of box 07
    (1.5, 15.0),     # pass west of box 08, north of box 10
    (GOAL[0], GOAL[1])  # final goal position (north of box 06)
]

def controller(t, pose, env):
    """Way-point following (_nominal_cmd) + a mover-aware safety filter (_safety_filter) + dynamic speed cap
    (_speed_caps). Last of all, a raw-sensor emergency brake (_emergency_brake, diagnosis): independent of the
    tracker, any moving-object point within 0.8 m and +-90 deg ahead forces v = 0.0 right before returning,
    overriding the safety filter and the speed caps (w is kept)."""
    x, y, yaw = pose
    S = _MV
    if S["prev"] is not None:
        dt = t - S["prev"][0]
        if dt > 1e-4:
            vi = ((x - S["prev"][1]) * math.cos(yaw) + (y - S["prev"][2]) * math.sin(yaw)) / dt
            S["v_est"] = _clamp(0.6 * S["v_est"] + 0.4 * vi, -1.5, 3.5)
            S["vmax_seen"] = max(S["vmax_seen"], S["v_est"])
    S["prev"] = (t, x, y)

    dyn, st = _sense(pose)
    obs = _dyn_summary(dyn)
    try:
        _track_update(t, pose, dyn)
    except Exception:
        S["tracks"] = []
    try:
        v_hi, v_lo = _speed_caps(t, pose)
    except Exception:
        v_hi, v_lo = _V_MAX, -_V_MAX
    S["cap"] = (v_hi, v_lo)
    v_nom, w_nom, done, target, align = _nominal_cmd(pose, st, obs)
    if done:
        return 0.0, 0.0, True
    v_nom = _clamp(v_nom, v_lo, v_hi)
    try:
        v, w = _safety_filter(t, pose, v_nom, w_nom, target, align)
    except Exception:
        v, w = v_nom, w_nom
    v = _clamp(v, v_lo, v_hi)

    # raw-sensor emergency brake (highest priority, after _safety_filter and _speed_caps)
    try:
        if _emergency_brake(pose):
            v = 0.0
    except Exception:
        v = 0.0

    S["last"] = (v, w)
    return v, w, False




_STATIC_RECTS = ([(b["x"] - b["width"] / 2.0, b["x"] + b["width"] / 2.0,
                   b["y"] - b["depth"] / 2.0, b["y"] + b["depth"] / 2.0) for b in MISSION["boxes"]]
                 + [(o[1], o[2], o[3], o[4]) for o in FIXED_OBSTACLES])


def _static_mask(wp, m=0.25):
    """World points (N, >=2) -> bool mask of points explained by the static map (walls, fixed obstacles, boxes)."""
    x, y = wp[:, 0], wp[:, 1]
    s = (x <= WALLS[0] + m) | (x >= WALLS[1] - m) | (y <= WALLS[2] + m) | (y >= WALLS[3] - m)
    for (x0, x1, y0, y1) in _STATIC_RECTS:
        s |= (x >= x0 - m) & (x <= x1 + m) & (y >= y0 - m) & (y <= y1 + m)
    return s


def _front_dynamic_min_dist(pose, half_angle_deg=90.0):
    """(min distance, bearing [rad], stop-cone min distance) of non-static front-lidar points in the robot frame.
    The first two are for the closest point with |bearing| <= half_angle_deg ((inf, 0.0) if none); the third is the
    closest point with |bearing| <= _STOP_CONE_DEG; inf if none."""
    inf = float("inf")
    pts = lidar2d_points("front")
    if pts is None or len(pts) == 0:
        return inf, 0.0, inf
    wp = to_world(pts, pose)
    dyn = pts[~_static_mask(wp)]
    if len(dyn) == 0:
        return inf, 0.0, inf
    r = np.hypot(dyn[:, 0], dyn[:, 1])
    bearing = np.arctan2(dyn[:, 1], dyn[:, 0])
    stop = np.abs(bearing) <= math.radians(_STOP_CONE_DEG)
    d_cone = float(np.min(r[stop])) if np.any(stop) else inf
    half = np.abs(bearing) <= math.radians(half_angle_deg)
    if not np.any(half):
        return inf, 0.0, d_cone
    rh = r[half]
    bh = bearing[half]
    i = int(np.argmin(rh))
    return float(rh[i]), float(bh[i]), d_cone


def _rear_clear():
    """True if the back lidar sees nothing in a strip just behind the chassis (static or moving)."""
    pts = lidar2d_points("back")
    if pts is None or len(pts) == 0:
        return False
    behind = (pts[:, 0] < -0.70) & (pts[:, 0] > -1.40) & (np.abs(pts[:, 1]) < 0.45)
    return not bool(np.any(behind))




_AVOID_R = 2.0          # avoidance / slow-down radius [m]


_STOP_R = 0.8           # full stop radius inside the stop cone [m]


_DPSI_MAX = 1.5         # side-object avoidance heading offset [rad] (1.2 -> 1.5, diagnosis direction 3)


_W_AVOID_MAX = 1.0      # |w| limit while avoiding [rad/s]


_B_TIE = 0.15           # |bearing| below this counts as dead ahead [rad]


_BACK_R = 0.45          # stop-zone object closer than this -> back up [m]


_BACK_AFTER_STEPS = 90  # blocked longer than this many control steps (1.5 s) -> back up


_V_BACK = 0.3           # back-up speed [m/s]


def _static_close_on_side(pose, turn_side):
    """True if a static-map point (wall, fixed obstacle, box) seen by the front or back lidar is near the region the
    robot would sweep when turning to turn_side (+1 = left, -1 = right): ahead/beside on the turn side
    (x -0.3..1.6 m, 0..1.2 m laterally) or behind on the opposite side where the rear of the chassis swings
    (x -1.05..-0.3 m, 0..0.85 m laterally). The world transform is applied to a copy (pts stays robot frame)."""
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return False
    pts = np.concatenate(parts)
    st = pts[_static_mask(to_world(pts.copy(), pose))]
    if len(st) == 0:
        return False
    lat = st[:, 1] * turn_side
    front_near = (st[:, 0] > -0.3) & (st[:, 0] < 1.6) & (lat > 0.0) & (lat < 1.2)
    rear_swing = (st[:, 0] > -1.05) & (st[:, 0] <= -0.3) & (lat < 0.0) & (lat > -0.85)
    return bool(np.any(front_near | rear_swing))




_STOP_CONE_DEG = 30.0   # stop zone half angle [deg]


_PATH_HALF_W = 0.45     # half width of the corridor ahead that also counts as stop zone [m]


_SIDE_SCALE_MIN = 0.25  # minimum speed scale for objects beside the path (prevents side-object stalls)


_V_AVOID_MAX = 0.7      # speed cap while a moving object is within _AVOID_R in +-90 deg [m/s]


_E_TURN = 0.75      # rotate in place when |e_cmd| exceeds this [rad] (0.6 -> 0.75, diagnosis direction 6)




def _get_dynamic_min(pose):
    """Moving-object summary from BOTH 2D lidars (front + back, 360 deg), robot frame. Returns a dict:
      d_all, b_all     closest moving-object point in any direction (inf, 0.0 if none)
      d_fs             closest moving-object point with |bearing| <= _SCALE_HALF_DEG (front/side, inf if none)
      d_front, b_front closest moving-object point with |bearing| <= 90 deg (inf, 0.0 if none)
      d_cone           closest moving-object point with |bearing| <= _STOP_CONE_DEG (inf if none)
      d_side           closest moving-object point with |bearing| > _STOP_CONE_DEG (inf if none)
      x_corr           smallest x of moving-object points in the path corridor (x > _CORR_X_MIN,
                       |y| < _PATH_HALF_W) (inf if none)
      ahead_blocked    any lidar point (static or moving) in the corridor _BODY_FRONT < x < _CLEAR_X,
                       |y| < _PATH_HALF_W
    The world transform is applied to a copy so pts stays in the robot frame."""
    inf = float("inf")
    out = {"d_all": inf, "b_all": 0.0, "d_fs": inf, "d_front": inf, "b_front": 0.0, "d_cone": inf,
           "d_side": inf, "x_corr": inf, "ahead_blocked": False}
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return out
    pts = np.concatenate(parts)
    ahead = (pts[:, 0] > _BODY_FRONT) & (pts[:, 0] < _CLEAR_X) & (np.abs(pts[:, 1]) < _PATH_HALF_W)
    out["ahead_blocked"] = bool(np.any(ahead))
    dyn = pts[~_static_mask(to_world(pts.copy(), pose))]
    if len(dyn) == 0:
        return out
    r = np.hypot(dyn[:, 0], dyn[:, 1])
    bearing = np.arctan2(dyn[:, 1], dyn[:, 0])
    i = int(np.argmin(r))
    out["d_all"], out["b_all"] = float(r[i]), float(bearing[i])
    fs = np.abs(bearing) <= math.radians(_SCALE_HALF_DEG)
    if np.any(fs):
        out["d_fs"] = float(np.min(r[fs]))
    half = np.abs(bearing) <= math.pi / 2.0
    if np.any(half):
        rh = r[half]
        bh = bearing[half]
        j = int(np.argmin(rh))
        out["d_front"], out["b_front"] = float(rh[j]), float(bh[j])
    cone = np.abs(bearing) <= math.radians(_STOP_CONE_DEG)
    if np.any(cone):
        out["d_cone"] = float(np.min(r[cone]))
    if np.any(~cone):
        out["d_side"] = float(np.min(r[~cone]))
    corr = (dyn[:, 0] > _CORR_X_MIN) & (np.abs(dyn[:, 1]) < _PATH_HALF_W)
    if np.any(corr):
        out["x_corr"] = float(np.min(dyn[corr, 0]))
    return out




_STOP_R_WIDE = 0.7      # wide stop radius: any moving object within +-90 deg closer than this -> v = 0 [m]




_EVADE_R = 1.5          # side approach: moving object outside the stop cone closer than this [m]


_EVADE_HOLD_STEPS = 30  # keep the evasion mode this many control steps (0.5 s) after the last detection


_V_EVADE_MIN = 0.6      # evasion forward floor / minimum escape speed [m/s] (0.3 -> 0.6, diagnosis direction 1)


_V_EVADE_STATIC = 0.4   # reduced floor when a static obstacle is close on the evasion side [m/s]


_DPSI_EVADE = 0.6       # evasion heading offset [rad]


_W_EVADE_MAX = 1.5      # |w| limit during active evasion [rad/s]


_BODY_FRONT = 0.25      # front face of the chassis in the robot frame [m]


_CLEAR_X = 1.15         # the corridor ahead must be free of any point up to this x for the evasion floor [m]


_CORR_X_MIN = 0.15      # moving-object points beyond this x within |y| < _PATH_HALF_W count as in the path [m]


_CORR_STOP_X = 0.70     # moving-object point in the path corridor closer than this x disables the evasion floor (no stop) [m]


_GOAL_REENTER = 0.6     # during goal alignment, drive back to GOAL if farther than this [m] (diagnosis direction 5)




def _static_close_turn(pose, w):
    """Static-map check for the side the robot is turning to: sign(w) > 0 -> left, < 0 -> right, w = 0 -> both."""
    if w > 1e-6:
        return _static_close_on_side(pose, 1.0)
    if w < -1e-6:
        return _static_close_on_side(pose, -1.0)
    return _static_close_on_side(pose, 1.0) or _static_close_on_side(pose, -1.0)




_SCALE_HALF_DEG = 120.0  # front/side sector (|bearing| <= this) used for the distance-based speed scale [deg]


_SCALE_STOP_R = 0.8     # stop-cone object closer than this -> speed scale 0 [m]


_SCALE_FULL_R = 2.2     # speed scale rises linearly to 1 at this stop-cone distance [m]




def _static_ahead(pose):
    """True if a static-map point (wall, fixed obstacle, box) seen by the front or back lidar lies in the corridor
    right in front of the chassis (_BODY_FRONT < x < 1.0 m, |y| < _PATH_HALF_W), robot frame.
    The world transform is applied to a copy (pts stays robot frame)."""
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return False
    pts = np.concatenate(parts)
    st = pts[_static_mask(to_world(pts.copy(), pose))]
    if len(st) == 0:
        return False
    ahead = (st[:, 0] > _BODY_FRONT) & (st[:, 0] < 1.0) & (np.abs(st[:, 1]) < _PATH_HALF_W)
    return bool(np.any(ahead))




def _sense(pose):
    """Front + back 2D lidar points (robot frame) split by the static map: returns (moving, static), each (N, 2)."""
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(np.asarray(p, dtype=np.float64)[:, :2])
    if not parts:
        empty = np.zeros((0, 2))
        return empty, empty
    pts = np.concatenate(parts)
    sm = _static_mask(to_world(pts, pose))
    return pts[~sm], pts[sm]


def _dyn_summary(dyn):
    """Moving-object summary in the robot frame:
      d_cone, b_cone  closest point with |bearing| <= _STOP_CONE_DEG (inf, 0.0 if none)
      g_side, b_side  smallest gap to the chassis rectangle _BODY among points outside the cone (inf, 0.0 if none)
      x_side, y_side  robot-frame position of that closest side point (0.0 if none)
      x_corr          smallest x of points in the corridor ahead (x > _CORR_X_MIN, |y| < _CORR_HALF_W) (inf if none)
      x_body          smallest x of points in the body-width corridor (x > _CORR_X_MIN, |y| < _CRIT_HALF_W)"""
    inf = float("inf")
    out = {"d_cone": inf, "b_cone": 0.0, "g_side": inf, "b_side": 0.0, "x_side": 0.0, "y_side": 0.0,
           "x_corr": inf, "x_body": inf}
    if len(dyn) == 0:
        return out
    r = np.hypot(dyn[:, 0], dyn[:, 1])
    b = np.arctan2(dyn[:, 1], dyn[:, 0])
    cone = np.abs(b) <= math.radians(_STOP_CONE_DEG)
    if np.any(cone):
        i = int(np.argmin(np.where(cone, r, inf)))
        out["d_cone"], out["b_cone"] = float(r[i]), float(b[i])
    side = ~cone
    if np.any(side):
        gx = np.maximum(np.maximum(_BODY[0] - dyn[:, 0], dyn[:, 0] - _BODY[1]), 0.0)
        gy = np.maximum(np.maximum(_BODY[2] - dyn[:, 1], dyn[:, 1] - _BODY[3]), 0.0)
        g = np.where(side, np.hypot(gx, gy), inf)
        j = int(np.argmin(g))
        out["g_side"], out["b_side"] = float(g[j]), float(b[j])
        out["x_side"], out["y_side"] = float(dyn[j, 0]), float(dyn[j, 1])
    corr = (dyn[:, 0] > _CORR_X_MIN) & (np.abs(dyn[:, 1]) < _CORR_HALF_W)
    if np.any(corr):
        out["x_corr"] = float(np.min(dyn[corr, 0]))
    body = (dyn[:, 0] > _CORR_X_MIN) & (np.abs(dyn[:, 1]) < _CRIT_HALF_W)
    if np.any(body):
        out["x_body"] = float(np.min(dyn[body, 0]))
    return out


def _static_side_pts(st, turn_side):
    """Static points (robot frame) near the region swept when turning to turn_side (+1 left, -1 right), range
    widened (diagnosis direction 5): ahead/beside on the turn side x -0.5..2.2 m, 0..1.3 m laterally, or where the
    rear swings x -1.4..-0.3 m, 0..1.0 m on the other side."""
    if len(st) == 0:
        return False
    lat = st[:, 1] * turn_side
    front_near = (st[:, 0] > -0.5) & (st[:, 0] < 2.2) & (lat > 0.0) & (lat < 1.3)
    rear_swing = (st[:, 0] > -1.4) & (st[:, 0] <= -0.3) & (lat < 0.0) & (lat > -1.0)
    return bool(np.any(front_near | rear_swing))


def _static_front_pts(st):
    """Same test as _static_ahead on already-sensed static points st: static point with _BODY_FRONT < x < 1.0 m,
    |y| < _PATH_HALF_W."""
    if len(st) == 0:
        return False
    ahead = (st[:, 0] > _BODY_FRONT) & (st[:, 0] < 1.0) & (np.abs(st[:, 1]) < _PATH_HALF_W)
    return bool(np.any(ahead))


_DPSI_CONE = 0.5        # stop-cone object avoidance heading offset [rad] (previous _DPSI_MAX value)


_EVADE_GAP = 1.5        # side/rear moving object closer than this to the chassis -> evasion mode [m] (raised from 1.2, diagnosis direction 2)


_CORR_HALF_W = 0.6      # half width of the corridor ahead in which a moving point disables the evasion floor [m]


_GOAL_YAW_DONE = 0.20   # finish when the goal yaw error is below this [rad] (tolerance 0.26, diagnosis direction 3)


_K_ALIGN = 1.6          # goal-alignment yaw gain (lowered from 2.0, diagnosis direction 4)


_W_ALIGN = 1.2          # goal-alignment |w| limit [rad/s]


_V_MAX = 3.0            # cruise speed limit [m/s]


_K_D = 3.5              # distance gain of the nominal speed


_K_W = 10.0             # heading gain




_V_ALIGN_FLOOR = 0.1    # forward creep during goal alignment while a side object closes in [m/s] (diagnosis direction 2)




_CRIT_GAP = 1.0         # side/rear moving object closer than this to the chassis -> critical danger state [m] (diagnosis direction 1)


_CRIT_HALF_W = 0.38     # body-width corridor half width for the critical forward floor [m] (relaxed, direction 6)




def _nominal_cmd(pose, st, obs):
    """Former controller logic (way-points, side evasion, stop cone, goal alignment) returning
    (v, w, done, target_xy, align). Changes: crit_ok uses the relaxed body corridor (_CRIT_STOP_X, _CRIT_HALF_W,
    direction 6); in the critical state a rotation toward the object is replaced by 0 (when driving forward) or by
    the opposite direction (direction 2)."""
    N = _NOM
    x, y, yaw = pose
    d_cone, b_cone = obs["d_cone"], obs["b_cone"]
    g_side, b_side = obs["g_side"], obs["b_side"]
    ox, oy = obs["x_side"], obs["y_side"]
    front_static = _static_front_pts(st)

    near_side = g_side < _EVADE_GAP
    if near_side:
        N["evade_steps"] = _EVADE_HOLD_STEPS
    elif N["evade_steps"] > 0:
        N["evade_steps"] -= 1
    evading = N["evade_steps"] > 0
    floor_ok = (evading and d_cone >= _STOP_R and obs["x_corr"] >= _CORR_STOP_X and not front_static)

    critical = g_side < _CRIT_GAP
    crit_ok = (critical and d_cone >= _STOP_R and obs["x_body"] >= _CRIT_STOP_X and not front_static)

    # 1) goal alignment
    if N["idx"] >= len(_WAYPOINTS):
        gd = math.hypot(GOAL[0] - x, GOAL[1] - y)
        eg = _wrap_angle(GOAL[2] - yaw)
        if gd > _GOAL_REENTER:
            N["idx"] = len(_WAYPOINTS) - 1
        else:
            if abs(eg) < _GOAL_YAW_DONE:
                return 0.0, 0.0, True, (GOAL[0], GOAL[1]), True
            w = _clamp(_K_ALIGN * eg, -_W_ALIGN, _W_ALIGN)
            v = _V_ALIGN_FLOOR if floor_ok else 0.0
            if critical and bool(_turn_toward(w, ox, oy)):
                w = 0.0
            return v, w, False, (GOAL[0], GOAL[1]), True

    # 2) way-point following
    tx, ty = _WAYPOINTS[N["idx"]]
    dist = math.hypot(tx - x, ty - y)
    if N["idx"] < len(_WAYPOINTS) - 1 and dist < 0.35:
        N["idx"] += 1
        tx, ty = _WAYPOINTS[N["idx"]]
        dist = math.hypot(tx - x, ty - y)
    e = _wrap_angle(math.atan2(ty - y, tx - x) - yaw)

    # 3) steering offset
    dpsi = 0.0
    steering = False
    if near_side and abs(b_side) <= math.pi / 2.0:
        side = 1.0 if b_side > 0.0 else -1.0
        k = _clamp((_EVADE_GAP - g_side) / (_EVADE_GAP - 0.2), 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k
        if _static_side_pts(st, -side):
            dpsi = 0.0
        steering = True
    elif d_cone < _AVOID_R:
        if abs(b_cone) >= _B_TIE:
            N["side"] = 1.0 if b_cone > 0.0 else -1.0
        elif N["side"] == 0.0:
            N["side"] = 1.0 if e <= 0.0 else -1.0
        side = N["side"]
        k = _clamp((_AVOID_R - d_cone) / (_AVOID_R - _STOP_R), 0.0, 1.0)
        dpsi = -side * _DPSI_CONE * k
        if _static_side_pts(st, -side):
            dpsi = 0.0
        steering = True
    else:
        N["side"] = 0.0

    if evading:
        w_lim = _W_EVADE_MAX
    elif steering:
        w_lim = _W_AVOID_MAX
    else:
        w_lim = 2.0
    w = _clamp(_K_W * (e + dpsi), -w_lim, w_lim)
    e_cmd = _wrap_angle(e + dpsi)

    # 4) speed
    if math.isfinite(d_cone):
        speed_scale = _clamp((d_cone - _SCALE_STOP_R) / (_SCALE_FULL_R - _SCALE_STOP_R), 0.0, 1.0)
    else:
        speed_scale = 1.0
    v_nom = min(_V_MAX, _K_D * dist + 0.5)
    if abs(e_cmd) > _E_TURN:
        v = _V_EVADE_MIN if crit_ok else 0.0
    else:
        v = v_nom * speed_scale
        if steering:
            v *= _clamp(math.cos(e_cmd), 0.3, 1.0)
    if steering and front_static:
        v = 0.0

    # 5) evasion floor
    if floor_ok or crit_ok:
        if w > 0.05:
            static_turn = _static_side_pts(st, 1.0)
        elif w < -0.05:
            static_turn = _static_side_pts(st, -1.0)
        else:
            static_turn = False
        v = max(v, _V_EVADE_STATIC if static_turn else _V_EVADE_MIN)

    # direction 2: no rotation toward the object in the critical state
    if critical and bool(_turn_toward(w, ox, oy)):
        w = 0.0 if v > 0.05 else -0.5 * w

    # 6) frontal stop cone
    if d_cone < _STOP_R:
        N["stop_steps"] += 1
        v = 0.0
        if (d_cone < _BACK_R or N["stop_steps"] > _BACK_AFTER_STEPS) and _rear_clear():
            v = -_V_BACK
            w = 0.0
    else:
        N["stop_steps"] = 0

    if N["idx"] == len(_WAYPOINTS) - 1 and dist < 0.25:
        N["idx"] += 1
    return v, w, False, (tx, ty), False


_CRIT_STOP_X = 0.5      # moving point in the body corridor closer than this disables the critical floor [m] (direction 6)


_NOM = {"idx": 0, "side": 0.0, "stop_steps": 0, "evade_steps": 0}


_MV = {"tracks": [], "prev": None, "v_est": 0.0, "vmax_seen": 0.0, "last": None}


_H = 2.5            # mover prediction horizon [s]


_DT = 0.1           # rollout step [s]


_H_STATIC = 1.2     # static-map check horizon [s]


_H_SHORT = 1.0      # repulsion horizon [s]


_H_PROG = 1.5       # progress evaluation time [s]


_ACC = 1.5          # assumed linear acceleration [m/s^2]


_G_SAFE = 0.4       # required gap chassis - moving object at tau = 0 [m] (0.22 -> 0.4, diagnosis direction 1)


_G_GROW = 0.15      # required gap growth with prediction time [m/s] (0.06 -> 0.15, diagnosis direction 1)


_G_STATIC = 0.10    # required gap to tracked objects that never moved [m]


_REPULSE_GAP = 0.8  # moving object closer than this -> repulsion has priority [m] (0.5 -> 0.8, diagnosis direction 2)


_STATIC_INFL = 0.05  # static map inflation for the outline check [m]


_STATIC_NEAR = 5.0  # only static rectangles within this distance are checked [m]


_CELL = 0.3         # clustering grid cell [m]


_MIN_PTS = 3        # minimum points per cluster


_ASSOC_R = 0.8      # track association radius [m]


_VEL_WIN = 0.45     # velocity estimation window [s]


_VEL_MIN_SPAN = 0.25  # minimum history span for a velocity estimate [s]


_TRACK_STALE = 0.5  # drop a track unseen for this long [s]


_YOUNG_S = 0.4      # tracks younger than this are treated as moving [s]


_MOVE_SPEED = 0.3   # a track faster than this is flagged as a moving object [m/s]


_U_MAX = 2.5        # velocity estimate clamp [m/s]


_TRACK_RANGE = 8.0  # cluster lidar points within this range [m]


_MOVER_RANGE = 6.0  # use tracks within this range in the safety filter [m]


_TRACK_PTS = 24     # points kept per track


_OUTLINE = np.array([(bx, by) for bx in (-0.70, -0.46, -0.225, 0.01, 0.25) for by in (-0.34, 0.34)]
                    + [(-0.70, 0.0), (0.25, 0.0)], dtype=np.float64)


def _turn_toward(w, ox, oy):
    """True where yaw rate w turns the chassis toward the object point (ox, oy) (robot frame), diagnosis direction 2:
    object ahead of the axle (ox >= 0): w with the same sign as oy (e.g. object right, w < 0); in any case also a w
    whose rotation moves the nearest chassis point toward the object (rear swing)."""
    w = np.asarray(w, dtype=np.float64)
    bx = min(max(ox, _BODY[0]), _BODY[1])
    by = min(max(oy, _BODY[2]), _BODY[3])
    k = -by * (ox - bx) + bx * (oy - by)
    swing = w * k > 1e-9
    front = (w * oy > 0.0) if ox >= 0.0 else np.zeros(w.shape, dtype=bool)
    return swing | front


def _cluster_dyn(dyn, pose):
    """Moving lidar points (robot frame) -> clusters [{"w": world points (M, 2), "c": centroid}] by grid adjacency."""
    if len(dyn) == 0:
        return []
    dyn = dyn[np.hypot(dyn[:, 0], dyn[:, 1]) < _TRACK_RANGE]
    if len(dyn) < _MIN_PTS:
        return []
    wp = to_world(dyn, pose)[:, :2]
    cells = np.floor(wp / _CELL).astype(np.int64)
    uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    index = {(int(u[0]), int(u[1])): k for k, u in enumerate(uniq)}
    label = [-1] * len(uniq)
    ncomp = 0
    for k0 in range(len(uniq)):
        if label[k0] >= 0:
            continue
        label[k0] = ncomp
        stack = [k0]
        while stack:
            k = stack.pop()
            cx, cy = int(uniq[k][0]), int(uniq[k][1])
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    nb = index.get((cx + ddx, cy + ddy))
                    if nb is not None and label[nb] < 0:
                        label[nb] = ncomp
                        stack.append(nb)
        ncomp += 1
    lab = np.asarray(label, dtype=np.int64)[inv]
    out = []
    for ci in range(ncomp):
        m = lab == ci
        if int(np.count_nonzero(m)) < _MIN_PTS:
            continue
        pw = wp[m]
        out.append({"w": pw, "c": pw.mean(axis=0)})
    return out


def _track_set(tr, cl, t):
    """Update a track with a cluster; velocity from the centroid history over _VEL_WIN."""
    c = cl["c"]
    tr["c"] = c
    tr["t"] = t
    pw = cl["w"]
    if len(pw) > _TRACK_PTS:
        pw = pw[np.linspace(0, len(pw) - 1, _TRACK_PTS).astype(np.int64)]
    tr["pw"] = pw
    hist = [h for h in tr["hist"] if t - h[0] <= _VEL_WIN] + [(t, c)]
    tr["hist"] = hist
    span = t - hist[0][0]
    if span >= _VEL_MIN_SPAN:
        u = (c - hist[0][1]) / span
        sp = float(math.hypot(u[0], u[1]))
        if sp > _U_MAX:
            u = u * (_U_MAX / sp)
        tr["u"] = u
        if sp > _MOVE_SPEED and t - tr["born"] >= _YOUNG_S:
            tr["moving"] = True


def _track_update(t, pose, dyn):
    """Associate this step's moving-point clusters with the tracks (greedy nearest, predicted position)."""
    tracks = _MV["tracks"]
    clusters = _cluster_dyn(dyn, pose)
    pairs = []
    for ci, cl in enumerate(clusters):
        for ti, tr in enumerate(tracks):
            pred = tr["c"] + tr["u"] * (t - tr["t"])
            d = math.hypot(cl["c"][0] - pred[0], cl["c"][1] - pred[1])
            if d < _ASSOC_R:
                pairs.append((d, ci, ti))
    pairs.sort(key=lambda p: p[0])
    used_c, used_t = set(), set()
    for d, ci, ti in pairs:
        if ci in used_c or ti in used_t:
            continue
        used_c.add(ci)
        used_t.add(ti)
        _track_set(tracks[ti], clusters[ci], t)
    for ci, cl in enumerate(clusters):
        if ci not in used_c:
            tr = {"born": t, "hist": [], "u": np.zeros(2), "moving": False}
            _track_set(tr, cl, t)
            tracks.append(tr)
    _MV["tracks"] = [tr for tr in tracks if t - tr["t"] <= _TRACK_STALE]


def _mover_set(t, pose):
    """Tracks near the robot as (points (M, 2) robot frame now, velocity (2,) robot frame, moving flag)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for tr in _MV["tracks"]:
        P = tr["pw"] + tr["u"] * (t - tr["t"])
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rx, ry = c * dx + s * dy, -s * dx + c * dy
        if float(np.min(np.hypot(rx, ry))) > _MOVER_RANGE:
            continue
        u = tr["u"]
        ur = np.array([c * u[0] + s * u[1], -s * u[0] + c * u[1]])
        mv = bool(tr["moving"] or (t - tr["born"] < _YOUNG_S))
        out.append((np.stack([rx, ry], axis=1), ur, mv))
    return out


def _nearest_moving(movers):
    """(gap to the chassis, x, y) of the closest point of a moving track, robot frame; (inf, 0, 0) if none."""
    best = (float("inf"), 0.0, 0.0)
    for P, u, mv in movers:
        if not mv:
            continue
        gx = np.maximum(np.maximum(_BODY[0] - P[:, 0], P[:, 0] - _BODY[1]), 0.0)
        gy = np.maximum(np.maximum(_BODY[2] - P[:, 1], P[:, 1] - _BODY[3]), 0.0)
        g = np.hypot(gx, gy)
        j = int(np.argmin(g))
        if float(g[j]) < best[0]:
            best = (float(g[j]), float(P[j, 0]), float(P[j, 1]))
    return best


def _rollout(cand, cap, n):
    """Robot poses (px, py, th), each (C, n), in the current robot frame for constant commands cand (C, 2) [v, w],
    speed ramped from the measured speed at _ACC and limited to the observed top speed cap."""
    C = len(cand)
    vc = np.clip(cand[:, 0], -0.8, cap)
    wc = cand[:, 1]
    px, py, th = np.zeros((C, n)), np.zeros((C, n)), np.zeros((C, n))
    xx, yy, aa = np.zeros(C), np.zeros(C), np.zeros(C)
    vv = np.full(C, float(_MV["v_est"]))
    step = _ACC * _DT
    for k in range(n):
        vn = vv + np.clip(vc - vv, -step, step)
        vm = 0.5 * (vv + vn)
        am = aa + 0.5 * wc * _DT
        xx = xx + vm * np.cos(am) * _DT
        yy = yy + vm * np.sin(am) * _DT
        aa = aa + wc * _DT
        vv = vn
        px[:, k], py[:, k], th[:, k] = xx, yy, aa
    return px, py, th


def _mover_gap_req(px, py, th, taus, movers):
    """(C, n) array: predicted gap chassis - tracked object minus the required gap (inf if no tracks)."""
    C, n = px.shape
    out = np.full((C, n), np.inf)
    if not movers:
        return out
    c = np.cos(th)[:, :, None]
    s = np.sin(th)[:, :, None]
    for P, u, mv in movers:
        qx = P[None, :, 0] + taus[:, None] * u[0]
        qy = P[None, :, 1] + taus[:, None] * u[1]
        dx = qx[None, :, :] - px[:, :, None]
        dy = qy[None, :, :] - py[:, :, None]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        gx = np.maximum(np.maximum(_BODY[0] - lx, lx - _BODY[1]), 0.0)
        gy = np.maximum(np.maximum(_BODY[2] - ly, ly - _BODY[3]), 0.0)
        g = np.sqrt(gx * gx + gy * gy).min(axis=2)
        req = (_G_SAFE + _G_GROW * taus) if mv else np.full(n, _G_STATIC)
        out = np.minimum(out, g - req[None, :])
    return out


def _static_pen(px, py, th, pose, rects):
    """(C, n) deepest penetration of the chassis outline into the inflated static map (walls, boxes, fixed
    obstacles); negative = clear."""
    x, y, yaw = pose
    c0, s0 = math.cos(yaw), math.sin(yaw)
    wx = x + c0 * px - s0 * py
    wy = y + s0 * px + c0 * py
    a = yaw + th
    ca = np.cos(a)[..., None]
    sa = np.sin(a)[..., None]
    ox, oy = _OUTLINE[:, 0], _OUTLINE[:, 1]
    X = wx[..., None] + ca * ox - sa * oy
    Y = wy[..., None] + sa * ox + ca * oy
    m = _STATIC_INFL
    pen = np.maximum(np.maximum((WALLS[0] + m) - X, X - (WALLS[1] - m)),
                     np.maximum((WALLS[2] + m) - Y, Y - (WALLS[3] - m)))
    for (x0, x1, y0, y1) in rects:
        d = np.minimum(np.minimum(X - (x0 - m), (x1 + m) - X), np.minimum(Y - (y0 - m), (y1 + m) - Y))
        pen = np.maximum(pen, d)
    return pen.max(axis=-1)


def _static_hits(px, py, th, pose):
    """(C,) True if the candidate drives the chassis deeper into the static map than it is now."""
    x, y, _ = pose
    rects = [r for r in _STATIC_RECTS
             if max(r[0] - x, 0.0, x - r[1]) ** 2 + max(r[2] - y, 0.0, y - r[3]) ** 2 < _STATIC_NEAR ** 2]
    z = np.zeros((1, 1))
    now = float(_static_pen(z, z, z, pose, rects)[0, 0])
    pen = _static_pen(px, py, th, pose, rects).max(axis=1)
    return pen > max(now, 0.0) + 0.01


def _safety_filter(t, pose, v_nom, w_nom, target, align):
    """Keep the nominal command if it stays clear of the predicted movers and the static map; otherwise pick a
    candidate (v, w). Every candidate v is clipped to the dynamic speed cap _MV["cap"].
    Repulsion first when a moving object is within _REPULSE_GAP (0.8 m): escape speed required up to the cap; score
    = escape gap + _K_PROG_REP * progress (raised weight, diagnosis direction 4); when the nearest moving object is
    beside the chassis (|n_x| < _SIDE_NX) and approaching, candidates with v < _SIDE_V_MIN get a strong penalty
    _SIDE_PEN (diagnosis direction 3, no slow 'sitting duck' in the object's path). Otherwise the safe candidate
    with the best progress, else the one keeping the largest gap. In the critical state rotations toward the
    object are excluded."""
    x, y, yaw = pose
    v_hi, v_lo = _MV.get("cap", (_V_MAX, -_V_MAX))
    v_nom = _clamp(v_nom, v_lo, v_hi)
    movers = _mover_set(t, pose)
    g_near, ox, oy = _nearest_moving(movers)
    repulse = g_near < _REPULSE_GAP
    critical = g_near < _CRIT_GAP
    cap = max(1.0, 1.05 * _MV["vmax_seen"])
    n = int(round(_H / _DT))
    ns = int(round(_H_STATIC / _DT))
    taus = _DT * (np.arange(n) + 1.0)

    if not repulse:
        c1 = np.array([[v_nom, w_nom]], dtype=np.float64)
        px, py, th = _rollout(c1, cap, n)
        gr1 = _mover_gap_req(px, py, th, taus, movers)
        if float(gr1.min()) >= 0.0 and not bool(_static_hits(px[:, :ns], py[:, :ns], th[:, :ns], pose)[0]):
            return v_nom, w_nom

    vs = (v_nom, 1.5, 1.0, _V_EVADE_MIN, 0.3, 0.0, -0.3, -_V_EVADE_MIN)
    ws = (w_nom, -1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5)
    cand = np.array([(v_nom, w_nom)] + [(a, b) for a in vs for b in ws], dtype=np.float64)
    cand[:, 0] = np.clip(cand[:, 0], v_lo, v_hi)
    C = len(cand)
    px, py, th = _rollout(cand, cap, n)
    shit = _static_hits(px[:, :ns], py[:, :ns], th[:, :ns], pose)
    gr = _mover_gap_req(px, py, th, taus, movers)
    margin = gr.min(axis=1)
    grc = np.minimum(gr, 1.0)
    ks = int(round(_H_SHORT / _DT))
    esc = grc[:, :ks].mean(axis=1) + 0.5 * grc[:, :ks].min(axis=1)
    esc_full = grc.mean(axis=1) + 0.5 * grc.min(axis=1)
    viol = gr < 0.0
    t_viol = np.where(viol.any(axis=1), taus[np.argmax(viol, axis=1)], taus[-1] + _DT)
    toward = _turn_toward(cand[:, 1], ox, oy) if critical else np.zeros(C, dtype=bool)

    kp = int(round(_H_PROG / _DT)) - 1
    c0, s0 = math.cos(yaw), math.sin(yaw)
    xe = x + c0 * px[:, kp] - s0 * py[:, kp]
    ye = y + s0 * px[:, kp] + c0 * py[:, kp]
    he = yaw + th[:, kp]
    tx, ty = target
    prog = math.hypot(tx - x, ty - y) - np.hypot(tx - xe, ty - ye)
    if align:
        ha = GOAL[2] - he
    else:
        ha = np.arctan2(ty - ye, tx - xe) - he
    herr = np.abs(np.arctan2(np.sin(ha), np.cos(ha)))
    same = np.zeros(C)
    if _MV["last"] is not None:
        lv, lw = _MV["last"]
        same = ((np.abs(cand[:, 0] - lv) < 0.05) & (np.abs(cand[:, 1] - lw) < 0.05)).astype(np.float64)
    ok_static = ~shit

    if repulse:
        thr = np.where(cand[:, 0] >= 0.0, min(_V_EVADE_MIN, v_hi), min(_V_EVADE_MIN, -v_lo))
        pool = ok_static & ~toward & (np.abs(cand[:, 0]) >= thr - 1e-6)
        if not pool.any():
            pool = ok_static & ~toward
        if not pool.any():
            pool = ok_static.copy()
        if not pool.any():
            pool = np.ones(C, dtype=bool)
        score = esc + 0.3 * esc_full + _K_PROG_REP * prog + 0.03 * same
        _, s_nx, s_close = _nearest_side_info(movers)
        if abs(s_nx) < _SIDE_NX and s_close > _SIDE_CLOSE:
            score = score - _SIDE_PEN * (cand[:, 0] < _SIDE_V_MIN).astype(np.float64)
    else:
        safe = ok_static & ~toward & (margin >= 0.0)
        if safe.any():
            pool = safe
            nominal = np.zeros(C)
            nominal[0] = 1.0
            score = (prog - 0.3 * herr - 0.03 * np.abs(cand[:, 1]) + 0.08 * same + 0.1 * nominal
                     - 0.1 * (cand[:, 0] < 0.0))
        else:
            pool = ok_static & ~toward
            if not pool.any():
                pool = ok_static.copy()
            if not pool.any():
                pool = np.ones(C, dtype=bool)
            score = esc_full + 0.05 * t_viol + 0.01 * prog + 0.03 * same
    i = int(np.argmax(np.where(pool, score, -np.inf)))
    return float(cand[i, 0]), float(cand[i, 1])




def _speed_caps(t, pose):
    """Dynamic speed cap (diagnosis direction): returns (v_hi >= 0, v_lo <= 0) [m/s].
    For every tracked moving object: nearest point to the chassis rectangle -> gap g and unit normal n (robot
    frame, from the chassis toward the object); object closing speed c_obj = -(n . u); total closing speed with
    the measured robot speed; TTC = g / closing. If TTC < _TTC_CAP or g < _CAP_NEAR, the allowed robot closing
    contribution is min(sqrt(2 * _A_CAP * (g - _G_CAP0)), (g - _G_CAP0) / _TTC_MIN) - max(c_obj, 0) (>= 0); the
    robot's forward speed adds v * n_x to the closing speed, so v <= allow / n_x for objects ahead (n_x > 0) and
    v >= -allow / |n_x| for objects behind. Goes to 0 as g -> _G_CAP0."""
    v_hi, v_lo = _V_MAX, -_V_MAX
    v_r = float(_MV["v_est"])
    for P, u, mv in _mover_set(t, pose):
        if not mv or len(P) == 0:
            continue
        bx = np.clip(P[:, 0], _BODY[0], _BODY[1])
        by = np.clip(P[:, 1], _BODY[2], _BODY[3])
        dx, dy = P[:, 0] - bx, P[:, 1] - by
        g = np.hypot(dx, dy)
        j = int(np.argmin(g))
        gj = float(g[j])
        if gj > 1e-3:
            nx, ny = float(dx[j]) / gj, float(dy[j]) / gj
        else:
            cx, cy = float(P[j, 0]) + 0.225, float(P[j, 1])
            nn = math.hypot(cx, cy)
            nx, ny = (cx / nn, cy / nn) if nn > 1e-6 else (1.0, 0.0)
        c_obj = -(nx * float(u[0]) + ny * float(u[1]))
        close = c_obj + v_r * nx
        ttc = gj / close if close > 1e-3 else float("inf")
        if ttc >= _TTC_CAP and gj >= _CAP_NEAR:
            continue
        room = max(gj - _G_CAP0, 0.0)
        c_lim = min(math.sqrt(2.0 * _A_CAP * room), room / _TTC_MIN)
        allow = max(c_lim - max(c_obj, 0.0), 0.0)
        if nx > _NX_MIN:
            v_hi = min(v_hi, allow / nx)
        elif nx < -_NX_MIN:
            v_lo = max(v_lo, -allow / (-nx))
    return max(v_hi, 0.0), min(v_lo, 0.0)


_A_CAP = 1.0        # assumed braking deceleration for the dynamic speed cap [m/s^2]


_G_CAP0 = 0.3       # gap at which the allowed closing speed reaches 0 [m]


_TTC_CAP = 2.0      # the dynamic cap is active when TTC is below this [s]


_TTC_MIN = 1.0      # keep at least this time-to-contact for the remaining gap (g - _G_CAP0) [s]


_CAP_NEAR = 1.2     # the dynamic cap is also active when the gap is below this [m]


_NX_MIN = 0.2       # objects whose normal has |n_x| below this are not affected by forward/backward speed (0.1 -> 0.2, diagnosis direction 5)




_SIDE_NX = 0.3      # nearest moving object counts as "side" when |n_x| of its chassis normal is below this (diagnosis direction 3)


_SIDE_CLOSE = 0.05  # ... and its closing speed toward the chassis exceeds this [m/s]


_SIDE_V_MIN = 0.5   # in that side-approach case, candidates with v below this get the slow penalty [m/s] (diagnosis direction 3)


_SIDE_PEN = 1.0     # strength of the slow-candidate penalty in the repulse score (diagnosis direction 3)


_K_PROG_REP = 0.12  # progress weight in the repulse score (0.02 -> 0.12, diagnosis direction 4)


def _nearest_side_info(movers):
    """(gap, n_x, closing speed) of the moving track point nearest to the chassis rectangle, robot frame.
    n_x is the x component of the unit normal from the chassis toward that point; closing speed is the object's
    velocity component toward the chassis (-(n . u), > 0 = approaching). (inf, 1.0, 0.0) if no moving track."""
    best = (float("inf"), 1.0, 0.0)
    for P, u, mv in movers:
        if not mv or len(P) == 0:
            continue
        bx = np.clip(P[:, 0], _BODY[0], _BODY[1])
        by = np.clip(P[:, 1], _BODY[2], _BODY[3])
        dx, dy = P[:, 0] - bx, P[:, 1] - by
        g = np.hypot(dx, dy)
        j = int(np.argmin(g))
        gj = float(g[j])
        if gj >= best[0]:
            continue
        if gj > 1e-3:
            nx, ny = float(dx[j]) / gj, float(dy[j]) / gj
        else:
            cx, cy = float(P[j, 0]) + 0.225, float(P[j, 1])
            nn = math.hypot(cx, cy)
            nx, ny = (cx / nn, cy / nn) if nn > 1e-6 else (1.0, 0.0)
        c_obj = -(nx * float(u[0]) + ny * float(u[1]))
        best = (gj, nx, c_obj)
    return best




_EBRAKE_R = 0.8         # raw-sensor emergency brake radius (robot frame distance from the base origin) [m] (diagnosis)


_EBRAKE_HALF_DEG = 90.0  # raw-sensor emergency brake half angle (front +-90 deg) [deg] (diagnosis)


def _emergency_brake(pose):
    """Tracker-independent virtual bumper (diagnosis): re-reads the front + back 2D lidars through _sense(pose)
    (static map already removed) and returns True if any moving-object point lies within _EBRAKE_R of the robot
    (robot frame) and within +-_EBRAKE_HALF_DEG of the forward direction."""
    dyn, _ = _sense(pose)
    if len(dyn) == 0:
        return False
    r = np.hypot(dyn[:, 0], dyn[:, 1])
    b = np.arctan2(dyn[:, 1], dyn[:, 0])
    hit = (r < _EBRAKE_R) & (np.abs(b) <= math.radians(_EBRAKE_HALF_DEG))
    return bool(np.any(hit))


# ========================= [END EDIT REGION] ===============================
