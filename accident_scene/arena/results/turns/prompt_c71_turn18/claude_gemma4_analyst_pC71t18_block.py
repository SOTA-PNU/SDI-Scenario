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
    """Follow the way-points (v_max _V_MAX, k_d _K_D, k_w _K_W) with moving-object handling (front + back 2D lidar,
    static map filtered out, sensed once per step):
      - frontal stop cone (+-_STOP_CONE_DEG): only objects here scale v down
        (speed_scale = clamp((d_cone - _SCALE_STOP_R) / (_SCALE_FULL_R - _SCALE_STOP_R), 0, 1)),
        stop inside _STOP_R and back up if very close or blocked long with the rear clear.
      - side / rear objects (outside the cone) closer than _EVADE_GAP to the chassis: evasion mode (held
        _EVADE_HOLD_STEPS). The forward speed is at least _V_EVADE_MIN (_V_EVADE_STATIC if a static obstacle is
        close on the turning side), and this floor takes priority over the speed scale and the in-place rotation.
        The floor is off when an object is in the stop cone, a moving point is in the corridor ahead, or a static
        obstacle is right ahead. Front-side objects (|bearing| <= 90 deg) also add a stronger steering offset
        (_DPSI_MAX) away from the object, cancelled when a static obstacle is close on that side.
      - goal alignment: rotate in place, finish when |yaw error| < _GOAL_YAW_DONE; while a side object closes in
        only a small forward floor _V_ALIGN_FLOOR is used (no hunting out of the goal disc); drive back to GOAL if
        farther than _GOAL_REENTER."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.side = 0.0
        controller.stop_steps = 0
        controller.evade_steps = 0

    x, y, yaw = pose
    dyn, st = _sense(pose)
    obs = _dyn_summary(dyn)
    d_cone, b_cone = obs["d_cone"], obs["b_cone"]
    g_side, b_side = obs["g_side"], obs["b_side"]
    front_static = _static_front_pts(st)

    # side / rear approach -> evasion mode (held for a short time after the last detection)
    near_side = g_side < _EVADE_GAP
    if near_side:
        controller.evade_steps = _EVADE_HOLD_STEPS
    elif controller.evade_steps > 0:
        controller.evade_steps -= 1
    evading = controller.evade_steps > 0
    floor_ok = (evading and d_cone >= _STOP_R and obs["x_corr"] >= _CORR_STOP_X and not front_static)

    # 1) goal alignment
    if controller.idx >= len(_WAYPOINTS):
        gd = math.hypot(GOAL[0] - x, GOAL[1] - y)
        eg = _wrap_angle(GOAL[2] - yaw)
        if gd > _GOAL_REENTER:
            controller.idx = len(_WAYPOINTS) - 1          # drifted off the goal: drive back to GOAL
        else:
            if abs(eg) < _GOAL_YAW_DONE:
                return 0.0, 0.0, True
            w = _clamp(_K_ALIGN * eg, -_W_ALIGN, _W_ALIGN)
            v = _V_ALIGN_FLOOR if floor_ok else 0.0       # small creep only: stay inside the goal disc
            return v, w, False

    # 2) way-point following
    tx, ty = _WAYPOINTS[controller.idx]
    dist = math.hypot(tx - x, ty - y)
    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dist = math.hypot(tx - x, ty - y)
    e = _wrap_angle(math.atan2(ty - y, tx - x) - yaw)

    # 3) steering offset: front-side object (stronger, _DPSI_MAX) or stop-cone object (_DPSI_CONE)
    dpsi = 0.0
    steering = False
    if near_side and abs(b_side) <= math.pi / 2.0:
        side = 1.0 if b_side > 0.0 else -1.0              # object on the left (+1) / right (-1)
        k = _clamp((_EVADE_GAP - g_side) / (_EVADE_GAP - 0.2), 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k
        if _static_side_pts(st, -side):
            dpsi = 0.0                                    # never steer into walls, boxes, racks
        steering = True
    elif d_cone < _AVOID_R:
        if abs(b_cone) >= _B_TIE:
            controller.side = 1.0 if b_cone > 0.0 else -1.0
        elif controller.side == 0.0:
            controller.side = 1.0 if e <= 0.0 else -1.0   # dead ahead: pass on the waypoint side
        side = controller.side
        k = _clamp((_AVOID_R - d_cone) / (_AVOID_R - _STOP_R), 0.0, 1.0)
        dpsi = -side * _DPSI_CONE * k
        if _static_side_pts(st, -side):
            dpsi = 0.0
        steering = True
    else:
        controller.side = 0.0

    if evading:
        w_lim = _W_EVADE_MAX
    elif steering:
        w_lim = _W_AVOID_MAX
    else:
        w_lim = 2.0
    w = _clamp(_K_W * (e + dpsi), -w_lim, w_lim)
    e_cmd = _wrap_angle(e + dpsi)

    # 4) speed: distance scale only from the frontal stop cone
    if math.isfinite(d_cone):
        speed_scale = _clamp((d_cone - _SCALE_STOP_R) / (_SCALE_FULL_R - _SCALE_STOP_R), 0.0, 1.0)
    else:
        speed_scale = 1.0
    v_nom = min(_V_MAX, _K_D * dist + 0.5)

    if abs(e_cmd) > _E_TURN:
        v = 0.0                                           # rotate in place (overridden by the evasion floor)
    else:
        v = v_nom * speed_scale
        if steering:
            v *= _clamp(math.cos(e_cmd), 0.3, 1.0)
    if steering and front_static:
        v = 0.0                                           # never push into a static obstacle while deviating

    # 5) side / rear evasion: minimum forward speed, takes priority over speed_scale and in-place rotation
    if floor_ok:
        if w > 0.05:
            static_turn = _static_side_pts(st, 1.0)
        elif w < -0.05:
            static_turn = _static_side_pts(st, -1.0)
        else:
            static_turn = False
        v = max(v, _V_EVADE_STATIC if static_turn else _V_EVADE_MIN)

    # 6) frontal stop cone: stop, back up if very close or blocked long and the rear is clear
    if d_cone < _STOP_R:
        controller.stop_steps += 1
        v = 0.0
        if (d_cone < _BACK_R or controller.stop_steps > _BACK_AFTER_STEPS) and _rear_clear():
            v = -_V_BACK
            w = 0.0
    else:
        controller.stop_steps = 0

    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1

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


_DPSI_MAX = 1.0         # side-object avoidance heading offset [rad] (raised from 0.8, diagnosis direction 3)


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


_E_TURN = 0.6           # rotate in place when |e_cmd| exceeds this [rad]




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


_V_EVADE_MIN = 0.3      # forward speed floor during active evasion [m/s] (lowered from 0.8, diagnosis direction 1)


_V_EVADE_STATIC = 0.3   # reduced floor when a static obstacle is close on the evasion side [m/s]


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
      d_cone, b_cone  closest point (range from base_link) with |bearing| <= _STOP_CONE_DEG (inf, 0.0 if none)
      g_side, b_side  smallest gap to the chassis rectangle _BODY among points outside the cone (inf, 0.0 if none)
      x_corr          smallest x of points in the corridor ahead (x > _CORR_X_MIN, |y| < _CORR_HALF_W) (inf if none)"""
    inf = float("inf")
    out = {"d_cone": inf, "b_cone": 0.0, "g_side": inf, "b_side": 0.0, "x_corr": inf}
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
    corr = (dyn[:, 0] > _CORR_X_MIN) & (np.abs(dyn[:, 1]) < _CORR_HALF_W)
    if np.any(corr):
        out["x_corr"] = float(np.min(dyn[corr, 0]))
    return out


def _static_side_pts(st, turn_side):
    """Same test as _static_close_on_side on already-sensed static points st (robot frame): a static point ahead/beside
    on the turn side (x -0.3..1.6 m, 0..1.2 m laterally) or where the rear swings (x -1.05..-0.3 m, 0..0.85 m)."""
    if len(st) == 0:
        return False
    lat = st[:, 1] * turn_side
    front_near = (st[:, 0] > -0.3) & (st[:, 0] < 1.6) & (lat > 0.0) & (lat < 1.2)
    rear_swing = (st[:, 0] > -1.05) & (st[:, 0] <= -0.3) & (lat < 0.0) & (lat > -0.85)
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


# ========================= [END EDIT REGION] ===============================
