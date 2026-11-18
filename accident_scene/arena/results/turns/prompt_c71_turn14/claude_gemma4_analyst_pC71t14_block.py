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
    (-5.0, -8.5),    # pass south-east of box 00
    (-4.0, 2.0),     # go north between forklift_01 / box 00 and boxes 01, 02
    (-2.8, 7.5),     # pass west of box 03, east of box 09
    (0.5, 11.0),     # pass east of box 10, west of box 07
    (1.5, 15.0),     # pass west of box 08, north of box 10
    (GOAL[0], GOAL[1])  # final goal position (north of box 06)
]

def controller(t, pose, env):
    """Follow the way-points (v_max 3.0, k_d 3.5, k_w 10.0).
    Moving objects (front + back 2D lidar, static map filtered out):
      - no hard speed cap while avoiding: v is multiplied by a distance-based scale
        speed_scale = clamp((d_front - _SCALE_STOP_R) / (_SCALE_FULL_R - _SCALE_STOP_R), 0, 1)
        computed from the closest moving-object point within +-90 deg (0.7 m -> v = 0, 1.5 m -> full speed).
      - speed_scale == 0 is decided first (v = 0), before the _E_TURN in-place-rotation rule.
      - object within +-90 deg closer than _AVOID_R (2.0 m): avoidance steering w_avoid (offset <= _DPSI_MAX)
        away from the object, v also scaled by cos(e_cmd) (at least 30 %); _static_close_on_side cancels steering
        toward static obstacles and _static_ahead stops forward motion if a static obstacle is right in front.
      - frontal stop cone (+-30 deg, _STOP_R 0.8 m): v = 0; back up if very close or blocked long and rear clear.
      - no moving object near: rotate in place when |e_cmd| > _E_TURN, else normal speed.
      - goal alignment: rotate in place (v = 0); re-approach GOAL if farther than _GOAL_REENTER."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.side = 0.0
        controller.stop_steps = 0

    x, y, yaw = pose
    obs = _get_dynamic_min(pose)
    stop_zone = obs["d_cone"] < _STOP_R

    # 1) Goal-alignment phase (v = 0 always)
    if controller.idx >= len(_WAYPOINTS):
        gd = math.hypot(GOAL[0] - x, GOAL[1] - y)
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if gd > _GOAL_REENTER:
            controller.idx = len(_WAYPOINTS) - 1  # drifted off the goal: drive back to GOAL
        else:
            if abs(e) < 0.05:
                return 0.0, 0.0, True
            return 0.0, w, False

    # 2) Normal waypoint-following
    tx, ty = _WAYPOINTS[controller.idx]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    K_W = 10.0
    span = _AVOID_R - _STOP_R

    d_front, b_front = obs["d_front"], obs["b_front"]
    d_cone = obs["d_cone"]
    steering = d_front < _AVOID_R

    # 3) avoidance steering (objects within +-90 deg) away from the object
    w_avoid = 0.0
    dpsi = 0.0
    if steering:
        if abs(b_front) >= _B_TIE:
            controller.side = 1.0 if b_front > 0.0 else -1.0     # object on the left (+1) / right (-1)
        elif controller.side == 0.0:
            controller.side = 1.0 if e <= 0.0 else -1.0          # dead ahead: pass on the waypoint side
        side = controller.side
        k = _clamp((_AVOID_R - d_front) / span, 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k * max(0.3, math.cos(b_front))
        if _static_close_on_side(pose, -side):
            dpsi = 0.0                                            # do not steer / swing into walls, boxes, racks
        w_avoid = K_W * dpsi
    else:
        controller.side = 0.0

    # 4) Control law: waypoint heading + avoidance yaw rate
    w_lim = _W_AVOID_MAX if steering else 2.0
    w = _clamp(K_W * e + w_avoid, -w_lim, w_lim)
    e_cmd = _wrap_angle(e + dpsi)

    # distance-based speed scale from the closest moving object within +-90 deg (replaces the 0.7 m/s hard cap)
    if math.isfinite(d_front):
        speed_scale = _clamp((d_front - _SCALE_STOP_R) / (_SCALE_FULL_R - _SCALE_STOP_R), 0.0, 1.0)
    else:
        speed_scale = 1.0
    v_nom = min(3.0, 3.5 * dist + 0.5)

    if speed_scale <= 0.0:
        v = 0.0                                                   # moving object within 0.7 m (+-90 deg): stop first
    elif steering:
        v = v_nom * speed_scale * _clamp(math.cos(e_cmd), 0.3, 1.0)
        if _static_ahead(pose):
            v = 0.0                                               # never push into a static obstacle
    elif abs(e_cmd) > _E_TURN:
        v = 0.0                                                   # rotate in place (no moving object near)
    else:
        v = v_nom * speed_scale

    # 5) Frontal stop only for the +-30 deg cone; back up if very close or blocked long, rear clear
    if stop_zone:
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


_DPSI_MAX = 0.5         # max avoidance heading offset [rad] (w_avoid = k_w * offset)


_W_AVOID_MAX = 1.0      # |w| limit while avoiding [rad/s]


_B_TIE = 0.15           # |bearing| below this counts as dead ahead [rad]


_BACK_R = 0.45          # stop-zone object closer than this -> back up [m]


_BACK_AFTER_STEPS = 90  # blocked longer than this many control steps (1.5 s) -> back up


_V_BACK = 0.3           # back-up speed [m/s]


def _static_close_on_side(pose, turn_side):
    """True if a static-map point (wall, fixed obstacle, box) seen by the front or back lidar is near the region the
    robot would sweep when turning to turn_side (+1 = left, -1 = right): ahead/beside on the turn side
    (x -0.3..1.6 m, 0..1.2 m laterally) or behind on the opposite side where the rear of the chassis swings
    (x -1.05..-0.3 m, 0..0.85 m laterally)."""
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return False
    pts = np.concatenate(parts)
    st = pts[_static_mask(to_world(pts, pose))]
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
                       |y| < _PATH_HALF_W"""
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
    dyn = pts[~_static_mask(to_world(pts, pose))]
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


_V_EVADE_MIN = 0.6      # forward speed floor during active evasion [m/s]


_V_EVADE_STATIC = 0.3   # reduced floor when a static obstacle is close on the evasion side [m/s]


_DPSI_EVADE = 0.6       # evasion heading offset [rad]


_W_EVADE_MAX = 1.5      # |w| limit during active evasion [rad/s]


_BODY_FRONT = 0.25      # front face of the chassis in the robot frame [m]


_CLEAR_X = 1.15         # the corridor ahead must be free of any point up to this x for the evasion floor [m]


_CORR_X_MIN = 0.15      # moving-object points beyond this x within |y| < _PATH_HALF_W count as in the path [m]


_CORR_STOP_X = 0.70     # moving-object point in the path corridor closer than this x disables the evasion floor (no stop) [m]


_GOAL_REENTER = 0.45    # after an evasion during goal alignment, drive back to GOAL if farther than this [m]




def _static_close_turn(pose, w):
    """Static-map check for the side the robot is turning to: sign(w) > 0 -> left, < 0 -> right, w = 0 -> both."""
    if w > 1e-6:
        return _static_close_on_side(pose, 1.0)
    if w < -1e-6:
        return _static_close_on_side(pose, -1.0)
    return _static_close_on_side(pose, 1.0) or _static_close_on_side(pose, -1.0)




_SCALE_HALF_DEG = 120.0  # front/side sector (|bearing| <= this) used for the distance-based speed scale [deg]


_SCALE_STOP_R = 0.7      # moving object within +-90 deg closer than this -> speed scale 0 (v = 0) [m]


_SCALE_FULL_R = 1.5      # speed scale rises linearly from 0 at _SCALE_STOP_R to 1 at this distance [m]




def _static_ahead(pose):
    """True if a static-map point (wall, fixed obstacle, box) seen by the front or back lidar lies in the corridor
    right in front of the chassis (_BODY_FRONT < x < 1.0 m, |y| < _PATH_HALF_W), robot frame."""
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return False
    pts = np.concatenate(parts)
    st = pts[_static_mask(to_world(pts, pose))]
    if len(st) == 0:
        return False
    ahead = (st[:, 0] > _BODY_FRONT) & (st[:, 0] < 1.0) & (np.abs(st[:, 1]) < _PATH_HALF_W)
    return bool(np.any(ahead))


# ========================= [END EDIT REGION] ===============================
