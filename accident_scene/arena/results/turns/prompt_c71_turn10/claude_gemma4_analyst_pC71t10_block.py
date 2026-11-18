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
    """Follow the way-points (v_max 3.0, k_d 3.5, k_w 10.0, rotate in place when |e_cmd| > _E_TURN = 0.6).
    Moving objects (front + back 2D lidar, 360 deg, static map filtered out):
      - slow-down: any moving object within _AVOID_R (2.0 m) in ANY direction scales v down and caps v at
        _V_AVOID_MAX; objects outside the stop cone (side / rear) never scale v below _SIDE_SCALE_MIN.
      - steering: avoidance yaw rate only for the closest object with |bearing| <= 90 deg (|dpsi| <= 0.35 rad,
        |w| <= 1.0); dropped when a static obstacle is near the turning front side or the swinging rear.
      - wide stop: v = 0 when the closest object with |bearing| <= 90 deg is closer than _STOP_R_WIDE (0.7 m);
        this overrides the _SIDE_SCALE_MIN floor.
      - v = 0 also when an object inside the stop cone (|bearing| <= 30 deg) is closer than _STOP_R (0.6 m).
      - back up (straight, w = 0) only when a stop-cone object stays > 1.5 s and the rear is clear."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.side = 0.0
        controller.stop_steps = 0

    x, y, yaw = pose

    # 1) Goal-alignment phase (rotate in place, v = 0)
    if controller.idx >= len(_WAYPOINTS):
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
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

    # 3) Moving objects: 360 deg closest (slow down), front-half closest (steer), stop cone (stop)
    d_all, b_all, d_front, b_front, d_cone = _get_dynamic_min(pose)
    slowing = d_all < _AVOID_R
    steering = d_front < _AVOID_R

    # 3a) avoidance steering: only for objects within +-90 deg
    w_avoid = 0.0
    if steering:
        if abs(b_front) >= _B_TIE:
            controller.side = 1.0 if b_front > 0.0 else -1.0     # object on the left (+1) / right (-1)
        elif controller.side == 0.0:
            controller.side = 1.0 if e <= 0.0 else -1.0          # dead ahead: pass on the waypoint side
        side = controller.side
        k = _clamp((_AVOID_R - d_front) / span, 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k * max(0.3, math.cos(b_front))  # turn away from the object (limited)
        if _static_close_on_side(pose, -side):
            dpsi = 0.0                                            # do not steer / swing into walls, boxes, racks
        w_avoid = K_W * dpsi
    else:
        controller.side = 0.0

    # 3b) speed scaling: for moving objects in any direction (360 deg)
    scale = 1.0
    if slowing:
        s_all = _clamp((d_all - _STOP_R) / span, 0.0, 1.0)
        in_path = abs(b_all) <= math.radians(_STOP_CONE_DEG)
        if not in_path:
            s_all = max(s_all, _SIDE_SCALE_MIN)                   # side / rear object: slow down, never stall
        s_cone = _clamp((d_cone - _STOP_R) / span, 0.0, 1.0) if d_cone < float("inf") else 1.0
        scale = min(s_all, s_cone)

    # 4) Control law: waypoint heading + limited avoidance yaw rate
    w_lim = _W_AVOID_MAX if steering else 2.0
    w = _clamp(K_W * e + w_avoid, -w_lim, w_lim)
    e_cmd = _wrap_angle(e + w_avoid / K_W)
    if abs(e_cmd) > _E_TURN:
        v = 0.0
    else:
        v = min(3.0, 3.5 * dist + 0.5) * scale
        if slowing:
            v = min(v, _V_AVOID_MAX)

    # 4b) Wide stop: any moving object within +-90 deg closer than _STOP_R_WIDE -> v = 0
    #     (applied after scaling, so it overrides the _SIDE_SCALE_MIN floor)
    if d_front < _STOP_R_WIDE:
        v = 0.0

    # 5) Full stop only for the frontal stop cone (unchanged); back up only if it lasts and the rear is clear
    if d_cone < _STOP_R:
        controller.stop_steps += 1
        v = 0.0
        if controller.stop_steps > _BACK_AFTER_STEPS and _rear_clear():
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


_STOP_R = 0.6           # full stop radius inside the stop cone [m]


_DPSI_MAX = 0.35        # max avoidance heading offset [rad] (w_avoid = k_w * offset)


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


_V_AVOID_MAX = 1.5      # speed cap while a moving object is within _AVOID_R [m/s]


_E_TURN = 0.6           # rotate in place when |e_cmd| exceeds this [rad]




def _get_dynamic_min(pose):
    """Closest non-static points from BOTH 2D lidars (front + back, full 360 deg), robot frame.
    Returns (d_all, b_all, d_front, b_front, d_cone):
      d_all, b_all     closest moving-object point in any direction (inf, 0.0 if none)
      d_front, b_front closest moving-object point with |bearing| <= 90 deg (inf, 0.0 if none)
      d_cone           closest moving-object point with |bearing| <= _STOP_CONE_DEG (inf if none)"""
    inf = float("inf")
    parts = []
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p is not None and len(p) > 0:
            parts.append(p)
    if not parts:
        return inf, 0.0, inf, 0.0, inf
    pts = np.concatenate(parts)
    dyn = pts[~_static_mask(to_world(pts, pose))]
    if len(dyn) == 0:
        return inf, 0.0, inf, 0.0, inf
    r = np.hypot(dyn[:, 0], dyn[:, 1])
    bearing = np.arctan2(dyn[:, 1], dyn[:, 0])
    i = int(np.argmin(r))
    d_all, b_all = float(r[i]), float(bearing[i])
    half = np.abs(bearing) <= math.pi / 2.0
    if np.any(half):
        rh = r[half]
        bh = bearing[half]
        j = int(np.argmin(rh))
        d_front, b_front = float(rh[j]), float(bh[j])
    else:
        d_front, b_front = inf, 0.0
    stop = np.abs(bearing) <= math.radians(_STOP_CONE_DEG)
    d_cone = float(np.min(r[stop])) if np.any(stop) else inf
    return d_all, b_all, d_front, b_front, d_cone




_STOP_R_WIDE = 0.7      # wide stop radius: any moving object within +-90 deg closer than this -> v = 0 [m]


# ========================= [END EDIT REGION] ===============================
