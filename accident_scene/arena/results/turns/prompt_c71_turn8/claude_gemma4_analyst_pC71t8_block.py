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
    Moving objects (front 2D lidar, static map filtered out):
      - within _AVOID_R (2.0 m) in the forward ±90° half-plane: add a limited avoidance yaw rate (|dpsi| <= 0.35 rad,
        |w| <= 1.0) and scale v down; v is capped at 1.5 m/s while avoiding; an object outside the ±_STOP_CONE_DEG
        stop cone never scales v below 0.25, so side traffic does not stall the robot.
      - v = 0 only when an object inside the stop cone (|bearing| <= 30 deg) is closer than _STOP_R (0.6 m).
      - avoidance steering is dropped when a static obstacle is near the turning front side or the swinging rear.
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

    # 3) Moving objects: closest in the ±90 deg half-plane (steer + slow) and closest in the stop cone (stop)
    d_obj, b_obj, d_cone = _front_dynamic_min_dist(pose, 90.0)
    avoiding = d_obj < _AVOID_R
    w_avoid = 0.0
    scale = 1.0
    span = _AVOID_R - _STOP_R
    if avoiding:
        if abs(b_obj) >= _B_TIE:
            controller.side = 1.0 if b_obj > 0.0 else -1.0       # object on the left (+1) / right (-1)
        elif controller.side == 0.0:
            controller.side = 1.0 if e <= 0.0 else -1.0          # dead ahead: pass on the waypoint side
        side = controller.side
        k = _clamp((_AVOID_R - d_obj) / span, 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k * max(0.3, math.cos(b_obj))  # turn away from the object (limited)
        if _static_close_on_side(pose, -side):
            dpsi = 0.0                                            # do not steer / swing into walls, boxes, racks
        w_avoid = K_W * dpsi
        s_all = _clamp((d_obj - _STOP_R) / span, 0.0, 1.0)
        in_path = abs(b_obj) <= math.radians(_STOP_CONE_DEG)
        if not in_path:
            s_all = max(s_all, _SIDE_SCALE_MIN)                   # side object: slow down, never stall
        s_cone = _clamp((d_cone - _STOP_R) / span, 0.0, 1.0) if d_cone < float("inf") else 1.0
        scale = min(s_all, s_cone)
    else:
        controller.side = 0.0

    # 4) Control law: waypoint heading + limited avoidance yaw rate
    w_lim = _W_AVOID_MAX if avoiding else 2.0
    w = _clamp(K_W * e + w_avoid, -w_lim, w_lim)
    e_cmd = _wrap_angle(e + w_avoid / K_W)
    if abs(e_cmd) > _E_TURN:
        v = 0.0
    else:
        v = min(3.0, 3.5 * dist + 0.5) * scale
        if avoiding:
            v = min(v, _V_AVOID_MAX)

    # 5) Full stop only for the frontal stop cone; back up only if it lasts and the rear is clear
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


# ========================= [END EDIT REGION] ===============================
