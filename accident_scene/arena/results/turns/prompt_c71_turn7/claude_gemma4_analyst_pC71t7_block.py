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
    """Follow the way-points (v_max 3.0, k_d 3.5, k_w 10.0, rotate in place when |e_cmd| > 0.5).
    Reactive steering avoidance (diagnosis): a non-static object within 2.5 m in the forward ±90° half-plane adds
    w_avoid that turns away from the object's bearing (dead ahead -> toward the waypoint side), |w| limited to 1.5;
    v is scaled by (d - 1.0) / 1.5 and is 0 only below 1.0 m. Avoidance steering is dropped when a static obstacle
    is close on the turn side. If blocked below 1.0 m for > 1.5 s (or the object is within 0.7 m), back up slowly
    when the rear strip is clear."""
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
    E_TURN = 0.5

    # 3) Reactive steering avoidance for moving objects (front 2D lidar, static map filtered out)
    d_obj, b_obj = _front_dynamic_min_dist(pose, 90.0)
    avoiding = d_obj < _AVOID_R
    w_avoid = 0.0
    scale = 1.0
    if avoiding:
        if abs(b_obj) >= _B_TIE:
            controller.side = 1.0 if b_obj > 0.0 else -1.0       # object on the left (+1) / right (-1)
        elif controller.side == 0.0:
            controller.side = 1.0 if e <= 0.0 else -1.0          # dead ahead: pass on the waypoint side
        side = controller.side
        k = _clamp((_AVOID_R - d_obj) / (_AVOID_R - _STOP_R), 0.0, 1.0)
        dpsi = -side * _DPSI_MAX * k * max(0.3, math.cos(b_obj))  # turn away from the object
        if _static_close_on_side(pose, -side):
            dpsi = 0.0                                            # do not steer into walls / boxes / racks
        w_avoid = K_W * dpsi
        scale = _clamp((d_obj - _STOP_R) / (_AVOID_R - _STOP_R), 0.0, 1.0)
    else:
        controller.side = 0.0

    # 4) Control law: waypoint heading + avoidance yaw rate, limited while avoiding
    w_lim = _W_AVOID_MAX if avoiding else 2.0
    w = _clamp(K_W * e + w_avoid, -w_lim, w_lim)
    e_cmd = _wrap_angle(e + w_avoid / K_W)
    if abs(e_cmd) > E_TURN:
        v = 0.0
    else:
        v = min(3.0, 3.5 * dist + 0.5) * scale

    # 5) Full stop only below 1.0 m; back up if it lasts or the object comes very close
    if d_obj < _STOP_R:
        controller.stop_steps += 1
        v = 0.0
        if (d_obj < _BACK_R or controller.stop_steps > _BACK_AFTER_STEPS) and _rear_clear():
            v = -_V_BACK
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
    """(min distance, bearing [rad]) in the robot frame of the closest non-static front-lidar point whose
    bearing is within ±half_angle_deg (default ±90°: the whole forward half-plane); (inf, 0.0) if none."""
    pts = lidar2d_points("front")
    if pts is None or len(pts) == 0:
        return float("inf"), 0.0
    wp = to_world(pts, pose)
    dyn = pts[~_static_mask(wp)]
    if len(dyn) == 0:
        return float("inf"), 0.0
    bearing = np.arctan2(dyn[:, 1], dyn[:, 0])
    cone = np.abs(bearing) <= math.radians(half_angle_deg)
    if not np.any(cone):
        return float("inf"), 0.0
    c = dyn[cone]
    r = np.hypot(c[:, 0], c[:, 1])
    i = int(np.argmin(r))
    return float(r[i]), float(math.atan2(c[i, 1], c[i, 0]))


def _rear_clear():
    """True if the back lidar sees nothing in a strip just behind the chassis (static or moving)."""
    pts = lidar2d_points("back")
    if pts is None or len(pts) == 0:
        return False
    behind = (pts[:, 0] < -0.70) & (pts[:, 0] > -1.40) & (np.abs(pts[:, 1]) < 0.45)
    return not bool(np.any(behind))




_AVOID_R = 2.5          # avoidance / slow-down radius [m]


_STOP_R = 1.0           # full stop radius [m]


_DPSI_MAX = 0.6         # max avoidance heading offset [rad] (w_avoid = k_w * offset)


_W_AVOID_MAX = 1.5      # |w| limit while avoiding [rad/s]


_B_TIE = 0.15           # |bearing| below this counts as dead ahead [rad]


_BACK_R = 0.7           # object closer than this while stopped -> back up [m]


_BACK_AFTER_STEPS = 90  # blocked longer than this many control steps (1.5 s) -> back up


_V_BACK = 0.3           # back-up speed [m/s]


def _static_close_on_side(pose, turn_side):
    """True if a static-map point (wall, fixed obstacle, box) is close beside/ahead of the robot on the side it
    would turn to (turn_side +1 = left, -1 = right)."""
    pts = lidar2d_points("front")
    if pts is None or len(pts) == 0:
        return False
    st = pts[_static_mask(to_world(pts, pose))]
    if len(st) == 0:
        return False
    lat = st[:, 1] * turn_side
    near = (st[:, 0] > -0.3) & (st[:, 0] < 1.3) & (lat > 0.0) & (lat < 0.95)
    return bool(np.any(near))


# ========================= [END EDIT REGION] ===============================
