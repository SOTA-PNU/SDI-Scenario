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
    """Follow the simplified way-points (v_max 3.0, k_d 3.5, k_w 10.0, rotate in place when |e| > 0.8);
    stop (v=0) when a non-static object is within 1.8 m inside the forward ±90° half-plane (release at 2.0 m).
    No creep mode: either full stop or normal speed."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.blocked = False

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

    # 3) Control law (diagnosis: v_max 3.0, E_TURN 0.8; k_d 3.5, k_w 10.0 unchanged)
    E_TURN = 0.8
    w = _clamp(10.0 * e, -2.0, 2.0)
    if abs(e) > E_TURN:
        v = 0.0
    else:
        v = min(3.0, 3.5 * dist + 0.5)

    # 4) Reactive stop for moving objects (front 2D lidar, static map filtered out)
    #    stop zone: forward half-plane (bearing within ±90°) closer than 1.8 m -> v = 0
    #    hysteresis: stay stopped until the object is farther than 2.0 m. No creep mode.
    d_stop, b_stop = _front_dynamic_min_dist(pose, 90.0)
    stop_r = 2.0 if controller.blocked else 1.8
    controller.blocked = (d_stop < stop_r) and (abs(b_stop) <= math.radians(90.0))
    if controller.blocked:
        v = 0.0                                   # stop; keep w so the heading stays on the waypoint

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


# ========================= [END EDIT REGION] ===============================
