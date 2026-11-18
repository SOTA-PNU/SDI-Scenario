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
    """Follow the pre-computed way-points (v_max 2.0, k_d 3.0, k_w 8.5, rotate in place when |e| > 0.3);
    brake (v=0) when a non-static object is within 2.0 m in the ±60° front cone, and cap v at 1.0 m/s
    when one is within 3.0 m."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.done_align = False
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

    # 3) Control law (diagnosis: E_TURN 0.3, v_max 2.0, k_d 3.0, k_w 8.5)
    E_TURN = 0.3
    w = _clamp(8.5 * e, -2.0, 2.0)
    if abs(e) > E_TURN:
        v = 0.0
    else:
        v = min(2.0, 3.0 * dist + 0.5)

    # 4) Reactive braking on moving objects (front 2D lidar, static map filtered out, ±60° cone): stop only, no reversing
    d_front = _front_dynamic_min_dist(pose)
    if d_front < 3.0:
        v = min(v, 1.0)                           # slow-down band so the higher speed does not lengthen braking
    controller.blocked = d_front < (2.2 if controller.blocked else 2.0)
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


def _front_dynamic_min_dist(pose):
    """Min distance (robot frame) to non-static front-lidar points inside the forward ±60° cone; inf if none."""
    pts = lidar2d_points("front")
    if pts is None or len(pts) == 0:
        return float("inf")
    wp = to_world(pts, pose)
    dyn = pts[~_static_mask(wp)]
    if len(dyn) == 0:
        return float("inf")
    cone = (dyn[:, 0] > 0.0) & (np.abs(dyn[:, 1]) < dyn[:, 0] * math.tan(math.radians(60.0)))
    if not np.any(cone):
        return float("inf")
    return float(np.min(np.hypot(dyn[cone, 0], dyn[cone, 1])))


def _rear_clear():
    """True if the back lidar sees nothing in a strip just behind the chassis (static or moving)."""
    pts = lidar2d_points("back")
    if pts is None or len(pts) == 0:
        return False
    behind = (pts[:, 0] < -0.70) & (pts[:, 0] > -1.40) & (np.abs(pts[:, 1]) < 0.45)
    return not bool(np.any(behind))


# ========================= [END EDIT REGION] ===============================
