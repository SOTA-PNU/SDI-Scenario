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
    """Waypoint navigation with close‑range stop‑back‑rotate‑wait avoidance.
    Uses the updated _CLOSE_DETECT_DIST, _AVOID_DURATION and _AVOID_COOLDOWN.
    """
    x, y, yaw = pose

    # ----- one‑time state initialization -----
    if not hasattr(controller, "initialized"):
        controller.initialized = True
        controller.idx = 0                     # waypoint index
        controller.state = "normal"            # normal, avoid_stop, avoid_back, avoid_rotate, avoid_wait
        controller.avoid_end = 0.0
        controller.avoid_dir = 1.0
        controller.wait_clear_end = 0.0
        controller.avoid_cooldown_end = 0.0
        controller.avoid_count = 0
        controller.low_speed = False

    # ----- detection -------------------------------------------------
    # close‑range points (within _CLOSE_DETECT_DIST)
    close_pts = _moving_points(pose)
    close = False
    bearing_close = 0.0
    if close_pts.shape[0]:
        robot_xy = np.array(pose[:2])
        dists = np.linalg.norm(close_pts - robot_xy, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] <= _CLOSE_DETECT_DIST:
            close = True
            dx = close_pts[min_idx, 0] - x
            dy = close_pts[min_idx, 1] - y
            # bearing in robot frame
            x_r =  math.cos(yaw) * dx + math.sin(yaw) * dy
            y_r = -math.sin(yaw) * dx + math.cos(yaw) * dy
            bearing_close = math.atan2(y_r, x_r)

    # medium‑range moving objects for repulsive field
    moving_any, _ = _detect_moving_any(pose)

    # ----- state machine for close‑range avoidance -----
    if controller.state == "normal":
        if close:
            controller.state = "avoid_stop"
            controller.avoid_end = t + _AVOID_DURATION
            # rotate away from the object (opposite bearing)
            dir_sign = -np.sign(bearing_close)
            controller.avoid_dir = dir_sign if dir_sign != 0 else 1.0
            return 0.0, 0.0, False

    elif controller.state == "avoid_stop":
        if t < controller.avoid_end:
            return 0.0, 0.0, False
        # after stop, back up
        controller.state = "avoid_back"
        controller.avoid_end = t + _BACK_TIME
        return _BACK_SPEED, 0.0, False

    elif controller.state == "avoid_back":
        if t < controller.avoid_end:
            return _BACK_SPEED, 0.0, False
        # after backup, rotate
        controller.state = "avoid_rotate"
        controller.avoid_end = t + _ROT_TIME
        return 0.0, controller.avoid_dir * _AVOID_ROT_SPEED, False

    elif controller.state == "avoid_rotate":
        if t < controller.avoid_end:
            return 0.0, controller.avoid_dir * _AVOID_ROT_SPEED, False
        # rotation finished – wait for the object to clear
        controller.state = "avoid_wait"
        controller.wait_clear_end = t + _WAIT_CLEAR_TIME
        return 0.0, 0.0, False

    elif controller.state == "avoid_wait":
        # stay still while any moving object is still present
        if moving_any:
            return 0.0, 0.0, False
        if t < controller.wait_clear_end:
            return 0.0, 0.0, False
        # object cleared – resume normal navigation
        controller.state = "normal"
        controller.avoid_cooldown_end = t + _AVOID_COOLDOWN
        controller.avoid_count += 1
        if controller.avoid_count > _MAX_AVOID_COUNT:
            controller.low_speed = True
        # fall through to normal waypoint navigation

    # ----- normal waypoint navigation (with optional repulsive field) -----
    if controller.idx < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[controller.idx]
    else:
        tx, ty = GOAL[0], GOAL[1]

    # advance waypoint if close enough
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

    # final goal alignment
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        yaw_err = _wrap_angle(GOAL[2] - yaw)
        if abs(yaw_err) < 0.05:
            return 0.0, 0.0, True
        w_align = _clamp(_KP_W * yaw_err, -0.6, 0.6)
        return 0.0, w_align, False

    # repulsive field from medium‑range moving objects
    rep_pts = _moving_points_repulsion(pose)
    rep_dir = _repulsive_vector(pose, rep_pts)
    att_dir = np.array([dx, dy]) / (dist + 1e-6)
    combined_dir = att_dir + _REPULSION_WEIGHT * rep_dir
    norm_comb = np.linalg.norm(combined_dir)
    if norm_comb > 0:
        combined_dir = combined_dir / norm_comb
    desired_heading = math.atan2(combined_dir[1], combined_dir[0])
    e = _wrap_angle(desired_heading - yaw)

    # speed command
    base_v = min(_MAX_SPEED, 2.5 * dist + 0.2)
    if controller.low_speed:
        base_v = _clamp(base_v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)
    v = base_v * max(0.0, math.cos(e))          # reduce forward speed for large heading error
    w = _clamp(_KP_W * e, -1.5, 1.5)

    return float(v), float(w), False




def _in_static_box(p, box):
    """Return True if world point p (x,y) lies inside the axis‑aligned box."""
    x, y = p
    xb = box["x"] - box["width"] / 2.0
    xe = box["x"] + box["width"] / 2.0
    yb = box["y"] - box["depth"] / 2.0
    ye = box["y"] + box["depth"] / 2.0
    return xb <= x <= xe and yb <= y <= ye


def _in_static_obstacle(p):
    """Check whether world point p is inside any static obstacle (boxes, walls, fixed obstacles)."""
    # boxes from MISSION
    for b in MISSION["boxes"]:
        if _in_static_box(p, b):
            return True
    # walls: points outside the interior rectangle are walls
    wx_min, wx_max, wy_min, wy_max = WALLS
    if p[0] < wx_min or p[0] > wx_max or p[1] < wy_min or p[1] > wy_max:
        return True
    # fixed obstacles list: (name, xmin, xmax, ymin, ymax)
    for _, ox_min, ox_max, oy_min, oy_max in FIXED_OBSTACLES:
        if ox_min <= p[0] <= ox_max and oy_min <= p[1] <= oy_max:
            return True
    return False


def _detect_moving_front(pose):
    """
    Scan front 2D lidar, return (has_moving, mean_bearing) for points that are:
      * within 1.2 m,
      * within ±30° of the robot's forward axis,
      * NOT inside any static obstacle.
    mean_bearing is the average angle (rad) of those points (positive = left).
    """
    pts = lidar2d_points("front")          # robot‑frame (x, y)
    if pts.shape[0] == 0:
        return False, 0.0

    # filter by distance and sector
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])
    sector_mask = (dist <= 1.2) & (np.abs(bearing) <= np.deg2rad(30.0))

    if not np.any(sector_mask):
        return False, 0.0

    # convert candidate points to world coordinates
    cand_robot = pts[sector_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]   # ignore z

    # keep only those not inside static map
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0

    moving_bearings = bearing[sector_mask][moving_mask]
    mean_bearing = float(np.mean(moving_bearings))
    return True, mean_bearing




def _detect_moving_any(pose):
    """
    Scan both front and back 2‑D lidars, return (has_moving, mean_bearing) for points that satisfy:
      * distance ≤ _DETECT_DIST,
      * absolute bearing ≤ _DETECT_ANGLE,
      * NOT inside any static obstacle (boxes, walls, fixed obstacles).
    Detection is valid only if at least _MIN_MOVING_POINTS points are classified as moving.
    """
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return False, 0.0

    pts = np.vstack([pts_front, pts_back]) if pts_front.shape[0] and pts_back.shape[0] else (
        pts_front if pts_front.shape[0] else pts_back
    )

    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])

    # Apply both distance and angle filters using the updated constants
    sector_mask = (dist <= _DETECT_DIST) & (np.abs(bearing) <= _DETECT_ANGLE)
    if not np.any(sector_mask):
        return False, 0.0

    cand_robot = pts[sector_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]

    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0

    if moving_mask.sum() < _MIN_MOVING_POINTS:
        return False, 0.0

    moving_bearings = bearing[sector_mask][moving_mask]
    mean_bearing = float(np.mean(moving_bearings))
    return True, mean_bearing




_DETECT_DIST = 2.5                     # [m] moving‑object detection radius (expanded)


_DETECT_ANGLE = np.pi                 # [rad] full 180° detection cone




_AVOID_DURATION   = 1.0   # [s] stop time before backup in close‑range avoidance


_AVOID_ROT_SPEED = 0.6                # [rad/s] rotation speed during avoidance




def _moving_points(pose):
    """Collect candidate moving points from front and back 2D lidars,
    filter out static obstacles, and return them in world coordinates.
    Only points within the close‑range distance are kept (no angular filter)."""
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return np.zeros((0, 2))

    pts = np.vstack([pts_front, pts_back]) if pts_front.shape[0] and pts_back.shape[0] else (
        pts_front if pts_front.shape[0] else pts_back
    )

    # distance in robot frame
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)

    # keep points within close detection distance (ignore angle)
    close_mask = dist <= _CLOSE_DETECT_DIST
    if not np.any(close_mask):
        return np.zeros((0, 2))

    cand_robot = pts[close_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]

    # keep only points not inside any static obstacle
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    return cand_world[moving_mask]




_MIN_MOVING_POINTS = 3                 # require at least three lidar points


_AVOID_TURN_TIME = 1.0                # [s] duration of rotation after stop_wait


_AVOID_MOVE_TIME = None


_AVOID_MOVE_SPEED = None




_BACK_TIME = 1.5                       # longer backup to clear moving object


_BACK_SPEED = -0.3                     # [m/s] backward speed during avoidance


_ROT_TIME = 1.0   # [s] rotation time during avoidance


_MAX_AVOID_COUNT = 3                   # maximum back‑up/rotate cycles per object


_LOW_SPEED_MIN = 0.3                   # [m/s] minimum forward speed after repeated avoidance


_LOW_SPEED_MAX = 0.6                   # [m/s] maximum forward speed when in low‑speed mode




_CLEAR_TIME = 1.0                     # [s] 원본 값 (현재 사용되지 않음)


_LATERAL_OFFSET = 2.0                  # [m] 회피 시 반대 방향 임시 경유점 거리




_AVOID_COOLDOWN   = 0.2   # [s] cooldown after a full avoidance cycle




_MAX_SPEED = 0.8                       # [m/s] maximum forward speed




_CLOSE_DETECT_DIST = 1.5   # [m] close‑range stop/rotate trigger (≥1.5 m as required)




_WAIT_CLEAR_TIME = 0.5                # [s] wait time after rotation until moving objects are cleared




_REPULSION_WEIGHT = 0.5               # weight of repulsive direction vs. attractive


_KP_W = 5.0                           # proportional gain for angular control


def _moving_points_repulsion(pose):
    """World points of moving objects within detection range (_DETECT_DIST)."""
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return np.zeros((0, 2))
    pts = np.vstack([pts_front, pts_back])
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    mask = dist <= _DETECT_DIST
    if not np.any(mask):
        return np.zeros((0, 2))
    cand_robot = pts[mask]
    cand_world = to_world(cand_robot, pose)[:, :2]
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    return cand_world[moving_mask]


def _repulsive_vector(pose, points):
    """Unit vector pointing away from moving points (weighted by inverse‑square distance)."""
    if points.shape[0] == 0:
        return np.zeros(2)
    robot_xy = np.array(pose[:2])
    vecs = robot_xy - points                     # away vectors
    dists = np.linalg.norm(vecs, axis=1)
    dists = np.maximum(dists, 0.05)              # avoid division by zero
    weights = 1.0 / (dists ** 2)
    rep = np.sum(vecs * weights[:, None], axis=0)
    norm = np.linalg.norm(rep)
    if norm < 1e-6:
        return np.zeros(2)
    return rep / norm


# ========================= [END EDIT REGION] ===============================
