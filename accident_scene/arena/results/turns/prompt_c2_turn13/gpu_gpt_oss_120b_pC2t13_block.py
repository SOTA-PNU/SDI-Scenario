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
    """Waypoint navigation with early moving‑object detection and simple stop‑rotate avoidance."""
    # One‑time initialisation
    if not hasattr(controller, "initialized"):
        controller.initialized = True
        controller.idx = 0                     # index in _WAYPOINTS
        controller.state = "normal"            # "normal", "avoid_stop", "avoid_rotate"
        controller.avoid_end = 0.0
        controller.avoid_dir = 1.0
        controller.consec_detect = 0
        controller.avoid_cooldown_end = 0.0

    # Detect moving objects only in normal state
    moving, bearing = _detect_moving_any(pose) if controller.state == "normal" else (False, 0.0)

    # Update consecutive detection counter
    if moving:
        controller.consec_detect += 1
    else:
        controller.consec_detect = 0

    # --------------------------------------------------------------------
    # Simple avoidance state machine
    # --------------------------------------------------------------------
    if controller.state == "normal":
        # Trigger avoidance after three consecutive detections and cooldown elapsed
        if moving and controller.consec_detect >= 3 and t >= controller.avoid_cooldown_end and controller.idx < len(_WAYPOINTS) - 1:
            controller.state = "avoid_stop"
            controller.avoid_end = t + _AVOID_DURATION
            # Rotate opposite to object's bearing
            dir_sign = -np.sign(bearing)
            controller.avoid_dir = dir_sign if dir_sign != 0 else 1.0
            return 0.0, 0.0, False

    elif controller.state == "avoid_stop":
        if t < controller.avoid_end:
            return 0.0, 0.0, False
        # Transition to rotate
        controller.state = "avoid_rotate"
        controller.avoid_end = t + _ROT_TIME
        return 0.0, controller.avoid_dir * _AVOID_ROT_SPEED, False

    elif controller.state == "avoid_rotate":
        if t < controller.avoid_end:
            return 0.0, controller.avoid_dir * _AVOID_ROT_SPEED, False
        # Rotation finished, resume normal navigation
        controller.state = "normal"
        controller.avoid_cooldown_end = t + _AVOID_COOLDOWN
        controller.consec_detect = 0
        # fall through to waypoint navigation

    # --------------------------------------------------------------------
    # Waypoint navigation (no lateral avoidance)
    # --------------------------------------------------------------------
    x, y, yaw = pose

    # Current navigation target
    if controller.idx < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[controller.idx]
    else:
        tx, ty = GOAL[0], GOAL[1]

    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # Final waypoint: align with goal orientation and finish
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        e_goal = _wrap_angle(GOAL[2] - yaw)
        w_align = _clamp(1.5 * e_goal, -0.6, 0.6)
        if abs(e_goal) < 0.05:
            return 0.0, 0.0, True
        else:
            return 0.0, w_align, False

    # Advance to next waypoint when close enough
    if dist < 0.35 and controller.idx < len(_WAYPOINTS) - 1:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # Proportional control law
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        base_v = min(_MAX_SPEED, 2.5 * dist + 0.5)
        v = base_v
        w = _clamp(7.5 * e, -2.0, 2.0)

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




_DETECT_DIST = 1.2                     # [m] detection distance reduced to avoid false triggers


_DETECT_ANGLE = np.deg2rad(30.0)       # [rad] ±30° detection cone




_AVOID_DURATION = 1.0                  # [s] stop duration before rotating


_AVOID_ROT_SPEED = 0.6                # [rad/s] rotation speed during avoidance




def _moving_points(pose):
    """Collect candidate moving points from front and back 2D lidars,
    filter out static obstacles, and return them in world coordinates."""
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return np.zeros((0, 2))

    pts = np.vstack([pts_front, pts_back]) if pts_front.shape[0] and pts_back.shape[0] else (
        pts_front if pts_front.shape[0] else pts_back
    )

    # distance & bearing in robot frame
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])

    # sector filter (same as detection)
    sector_mask = (dist <= _DETECT_DIST) & (np.abs(bearing) <= _DETECT_ANGLE)
    if not np.any(sector_mask):
        return np.zeros((0, 2))

    cand_robot = pts[sector_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]

    # keep only points not inside any static obstacle
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    return cand_world[moving_mask]




_MIN_MOVING_POINTS = 3                # 최소 감지 포인트 수 (연속 감지를 위해)


_AVOID_TURN_TIME = 1.0                # [s] duration of rotation after stop_wait


_AVOID_MOVE_TIME = None


_AVOID_MOVE_SPEED = None




_BACK_TIME = 0.3                       # [s] backup time (unused after simplification)


_BACK_SPEED = -0.3                     # [m/s] backward speed during avoidance


_ROT_TIME = 0.5                        # [s] rotation time after stop


_MAX_AVOID_COUNT = 3                   # maximum back‑up/rotate cycles per object


_LOW_SPEED_MIN = 0.3                   # [m/s] minimum forward speed after repeated avoidance


_LOW_SPEED_MAX = 0.6                   # [m/s] maximum forward speed when in low‑speed mode




_CLEAR_TIME = 1.0                      # [s] (kept for compatibility, not used in simplified logic)


_LATERAL_OFFSET = 2.0                  # [m] 회피 시 반대 방향 임시 경유점 거리




_AVOID_COOLDOWN = 0.2                  # [s] brief ignore period after avoidance




_MAX_SPEED = 0.8                       # [m/s] maximum forward speed


# ========================= [END EDIT REGION] ===============================
