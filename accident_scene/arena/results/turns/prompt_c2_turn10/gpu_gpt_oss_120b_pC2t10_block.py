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
    """Waypoint navigation with enhanced dynamic‑obstacle detection and
    multi‑stage avoidance.

    States:
        normal        – follow waypoints.
        avoid_stop    – initial stop for _AVOID_DURATION.
        avoid_wait    – stay stopped while a moving object remains in the detection zone;
                        resume only after it has been absent for _CLEAR_TIME.
        avoid_back    – back up for _BACK_TIME.
        avoid_rotate  – rotate for _ROT_TIME away from the object.
        avoid_lateral – navigate to a temporary waypoint offset opposite the object's bearing.
    """
    # One‑time initialisation
    if not hasattr(controller, "initialized"):
        controller.initialized = True
        controller.idx = 0                     # index in _WAYPOINTS
        controller.state = "normal"
        controller.avoid_count = 0
        controller.low_speed = False
        controller.last_moving_time = -float("inf")
        controller.wait_clear_end = 0.0
        controller.lateral_active = False
        controller.lateral_target = None
        controller.orig_idx = None
        controller.avoid_dir = 1.0
        controller.avoid_end = 0.0

    # Detect moving objects (skip detection while executing lateral avoidance)
    moving, bearing = _detect_moving_any(pose) if controller.state != "avoid_lateral" else (False, 0.0)

    # Record the most recent detection time
    if moving:
        controller.last_moving_time = t

    # --------------------------------------------------------------------
    # State machine for dynamic‑obstacle avoidance
    # --------------------------------------------------------------------
    if controller.state == "normal":
        if moving:
            if controller.avoid_count < _MAX_AVOID_COUNT:
                # Begin stop‑wait phase
                controller.avoid_count += 1
                controller.state = "avoid_stop"
                controller.avoid_end = t + _AVOID_DURATION
                # Rotate opposite to the object's bearing
                dir_sign = -np.sign(bearing)
                controller.avoid_dir = dir_sign if dir_sign != 0 else 1.0
                return 0.0, 0.0, False
            else:
                # Too many attempts – create a lateral waypoint opposite the object
                x, y, yaw = pose
                offset_angle = yaw + bearing + math.pi
                tx = x + _LATERAL_OFFSET * math.cos(offset_angle)
                ty = y + _LATERAL_OFFSET * math.sin(offset_angle)
                # Clamp target inside the interior rectangle to avoid walls
                wx_min, wx_max, wy_min, wy_max = WALLS
                tx = np.clip(tx, wx_min + 0.5, wx_max - 0.5)
                ty = np.clip(ty, wy_min + 0.5, wy_max - 0.5)
                controller.lateral_target = (tx, ty)
                controller.lateral_active = True
                controller.orig_idx = controller.idx
                controller.state = "avoid_lateral"
                controller.avoid_count = 0
                # Continue to navigation (target will be the lateral waypoint)
        else:
            # No moving object – reset avoidance counter
            controller.avoid_count = 0

    elif controller.state == "avoid_stop":
        if t < controller.avoid_end:
            return 0.0, 0.0, False
        # Transition to waiting‑until‑clear phase
        controller.state = "avoid_wait"
        controller.wait_clear_end = t + _CLEAR_TIME
        return 0.0, 0.0, False

    elif controller.state == "avoid_wait":
        if moving:
            # Object still present – extend clear timer
            controller.wait_clear_end = t + _CLEAR_TIME
            return 0.0, 0.0, False
        if t >= controller.wait_clear_end:
            # Object cleared long enough – back up
            controller.state = "avoid_back"
            controller.avoid_end = t + _BACK_TIME
            return _BACK_SPEED, 0.0, False
        return 0.0, 0.0, False

    elif controller.state == "avoid_back":
        if t < controller.avoid_end:
            return _BACK_SPEED, 0.0, False
        # Back‑up finished – start rotation
        controller.state = "avoid_rotate"
        controller.avoid_end = t + _ROT_TIME
        return 0.0, 0.0, False

    elif controller.state == "avoid_rotate":
        if t < controller.avoid_end:
            return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False
        # Rotation finished – resume normal navigation
        controller.state = "normal"
        # fall through to waypoint navigation

    # --------------------------------------------------------------------
    # Waypoint (or lateral) navigation
    # --------------------------------------------------------------------
    x, y, yaw = pose

    # Choose current navigation target
    if controller.lateral_active and controller.lateral_target is not None:
        tx, ty = controller.lateral_target
    else:
        if controller.idx >= len(_WAYPOINTS):
            # Goal alignment and termination
            e = _wrap_angle(GOAL[2] - yaw)
            w = _clamp(1.5 * e, -0.6, 0.6)
            if abs(e) < 0.05:
                return 0.0, 0.0, True
            return 0.0, w, False
        tx, ty = _WAYPOINTS[controller.idx]

    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # Advance to next waypoint when close enough (only for normal waypoints)
    if not controller.lateral_active:
        if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
            controller.idx += 1
            tx, ty = _WAYPOINTS[controller.idx]
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)
            e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # If lateral avoidance waypoint reached, resume original path
    if controller.lateral_active and dist < 0.35:
        controller.lateral_active = False
        controller.lateral_target = None
        controller.idx = controller.orig_idx if controller.orig_idx is not None else controller.idx
        controller.state = "normal"
        # Re‑compute target after returning to the original waypoint
        if controller.idx < len(_WAYPOINTS):
            tx, ty = _WAYPOINTS[controller.idx]
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)
            e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # Control law
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        base_v = min(1.5, 2.5 * dist + 0.5)
        if controller.low_speed:
            v = max(_LOW_SPEED_MIN, min(_LOW_SPEED_MAX, base_v))
        else:
            v = base_v
        w = _clamp(7.5 * e, -2.0, 2.0)

    # Trigger alignment after reaching the final waypoint
    if not controller.lateral_active and controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1

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




_DETECT_DIST = 2.5                     # [m] 확대된 탐지 거리


_DETECT_ANGLE = np.deg2rad(120.0)  # [rad] detection half‑plane ±120°




_AVOID_DURATION = 1.0                  # [s] 물체 감지 시 정지 지속 시간


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




_MIN_MOVING_POINTS = 1                # 최소 감지 포인트 수


_AVOID_TURN_TIME = 1.0                # [s] duration of rotation after stop_wait


_AVOID_MOVE_TIME = None


_AVOID_MOVE_SPEED = None




_BACK_TIME = 0.5                       # [s] time to back up after detection


_BACK_SPEED = -0.3                     # [m/s] backward speed during avoidance


_ROT_TIME = 1.0                        # [s] rotation time after backing up


_MAX_AVOID_COUNT = 3                   # maximum back‑up/rotate cycles per object


_LOW_SPEED_MIN = 0.3                   # [m/s] minimum forward speed after repeated avoidance


_LOW_SPEED_MAX = 0.6                   # [m/s] maximum forward speed when in low‑speed mode




_CLEAR_TIME = 1.0                      # [s] 물체 사라짐을 확인하는 대기 시간


_LATERAL_OFFSET = 2.0                  # [m] 회피 시 반대 방향 임시 경유점 거리


# ========================= [END EDIT REGION] ===============================
