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
    """Simple waypoint navigation with immediate stop‑wait on moving object detection.
    When any moving object is detected within the detection range, the robot stops
    (v=0, w=0) and stays stopped until the object disappears for _WAIT_CLEAR_TIME
    seconds, after which a short cooldown is applied before resuming waypoint
    following."""
    x, y, yaw = pose

    # ---------- initialization ----------
    if not hasattr(controller, "state"):
        controller.state = "normal"        # normal | stop_wait
        controller.idx = 0                # waypoint index
        controller.state_end = 0.0        # cooldown end time
        controller.wait_clear_end = None  # time when clear period started

    # ---------- cooldown after a stop‑wait ----------
    if controller.state == "normal" and t < controller.state_end:
        return 0.0, 0.0, False

    # ---------- moving‑object detection ----------
    moving, _ = _detect_moving_any(pose)

    if controller.state == "normal":
        if moving:
            # enter stop‑wait state with a mandatory stop duration
            controller.state = "stop_wait"
            controller.state_end = t + _AVOID_DURATION
            controller.wait_clear_end = None
            return 0.0, 0.0, False

        # ---------- waypoint navigation ----------
        # choose current target (waypoint or final goal)
        if controller.idx < len(_WAYPOINTS):
            tx, ty = _WAYPOINTS[controller.idx]
        else:
            tx, ty = GOAL[0], GOAL[1]

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # advance to next waypoint when close enough
        if controller.idx < len(_WAYPOINTS) and dist < 0.4:
            controller.idx += 1
            if controller.idx < len(_WAYPOINTS):
                tx, ty = _WAYPOINTS[controller.idx]
            else:
                tx, ty = GOAL[0], GOAL[1]
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

        # final goal alignment
        if controller.idx == len(_WAYPOINTS) and dist < 0.3:
            yaw_err = _wrap_angle(GOAL[2] - yaw)
            if abs(yaw_err) < 0.05:
                return 0.0, 0.0, True
            w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
            return 0.0, w, False

        # simple proportional steering towards target
        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - yaw)
        v = _MAX_SPEED * max(0.0, math.cos(heading_error))
        w = _clamp(_KP_W * heading_error, -1.5, 1.5)
        return float(v), float(w), False

    # ---------- stop‑wait state ----------
    # remain stopped while any moving object is present
    if moving:
        controller.wait_clear_end = None
        return 0.0, 0.0, False

    # no moving object detected – start/continue clear‑period timer
    if controller.wait_clear_end is None:
        controller.wait_clear_end = t + _WAIT_CLEAR_TIME
        return 0.0, 0.0, False

    if t >= controller.wait_clear_end:
        # clear period finished – return to normal navigation with a short cooldown
        controller.state = "normal"
        controller.state_end = t + _AVOID_COOLDOWN
        controller.wait_clear_end = None
        return 0.0, 0.0, False

    # still waiting for clear period
    return 0.0, 0.0, False




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
    """Detect moving objects in a narrow front cone for immediate avoidance.
    Returns (has_moving, mean_bearing) where bearing is relative to robot forward.
    """
    pts = lidar2d_points("front")
    if pts.shape[0] == 0:
        return False, 0.0

    # distance and bearing in robot frame
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])

    # keep points inside the close detection cone
    sector_mask = (dist <= _CLOSE_DETECT_DIST) & (np.abs(bearing) <= _CLOSE_DETECT_ANGLE)
    if not np.any(sector_mask):
        return False, 0.0

    cand_robot = pts[sector_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]

    # discard points that belong to static obstacles
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0

    # require a minimum number of moving points to avoid false positives
    if moving_mask.sum() < _MIN_MOVING_POINTS:
        return False, 0.0

    moving_bearings = bearing[sector_mask][moving_mask]
    mean_bearing = float(np.mean(moving_bearings))
    return True, mean_bearing




def _detect_moving_any(pose):
    """Scan both front and back 2‑D lidars, return (has_moving, mean_bearing) for points that satisfy:
       * distance ≤ _DETECT_DIST,
       * absolute bearing ≤ _DETECT_ANGLE,
       * NOT inside any static obstacle (boxes, walls, fixed obstacles).
       Detection is valid only if at least _MIN_MOVING_POINTS points are classified as moving."""
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

    # angle filter restored: only points within +/- _DETECT_ANGLE are considered
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




_DETECT_DIST = 2.0                # [m] radius for repulsion / general moving‑object detection


_DETECT_ANGLE = np.pi




_AVOID_DURATION = 1.0  # [s] stop duration before clear‑period check


_AVOID_ROT_SPEED = 1.2        # [rad/s] rotation speed during avoidance (≈68° in 1 s)




def _moving_points(pose):
    """Collect candidate moving points from front and back 2D lidars,
    filter out static obstacles, and return them in world coordinates.
    Points within _DETECT_DIST are kept (no angular filter)."""
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return np.zeros((0, 2))

    pts = (np.vstack([pts_front, pts_back])
           if pts_front.shape[0] and pts_back.shape[0]
           else (pts_front if pts_front.shape[0] else pts_back))

    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    mask = dist <= _DETECT_DIST
    if not np.any(mask):
        return np.zeros((0, 2))

    cand_robot = pts[mask]
    cand_world = to_world(cand_robot, pose)[:, :2]
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    return cand_world[moving_mask]




_MIN_MOVING_POINTS = 1  # require at least one lidar point to consider a moving object


_AVOID_TURN_TIME = 1.0                # [s] duration of rotation after stop_wait


_AVOID_MOVE_TIME = None


_AVOID_MOVE_SPEED = None




_BACK_TIME = 2.0  # [s] backing‑up duration


_BACK_SPEED = -0.5  # [m/s] backward speed when backing up after a long wait


_ROT_TIME = 1.0   # [s] rotation time during avoidance


_MAX_AVOID_COUNT = 3              # after this many stops switch to low‑speed mode


_LOW_SPEED_MIN = 0.2              # [m/s] lower bound in low‑speed mode


_LOW_SPEED_MAX = 0.5              # [m/s] upper bound in low‑speed mode




_CLEAR_TIME = 1.0                     # [s] 원본 값 (현재 사용되지 않음)


_LATERAL_OFFSET = 2.0                       # [m] lateral offset for bypass waypoint




_AVOID_COOLDOWN = 0.2  # allow prompt re‑detection after an avoidance maneuver




_MAX_SPEED = 0.8                  # [m/s] nominal forward speed




_CLOSE_DETECT_DIST = 1.5          # [m] distance for speed scaling near moving objects




_WAIT_CLEAR_TIME = 1.0  # [s] required clear period before resuming motion




_REPULSION_WEIGHT = 1.0  # reduced to avoid over‑steering


_KP_W = 5.0                       # proportional gain for angular control


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




_CLOSE_DETECT_ANGLE = np.pi                # [rad] full 360° detection for close‑range checks


_AVOID_ROT_TIME = 1.0          # [s] rotation duration during avoidance




_CLOSE_STOP_DIST = 1.0                     # [m] distance at which we stop for a moving object


_MAX_WAIT_TIME = 2.0  # [s] max wait in avoid_wait state


def _detect_moving_close(pose):
    """Detect moving objects within _CLOSE_STOP_DIST in a front cone.
    Returns (has_moving, mean_bearing) where bearing is robot‑frame angle."""
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return False, 0.0
    pts = np.vstack([pts_front, pts_back]) if pts_front.shape[0] and pts_back.shape[0] else (
        pts_front if pts_front.shape[0] else pts_back
    )
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    close_mask = dist <= _CLOSE_STOP_DIST
    if not np.any(close_mask):
        return False, 0.0
    cand_robot = pts[close_mask]
    bearing = np.arctan2(cand_robot[:, 1], cand_robot[:, 0])
    # restrict to front cone
    angle_mask = np.abs(bearing) <= _CLOSE_DETECT_ANGLE
    if not np.any(angle_mask):
        return False, 0.0
    cand_robot = cand_robot[angle_mask]
    bearing = bearing[angle_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0
    mean_bearing = float(np.mean(bearing[moving_mask]))
    return True, mean_bearing




_WAIT_TIME = 1.0                           # [s] clear‑period before resuming normal navigation




_WAIT_MAX_TIME = 5.0              # [s] max time to stay in wait state before backing up


_detect_moving_front_close = _detect_moving_front




def _line_is_clear(p_start, p_end, step=0.2):
    """Return True if a straight line from p_start to p_end, sampled every `step` meters,
    does not intersect any static obstacle."""
    p_start = np.asarray(p_start, dtype=np.float64)
    p_end = np.asarray(p_end, dtype=np.float64)
    vec = p_end - p_start
    dist = np.linalg.norm(vec)
    if dist == 0:
        return not _in_static_obstacle(p_start)
    direction = vec / dist
    n_steps = int(dist / step)
    for i in range(1, n_steps + 1):
        pt = p_start + direction * (i * step)
        if _in_static_obstacle(pt):
            return False
    return True




_DETECT_CONSECUTIVE = 2                     # require this many consecutive detections to trigger avoidance




_CRITICAL_DIST = 0.8  # [m] distance that triggers an immediate stop


_CRITICAL_ANGLE = np.deg2rad(30)  # [rad] front cone for the critical stop


_MIN_MOVING_POINTS_CRITICAL = 2   # require at least two points to consider a critical object


def _detect_critical(pose):
    """Detect moving objects within a close range in any direction.
    Returns (has_moving, mean_bearing) where bearing is relative to robot forward."""
    # combine front and back 2‑D lidar scans
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return False, 0.0

    pts = (np.vstack([pts_front, pts_back])
           if pts_front.shape[0] and pts_back.shape[0]
           else (pts_front if pts_front.shape[0] else pts_back))

    # distance filter – use the tighter critical distance to trigger a stop
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    mask = dist <= _CRITICAL_DIST          # _CRITICAL_DIST = 0.5 m
    if not np.any(mask):
        return False, 0.0

    cand_robot = pts[mask]
    bearing = np.arctan2(cand_robot[:, 1], cand_robot[:, 0])   # robot‑frame angle

    # world coordinates for static‑obstacle filtering
    cand_world = to_world(cand_robot, pose)[:, :2]
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0
    if moving_mask.sum() < _MIN_MOVING_POINTS:
        return False, 0.0

    mean_bearing = float(np.mean(bearing[moving_mask]))
    return True, mean_bearing




_BYPASS_TOL = 0.2  # [m] distance within which the bypass point is considered reached


def _compute_bypass_target(pose, bearing):
    """Return a lateral bypass waypoint using the first feasible offset from
    _LATERAL_OFFSETS. For each offset, evaluate left and right candidates;
    the first that is free of static obstacles and has a clear line from the
    robot is returned. If none are feasible, return None."""
    x, y, yaw = pose
    world_angle = yaw + bearing

    for offset in _LATERAL_OFFSETS:
        # left (90° CCW from object's direction)
        dx_left = -math.sin(world_angle)
        dy_left =  math.cos(world_angle)
        cand_left = (x + offset * dx_left, y + offset * dy_left)

        # right (90° CW from object's direction)
        dx_right =  math.sin(world_angle)
        dy_right = -math.cos(world_angle)
        cand_right = (x + offset * dx_right, y + offset * dy_right)

        left_ok = not _in_static_obstacle(cand_left) and _line_is_clear((x, y), cand_left)
        right_ok = not _in_static_obstacle(cand_right) and _line_is_clear((x, y), cand_right)

        if left_ok and right_ok:
            # choose side with larger clearance from static obstacles
            d_left = _distance_to_nearest_static(cand_left)
            d_right = _distance_to_nearest_static(cand_right)
            return cand_left if d_left >= d_right else cand_right
        elif left_ok:
            return cand_left
        elif right_ok:
            return cand_right
    return None




def _distance_to_nearest_static(p):
    """Return the Euclidean distance from point p to the closest static obstacle."""
    x, y = p
    min_dist = float('inf')

    # static boxes (MISSION["boxes"])
    for b in MISSION["boxes"]:
        xb = b["x"] - b["width"] / 2.0
        xe = b["x"] + b["width"] / 2.0
        yb = b["y"] - b["depth"] / 2.0
        ye = b["y"] + b["depth"] / 2.0
        dx = max(xb - x, 0.0, x - xe)
        dy = max(yb - y, 0.0, y - ye)
        d = math.hypot(dx, dy)
        if d < min_dist:
            min_dist = d

    # walls – interior rectangle defined by WALLS
    wx_min, wx_max, wy_min, wy_max = WALLS
    if wx_min <= x <= wx_max and wy_min <= y <= wy_max:
        # inside the arena; distance to the nearest wall side
        d_wall = min(x - wx_min, wx_max - x, y - wy_min, wy_max - y)
        if d_wall < min_dist:
            min_dist = d_wall
    else:
        # outside interior means we are at or beyond a wall → zero clearance
        return 0.0

    # fixed obstacles (axis‑aligned rectangles)
    for _, ox_min, ox_max, oy_min, oy_max in FIXED_OBSTACLES:
        dx = max(ox_min - x, 0.0, x - ox_max)
        dy = max(oy_min - y, 0.0, y - oy_max)
        d = math.hypot(dx, dy)
        if d < min_dist:
            min_dist = d

    return min_dist




_LATERAL_OFFSETS = [0.5, 1.0, 1.5, 2.0]




_MOVING_PERSIST_TIME = 3.0        # [s] treat moving object as present for this duration after last detection


# ========================= [END EDIT REGION] ===============================
