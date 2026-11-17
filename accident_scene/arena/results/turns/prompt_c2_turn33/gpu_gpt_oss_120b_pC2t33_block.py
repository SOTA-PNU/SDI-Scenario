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
    """Waypoint navigation with early moving‑object detection and lateral bypass."""
    x, y, yaw = pose

    # ---------- initialization ----------
    if not hasattr(controller, "state"):
        controller.state = "normal"          # normal | avoid_stop | avoid_back | avoid_rotate | avoid_bypass | avoid_return
        controller.state_end = 0.0
        controller.idx = 0
        controller.avoid_dir = 1.0
        controller.avoid_count = 0
        controller.low_speed = False
        controller.bypass_target = None
        controller.last_bearing = None      # bearing of the most recent detection

    # ---------- cooldown after any avoidance ----------
    if controller.state == "normal" and t < controller.state_end:
        return 0.0, 0.0, False

    # ---------- detection (only in normal state) ----------
    if controller.state == "normal":
        moving, bearing = _detect_critical(pose)
        if moving:
            # store bearing for later bypass computation
            controller.last_bearing = bearing
            # fallback rotation direction opposite to the object's bearing
            dir_sign = -np.sign(bearing)
            if dir_sign == 0:
                dir_sign = 1.0
            controller.avoid_dir = dir_sign
            controller.state = "avoid_stop"
            controller.state_end = t + _AVOID_DURATION
            controller.avoid_count += 1
            if controller.avoid_count > _MAX_AVOID_COUNT:
                controller.low_speed = True
            return 0.0, 0.0, False

    # ---------- avoidance state machine ----------
    if controller.state == "avoid_stop":
        if t < controller.state_end:
            return 0.0, 0.0, False
        # after stopping, try to create a lateral bypass waypoint
        if controller.last_bearing is not None:
            target = _compute_bypass_target(pose, controller.last_bearing)
            if target is not None:
                controller.bypass_target = target
                controller.state = "avoid_bypass"
                controller.state_end = t + 0.1   # brief pause before moving
                return 0.0, 0.0, False
        # fallback to original back‑rotate sequence if bypass not possible
        controller.state = "avoid_back"
        controller.state_end = t + _BACK_TIME
        return float(_BACK_SPEED), 0.0, False

    if controller.state == "avoid_back":
        if t < controller.state_end:
            return float(_BACK_SPEED), 0.0, False
        controller.state = "avoid_rotate"
        controller.state_end = t + _AVOID_ROT_TIME
        return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False

    if controller.state == "avoid_rotate":
        if t < controller.state_end:
            return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False
        controller.state = "normal"
        controller.state_end = t + _AVOID_COOLDOWN
        return 0.0, 0.0, False

    # ---------- bypass navigation ----------
    if controller.state == "avoid_bypass":
        # pause before starting movement
        if t < controller.state_end:
            return 0.0, 0.0, False
        tx, ty = controller.bypass_target
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if dist < _BYPASS_TOL:
            # reached bypass point, now head back to the original waypoint
            controller.state = "avoid_return"
            controller.state_end = 0.0
            return 0.0, 0.0, False
        # simple heading control toward bypass point
        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - yaw)
        v = _MAX_SPEED * max(0.0, math.cos(heading_error))
        if controller.low_speed:
            v = _clamp(v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)
        w = _clamp(_KP_W * heading_error, -1.5, 1.5)
        return float(v), float(w), False

    if controller.state == "avoid_return":
        # navigate back to the original waypoint (or goal)
        if controller.idx < len(_WAYPOINTS):
            tx, ty = _WAYPOINTS[controller.idx]
        else:
            tx, ty = GOAL[0], GOAL[1]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if dist < 0.3:
            controller.state = "normal"
            controller.state_end = t + _AVOID_COOLDOWN
            return 0.0, 0.0, False
        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - yaw)
        v = _MAX_SPEED * max(0.0, math.cos(heading_error))
        if controller.low_speed:
            v = _clamp(v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)
        w = _clamp(_KP_W * heading_error, -1.5, 1.5)
        return float(v), float(w), False

    # ---------- waypoint navigation (normal state) ----------
    if controller.idx < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[controller.idx]
    else:
        tx, ty = GOAL[0], GOAL[1]

    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)

    # advance to next waypoint
    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.4:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

    # final goal alignment
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.3:
        yaw_err = _wrap_angle(GOAL[2] - yaw)
        if abs(yaw_err) < 0.05:
            return 0.0, 0.0, True
        w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
        return 0.0, w, False

    # attraction direction toward waypoint
    att_dir = np.array([dx, dy]) / (dist + 1e-6)

    # repulsive direction from moving objects
    rep_pts = _moving_points(pose)
    rep_dir = _repulsive_vector(pose, rep_pts)

    combined_dir = att_dir + _REPULSION_WEIGHT * rep_dir
    norm_comb = np.linalg.norm(combined_dir)
    if norm_comb:
        combined_dir = combined_dir / norm_comb

    desired_heading = math.atan2(combined_dir[1], combined_dir[0])
    heading_error = _wrap_angle(desired_heading - yaw)

    # base forward speed
    base_v = _MAX_SPEED
    if controller.low_speed:
        base_v = _clamp(base_v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)

    # speed scaling based on heading error
    v = base_v * max(0.0, math.cos(heading_error))

    # additional slowdown when moving objects are very close
    if rep_pts.shape[0]:
        robot_xy = np.array([x, y])
        dists = np.linalg.norm(rep_pts - robot_xy, axis=1)
        min_dist = np.min(dists)
        if min_dist < _CLOSE_DETECT_DIST:
            factor = (min_dist - _CRITICAL_DIST) / (_CLOSE_DETECT_DIST - _CRITICAL_DIST)
            factor = np.clip(factor, 0.0, 1.0)
            v *= factor

    # angular velocity command
    w = _clamp(_KP_W * heading_error, -1.5, 1.5)

    # safety: prevent stepping into a static obstacle
    step = 0.2
    next_x = x + step * math.cos(yaw)
    next_y = y + step * math.sin(yaw)
    if _in_static_obstacle((next_x, next_y)):
        return 0.0, 0.0, False

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




_AVOID_DURATION = 0.8                      # [s] stop duration before backing up


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




_BACK_TIME = 1.0                           # [s] backing‑up duration


_BACK_SPEED = -0.3                # [m/s] backward speed when backing up after a long wait


_ROT_TIME = 1.0   # [s] rotation time during avoidance


_MAX_AVOID_COUNT = 3              # after this many stops switch to low‑speed mode


_LOW_SPEED_MIN = 0.2              # [m/s] lower bound in low‑speed mode


_LOW_SPEED_MAX = 0.5              # [m/s] upper bound in low‑speed mode




_CLEAR_TIME = 1.0                     # [s] 원본 값 (현재 사용되지 않음)


_LATERAL_OFFSET = 2.0                       # [m] lateral offset for bypass waypoint




_AVOID_COOLDOWN = 0.2  # allow prompt re‑detection after an avoidance maneuver




_MAX_SPEED = 0.8                  # [m/s] nominal forward speed




_CLOSE_DETECT_DIST = 1.5          # [m] distance for speed scaling near moving objects




_WAIT_CLEAR_TIME = 0.5            # [s] required clear period before resuming motion




_REPULSION_WEIGHT = 8.0           # weight of the repulsive vector from moving objects


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


_MAX_WAIT_TIME = 5.0    # [s] avoid_wait 상태에서 최대 대기 시간 (bypass 전환 기준)


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




_CRITICAL_DIST = 0.5              # [m] distance that triggers an immediate stop


_CRITICAL_ANGLE = np.deg2rad(30)  # [rad] front cone for the critical stop


_MIN_MOVING_POINTS_CRITICAL = 2   # require at least two points to consider a critical object


def _detect_critical(pose):
    """Detect moving objects in a front cone of +/-45° within _DETECT_DIST.
    Returns (has_moving, mean_bearing) where bearing is relative to robot forward."""
    pts = lidar2d_points("front")
    if pts.shape[0] == 0:
        return False, 0.0

    # distance and bearing in robot frame
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])

    # keep points inside detection cone and distance limit
    cone_mask = (dist <= _DETECT_DIST) & (np.abs(bearing) <= np.deg2rad(45))
    if not np.any(cone_mask):
        return False, 0.0

    cand_robot = pts[cone_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]

    # discard points that belong to static obstacles
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0
    if moving_mask.sum() < _MIN_MOVING_POINTS:
        return False, 0.0

    mean_bearing = float(np.mean(bearing[cone_mask][moving_mask]))
    return True, mean_bearing




_BYPASS_TOL = 0.2  # [m] distance within which the bypass point is considered reached


def _compute_bypass_target(pose, bearing):
    """Return a lateral bypass waypoint offset by _LATERAL_OFFSET to the left or right
    of the detected moving object's bearing. Only left/right perpendicular candidates
    are considered; the opposite direction is deliberately omitted."""
    x, y, yaw = pose
    world_angle = yaw + bearing

    # Left perpendicular (90° CCW from object's direction)
    dx_left = -math.sin(world_angle)
    dy_left =  math.cos(world_angle)
    cand_left = (x + _LATERAL_OFFSET * dx_left, y + _LATERAL_OFFSET * dy_left)
    if not _in_static_obstacle(cand_left) and _line_is_clear((x, y), cand_left):
        return cand_left

    # Right perpendicular (90° CW from object's direction)
    dx_right =  math.sin(world_angle)
    dy_right = -math.cos(world_angle)
    cand_right = (x + _LATERAL_OFFSET * dx_right, y + _LATERAL_OFFSET * dy_right)
    if not _in_static_obstacle(cand_right) and _line_is_clear((x, y), cand_right):
        return cand_right

    # No viable lateral bypass point found
    return None


# ========================= [END EDIT REGION] ===============================
