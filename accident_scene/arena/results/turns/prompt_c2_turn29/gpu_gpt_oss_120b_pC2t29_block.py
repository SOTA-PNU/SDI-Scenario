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
    """Waypoint navigation with robust moving‑object detection and simple lateral bypass."""
    x, y, yaw = pose

    # ---------- one‑time initialization ----------
    if not hasattr(controller, "initialized"):
        controller.initialized = True
        controller.idx = 0                     # waypoint index
        controller.state = "normal"            # normal | avoid_stop | avoid_back | avoid_rotate | avoid_bypass
        controller.state_end = 0.0             # time when current state ends
        controller.avoid_dir = 1.0             # rotation direction (+1 left, -1 right)
        controller.avoid_bearing = 0.0         # bearing of detected moving object
        controller.avoid_count = 0             # number of avoidance attempts
        controller.low_speed = False           # switch to low‑speed after many avoids
        controller.bypass_target = None        # temporary lateral waypoint (x, y)
        controller.moving_counter = 0          # consecutive front‑cone detections

    # ---------- avoidance state machine ----------
    if controller.state != "normal":
        if controller.state == "avoid_stop":
            if t < controller.state_end:
                return 0.0, 0.0, False
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
            # after rotation, check if the object is still present
            still_close, _ = _detect_moving_front(pose)
            if not still_close:
                controller.state = "normal"
                controller.state_end = t + _AVOID_COOLDOWN
                return 0.0, 0.0, False

            # object persists → try lateral bypass
            angle_left = controller.avoid_bearing + np.pi / 2
            angle_right = controller.avoid_bearing - np.pi / 2
            candidates = [
                (x + _LATERAL_OFFSET * math.cos(angle_left),
                 y + _LATERAL_OFFSET * math.sin(angle_left)),
                (x + _LATERAL_OFFSET * math.cos(angle_right),
                 y + _LATERAL_OFFSET * math.sin(angle_right)),
                (x + _LATERAL_OFFSET * math.cos(controller.avoid_bearing + math.pi),
                 y + _LATERAL_OFFSET * math.sin(controller.avoid_bearing + math.pi)),
            ]
            target = None
            for cand in candidates:
                if _line_is_clear((x, y), cand):
                    target = cand
                    break
            if target is None:
                controller.state = "normal"
                controller.state_end = t + _AVOID_COOLDOWN
                return 0.0, 0.0, False

            controller.bypass_target = target
            controller.state = "avoid_bypass"
            return 0.0, 0.0, False

        if controller.state == "avoid_bypass":
            # navigate toward the selected bypass waypoint
            if controller.bypass_target is None:
                controller.state = "normal"
                controller.state_end = t + _AVOID_COOLDOWN
                return 0.0, 0.0, False

            tx, ty = controller.bypass_target
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

            if dist < 0.3:
                controller.bypass_target = None
                controller.state = "normal"
                controller.state_end = t + _AVOID_COOLDOWN
                return 0.0, 0.0, False

            # attraction to bypass point + repulsion from moving points
            att_dir = np.array([dx, dy]) / (dist + 1e-6)
            rep_pts = _moving_points_repulsion(pose)
            rep_dir = _repulsive_vector(pose, rep_pts)
            combined_dir = att_dir + _REPULSION_WEIGHT * rep_dir
            norm_comb = np.linalg.norm(combined_dir)
            if norm_comb:
                combined_dir = combined_dir / norm_comb

            desired_heading = math.atan2(combined_dir[1], combined_dir[0])
            heading_error = _wrap_angle(desired_heading - yaw)

            base_v = min(_MAX_SPEED, 2.5 * dist + 0.2)
            if controller.low_speed:
                base_v = _clamp(base_v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)

            if rep_pts.shape[0]:
                robot_xy = np.array([x, y])
                dists = np.linalg.norm(rep_pts - robot_xy, axis=1)
                min_dist = np.min(dists)
                factor = np.clip((min_dist - 0.3) / (_CLOSE_DETECT_DIST - 0.3), 0.0, 1.0)
                base_v *= factor

            v = base_v * max(0.0, math.cos(heading_error))
            w = _clamp(_KP_W * heading_error, -1.5, 1.5)
            return float(v), float(w), False

        # unexpected state → reset to normal
        controller.state = "normal"
        controller.state_end = t + _AVOID_COOLDOWN
        return 0.0, 0.0, False

    # ---------- normal navigation ----------
    # front‑cone moving‑object detection
    close, bearing = _detect_moving_front(pose)
    if close:
        controller.moving_counter = min(controller.moving_counter + 1, 10)
    else:
        controller.moving_counter = 0

    # start avoidance only after consecutive detections
    if controller.moving_counter >= 2 and t >= controller.state_end:
        dir_sign = -np.sign(bearing)
        if dir_sign == 0:
            dir_sign = 1.0
        controller.avoid_dir = dir_sign
        controller.avoid_bearing = bearing
        controller.state = "avoid_stop"
        controller.state_end = t + _AVOID_DURATION
        controller.avoid_count += 1
        if controller.avoid_count > _MAX_AVOID_COUNT:
            controller.low_speed = True
        controller.moving_counter = 0
        return 0.0, 0.0, False

    # waypoint handling
    if controller.idx < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[controller.idx]
    else:
        tx, ty = GOAL[0], GOAL[1]

    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)

    # advance to next waypoint
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
        w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
        return 0.0, w, False

    # attraction toward waypoint
    att_dir = np.array([dx, dy]) / (dist + 1e-6)

    # repulsion from moving points (longer range)
    rep_pts = _moving_points_repulsion(pose)
    rep_dir = _repulsive_vector(pose, rep_pts)

    combined_dir = att_dir + _REPULSION_WEIGHT * rep_dir
    norm_comb = np.linalg.norm(combined_dir)
    if norm_comb:
        combined_dir = combined_dir / norm_comb

    desired_heading = math.atan2(combined_dir[1], combined_dir[0])
    heading_error = _wrap_angle(desired_heading - yaw)

    base_v = min(_MAX_SPEED, 2.5 * dist + 0.2)
    if controller.low_speed:
        base_v = _clamp(base_v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)

    # speed scaling based on nearest moving point
    if rep_pts.shape[0]:
        robot_xy = np.array([x, y])
        dists = np.linalg.norm(rep_pts - robot_xy, axis=1)
        min_dist = np.min(dists)
        factor = np.clip((min_dist - 0.3) / (_CLOSE_DETECT_DIST - 0.3), 0.0, 1.0)
        base_v *= factor

    v = base_v * max(0.0, math.cos(heading_error))
    w = _clamp(_KP_W * heading_error, -1.5, 1.5)
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
    Requires at least _MIN_MOVING_CLOSE_POINTS points to reduce false positives."""
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
    if moving_mask.sum() < _MIN_MOVING_CLOSE_POINTS:
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




_DETECT_DIST = 2.0                     # [m] repulsion / any‑direction detection radius


_DETECT_ANGLE = np.pi                  # [rad] full 180° each side (covers 360° total)




_AVOID_DURATION = 0.5                  # [s] stop time before backing up


_AVOID_ROT_SPEED = 0.8                 # [rad/s] rotation speed during avoidance




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




_MIN_MOVING_POINTS = 1                 # require at least one lidar point to consider moving


_AVOID_TURN_TIME = 1.0                # [s] duration of rotation after stop_wait


_AVOID_MOVE_TIME = None


_AVOID_MOVE_SPEED = None




_BACK_TIME = 1.0                       # [s] backing‑up duration


_BACK_SPEED = -0.5                     # [m/s] backward speed during avoidance


_ROT_TIME = 1.0   # [s] rotation time during avoidance


_MAX_AVOID_COUNT = 5                   # after this many avoids switch to low‑speed mode


_LOW_SPEED_MIN = 0.2                   # [m/s] low‑speed mode minimum


_LOW_SPEED_MAX = 0.5                   # [m/s] low‑speed mode maximum




_CLEAR_TIME = 1.0                     # [s] 원본 값 (현재 사용되지 않음)


_LATERAL_OFFSET = 1.5                  # [m] lateral offset for bypass waypoint




_AVOID_COOLDOWN = 0.5                  # [s] pause before re‑checking for moving objects




_MAX_SPEED = 0.8                       # [m/s] maximum forward speed




_CLOSE_DETECT_DIST = 1.0               # [m] front‑cone stop‑detection distance




_WAIT_CLEAR_TIME = 0.5                     # [s] object must be absent this long before resuming




_REPULSION_WEIGHT = 3.0                # stronger repulsion from moving points


_KP_W = 5.0                            # proportional gain for angular control


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




_CLOSE_DETECT_ANGLE = np.deg2rad(45)   # [rad] front cone for immediate stop


_AVOID_ROT_TIME = 0.8                  # [s] rotation duration after backing up




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




_WAIT_MAX_TIME = 2.0                       # [s] max wait before attempting a lateral bypass


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




_MIN_MOVING_CLOSE_POINTS = 3           # require ≥3 lidar points to deem an object moving


# ========================= [END EDIT REGION] ===============================
