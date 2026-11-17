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
    """Dynamic‑Window‑Approach controller with waypoint tracking and
    stop‑and‑wait fallback when no safe motion is found."""
    x, y, yaw = pose

    # ---- persistent state -------------------------------------------------
    if not hasattr(controller, "idx"):
        controller.idx = 0               # waypoint index
        controller.last_rot_dir = 1.0    # direction for fallback rotation (kept for critical stop)
        controller.wait_start = None     # timestamp when we entered wait‑stop
        controller.max_speed = _MAX_SPEED
        controller.max_rot = _MAX_ROT

    # ---- wait after dead‑end (no safe candidate) -------------------------
    if controller.wait_start is not None:
        if t - controller.wait_start < 1.0:
            return 0.0, 0.0, False
        else:
            controller.wait_start = None   # end waiting, resume normal planning

    # ---- waypoint handling ------------------------------------------------
    if controller.idx < len(_WAYPOINTS):
        wp_x, wp_y = _WAYPOINTS[controller.idx]
    else:
        wp_x, wp_y = GOAL[0], GOAL[1]

    # advance to next waypoint when close enough
    dx_wp, dy_wp = wp_x - x, wp_y - y
    dist_wp = math.hypot(dx_wp, dy_wp)
    if controller.idx < len(_WAYPOINTS) and dist_wp < 0.4:
        controller.idx += 1
        if controller.idx < len(_WAYPOINTS):
            wp_x, wp_y = _WAYPOINTS[controller.idx]
        else:
            wp_x, wp_y = GOAL[0], GOAL[1]
        dx_wp, dy_wp = wp_x - x, wp_y - y
        dist_wp = math.hypot(dx_wp, dy_wp)

    # ---- final goal alignment --------------------------------------------
    if controller.idx == len(_WAYPOINTS) and dist_wp < 0.3:
        yaw_err = _wrap_angle(GOAL[2] - yaw)
        if abs(yaw_err) < 0.05:
            return 0.0, 0.0, True
        w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
        return 0.0, w, False

    # ---- urgent critical stop (very close moving object) -----------------
    critical, crit_bearing = _detect_critical(pose)
    if critical:
        # rotate away from the detected bearing
        dir_sign = np.sign(crit_bearing) if crit_bearing != 0 else controller.last_rot_dir
        controller.last_rot_dir = dir_sign
        return 0.0, _AVOID_ROT_SPEED * dir_sign, False

    # ---- moving‑object perception -----------------------------------------
    moving_pts, moving_vels = _estimate_moving_objects(pose, t)

    # ---- DWA candidate evaluation -----------------------------------------
    dt = 1.0 / 60.0
    steps = max(1, int(_AVOID_HORIZON / dt))
    best_score = -float('inf')
    best_v, best_w = 0.0, 0.0
    wp = (wp_x, wp_y)

    for v, w in _sample_velocities():
        traj = _simulate_trajectory(pose, v, w, dt, steps)
        if not _is_trajectory_safe(traj, moving_pts, moving_vels, dt):
            continue
        score = _score_candidate(v, w, traj, wp)
        if score > best_score:
            best_score, best_v, best_w = score, v, w

    # ---- fallback when no safe motion is found ----------------------------
    if best_score == -float('inf'):
        # enter stop‑and‑wait state
        if controller.wait_start is None:
            controller.wait_start = t
        return 0.0, 0.0, False

    # ---- command the best safe candidate ----------------------------------
    return float(best_v), float(best_w), False




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




_REPULSION_WEIGHT = 1.5  # increased from 1.0


_KP_W = 5.0                       # proportional gain for angular control


def _moving_points_repulsion(pose):
    """Deprecated: function removed; returns empty array."""
    return np.zeros((0, 2))


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
    """Return a lateral bypass waypoint 2 m offset from the robot.
    Chooses left or right side of the object's bearing that is free of static obstacles
    and has a clear line from the robot. If both sides are feasible, picks the side with
    larger clearance to static obstacles."""
    x, y, yaw = pose
    world_angle = yaw + bearing
    offset = 2.0  # fixed lateral offset (meters)

    # left side (90° CCW from object's direction)
    dx_left = -math.sin(world_angle)
    dy_left =  math.cos(world_angle)
    cand_left = (x + offset * dx_left, y + offset * dy_left)

    # right side (90° CW from object's direction)
    dx_right =  math.sin(world_angle)
    dy_right = -math.cos(world_angle)
    cand_right = (x + offset * dx_right, y + offset * dy_right)

    left_ok = not _in_static_obstacle(cand_left) and _line_is_clear((x, y), cand_left)
    right_ok = not _in_static_obstacle(cand_right) and _line_is_clear((x, y), cand_right)

    if left_ok and right_ok:
        d_left = _distance_to_nearest_static(cand_left)
        d_right = _distance_to_nearest_static(cand_right)
        return cand_left if d_left >= d_right else cand_right
    elif left_ok:
        return cand_left
    elif right_ok:
        return cand_right
    else:
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




_V_SAMPLES = 5                # number of forward speed samples


_W_SAMPLES = 7                # number of angular speed samples


_AVOID_HORIZON = 0.5          # seconds to look ahead when testing candidates


_SAFE_DIST = 0.5              # minimum clearance to moving objects (m)




def _estimate_moving_objects(pose, t):
    """Estimate per‑point velocities of moving objects using two consecutive lidar scans.
    Returns (moving_pts, moving_vels) where:
      moving_pts  – world coordinates of moving points (filtered by static obstacles)
      moving_vels – Nx2 array of velocity vectors (m/s) for each point.
    The function updates controller.prev_moving_pts and controller.t_last_est for the next call."""
    moving_pts = _moving_points(pose)   # world points within _DETECT_DIST
    moving_vels = np.zeros((0, 2))

    # Use stored previous scan to estimate per‑point velocity
    if moving_pts.shape[0] > 0 and hasattr(controller, "prev_moving_pts") and controller.prev_moving_pts is not None:
        dt = t - controller.t_last_est
        if dt > 1e-3:
            prev = controller.prev_moving_pts
            # nearest‑neighbor matching from current points to previous points
            # distance matrix (N_cur, N_prev)
            dmat = np.linalg.norm(moving_pts[:, None, :] - prev[None, :, :], axis=2)
            nearest_idx = np.argmin(dmat, axis=1)
            disp = moving_pts - prev[nearest_idx]
            moving_vels = disp / dt
        else:
            moving_vels = np.zeros_like(moving_pts)
    else:
        moving_vels = np.zeros((moving_pts.shape[0], 2))

    # Store for the next iteration
    controller.prev_moving_pts = moving_pts.copy() if moving_pts.shape[0] else None
    controller.t_last_est = t

    return moving_pts, moving_vels




_TTC_THRESH = 1.5  # [s] 충돌 예상 시 회피 시작 임계값


def _min_ttc(moving_pts, moving_vels, robot_xy):
    """Return the smallest time‑to‑collision (seconds) among all moving points.
    Robot is assumed stationary; only points moving toward the robot are considered."""
    if moving_pts.shape[0] == 0:
        return float('inf')
    if moving_vels.shape[0] == 0:
        return float('inf')

    r = moving_pts - robot_xy                     # (N,2) vector robot → point
    r_norm = np.linalg.norm(r, axis=1)
    r_norm = np.maximum(r_norm, 1e-6)            # avoid division by zero

    # closing speed = - (r · v) / |r|
    closing_speed = -np.einsum('ij,ij->i', r, moving_vels) / r_norm
    mask = closing_speed > 0.0
    if not np.any(mask):
        return float('inf')
    ttc = r_norm[mask] / closing_speed[mask]
    return float(np.min(ttc))


def _should_bypass(pose, t):
    """Determine whether a lateral bypass is required using TTC.
    Returns (need_bypass, moving_pts, moving_vels)."""
    moving_pts, moving_vels = _estimate_moving_objects(pose, t)
    robot_xy = np.array(pose[:2])
    ttc = _min_ttc(moving_pts, moving_vels, robot_xy)
    return ttc < _TTC_THRESH, moving_pts, moving_vels


def _get_moving_bearing(pose):
    """움직이는 물체들의 평균 베어링을 반환.
    _detect_moving_any 로 얻지 못하면 가장 가까운 점의 베어링을 사용."""
    has, mean_bearing = _detect_moving_any(pose)
    if has:
        return mean_bearing
    pts = _moving_points(pose)
    if pts.shape[0] == 0:
        return 0.0
    robot_xy = np.array(pose[:2])
    vec = pts - robot_xy
    idx = int(np.argmin(np.linalg.norm(vec, axis=1)))
    return math.atan2(vec[idx, 1], vec[idx, 0])




_ESCAPE_WEIGHT = 0.8  # balance between waypoint tracking and side‑steering


def _lateral_escape_vector(pose, moving_pts):
    """Return a unit vector perpendicular to the bearing of the nearest moving point.
    Chooses the side (left/right) with larger clearance from static obstacles.
    If no moving points are detected, returns a zero vector."""
    if moving_pts.shape[0] == 0:
        return np.zeros(2)

    robot_xy = np.array(pose[:2])
    # vector from robot to each moving point
    diffs = moving_pts - robot_xy
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = int(np.argmin(dists))
    bearing = math.atan2(diffs[nearest_idx, 1], diffs[nearest_idx, 0])

    # perpendicular unit vectors (left = +90°, right = -90°)
    left_vec = np.array([-math.sin(bearing),  math.cos(bearing)])
    right_vec = np.array([ math.sin(bearing), -math.cos(bearing)])

    # evaluate static clearance a short distance (0.5 m) along each side
    offset = 0.5
    left_pt = robot_xy + offset * left_vec
    right_pt = robot_xy + offset * right_vec
    left_clear = _distance_to_nearest_static(tuple(left_pt))
    right_clear = _distance_to_nearest_static(tuple(right_pt))

    # pick the side with larger clearance; if both are blocked, return zero
    if left_clear > 0.1 or right_clear > 0.1:
        return left_vec if left_clear >= right_clear else right_vec
    else:
        return np.zeros(2)




_MAX_ROT = 1.5


def _sample_velocities():
    """Generate a grid of (v, w) candidates for DWA."""
    vs = np.linspace(0.0, _MAX_SPEED, _V_SAMPLES)
    ws = np.linspace(-_MAX_ROT, _MAX_ROT, _W_SAMPLES)
    return [(float(v), float(w)) for v in vs for w in ws]


def _simulate_trajectory(pose, v, w, dt, steps):
    """Unicycle forward integration for a short horizon."""
    x, y, yaw = pose
    traj = [(x, y, yaw)]
    for _ in range(steps):
        if abs(w) < 1e-6:                     # straight line
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:                                 # exact circular arc
            x += (v / w) * (math.sin(yaw + w * dt) - math.sin(yaw))
            y += (v / w) * (-math.cos(yaw + w * dt) + math.cos(yaw))
            yaw = _wrap_angle(yaw + w * dt)
        traj.append((x, y, yaw))
    return traj


def _is_trajectory_safe(traj, moving_pts, moving_vels, dt):
    """Check static and predicted moving obstacles for a trajectory."""
    for i, (x, y, _) in enumerate(traj):
        # static obstacles (boxes, walls, fixed structures)
        if _in_static_obstacle((x, y)):
            return False
        # moving objects: predict linear motion over the same horizon
        if moving_pts.shape[0] > 0:
            pts_t = moving_pts + moving_vels * (i * dt)
            dists = np.linalg.norm(pts_t - np.array([x, y]), axis=1)
            if np.any(dists < _SAFE_DIST):
                return False
    return True


def _score_candidate(v, w, traj, wp):
    """Higher score for candidates that reduce distance to the waypoint
    and that move faster."""
    x0, y0 = traj[0][0], traj[0][1]
    xf, yf = traj[-1][0], traj[-1][1]
    dist_before = math.hypot(wp[0] - x0, wp[1] - y0)
    dist_after  = math.hypot(wp[0] - xf, wp[1] - yf)
    progress = dist_before - dist_after          # positive if we get closer
    return progress + 0.1 * v                    # modest bias toward speed


# ========================= [END EDIT REGION] ===============================
