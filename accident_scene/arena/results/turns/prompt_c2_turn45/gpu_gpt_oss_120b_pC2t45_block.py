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
    """TTC 기반 우회와 기존 repulsion·웨이포인트 추적을 결합한 상태머신."""
    x, y, yaw = pose

    # ---- persistent state 초기화 ----
    if not hasattr(controller, "state"):
        controller.state = "normal"          # normal | bypass
        controller.idx = 0
        controller.rot_end = 0.0
        controller.rot_dir = 0.0
        controller.bypass_target = None
        controller.prev_moving_pts = None
        controller.t_last_est = None

    # ---- 짧은 회전 처리 (critical 회피) ----
    if t < controller.rot_end:
        return 0.0, float(_AVOID_ROT_SPEED * controller.rot_dir), False

    # ---- 매우 근접 물체에 대한 즉시 정지·회전 (critical) ----
    critical, crit_bearing = _detect_critical(pose)
    if critical:
        controller.rot_end = t + 0.5               # 0.5 s 회전
        controller.rot_dir = np.sign(crit_bearing) if crit_bearing != 0 else 1.0
        return 0.0, float(_AVOID_ROT_SPEED * controller.rot_dir), False

    # ---- bypass 상태 처리 ----
    target_x, target_y = None, None
    if controller.state == "bypass":
        tx, ty = controller.bypass_target
        # TTC 안전성 검사: 필요 없으면 normal 로 복귀
        need_bypass, _, _ = _should_bypass(pose, t)
        if not need_bypass:
            controller.state = "normal"
            controller.bypass_target = None
        elif math.hypot(tx - x, ty - y) < _BYPASS_TOL:
            # 우회점 도달
            controller.state = "normal"
            controller.bypass_target = None
        else:
            target_x, target_y = tx, ty   # 현재 목표는 bypass 점

    # ---- normal 상태 (웨이포인트 추적) ----
    if controller.state != "bypass":
        # 현재 웨이포인트 선택
        if controller.idx < len(_WAYPOINTS):
            wp_x, wp_y = _WAYPOINTS[controller.idx]
        else:
            wp_x, wp_y = GOAL[0], GOAL[1]

        # 웨이포인트 도달 시 인덱스 전진
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

        # 목표에 거의 도착했을 때 정렬 및 종료
        if controller.idx == len(_WAYPOINTS) and dist_wp < 0.3:
            yaw_err = _wrap_angle(GOAL[2] - yaw)
            if abs(yaw_err) < 0.05:
                return 0.0, 0.0, True
            w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
            return 0.0, w, False

        # ---- TTC 기반 사전 회피 판단 ----
        need_bypass, _, _ = _should_bypass(pose, t)
        if need_bypass:
            bearing = _get_moving_bearing(pose)
            bypass = _compute_bypass_target(pose, bearing)
            if bypass is not None:
                controller.state = "bypass"
                controller.bypass_target = bypass
                # 즉시 정지, 회전 없이 bypass 로 이동
                return 0.0, 0.0, False
            else:
                # 우회점 생성 실패 시 간단 회전 회피
                controller.rot_end = t + 0.5
                controller.rot_dir = np.sign(bearing) if bearing != 0 else 1.0
                return 0.0, float(_AVOID_ROT_SPEED * controller.rot_dir), False

        # normal 상태에서 현재 목표는 웨이포인트
        if target_x is None:
            target_x, target_y = wp_x, wp_y

    # ---- 목표(웨이포인트 또는 bypass) 향한 이동 벡터 계산 ----
    dx, dy = target_x - x, target_y - y
    dist_target = math.hypot(dx, dy)

    # 움직이는 물체에 대한 repulsion 및 속도 스케일링
    moving_pts = _moving_points(pose)
    rep_vec = _repulsive_vector(pose, moving_pts)

    # 목표 방향 단위 벡터
    wp_dir = np.array([dx, dy]) / max(dist_target, 1e-6)

    # repulsion 과 목표 방향 결합
    dir_vec = wp_dir + _REPULSION_WEIGHT * rep_vec
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-6:
        return 0.0, 0.0, False
    dir_vec /= norm

    desired_yaw = math.atan2(dir_vec[1], dir_vec[0])
    yaw_err = _wrap_angle(desired_yaw - yaw)

    # 가장 가까운 움직이는 점 거리 → 전진 속도 스케일링
    if moving_pts.shape[0] > 0:
        robot_xy = np.array([x, y])
        nearest = float(np.min(np.linalg.norm(moving_pts - robot_xy, axis=1)))
    else:
        nearest = float('inf')
    v = _MAX_SPEED * np.clip(nearest / _SAFE_DIST, 0.0, 1.0)
    v = v * max(0.0, math.cos(yaw_err))

    w = _clamp(_KP_W * yaw_err, -1.5, 1.5)

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


_AVOID_HORIZON = 0.6          # seconds to look ahead when testing candidates


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


# ========================= [END EDIT REGION] ===============================
