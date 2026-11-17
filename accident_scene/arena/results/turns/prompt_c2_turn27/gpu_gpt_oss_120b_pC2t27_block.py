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
    """Waypoint tracking with multi‑stage avoidance (stop → back → rotate → bypass)."""
    x, y, yaw = pose

    # ---------- 초기화 ----------
    if not hasattr(controller, "initialized"):
        controller.initialized = True
        controller.idx = 0                     # 현재 waypoint 인덱스
        controller.state = "normal"            # normal | avoid_stop | avoid_back | avoid_rotate | avoid_bypass
        controller.state_end = 0.0             # 상태 전환 시점 (시간)
        controller.avoid_dir = 1.0             # 회피 회전 방향 (+1 left, -1 right)
        controller.avoid_bearing = 0.0         # 감지된 물체 베어링 (rad)
        controller.avoid_count = 0             # 회피 횟수
        controller.low_speed = False           # 반복 회피 시 저속 전환 플래그
        controller.bypass_target = None        # 우회용 임시 목표점 (x, y)

    # ---------- 회피 쿨다운 중 정지 ----------
    if controller.state == "normal" and t < controller.state_end:
        return 0.0, 0.0, False

    # ---------- 근접 물체 감지 ----------
    close, bearing = _detect_moving_close(pose)

    # ---------- 상태 머신 ----------
    if controller.state == "normal":
        # 근접 물체 감지 → 회피 시작
        if close:
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
            return 0.0, 0.0, False

        # ---------- 웨이포인트 추적 ----------
        if controller.idx < len(_WAYPOINTS):
            tx, ty = _WAYPOINTS[controller.idx]
        else:
            tx, ty = GOAL[0], GOAL[1]

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # 다음 웨이포인트 전환
        if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
            controller.idx += 1
            tx, ty = _WAYPOINTS[controller.idx]
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

        # 최종 목표 정렬
        if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
            yaw_err = _wrap_angle(GOAL[2] - yaw)
            if abs(yaw_err) < 0.05:
                return 0.0, 0.0, True
            w = _clamp(_KP_W * yaw_err, -0.6, 0.6)
            return 0.0, w, False

        # 유인 벡터
        att_dir = np.array([dx, dy]) / (dist + 1e-6)

        # 이동 물체에 대한 반발 벡터
        rep_pts = _moving_points(pose)
        rep_dir = _repulsive_vector(pose, rep_pts)

        combined_dir = att_dir + _REPULSION_WEIGHT * rep_dir
        norm_comb = np.linalg.norm(combined_dir)
        if norm_comb:
            combined_dir = combined_dir / norm_comb

        desired_heading = math.atan2(combined_dir[1], combined_dir[0])
        heading_error = _wrap_angle(desired_heading - yaw)

        # 기본 전진 속도
        base_v = min(_MAX_SPEED, 2.5 * dist + 0.2)
        if controller.low_speed:
            base_v = _clamp(base_v, _LOW_SPEED_MIN, _LOW_SPEED_MAX)

        # 가까운 이동 물체에 따라 속도 스케일링
        if rep_pts.shape[0]:
            robot_xy = np.array([x, y])
            dists = np.linalg.norm(rep_pts - robot_xy, axis=1)
            min_dist = np.min(dists)
            factor = np.clip((min_dist - 0.3) / (_CLOSE_DETECT_DIST - 0.3), 0.0, 1.0)
            base_v *= factor

        v = base_v * max(0.0, math.cos(heading_error))
        w = _clamp(_KP_W * heading_error, -1.5, 1.5)
        return float(v), float(w), False

    elif controller.state == "avoid_stop":
        # 정지 유지
        if t < controller.state_end:
            return 0.0, 0.0, False
        # 뒤로 물러나기
        controller.state = "avoid_back"
        controller.state_end = t + _BACK_TIME
        return float(_BACK_SPEED), 0.0, False

    elif controller.state == "avoid_back":
        # 후진 지속
        if t < controller.state_end:
            return float(_BACK_SPEED), 0.0, False
        # 회전 단계 진입
        controller.state = "avoid_rotate"
        controller.state_end = t + _AVOID_ROT_TIME
        return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False

    elif controller.state == "avoid_rotate":
        # 회전 지속
        if t < controller.state_end:
            return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False
        # 회전 종료 후 물체 존재 여부 확인
        still_close, _ = _detect_moving_close(pose)
        if not still_close:
            # 물체가 사라졌으면 정상 복귀 (쿨다운 적용)
            controller.state = "normal"
            controller.state_end = t + _AVOID_COOLDOWN
            return 0.0, 0.0, False

        # 물체가 여전히 있으면 측면 우회 경로 생성 (±90°)
        angle1 = controller.avoid_bearing + np.pi / 2
        angle2 = controller.avoid_bearing - np.pi / 2
        cand1 = (x + _LATERAL_OFFSET * math.cos(angle1),
                 y + _LATERAL_OFFSET * math.sin(angle1))
        cand2 = (x + _LATERAL_OFFSET * math.cos(angle2),
                 y + _LATERAL_OFFSET * math.sin(angle2))

        if not _in_static_obstacle(cand1):
            target = cand1
        elif not _in_static_obstacle(cand2):
            target = cand2
        else:
            # 양쪽 모두 막혀 있으면 반대 방향 사용
            opp_angle = controller.avoid_bearing + math.pi
            target = (x + _LATERAL_OFFSET * math.cos(opp_angle),
                      y + _LATERAL_OFFSET * math.sin(opp_angle))

        controller.bypass_target = target
        controller.state = "avoid_bypass"
        return 0.0, 0.0, False

    elif controller.state == "avoid_bypass":
        # 우회 목표점으로 이동
        if controller.bypass_target is None:
            controller.state = "normal"
            controller.state_end = t + _AVOID_COOLDOWN
            return 0.0, 0.0, False

        tx, ty = controller.bypass_target
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        if dist < 0.3:
            # 우회점 도착 → 정상 복귀
            controller.bypass_target = None
            controller.state = "normal"
            controller.state_end = t + _AVOID_COOLDOWN
            return 0.0, 0.0, False

        # 우회 중에도 repulsion 적용
        att_dir = np.array([dx, dy]) / (dist + 1e-6)
        rep_pts = _moving_points(pose)
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

    # ---------- fallback ----------
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




_DETECT_DIST = 2.0                     # [m] detection radius for repulsion / any detection


_DETECT_ANGLE = np.pi




_AVOID_DURATION = 2.0        # [s] 정지 후 후진 시작 전 대기 시간 (연장)


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




_BACK_TIME = 1.0             # [s] 물체를 피하기 위한 후진 지속 시간 (연장)


_BACK_SPEED = -0.3                     # [m/s] backward speed during avoidance


_ROT_TIME = 1.0   # [s] rotation time during avoidance


_MAX_AVOID_COUNT = 5                        # 회피 횟수가 이 값을 초과하면 저속 모드 전환


_LOW_SPEED_MIN = 0.2                      # [m/s] 반복 회피 후 최소 전진 속도


_LOW_SPEED_MAX = 0.5                      # [m/s] 반복 회피 후 최대 전진 속도




_CLEAR_TIME = 1.0                     # [s] 원본 값 (현재 사용되지 않음)


_LATERAL_OFFSET = 3.0                      # [m] lateral offset for bypass waypoint




_AVOID_COOLDOWN = 0.3                     # [s] 회피 후 재감지 방지 쿨다운




_MAX_SPEED = 0.8                       # [m/s] maximum forward speed




_CLOSE_DETECT_DIST = 1.5                  # [m] 이동 물체 반발·속도 스케일링 거리 (축소)




_WAIT_CLEAR_TIME = 0.5                 # [s] wait after rotation until moving objects cleared




_REPULSION_WEIGHT = 1.2      # 이동 물체에 대한 반발 가중치 (증가)


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




_CLOSE_DETECT_ANGLE = np.pi  # [rad] 전방 근접 감지 각도 (전체)


_AVOID_ROT_TIME = 0.6                     # [s] 물체 회피 회전 지속 시간




_CLOSE_STOP_DIST = 1.5       # [m] 즉시 정지·회피 트리거 거리 (확대)


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


# ========================= [END EDIT REGION] ===============================
