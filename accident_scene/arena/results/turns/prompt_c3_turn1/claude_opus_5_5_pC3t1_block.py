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
    """경유점 추종 + 2D 라이다 기반 움직이는 물체 안전 계층(감속·정지·회전 정지·후진)."""
    if not hasattr(controller, "idx"):
        controller.idx = 0
        controller.retreat = False
        controller.ready = 0

    x, y, yaw = pose
    n = len(_WAYPOINTS)

    # ---- 센서: 움직이는 물체 감지 ----
    movers, raw, ok = _movers_robot_frame(pose)
    controller.ready = controller.ready + 1 if ok else 0
    if controller.ready < 12:          # 두 라이다가 모두 들어올 때까지 대기
        return 0.0, 0.0, False

    if movers.shape[0] > 0:
        bd = _body_dist(movers)
        fs = movers[:, 0] > -0.225     # 몸체 중심보다 앞쪽(앞·옆), 뒤쪽 제외
        near_fs = float(bd[fs].min()) if fs.any() else 99.0
        near_all = float(bd.min())
        corr = movers[(movers[:, 0] > 0.2) & (np.abs(movers[:, 1]) < 0.6)]
        d_front = float(corr[:, 0].min()) if corr.shape[0] > 0 else 99.0
    else:
        near_fs, near_all, d_front = 99.0, 99.0, 99.0

    # ---- 근접 시 후진(히스테리시스 0.7 m 진입 / 1.2 m 해제) ----
    if controller.retreat:
        if near_fs > 1.2:
            controller.retreat = False
    elif near_fs < 0.7:
        controller.retreat = True
    if controller.retreat:
        rear = raw[(raw[:, 0] < _BODY[0]) & (raw[:, 0] > _BODY[0] - 0.45) & (np.abs(raw[:, 1]) < 0.45)]
        v = -0.3 if rear.shape[0] == 0 else 0.0
        return v, 0.0, False

    # ---- GOAL 정렬 단계 ----
    if controller.idx >= n:
        if math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.45:
            controller.idx = n - 1     # 후진 등으로 밀려났으면 GOAL 로 복귀
        else:
            e = _wrap_angle(GOAL[2] - yaw)
            if abs(e) < 0.05:
                return 0.0, 0.0, True
            w = _clamp(1.5 * e, -0.6, 0.6)
            if near_all < 0.9:
                w = 0.0
            return 0.0, w, False

    # ---- 경유점 추종 ----
    tx, ty = _WAYPOINTS[controller.idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if controller.idx < n - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -1.2, 1.2)
        if near_all < 0.9:             # 제자리 회전 중 물체가 0.9 m 안이면 회전 정지
            w = 0.0
    else:
        v_wp = min(1.5, 2.5 * dist + 0.5)
        v = min(v_wp, 0.8, 0.8 * (d_front - 0.9))
        if v < 0.0:
            v = 0.0
        w = _clamp(7.5 * e, -1.2, 1.2)

    if controller.idx == n - 1 and dist < 0.25:
        controller.idx += 1

    return v, w, False




_STATIC_RECTS = np.array(
    [(b["x"] - b["width"] / 2.0 - 0.2, b["x"] + b["width"] / 2.0 + 0.2,
      b["y"] - b["depth"] / 2.0 - 0.2, b["y"] + b["depth"] / 2.0 + 0.2) for b in MISSION["boxes"]]
    + [(x0 - 0.2, x1 + 0.2, y0 - 0.2, y1 + 0.2) for _, x0, x1, y0, y1 in FIXED_OBSTACLES],
    dtype=np.float64)


def _body_dist(p):
    """로봇 기준 점 (N, 2) 각각의 몸체 사각형 _BODY 바깥면까지 거리."""
    dx = np.maximum(np.maximum(_BODY[0] - p[:, 0], p[:, 0] - _BODY[1]), 0.0)
    dy = np.maximum(np.maximum(_BODY[2] - p[:, 1], p[:, 1] - _BODY[3]), 0.0)
    return np.hypot(dx, dy)


def _movers_robot_frame(pose):
    """앞·뒤 2D 라이다 점에서 정적 지도(WALLS 띠·바깥, 0.2 m 부풀린 boxes/FIXED_OBSTACLES)를 빼고,
    0.3 m 안에 3점 이상 모인 점만 움직이는 물체로 남겨 로봇 기준 (K, 2) 로 돌려줍니다.
    반환: (물체 점, 3 m 안의 전체 라이다 점(로봇 기준), 스캔 수신 여부)."""
    scans = [lidar2d_points("front"), lidar2d_points("back")]
    scans = [s for s in scans if s is not None and len(s) > 0]
    empty = np.zeros((0, 2))
    if not scans:
        return empty, empty, False
    rob = np.concatenate(scans, axis=0)
    rob = rob[np.hypot(rob[:, 0], rob[:, 1]) < 3.0]
    if rob.shape[0] == 0:
        return empty, rob, True
    wpts = to_world(rob, pose)
    wx, wy = wpts[:, 0], wpts[:, 1]
    keep = ((wx > WALLS[0] + 0.2) & (wx < WALLS[1] - 0.2) &
            (wy > WALLS[2] + 0.2) & (wy < WALLS[3] - 0.2))
    R = _STATIC_RECTS
    inside = ((wx[:, None] >= R[None, :, 0]) & (wx[:, None] <= R[None, :, 1]) &
              (wy[:, None] >= R[None, :, 2]) & (wy[:, None] <= R[None, :, 3])).any(axis=1)
    keep &= ~inside
    m = rob[keep]
    if m.shape[0] > 600:
        m = m[::int(math.ceil(m.shape[0] / 600.0))]
    if m.shape[0] < 3:
        return empty, rob, True
    d2 = ((m[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
    m = m[(d2 < 0.09).sum(axis=1) >= 3]
    return m, rob, True


# ========================= [END EDIT REGION] ===============================
