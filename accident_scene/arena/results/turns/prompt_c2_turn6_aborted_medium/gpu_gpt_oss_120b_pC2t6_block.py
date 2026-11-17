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
    """Waypoint‑following with a simple stop‑wait‑rotate avoidance loop.

    States:
        "normal" – follow way‑points.
        "stop"   – stopped; may be waiting or rotating (handled internally).
    """
    # ---- one‑time static initialisation ---------------------------------
    if not hasattr(controller, "idx"):
        controller.idx = 0                # current waypoint index
        controller.state = "normal"       # avoidance state
        controller.timer = 0.0            # generic timer for stop / rotation
        controller.rotating = False       # flag: currently in rotation phase
        controller.avoid_dir = 0.0        # rotation direction (set when detection occurs)

    # ---- helper: enter stop‑wait -----------------------------------------
    def _enter_stop(bearing):
        controller.state = "stop"
        controller.rotating = False
        controller.timer = t + _MAX_WAIT   # maximum waiting period
        controller.avoid_dir = -np.sign(bearing) if bearing != 0 else 1.0

    # ---- avoidance handling -----------------------------------------------
    if controller.state == "stop":
        # ----- rotating phase ------------------------------------------------
        if controller.rotating:
            if t < controller.timer:                     # keep rotating
                return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False
            # rotation finished → re‑enter waiting phase
            controller.rotating = False
            controller.timer = t + _MAX_WAIT

        # ----- waiting phase -------------------------------------------------
        if t < controller.timer:
            return 0.0, 0.0, False

        # timer expired – re‑evaluate surroundings
        moving, bearing = _detect_moving_any(pose)
        if moving:
            # object still present – check its distance
            if _closest_moving_dist(pose) < _SAFE_DIST:
                # need a corrective rotation
                controller.rotating = True
                controller.timer = t + _AVOID_ROT_TIME
                controller.avoid_dir = -np.sign(bearing) if bearing != 0 else 1.0
                return 0.0, float(controller.avoid_dir * _AVOID_ROT_SPEED), False
            else:
                # object is far enough → resume normal navigation
                controller.state = "normal"
                controller.rotating = False
        else:
            # no moving object detected → resume normal navigation
            controller.state = "normal"
            controller.rotating = False

        # fall through to normal navigation after state change

    # ---- normal navigation -------------------------------------------------
    # detect moving objects – if found, decide whether to stop or keep going
    moving, bearing = _detect_moving_any(pose)
    if moving:
        if _closest_moving_dist(pose) < _SAFE_DIST:
            _enter_stop(bearing)
            return 0.0, 0.0, False
        # object already beyond safe distance → ignore and continue

    # ---- goal‑alignment phase (after last waypoint) -----------------------
    x, y, yaw = pose
    if controller.idx >= len(_WAYPOINTS):
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, w, False

    # ---- waypoint‑following ------------------------------------------------
    tx, ty = _WAYPOINTS[controller.idx]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # advance to next waypoint if close enough (except for the goal)
    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ---- control law (same as proven static version) --------------------
    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        v = min(1.5, 2.5 * dist + 0.5)
        w = _clamp(7.5 * e, -2.0, 2.0)

    # trigger alignment after reaching the final waypoint
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
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
      * absolute bearing ≤ _DETECT_ANGLE relative to the robot's forward axis,
      * NOT inside any static obstacle (boxes, walls, fixed obstacles).
    mean_bearing is the average bearing (rad) of the detected moving points
    (positive = left of forward direction).
    """
    # collect points from both lidars
    pts_front = lidar2d_points("front")
    pts_back = lidar2d_points("back")
    if pts_front.shape[0] == 0 and pts_back.shape[0] == 0:
        return False, 0.0

    pts = np.vstack([pts_front, pts_back]) if pts_front.shape[0] and pts_back.shape[0] else (
        pts_front if pts_front.shape[0] else pts_back
    )

    # distance and bearing in robot frame
    d2 = np.einsum("ij,ij->i", pts, pts)
    dist = np.sqrt(d2)
    bearing = np.arctan2(pts[:, 1], pts[:, 0])

    # sector filter: within detection distance and half‑plane angle
    sector_mask = (dist <= _DETECT_DIST) & (np.abs(bearing) <= _DETECT_ANGLE)
    if not np.any(sector_mask):
        return False, 0.0

    # convert candidate points to world coordinates
    cand_robot = pts[sector_mask]
    cand_world = to_world(cand_robot, pose)[:, :2]   # (x, y) in world

    # keep only those not inside any static obstacle
    moving_mask = np.array([not _in_static_obstacle(p) for p in cand_world])
    if not np.any(moving_mask):
        return False, 0.0

    moving_bearings = bearing[sector_mask][moving_mask]
    mean_bearing = float(np.mean(moving_bearings))
    return True, mean_bearing




_DETECT_DIST = 2.0                     # [m] consider points up to 2 m away


_DETECT_ANGLE = np.deg2rad(60.0)       # [rad] half‑plane (±60°) in front/back of robot




_AVOID_DURATION = 2.5                  # [s] stop time after detection


_AVOID_ROT_SPEED = 0.4                 # [rad/s] rotation speed during avoidance




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




_AVOID_ROT_TIME = 1.0                  # [s] rotation duration (≈0.4 rad at 0.4 rad/s)




_SAFE_DIST = 1.5                       # [m] distance at which a moving object is considered safe


_MAX_WAIT = 3.0                        # [s] maximum stop‑wait time before rotating


def _closest_moving_dist(pose):
    """Return the Euclidean distance from the robot to the nearest detected moving point.
    If no moving points are present, returns a large number."""
    mv_pts = _moving_points(pose)      # world‑frame (N,2) points not belonging to static map
    if mv_pts.shape[0] == 0:
        return 1e6
    robot_xy = np.array(pose[:2])
    return float(np.min(np.linalg.norm(mv_pts - robot_xy, axis=1)))


def _rotate_away(bearing):
    """Determine rotation direction based on the bearing of the detected object.
    Positive bearing → object on left → rotate right (negative angular velocity)."""
    dir_sign = -np.sign(bearing) if bearing != 0 else 1.0
    return float(dir_sign * _AVOID_ROT_SPEED)


# ========================= [END EDIT REGION] ===============================
