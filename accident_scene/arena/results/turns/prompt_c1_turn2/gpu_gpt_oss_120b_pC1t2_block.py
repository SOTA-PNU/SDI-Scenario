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
    """Waypoint‑following with proactive dynamic‑obstacle avoidance."""
    # ---- one‑time state initialisation ---------------------------------
    if not hasattr(controller, "idx"):
        controller.idx = 0          # current waypoint index
        controller.avoid = False    # avoidance mode flag
        controller.avoid_dir = 0.0  # rotation direction during avoidance
        controller.align = False    # goal‑alignment phase flag

    x, y, yaw = pose

    # ---- sense dynamic obstacles ---------------------------------------
    front_dist, front_bearing = _nearest_dynamic_front(pose)
    back_dist, back_bearing = _nearest_dynamic_back(pose)

    # ---- compute provisional forward speed (based on current waypoint) --
    tx, ty = _WAYPOINTS[controller.idx]
    dx, dy = tx - x, ty - y
    dist_to_wp = math.hypot(dx, dy)
    # speed law identical to the original controller
    v_target = min(1.5, 2.5 * dist_to_wp + 0.5)

    # ---- required stopping distance ------------------------------------
    required_stop = max(SAFE_DIST, v_target * REACTION_TIME + BUFFER)

    # ---- avoidance logic ------------------------------------------------
    # decide whether we must start/continue avoidance
    if front_dist < required_stop:
        controller.avoid = True
        # choose rotation direction opposite to bearing (left if obstacle on left)
        controller.avoid_dir = 1.0 if front_bearing > 0 else -1.0
        # if rotating that way would hit something behind, flip direction
        if back_dist < required_stop and (back_bearing * controller.avoid_dir) > 0:
            controller.avoid_dir *= -1.0
    elif controller.avoid and front_dist > EXIT_DIST and back_dist > EXIT_DIST:
        controller.avoid = False
        controller.avoid_dir = 0.0

    if controller.avoid:
        # stay in place and rotate away from the obstacle
        v = 0.0
        w = _clamp(1.0 * controller.avoid_dir, -0.8, 0.8)
        return v, w, False

    # ---- goal‑alignment phase -------------------------------------------
    if controller.align:
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if abs(e) < 0.05:
            return 0.0, 0.0, True          # finished
        return 0.0, w, False

    # ---- normal waypoint‑following --------------------------------------
    # advance to next waypoint when close enough (except final goal)
    if controller.idx < len(_WAYPOINTS) - 1 and dist_to_wp < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx, dy = tx - x, ty - y
        dist_to_wp = math.hypot(dx, dy)

    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if abs(e) > 0.35:
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)
    else:
        v = min(1.5, 2.5 * dist_to_wp + 0.5)
        w = _clamp(7.5 * e, -2.0, 2.0)

    # when the last waypoint (the goal) is reached, switch to alignment mode
    if controller.idx == len(_WAYPOINTS) - 1 and dist_to_wp < 0.25:
        controller.align = True

    return v, w, False




SAFE_DIST = 1.5          # minimum distance to start avoidance (m)


EXIT_DIST = 2.0          # distance to consider obstacle cleared (m)


def _is_static_world(pt):
    """Return True if the world‑frame point lies inside any static box from MISSION."""
    x, y = pt[0], pt[1]
    for b in MISSION["boxes"]:
        half_w, half_d = b["width"] / 2.0, b["depth"] / 2.0
        if (b["x"] - half_w <= x <= b["x"] + half_w and
                b["y"] - half_d <= y <= b["y"] + half_d):
            return True
    return False


def _nearest_dynamic(pose):
    """
    Scan the front 2‑D lidar, filter out points that belong to static boxes,
    and return the distance (m) and bearing (rad, robot‑frame) of the nearest
    remaining point. If none exist, return (inf, 0.0).
    """
    pts = lidar2d_points('front')          # robot‑frame (x, y)
    if pts.shape[0] == 0:
        return float('inf'), 0.0

    # transform to world to test against static boxes
    world_pts = to_world(pts, pose)

    # keep only points that are NOT inside any static box
    dynamic_mask = np.array([not _is_static_world(p) for p in world_pts])
    dyn_pts = pts[dynamic_mask]

    if dyn_pts.shape[0] == 0:
        return float('inf'), 0.0

    dists = np.hypot(dyn_pts[:, 0], dyn_pts[:, 1])
    idx = np.argmin(dists)
    bearing = math.atan2(dyn_pts[idx, 1], dyn_pts[idx, 0])  # robot‑frame angle
    return dists[idx], bearing




REACTION_TIME = 0.5      # assumed reaction time before full stop (s)


BUFFER = 0.3             # extra safety margin (m)


def _nearest_dynamic_front(pose):
    """Return (dist, bearing) of nearest non‑static point from the front 2‑D lidar."""
    pts = lidar2d_points('front')
    if pts.shape[0] == 0:
        return float('inf'), 0.0
    world_pts = to_world(pts, pose)
    dynamic_mask = np.array([not _is_static_world(p) for p in world_pts])
    dyn_pts = pts[dynamic_mask]
    if dyn_pts.shape[0] == 0:
        return float('inf'), 0.0
    dists = np.hypot(dyn_pts[:, 0], dyn_pts[:, 1])
    idx = np.argmin(dists)
    bearing = math.atan2(dyn_pts[idx, 1], dyn_pts[idx, 0])
    return dists[idx], bearing


def _nearest_dynamic_back(pose):
    """Return (dist, bearing) of nearest non‑static point from the back 2‑D lidar."""
    pts = lidar2d_points('back')
    if pts.shape[0] == 0:
        return float('inf'), 0.0
    world_pts = to_world(pts, pose)
    dynamic_mask = np.array([not _is_static_world(p) for p in world_pts])
    dyn_pts = pts[dynamic_mask]
    if dyn_pts.shape[0] == 0:
        return float('inf'), 0.0
    dists = np.hypot(dyn_pts[:, 0], dyn_pts[:, 1])
    idx = np.argmin(dists)
    bearing = math.atan2(dyn_pts[idx, 1], dyn_pts[idx, 0])
    return dists[idx], bearing


# ========================= [END EDIT REGION] ===============================
