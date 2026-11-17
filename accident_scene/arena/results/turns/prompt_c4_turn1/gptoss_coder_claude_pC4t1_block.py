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
    """Follow way‑points while reacting to moving obstacles."""
    # ---- one‑time initialisation -----------------------------------------
    if not hasattr(controller, "idx"):
        controller.idx = 0               # waypoint index
        controller.done_align = False
        controller.state = "drive"       # drive | slow | stop | back
        controller.prev_state = None

    x, y, yaw = pose

    # ---- goal‑alignment phase --------------------------------------------
    if controller.idx >= len(_WAYPOINTS):
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, w, False

    # ---- waypoint navigation ---------------------------------------------
    tx, ty = _WAYPOINTS[controller.idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ---- nominal control law (reduced turning gain) ----------------------
    if abs(e) > 0.35:
        v_nom = 0.0
        w_nom = _clamp(3.0 * e, -2.0, 2.0)   # reduced gain for smoother turn
    else:
        v_nom = min(1.5, 2.5 * dist + 0.5)   # speed cap 1.5 m/s
        w_nom = _clamp(3.0 * e, -2.0, 2.0)

    # ---- moving‑obstacle reaction ----------------------------------------
    mover_pts, raw_pts = _mover_points(pose)

    v = v_nom
    w = w_nom
    d = -1  # distance for debug output; -1 means “no forward mover”

    # forward corridor check (0.25 m – 3.0 m ahead, |y| < 0.6 m)
    if mover_pts.shape[0] >= 3:
        fwd_mask = (mover_pts[:, 0] >= 0.25) & (mover_pts[:, 0] <= 3.0) & (np.abs(mover_pts[:, 1]) <= 0.6)
        if np.count_nonzero(fwd_mask) >= 3:
            d_min = mover_pts[fwd_mask, 0].min()
            d = d_min
            safe_speed = max(0.0, 0.9 * (d_min - 1.0))   # keep ≥1 m clearance
            v = min(v_nom, safe_speed)

            # stop → possibly back up if rear corridor is clear (using raw points)
            if v < 0.01 and d_min < 0.8:
                rear_mask = (raw_pts[:, 0] >= -1.3) & (raw_pts[:, 0] <= -0.7) & (np.abs(raw_pts[:, 1]) <= 0.6)
                if not np.any(rear_mask):
                    v = -0.4
                    w = 0.0

    # ---- state handling & debug printing ---------------------------------
    # decide new state
    if v < -0.01:
        new_state = "back"
    elif abs(v) < 0.01 and abs(w) < 0.01:
        new_state = "stop"
    elif v < v_nom - 1e-3:
        new_state = "slow"
    else:
        new_state = "drive"

    if new_state != controller.state:
        # print only on state change
        print('DBG', round(t, 2), new_state, d if d >= 0 else -1)
        controller.prev_state = controller.state
        controller.state = new_state

    # ---- transition to goal‑alignment when final waypoint is reached -----
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1   # trigger alignment on next call

    return v, w, False




def _mover_points(pose):
    """Return a tuple (mover_pts, raw_pts).
    mover_pts – points belonging to moving obstacles after static‑map filtering.
    raw_pts   – the combined front and back 2‑D lidar points in robot frame (unfiltered)."""
    # 1) acquire latest scans
    front = lidar2d_points('front')
    back = lidar2d_points('back')
    raw_pts = np.vstack([front, back]) if (front.shape[0] or back.shape[0]) else np.zeros((0, 2))

    # 2) world coordinates for static‑map filtering
    pts_w = to_world(raw_pts, pose)

    # 3) keep points that lie inside the *shrunken* free‑space rectangle
    mask = (pts_w[:, 0] >= WALLS[0] + 0.2) & (pts_w[:, 0] <= WALLS[1] - 0.2) & \
           (pts_w[:, 1] >= WALLS[2] + 0.2) & (pts_w[:, 1] <= WALLS[3] - 0.2)

    margin = 0.25  # enlarged safety margin for static obstacles

    # 4) remove floor boxes
    for b in MISSION["boxes"]:
        bx, by, bw, bd = b["x"], b["y"], b["width"], b["depth"]
        x_min = bx - bw / 2 - margin
        x_max = bx + bw / 2 + margin
        y_min = by - bd / 2 - margin
        y_max = by + bd / 2 + margin
        mask &= ~((pts_w[:, 0] >= x_min) & (pts_w[:, 0] <= x_max) &
                  (pts_w[:, 1] >= y_min) & (pts_w[:, 1] <= y_max))

    # 5) remove fixed obstacles (racks, pillars, …)
    for obs in FIXED_OBSTACLES:
        _, x_min_o, x_max_o, y_min_o, y_max_o = obs
        mask &= ~((pts_w[:, 0] >= x_min_o - margin) & (pts_w[:, 0] <= x_max_o + margin) &
                  (pts_w[:, 1] >= y_min_o - margin) & (pts_w[:, 1] <= y_max_o + margin))

    mover_pts = raw_pts[mask]
    return mover_pts, raw_pts


# ========================= [END EDIT REGION] ===============================
