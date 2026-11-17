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
    """Follow way‑points while predicting mover trajectories and picking a safe speed."""
    # ---- one‑time initialisation -----------------------------------------
    if not hasattr(controller, "idx"):
        controller.idx = 0               # waypoint index
        controller.state = "drive"       # drive | slow | stop | back
        controller.prev_state = None

        # reset mover‑tracking cache
        for attr in ("prev_clusters", "last_count", "last_t", "last_raw_pts"):
            if hasattr(_track_movers, attr):
                delattr(_track_movers, attr)

    x, y, yaw = pose

    # ---- waypoint handling ------------------------------------------------
    if controller.idx < len(_WAYPOINTS):
        tx, ty = _WAYPOINTS[controller.idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

        # advance to next waypoint
        if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
            controller.idx += 1
            tx, ty = _WAYPOINTS[controller.idx]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            e = _wrap_angle(math.atan2(dy, dx) - yaw)
    else:
        # ---- goal‑alignment phase -----------------------------------------
        tx, ty = GOAL[0], GOAL[1]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(GOAL[2] - yaw)

    # ---- nominal control --------------------------------------------------
    if controller.idx >= len(_WAYPOINTS):
        # alignment (rotation only)
        v_nom = 0.0
        w_nom = _clamp(1.5 * e, -0.6, 0.6)
    else:
        # waypoint navigation
        if abs(e) > 0.35:
            v_nom = 0.0
        else:
            v_nom = min(1.5, 2.5 * dist + 0.5)
        w_nom = _clamp(3.0 * e, -2.0, 2.0)

    # ---- moving‑object detection & safe‑speed selection --------------------
    clusters, raw_pts = _track_movers(t, pose)
    v_safe, pred_min_dist, zero_dist = _pick_speed(pose, w_nom, v_nom, clusters, raw_pts)

    # ---- combine nominal and safe speed -----------------------------------
    if v_nom == 0.0:
        # alignment: stay still if zero‑speed candidate is safe (≥1 m)
        if zero_dist is not None and zero_dist >= 1.0:
            v = 0.0
        else:
            v = v_safe
    else:
        v = min(v_nom, v_safe)

    # ---- state handling & debug output ------------------------------------
    if v < -0.01:
        new_state = "back"
    elif abs(v) < 0.01 and abs(w_nom) < 0.01:
        new_state = "stop"
    elif v < v_nom - 1e-3:
        new_state = "slow"
    else:
        new_state = "drive"

    if new_state != controller.state:
        n_clusters = len(clusters)
        if clusters:
            dists = [np.linalg.norm(cl['pos'] - np.array([x, y])) for cl in clusters]
            nearest_dist = min(dists)
            nearest_vel = np.linalg.norm(clusters[dists.index(nearest_dist)]['vel'])
        else:
            nearest_dist = -1.0
            nearest_vel = 0.0
        print('DBG', round(t, 2), new_state,
              n_clusters, round(nearest_dist, 2), round(nearest_vel, 2),
              round(v, 2), round(pred_min_dist, 2))
        controller.prev_state = controller.state
        controller.state = new_state

    # ---- transition to goal‑alignment after final waypoint -----------------
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1   # trigger alignment on next call

    # ---- goal‑alignment termination & fallback ---------------------------
    if controller.idx >= len(_WAYPOINTS):
        # alignment stage
        if dist < 0.75 and abs(e) < 0.05 and abs(v) < 1e-3:
            return 0.0, 0.0, True
        # if we have drifted away from the goal, resume navigation
        if dist >= 0.75:
            controller.idx = len(_WAYPOINTS) - 1

    return v, w_nom, False




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




def _cluster_points(pts, radius=0.5):
    """Cluster 2‑D points into connected components where any two points
    within `radius` are considered connected (BFS). Returns only components
    containing at least three points. Each component is a dict with keys
    'pts' (list of point arrays) and 'pos' (centroid as np.array)."""
    pts = np.asarray(pts)
    if pts.shape[0] == 0:
        return []
    N = pts.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]:
            continue
        # start new component
        queue = [i]
        visited[i] = True
        comp_idx = []
        while queue:
            idx = queue.pop()
            comp_idx.append(idx)
            # neighbours within radius
            diff = pts - pts[idx]
            d2 = np.einsum('ij,ij->i', diff, diff)
            neigh = np.where((d2 <= radius ** 2) & (~visited))[0]
            visited[neigh] = True
            queue.extend(neigh.tolist())
        if len(comp_idx) >= 3:
            comp_pts = pts[comp_idx]
            centroid = np.mean(comp_pts, axis=0)
            clusters.append({'pts': comp_pts.tolist(), 'pos': centroid})
    return clusters


def _track_movers(t, pose):
    """
    Detect moving objects, cluster them, and estimate their world‑frame velocities.
    Velocity is updated only when a new lidar scan arrives.
    Returns (clusters, raw_pts) where clusters is a list of dicts:
        {'pos': np.array([x, y]), 'vel': np.array([vx, vy])}
    raw_pts are the unfiltered 2‑D lidar points (robot frame) used for rear‑corridor checks.
    """
    mover_pts, raw_pts = _mover_points(pose)

    cur_count = _S['count']
    last_count = getattr(_track_movers, 'last_count', None)
    dt = 0.0

    if last_count is None or cur_count != last_count:
        # New scan arrived – recompute clusters and velocities
        dt = t - getattr(_track_movers, 'last_t', t)
        if dt <= 0.0:
            dt = 1.0 / 60.0      # fallback to one control step

        # world coordinates for clustering
        if mover_pts.shape[0] > 0:
            mover_world = to_world(mover_pts, pose)[:, :2]
            clusters = _cluster_points(mover_world, radius=0.5)
        else:
            clusters = []

        # velocity estimation (exponential smoothing, speed clipped to 1.5 m/s)
        alpha = 0.5
        prev = getattr(_track_movers, 'prev_clusters', [])
        for cl in clusters:
            # find matching previous cluster (within 0.6 m)
            best = None
            best_dist = float('inf')
            for pc in prev:
                d = np.linalg.norm(cl['pos'] - pc['pos'])
                if d < 0.6 and d < best_dist:
                    best = pc
                    best_dist = d
            if best is not None:
                raw_vel = (cl['pos'] - best['pos']) / dt
                vel = alpha * raw_vel + (1 - alpha) * best.get('vel', np.zeros(2))
                speed = np.linalg.norm(vel)
                if speed > 1.5:
                    vel = vel * (1.5 / speed)
                cl['vel'] = vel
            else:
                cl['vel'] = np.zeros(2)

        # cache for the next call
        _track_movers.prev_clusters = clusters
        _track_movers.last_count = cur_count
        _track_movers.last_t = t
        _track_movers.last_raw_pts = raw_pts
    else:
        # No new scan – reuse previous data
        clusters = getattr(_track_movers, 'prev_clusters', [])
        raw_pts = getattr(_track_movers, 'last_raw_pts', raw_pts)

    return clusters, raw_pts


def _pick_speed(pose, w_nom, v_nom, clusters, raw_pts):
    """
    Evaluate a set of candidate forward speeds and pick the fastest one that
    keeps a minimum centre‑to‑centre clearance of at least `safe_dist`
    (1.0 m) from any predicted mover position over a 2.5 s horizon.
    The robot is approximated by two circles: its centre (0,0) and a rear
    point at (-0.45,0) in the robot frame.
    Returns (chosen_speed, predicted_min_distance, zero_candidate_min_dist).
    """
    base_candidates = [1.5, 1.0, 0.6, 0.3, 0.0, -0.3, -0.5]
    # ensure v_nom itself is examined
    candidates = list(set(base_candidates + [v_nom]))
    # when a positive nominal speed is required, discard faster forward candidates
    if v_nom > 0:
        candidates = [c for c in candidates if c <= v_nom]
    # (v_nom == 0) – keep all candidates; we may need to move if staying still is unsafe
    candidates = sorted(candidates, reverse=True)   # deterministic order, fast → slow

    dt = 0.1
    horizon = 2.5
    steps = int(horizon / dt)
    safe_dist = 1.0  # metres

    # rear‑corridor mask for backward motion (allow only if empty)
    rear_mask = (raw_pts[:, 0] >= -1.3) & (raw_pts[:, 0] <= -0.7) & (np.abs(raw_pts[:, 1]) <= 0.6)

    safe_candidates = []
    unsafe_candidates = []
    zero_dist = None

    for v in candidates:
        # allow backward only when rear corridor is clear
        if v < 0 and rear_mask.any():
            continue

        # simulate robot trajectory over the horizon
        x, y, yaw = pose
        min_dist = float('inf')
        elapsed = 0.0

        for _ in range(steps):
            elapsed += dt
            yaw += w_nom * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt

            # robot body points in world frame
            front_pt = np.array([x, y])
            rear_pt = np.array([x - 0.45 * math.cos(yaw), y - 0.45 * math.sin(yaw)])

            for cl in clusters:
                obj_pos = cl['pos'] + cl['vel'] * elapsed
                d_front = np.linalg.norm(front_pt - obj_pos)
                d_rear = np.linalg.norm(rear_pt - obj_pos)
                d = d_front if d_front < d_rear else d_rear
                if d < min_dist:
                    min_dist = d

        # remember distance for the zero‑speed candidate
        if abs(v) < 1e-9:
            zero_dist = min_dist

        # classify candidate
        if min_dist >= safe_dist:
            safe_candidates.append((v, min_dist))
        else:
            unsafe_candidates.append((v, min_dist))

    if safe_candidates:
        # fastest safe candidate
        best_speed, best_min_dist = max(safe_candidates, key=lambda x: x[0])
    else:
        # least‑dangerous unsafe candidate (largest min distance)
        best_speed, best_min_dist = max(unsafe_candidates, key=lambda x: x[1])

    return best_speed, best_min_dist, zero_dist


# ========================= [END EDIT REGION] ===============================
