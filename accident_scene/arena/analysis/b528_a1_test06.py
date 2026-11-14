"""Analysis-only exact saved B525 a1 test06 hooks; never model input."""

def plan_path(start_xy, goal_xy, clearance_m):
    """A* planner without aggressive simplification.
    Returns the full node sequence (including start and goal) if a path is found.
    For clearances >0.35 returns None to let the base try smaller clearances.
    """
    import math, heapq, numpy as np

    if clearance_m > 0.35:
        return None

    sx, sy = start_xy
    gx, gy = goal_xy

    # Direct line
    if segment_clearance(start_xy, goal_xy) >= clearance_m:
        return [tuple(start_xy), tuple(goal_xy)]

    # Simple L‑shaped
    if (segment_clearance(start_xy, (sx, gy)) >= clearance_m and
        segment_clearance((sx, gy), goal_xy) >= clearance_m):
        return [tuple(start_xy), (sx, gy), tuple(goal_xy)]
    if (segment_clearance(start_xy, (gx, sy)) >= clearance_m and
        segment_clearance((gx, sy), goal_xy) >= clearance_m):
        return [tuple(start_xy), (gx, sy), tuple(goal_xy)]

    # Grid A* (0.1 m cells)
    step = 0.1
    wx0, wx1, wy0, wy1 = WALLS
    xs = np.arange(wx0, wx1 + step, step)
    ys = np.arange(wy0, wy1 + step, step)
    free_pts = []
    grid_index = {}
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if map_clearance(float(x), float(y)) >= clearance_m:
                idx = len(free_pts)
                free_pts.append((float(x), float(y)))
                grid_index[(ix, iy)] = idx
    free_pts = np.array(free_pts, dtype=np.float64)

    def edge_ok(p, q):
        return segment_clearance(p, q) >= clearance_m

    start_idx = len(free_pts)
    goal_idx = start_idx + 1
    points = np.vstack([free_pts, start_xy, goal_xy])

    neigh_offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]

    def neighbors(idx):
        if idx == start_idx:
            for i, pt in enumerate(free_pts):
                if edge_ok(start_xy, pt):
                    yield i
            if edge_ok(start_xy, goal_xy):
                yield goal_idx
        elif idx == goal_idx:
            for i, pt in enumerate(free_pts):
                if edge_ok(goal_xy, pt):
                    yield i
        else:
            ix = int(round((points[idx][0] - wx0) / step))
            iy = int(round((points[idx][1] - wy0) / step))
            for dx, dy in neigh_offsets:
                nkey = (ix + dx, iy + dy)
                nidx = grid_index.get(nkey)
                if nidx is not None and edge_ok(points[idx], points[nidx]):
                    yield nidx
            if edge_ok(points[idx], start_xy):
                yield start_idx
            if edge_ok(points[idx], goal_xy):
                yield goal_idx

    goal_pt = points[goal_idx]
    open_set = [(0.0, 0.0, start_idx)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {start_idx: 0.0}
    closed = set()
    while open_set:
        _, cur_g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal_idx:
            # reconstruct full path
            path_idx = [goal_idx]
            while path_idx[-1] != start_idx:
                path_idx.append(came_from[path_idx[-1]])
            path_idx.reverse()
            raw_path = [tuple(points[i]) for i in path_idx]
            return raw_path
        closed.add(current)
        for nb in neighbors(current):
            if nb in closed:
                continue
            tentative_g = cur_g + math.hypot(points[nb][0] - points[current][0],
                                            points[nb][1] - points[current][1])
            if nb not in g_score or tentative_g < g_score[nb] - 1e-12:
                g_score[nb] = tentative_g
                f = tentative_g + math.hypot(goal_pt[0] - points[nb][0], goal_pt[1] - points[nb][1])
                heapq.heappush(open_set, (f, tentative_g, nb))
                came_from[nb] = current
    return None

def lookahead_target(path, pose):
    """Return a point on *path* 0.3–2.0 m ahead of the robot projection.
    If remaining path from the projection is shorter than 0.3 m, the last
    point (goal) is returned.  The base will later guard the point.
    """
    proj = _base_loop_projection(path, pose[:2], _BASE_LOOP_ST["path_progress_s"])
    s = proj["s"]
    total = proj["total"]
    remaining = total - s
    if remaining < 0.3:
        target_s = total
    else:
        ahead = max(0.3, min(2.0, remaining))
        target_s = s + ahead
    return _base_loop_point_at_s(path, target_s)

def follow_cmd(pose, target_xy):
    """Pure‑pursuit controller as specified.
    Returns (v, w) before clipping.
    """

    dx = target_xy[0] - pose[0]
    dy = target_xy[1] - pose[1]
    dist = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    e = angle_to_target - pose[2]
    e = math.atan2(math.sin(e), math.cos(e))
    if abs(e) > 0.35:
        v = 0.0
    else:
        v = min(0.3, 0.5 * dist + 0.1)
    w = max(-0.4, min(0.4, 1.5 * e))
    return v, w

def controller(t, pose, env): return _base_control_loop(t, pose, env)
