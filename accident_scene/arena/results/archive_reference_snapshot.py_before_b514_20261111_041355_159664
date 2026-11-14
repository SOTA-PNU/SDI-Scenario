# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# Analysis-only reference. Never include filled hooks in model input.
def plan_path(start_xy, goal_xy, clearance_m):
    """Deterministic A* using public map tools; search stays in this model hook."""
    start_xy, goal_xy = tuple(map(float, start_xy)), tuple(map(float, goal_xy))
    need = float(clearance_m)
    if (len(start_xy) != 2 or len(goal_xy) != 2 or not math.isfinite(need)
            or need < 0.25 or not all(math.isfinite(v) for v in start_xy + goal_xy)):
        return []
    if not segment_free(start_xy, start_xy, need) or not segment_free(goal_xy, goal_xy, need):
        return []
    def _search_path(center_clearance):
        blocked, origin, resolution = occupancy_grid(center_clearance, 0.5)
        ny, nx = blocked.shape

        def point(cell):
            return xy_of(cell, origin, resolution)

        def containing(xy):
            return cell_of(xy, origin, resolution)
        need = float(center_clearance)
        free_cache, edge_cache = ({}, {})
    
        def free(cell):
            if cell not in free_cache:
                free_cache[cell] = 0 <= cell[0] < nx and 0 <= cell[1] < ny and (not blocked[cell[1], cell[0]])
            return free_cache[cell]
    
        def connected(p, q):
            return segment_free(p, q, need)
    
        def anchor(xy):
            cell = containing(xy)
            if free(cell) and connected(xy, point(cell)):
                return cell
            alternatives = sorted(((i, j) for i in range(nx) for j in range(ny)), key=lambda c: (math.dist(xy, point(c)), c))
            return next((c for c in alternatives if free(c) and connected(xy, point(c))), None)
        start, goal = (anchor(start_xy), anchor(goal_xy))
        if start is None or goal is None:
            return None
        costs, came, opened, closed = ({start: 0.0}, {start: None}, [(0.0, start)], set())
        while opened:
            _, cell = heapq.heappop(opened)
            if cell in closed:
                continue
            closed.add(cell)
            if cell == goal:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nxt = (cell[0] + di, cell[1] + dj)
                    if nxt == cell or not free(nxt):
                        continue
                    edge = tuple(sorted((cell, nxt)))
                    if edge not in edge_cache:
                        edge_cache[edge] = connected(point(cell), point(nxt))
                    if not edge_cache[edge]:
                        continue
                    cost = costs[cell] + math.hypot(di, dj) * resolution
                    if cost < costs.get(nxt, float('inf')):
                        costs[nxt], came[nxt] = (cost, cell)
                        heuristic = math.hypot(nxt[0] - goal[0], nxt[1] - goal[1]) * resolution
                        heapq.heappush(opened, (cost + heuristic, nxt))
        if goal not in came:
            return None
        path, cell = ([goal_xy], goal)
        while cell is not None:
            path.append(point(cell))
            cell = came[cell]
        path = [start_xy] + path[::-1]
        simple, i = ([path[0]], 0)
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and (not connected(path[i], path[j])):
                j -= 1
            if not connected(path[i], path[j]):
                break
            simple.append(path[j])
            i = j
        if i != len(path) - 1:
            return None
        return simple
    candidate = _search_path(need)
    if candidate is None:
        return []
    checked = validate_plan(candidate)
    return checked['path'] if checked['accepted'] else []

def lookahead_target(path, pose):
    """Project onto the polyline; keep the forward target on one segment."""
    if path is None or len(path) < 2:
        return None
    xy = (float(pose[0]), float(pose[1]))
    projections = []
    for index, (p, q) in enumerate(zip(path, path[1:])):
        dx, dy = (q[0] - p[0], q[1] - p[1])
        length2 = dx * dx + dy * dy
        if length2 <= 1e-16:
            continue
        u = max(0.0, min(1.0, ((xy[0] - p[0]) * dx + (xy[1] - p[1]) * dy) / length2))
        projected = (p[0] + u * dx, p[1] + u * dy)
        projections.append((math.dist(xy, projected), -index, index, u))
    if not projections:
        return None
    _, _, index, u = min(projections)
    if index < len(path) - 2 and math.dist(xy, path[index + 1]) < 0.35:
        index, u = (index + 1, 0.0)
    p, q = (path[index], path[index + 1])
    length = math.dist(p, q)
    if length <= 1e-08:
        return None
    ahead = min(3.0, (1.0 - u) * length)
    if ahead <= 1e-08:
        return None
    for travel in (ahead, ahead * 0.5, ahead * 0.25):
        fraction = min(1.0, u + travel / length)
        target = (p[0] + fraction * (q[0] - p[0]), p[1] + fraction * (q[1] - p[1]))
        if math.dist(xy, target) > 1e-08 and segment_free(xy, target, 0.35):
            return target
    return None

def follow_cmd(pose, target_xy):
    """The verified simple gate's nominal speed/angle rule, without object logic."""
    x, y, yaw = pose
    dx, dy = (target_xy[0] - x, target_xy[1] - y)
    distance = math.hypot(dx, dy)
    error = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
    v = 0.0 if abs(error) > 0.35 else min(0.3, 0.5 * distance + 0.1)
    w = max(-0.4, min(0.4, 1.5 * error))
    return (v, w)

def controller(t, pose, env): return _base_control_loop(t, pose, env)
# ========================= [END EDIT REGION] ===============================
