"""Analysis-only exact completed B526 a0 test03 planner: .1m grid and broken simplification."""

def plan_path(start_xy, goal_xy, clearance_m):
    """A* on the occupancy grid (8‑neighbour moves).  The search stops when a cell
    centre is within ``plan_endpoint_limit_m`` (0.5 m) of the real goal.  After a
    path is found it is simplified with a line‑of‑sight test and finally passed
    through ``validate_plan``.  If validation fails ``None`` is returned.
    """
    # Fixed resolution – a compromise between accuracy and speed.
    res = 0.10
    blocked, origin, res = occupancy_grid(clearance_m, res)
    sx, sy = cell_of(start_xy, origin, res)
    gx, gy = cell_of(goal_xy, origin, res)
    h, w = blocked.shape
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None

    # ------------------------------------------------------------------
    # A* search (8‑neighbour connectivity).  We deliberately omit the
    # ``segment_free`` test for individual edges – the final ``validate_plan``
    # call will reject any path that violates the true clearance constraints.
    # ------------------------------------------------------------------
    open_set = []
    heapq.heappush(open_set, (0.0, (sx, sy)))
    came_from = {}
    g_score = {(sx, sy): 0.0}

    def heuristic(ix, iy):
        return math.hypot((ix - gx) * res, (iy - gy) * res)

    goal_reached = None
    while open_set:
        _, current = heapq.heappop(open_set)
        cx, cy = current
        pc = xy_of((cx, cy), origin, res)
        if math.hypot(pc[0] - goal_xy[0], pc[1] - goal_xy[1]) <= _BASE_LOOP_CFG["plan_endpoint_limit_m"]:
            goal_reached = current
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if blocked[ny, nx]:
                    continue
                tentative_g = g_score[current] + math.hypot(dx * res, dy * res)
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
    else:
        return None

    # Reconstruct world‑space polyline.
    path_cells = []
    cur = goal_reached
    while cur != (sx, sy):
        path_cells.append(cur)
        cur = came_from.get(cur)
        if cur is None:
            return None
    path_cells.append((sx, sy))
    path_cells.reverse()
    points = [xy_of(cell, origin, res) for cell in path_cells]

    # Simple line‑of‑sight reduction.
    if len(points) > 2:
        reduced = [points[0]]
        last = points[0]
        for pt in points[1:]:
            if segment_free(last, pt, clearance_m):
                continue
            reduced.append(last)
            last = pt
        reduced.append(points[-1])
        points = [reduced[0]] + [p for i, p in enumerate(reduced[1:], 1) if p != reduced[i - 1]]

    # Final validation – ``validate_plan`` adds the required start/goal
    # connectors and checks the whole polyline.
    result = validate_plan(points)
    if result.get("accepted"):
        return result["path"]
    return None
