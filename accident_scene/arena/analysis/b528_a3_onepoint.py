def plan_path(start_xy, goal_xy, clearance_m):
    """
    A* planner on the occupancy grid.  Tries a coarse (0.5 m) resolution first;
    if no path is found it retries with a finer (0.25 m) resolution.  The
    algorithm uses 8‑connectivity and checks each neighbour edge with
    ``segment_free`` to guarantee geometric feasibility.  After a path of cell
    centres is found it is simplified by removing intermediate points that are
    still segment‑free.
    """
    for res in (0.5, 0.25):                         # coarse → fine
        try:
            blocked, origin, _ = occupancy_grid(clearance_m, res)
        except Exception:
            continue

        try:
            sx, sy = cell_of(start_xy, origin, res)
            gx, gy = cell_of(goal_xy, origin, res)
        except Exception:
            continue

        h, w = blocked.shape                         # rows (y), cols (x)
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            continue
        if blocked[sy, sx] or blocked[gy, gx]:
            continue

        def world(ix, iy):
            return xy_of((ix, iy), origin, res)

        # cache for segment_free results
        seg_cache = {}

        def seg_free(p, q):
            key = (tuple(p), tuple(q))
            if key in seg_cache:
                return seg_cache[key]
            ok = segment_free(p, q, clearance_m)
            seg_cache[key] = ok
            return ok

        # A* search
        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, (sx, sy), None))  # f, g, cell, parent
        came_from = {}
        g_score = {(sx, sy): 0.0}
        neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if not (dx == 0 and dy == 0)]

        while open_heap:
            _, g, (cx, cy), parent = heapq.heappop(open_heap)
            if (cx, cy) in came_from:
                continue
            came_from[(cx, cy)] = parent
            if (cx, cy) == (gx, gy):
                break
            for dx, dy in neigh:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if blocked[ny, nx]:
                    continue
                p_world = world(cx, cy)
                q_world = world(nx, ny)
                if not seg_free(p_world, q_world):
                    continue
                tentative_g = g + math.hypot(dx * res, dy * res)
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    heuristic = math.hypot((gx - nx) * res, (gy - ny) * res)
                    f = tentative_g + heuristic
                    heapq.heappush(open_heap, (f, tentative_g, (nx, ny), (cx, cy)))

        if (gx, gy) not in came_from:
            continue

        # reconstruct cell path
        cell_path = []
        cur = (gx, gy)
        while cur is not None:
            cell_path.append(cur)
            cur = came_from[cur]
        cell_path.reverse()

        raw_path = [world(ix, iy) for (ix, iy) in cell_path]

        # simplify – drop intermediate points that stay segment‑free
        if not raw_path:
            continue
        simplified = [raw_path[0]]
        for pt in raw_path[1:]:
            if seg_free(simplified[-1], pt):
                simplified[-1] = pt
            else:
                simplified.append(pt)

        return simplified

    return None
