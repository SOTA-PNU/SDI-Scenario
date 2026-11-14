"""Analysis-only exact original B524 a1 test01 planner; checks grid nodes, not edges."""

def plan_path(start_xy, goal_xy, clearance_m):
    """Deterministic grid A* planner.

    A uniform square grid (step 0.5 m) covering the arena is built.  Cells whose
    centre clearance (``map_clearance``) is at least ``clearance_m`` are free.
    The start and goal are snapped to the nearest free cell within 0.5 m; if no
    such cell exists the function returns an empty list, letting the base try the
    next (smaller) clearance.
    The search uses 8‑connectivity with a fixed neighbour order for
    reproducibility.  The heuristic is Euclidean distance.  The resulting path
    consists of the exact start point, the centre of each visited cell and the
    exact goal point.
    """
    import math, heapq, numpy as np

    # arena bounds (inner walls)
    wx0, wx1, wy0, wy1 = WALLS
    step = 0.5  # grid resolution (metres)

    # helper: convert world coord to grid index
    def to_idx(pt):
        ix = int(round((pt[0] - wx0) / step))
        iy = int(round((pt[1] - wy0) / step))
        return ix, iy

    # helper: grid index to world centre
    def to_world(ix, iy):
        return wx0 + ix * step, wy0 + iy * step

    # helper: is cell free?
    def cell_free(ix, iy):
        x, y = to_world(ix, iy)
        # centre must be inside walls
        if not (wx0 <= x <= wx1 and wy0 <= y <= wy1):
            return False
        return map_clearance(x, y) + 1e-12 >= clearance_m

    # find nearest free cell within 0.5 m of a point (including the point itself)
    def nearest_free(pt):
        ix0, iy0 = to_idx(pt)
        # search radius in cells (0.5 m -> 1 cell)
        max_r = 1
        best = None
        best_dist2 = float('inf')
        for dx in range(-max_r, max_r + 1):
            for dy in range(-max_r, max_r + 1):
                ix, iy = ix0 + dx, iy0 + dy
                if not cell_free(ix, iy):
                    continue
                cx, cy = to_world(ix, iy)
                d2 = (cx - pt[0]) ** 2 + (cy - pt[1]) ** 2
                if d2 <= (0.5 + 1e-12) ** 2 and d2 < best_dist2:
                    best = (ix, iy)
                    best_dist2 = d2
        return best

    start_cell = nearest_free(start_xy)
    goal_cell = nearest_free(goal_xy)
    if start_cell is None or goal_cell is None:
        return []

    # A* search
    open_heap = []
    start_idx = start_cell
    goal_idx = goal_cell
    heapq.heappush(open_heap, (0.0, 0.0, start_idx, None))  # (f, g, node, parent)
    came_from = {}
    g_score = {start_idx: 0.0}

    # 8‑connected neighbour offsets in a deterministic order (clockwise)
    neigh_offsets = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)
        if current in came_from:
            continue  # already processed with a better g
        came_from[current] = parent
        if current == goal_idx:
            break
        cx, cy = current
        for dx, dy in neigh_offsets:
            nx, ny = cx + dx, cy + dy
            if not cell_free(nx, ny):
                continue
            # movement cost (Euclidean)
            step_cost = math.hypot(dx * step, dy * step)
            tentative_g = g + step_cost
            if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)] - 1e-12:
                g_score[(nx, ny)] = tentative_g
                # heuristic: straight‑line to goal
                hx = wx0 + nx * step - (wx0 + goal_idx[0] * step)
                hy = wy0 + ny * step - (wy0 + goal_idx[1] * step)
                h = math.hypot(hx, hy)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, (nx, ny), current))

    # reconstruct path if goal reached
    if goal_idx not in came_from:
        return []
    rev = []
    cur = goal_idx
    while cur is not None:
        rev.append(cur)
        cur = came_from[cur]
    rev.reverse()
    # convert to world points, insert exact start and goal
    path = []
    # exact start
    path.append((float(start_xy[0]), float(start_xy[1])))
    for ix, iy in rev:
        x, y = to_world(ix, iy)
        path.append((float(x), float(y)))
    # exact goal
    path.append((float(goal_xy[0]), float(goal_xy[1])))
    return path
