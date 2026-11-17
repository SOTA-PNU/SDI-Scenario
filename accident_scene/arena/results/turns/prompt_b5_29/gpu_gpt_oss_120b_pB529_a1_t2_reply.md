```python
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
def plan_path(start_xy, goal_xy, clearance_m):
    """
    A* search on the occupancy grid (resolution 0.5 m) with 8‑connectivity.
    Every neighbour edge is checked with ``segment_free`` to avoid diagonal
    shortcuts through obstacles.  The raw grid path is then line‑of‑sight
    simplified while preserving collision‑free segments, and finally validated
    with ``validate_plan``.  Planning stages are recorded via ``plan_stage``.
    The function returns the accepted list of (x, y) world points or ``None``.
    """
    import math, heapq

    # -----------------------------------------------------------------
    # Stage: grid – obtain the occupancy grid for the requested clearance.
    # -----------------------------------------------------------------
    plan_stage("grid")
    blocked, origin, res = occupancy_grid(clearance_m, 0.5)

    # Convert start/goal world positions to cell indices.
    sx, sy = cell_of(start_xy, origin, res)
    gx, gy = cell_of(goal_xy, origin, res)

    max_ix, max_iy = blocked.shape[1] - 1, blocked.shape[0] - 1
    # If either endpoint lies in a blocked cell we cannot plan.
    if not (0 <= sx <= max_ix and 0 <= sy <= max_iy) or blocked[sy, sx]:
        return None
    if not (0 <= gx <= max_ix and 0 <= gy <= max_iy) or blocked[gy, gx]:
        return None

    # ---------------------------------------------------------------
    # Stage: search – A* on the grid, caching segment‑free results.
    # ---------------------------------------------------------------
    plan_stage("search")
    start_center = xy_of((sx, sy), origin, res)
    goal_center = xy_of((gx, gy), origin, res)

    # Cache for segment_free checks to avoid duplicate geometry work.
    _seg_cache = {}

    def edge_free(c1, c2):
        key = (c1, c2) if c1 <= c2 else (c2, c1)
        if key in _seg_cache:
            return _seg_cache[key]
        p1 = xy_of(c1, origin, res)
        p2 = xy_of(c2, origin, res)
        ok = segment_free(p1, p2, clearance_m)
        _seg_cache[key] = ok
        return ok

    frontier = [(0.0, 0.0, (sx, sy))]          # (f, g, cell)
    came_from = {}
    cost_so_far = {(sx, sy): 0.0}
    found = False

    while frontier:
        _, g, current = heapq.heappop(frontier)
        if current == (gx, gy):
            found = True
            break

        cx, cy = current
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx <= max_ix and 0 <= ny <= max_iy):
                    continue
                if blocked[ny, nx]:
                    continue
                if not edge_free((cx, cy), (nx, ny)):
                    continue
                step = math.hypot(dx * res, dy * res)
                new_cost = g + step
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    hx, hy = xy_of((nx, ny), origin, res)
                    heuristic = math.hypot(hx - goal_center[0], hy - goal_center[1])
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, (nx, ny)))
                    came_from[(nx, ny)] = (cx, cy)

    if not found:
        return None

    # Reconstruct the raw grid path (list of world points).
    raw_cells = []
    cur = (gx, gy)
    while cur != (sx, sy):
        raw_cells.append(cur)
        cur = came_from[cur]
    raw_cells.append((sx, sy))
    raw_cells.reverse()
    raw_path = [xy_of(c, origin, res) for c in raw_cells]

    # ---------------------------------------------------------------
    # Stage: simplify – line‑of‑sight simplifier that keeps start/goal.
    # ---------------------------------------------------------------
    plan_stage("simplify", input_points=len(raw_path))
    if len(raw_path) <= 2:
        simple_path = raw_path
    else:
        simple_path = [raw_path[0]]
        for i in range(2, len(raw_path)):
            if not segment_free(simple_path[-1], raw_path[i], clearance_m):
                simple_path.append(raw_path[i - 1])
        simple_path.append(raw_path[-1])
    plan_stage("simplify", input_points=len(raw_path), output_points=len(simple_path))

    # ---------------------------------------------------------------
    # Stage: validate – ensure the path (with start/goal connectors) is safe.
    # ---------------------------------------------------------------
    plan_stage("validate")
    validation = validate_plan(simple_path)
    if validation["accepted"]:
        return validation["path"]
    # If the simplified path is rejected, fall back to the raw grid path.
    validation = validate_plan(raw_path)
    if validation["accepted"]:
        return validation["path"]
    return None


def lookahead_target(path, pose):
    """
    Compute the forward target exactly as the base controller would,
    but return it directly so the guard sees a safe connector.
    """
    # Project the robot onto the path.
    proj = _base_loop_projection(path, pose[:2])
    # Use the base forward‑target logic (which respects the 0.3–2.0 m limits).
    forward = _base_loop_forward_target(path, pose, proj)
    # If forward target cannot be obtained (e.g., near the end), fall back
    # to the projection point – the guard will handle the fallback.
    return forward if forward is not None else proj["point"]


def follow_cmd(pose, target_xy):
    """
    Pure geometric follower.
    - ``dist`` – Euclidean distance to the target.
    - ``e`` – signed yaw error wrapped to [-π, π].
    - If |e| > 0.35 rad, stop forward motion (v = 0).
      Otherwise v = min(1.5, 2.5·dist + 0.5).
    - Angular speed w = clamp(7.5·e, ‑2.0, 2.0).
    """
    dx = target_xy[0] - pose[0]
    dy = target_xy[1] - pose[1]
    dist = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    e = angle_to_target - pose[2]
    e = math.atan2(math.sin(e), math.cos(e))  # wrap to [-π,π]

    v = 0.0 if abs(e) > 0.35 else min(1.5, 2.5 * dist + 0.5)
    w = max(-2.0, min(2.0, 7.5 * e))
    return v, w


def controller(t, pose, env):
    return _base_control_loop(t, pose, env)
# ========================= [END EDIT REGION] ===============================
```