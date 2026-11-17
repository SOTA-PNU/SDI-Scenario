```python
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
def plan_path(start_xy, goal_xy, clearance_m):
    """
    A* on the occupancy grid (resolution 0.5 m) with 8‑connectivity.
    All neighbour edges are verified with ``segment_free`` to avoid diagonal
    shortcuts through obstacles.  After a raw grid path is found it is
    line‑of‑sight simplified, then validated with ``validate_plan``.
    The function records the planning stages via ``plan_stage`` and returns
    the accepted list of (x, y) world points or ``None`` on failure.
    """
    import math, heapq
    # ---- stage: grid -------------------------------------------------
    plan_stage("grid")
    blocked, origin, res = occupancy_grid(clearance_m, 0.5)

    # convert world start/goal to cell indices
    sx, sy = cell_of(start_xy, origin, res)
    gx, gy = cell_of(goal_xy, origin, res)

    # sanity checks – if either endpoint lies in a blocked cell we give up
    max_ix, max_iy = blocked.shape[1] - 1, blocked.shape[0] - 1
    if not (0 <= sx <= max_ix and 0 <= sy <= max_iy) or blocked[sy, sx]:
        return None
    if not (0 <= gx <= max_ix and 0 <= gy <= max_iy) or blocked[gy, gx]:
        return None

    # ---- stage: search -----------------------------------------------
    plan_stage("search")
    start_center = xy_of((sx, sy), origin, res)
    goal_center = xy_of((gx, gy), origin, res)

    # cache for segment_free results to avoid duplicate geometry checks
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

    # A* structures
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
                # movement cost (Euclidean)
                step = math.hypot(dx * res, dy * res)
                new_cost = g + step
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    # heuristic: Euclidean distance to goal centre
                    hx, hy = xy_of((nx, ny), origin, res)
                    heuristic = math.hypot(hx - goal_center[0], hy - goal_center[1])
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, (nx, ny)))
                    came_from[(nx, ny)] = (cx, cy)

    if not found:
        return None

    # reconstruct raw grid path (list of world points)
    raw_cells = []
    cur = (gx, gy)
    while cur != (sx, sy):
        raw_cells.append(cur)
        cur = came_from[cur]
    raw_cells.append((sx, sy))
    raw_cells.reverse()
    raw_path = [xy_of(c, origin, res) for c in raw_cells]

    # ---- stage: simplify ---------------------------------------------
    plan_stage("simplify", input_points=len(raw_path))
    # simple line‑of‑sight simplifier preserving start and goal
    if len(raw_path) <= 2:
        simple_path = raw_path
    else:
        simple_path = [raw_path[0]]
        anchor_idx = 0
        for i in range(2, len(raw_path)):
            if not segment_free(simple_path[-1], raw_path[i], clearance_m):
                # last point that was reachable becomes a new anchor
                simple_path.append(raw_path[i - 1])
                anchor_idx = i - 1
        simple_path.append(raw_path[-1])
    plan_stage("simplify", input_points=len(raw_path), output_points=len(simple_path))

    # ---- stage: validate ---------------------------------------------
    plan_stage("validate")
    validation = validate_plan(simple_path)
    if validation["accepted"]:
        return validation["path"]
    # If the simplified path is rejected, fall back to the raw grid path.
    validation = validate_plan(raw_path)
    if validation["accepted"]:
        return validation["path"]
    # Nothing works – signal failure.
    return None


def lookahead_target(path, pose):
    """
    Return a point on *path* that lies 0.3 – 2.0 m ahead of the robot
    (measured along the polyline).  The implementation mirrors the base
    controller’s forward‑target logic: project the robot onto the path,
    then advance the required distance.  If the forward point cannot be
    obtained (e.g., we are too close to the end), the projection point
    itself is returned – the base guard will handle the fallback.
    """
    # reuse the base utilities that are already in the global namespace
    proj = _base_loop_projection(path, pose[:2])
    forward = _base_loop_forward_target(path, pose, proj)
    if forward is None:
        # No valid forward point – use the projection point as a safe fallback.
        forward = proj["point"]
    return forward


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

    if abs(e) > 0.35:
        v = 0.0
    else:
        v = min(1.5, 2.5 * dist + 0.5)

    w = max(-2.0, min(2.0, 7.5 * e))
    return v, w


def controller(t, pose, env):
    return _base_control_loop(t, pose, env)
# ========================= [END EDIT REGION] ===============================
```