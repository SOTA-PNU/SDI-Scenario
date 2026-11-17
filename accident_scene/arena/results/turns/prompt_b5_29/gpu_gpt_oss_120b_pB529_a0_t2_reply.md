We'll adjust plan_path to use 8-connected with res=0.5 and accept any clearance.
```python
# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
import math
import heapq

def _find_start_cell(start_xy, blocked, origin, res, clearance_m):
    sx, sy = start_xy
    start_cell = cell_of(start_xy, origin, res)
    ny, nx = blocked.shape
    def in_bounds(c):
        ix, iy = c
        return 0 <= ix < nx and 0 <= iy < ny
    if in_bounds(start_cell) and not blocked[start_cell[1], start_cell[0]]:
        return start_cell
    cx, cy = start_cell
    best = None
    best_dist = float('inf')
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cand = (cx + dx, cy + dy)
            if not in_bounds(cand):
                continue
            if blocked[cand[1], cand[0]]:
                continue
            pt = xy_of(cand, origin, res)
            if not segment_free(start_xy, pt, clearance_m):
                continue
            d = math.hypot(pt[0] - sx, pt[1] - sy)
            if d < best_dist:
                best_dist = d
                best = cand
    return best

def plan_path(start_xy, goal_xy, clearance_m):
    """A* planner (8‑connected) on a 0.5 m grid respecting the given clearance."""
    res = 0.5  # grid resolution
    plan_stage("grid")
    blocked, origin, _ = occupancy_grid(clearance_m, res)
    plan_stage("search")
    start_cell = _find_start_cell(start_xy, blocked, origin, res, clearance_m)
    if start_cell is None:
        return None
    ny, nx = blocked.shape
    def in_bounds(c):
        ix, iy = c
        return 0 <= ix < nx and 0 <= iy < ny
    def centre(cell):
        return xy_of(cell, origin, res)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = []
    heapq.heappush(open_set, (0.0, start_cell))
    came_from = {}
    g_score = {start_cell: 0.0}
    goal_reached = None
    while open_set:
        _, current = heapq.heappop(open_set)
        cur_pt = centre(current)
        if (math.hypot(cur_pt[0] - goal_xy[0], cur_pt[1] - goal_xy[1]) <= 0.5 and
                segment_free(cur_pt, goal_xy, clearance_m)):
            goal_reached = current
            break
        cx, cy = current
        for dx, dy in moves:
            neighbor = (cx + dx, cy + dy)
            if not in_bounds(neighbor):
                continue
            if blocked[neighbor[1], neighbor[0]]:
                continue
            p = centre(current)
            q = centre(neighbor)
            if not segment_free(p, q, clearance_m):
                continue
            step_cost = math.hypot(dx * res, dy * res)
            tentative_g = g_score[current] + step_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor] - 1e-12:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = math.hypot(q[0] - goal_xy[0], q[1] - goal_xy[1])
                heapq.heappush(open_set, (tentative_g + h, neighbor))
    if goal_reached is None:
        return None
    # Reconstruct path
    path_cells = []
    cur = goal_reached
    while cur != start_cell:
        path_cells.append(cur)
        cur = came_from.get(cur)
        if cur is None:
            return None
    path_cells.append(start_cell)
    path_cells.reverse()
    raw_path = [centre(c) for c in path_cells]
    # Simplify collinear points
    plan_stage("simplify", input_points=len(raw_path))
    simplified = []
    for pt in raw_path:
        if not simplified:
            simplified.append(pt)
            continue
        if len(simplified) >= 2:
            a = simplified[-2]
            b = simplified[-1]
            c = pt
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-6:
                simplified[-1] = c
                continue
        simplified.append(pt)
    plan_stage("simplify", input_points=len(raw_path), output_points=len(simplified))
    plan_stage("validate")
    return simplified

def lookahead_target(path, pose):
    proj = _base_loop_projection(path, pose[:2], _BASE_LOOP_ST.get("path_progress_s"))
    target = _base_loop_forward_target(path, pose, proj)
    if target is None:
        return pose[:2]
    return target

def follow_cmd(pose, target_xy):
    tx, ty = target_xy
    x, y, yaw = pose
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    e = math.atan2(dy, dx) - yaw
    e = math.atan2(math.sin(e), math.cos(e))
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