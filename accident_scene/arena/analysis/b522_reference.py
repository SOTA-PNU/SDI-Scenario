# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
_PLAN_POSE = None
_PLAN_CLEARANCE = 1.0  # 탐색 중심 여유(m); 검사 최소 기준은 0.5 m
_PATH = None
_REPLAN_PENDING = True
_PLAN_FAILED = True
_BLOCK_SINCE = None
_LAST_T = None
_NEXT_PLAN_T = 0.0

def plan_path():
    global _PLAN_CLEARANCE
    if _PLAN_POSE is None or len(_PLAN_POSE) != 3 or not all(math.isfinite(float(q)) for q in _PLAN_POSE):
        return None
    start_xy, goal_xy = tuple(_PLAN_POSE[:2]), tuple(GOAL[:2])
    if min(map_clearance(*start_xy), map_clearance(*goal_xy)) < 0.5 - 1e-12:
        return None
    def _search_path(center_clearance):
        resolution = 0.5
        x0, x1, y0, y1 = WALLS
        nx, ny = (int((x1 - x0) / resolution), int((y1 - y0) / resolution))

        def point(cell):
            return (x0 + (cell[0] + 0.5) * resolution, y0 + (cell[1] + 0.5) * resolution)

        def containing(xy):
            return (int(math.floor((xy[0] - x0) / resolution)), int(math.floor((xy[1] - y0) / resolution)))
        need = float(center_clearance)
        free_cache, edge_cache = ({}, {})

        def free(cell):
            if cell not in free_cache:
                free_cache[cell] = 0 <= cell[0] < nx and 0 <= cell[1] < ny and (map_clearance(*point(cell)) >= need - 1e-12)
            return free_cache[cell]

        def connected(p, q):
            return segment_clearance(p, q) >= need - 1e-12

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
    requested = min(2.0, max(0.5, float(_PLAN_CLEARANCE)))
    needs = [max(0.5, round(requested - 0.1 * i, 10)) for i in range(int(math.ceil((requested - 0.5) / 0.1)) + 1)]
    if needs[-1] != 0.5:
        needs.append(0.5)
    for need in dict.fromkeys(needs):  # 0.5 m 시도도 한 번뿐; 끝나면 반드시 반환
        if min(map_clearance(*start_xy), map_clearance(*goal_xy)) < need - 1e-12:
            continue
        candidate = _search_path(need)
        if candidate is None:
            continue
        try:
            path = [tuple(float(q) for q in point) for point in candidate]
            if len(path) < 2 or any(len(point) != 2 or not all(math.isfinite(q) for q in point) for point in path):
                continue
            if math.dist(path[0], start_xy) > 1e-9 or math.dist(path[-1], goal_xy) > 1e-9:
                continue
            checked = check_path(path)
            if (checked['first_bad_segment'] is None and checked['min_gap_map'] >= 0.25 - 1e-12
                    and all(segment_clearance(p, q) >= need - 1e-12 for p, q in zip(path, path[1:]))):
                _PLAN_CLEARANCE = need
                return path
        except (TypeError, ValueError, OverflowError):
            continue
    return None

def lookahead_target(path, pose):
    """Project onto the polyline; keep the forward target on one segment."""
    if path is None or len(path) < 2:
        return None
    xy = (float(pose[0]), float(pose[1]))
    projections = []
    for index, (p, q) in enumerate(zip(path, path[1:])):
        dx, dy = q[0] - p[0], q[1] - p[1]
        length2 = dx * dx + dy * dy
        if length2 <= 1e-16:
            continue
        u = max(0.0, min(1.0, ((xy[0]-p[0])*dx + (xy[1]-p[1])*dy) / length2))
        projected = (p[0] + u*dx, p[1] + u*dy)
        projections.append((math.dist(xy, projected), -index, index, u))
    if not projections:
        return None
    _, _, index, u = min(projections)
    if index < len(path)-2 and math.dist(xy, path[index+1]) < 0.35:
        index, u = index+1, 0.0
    p, q = path[index], path[index+1]
    length = math.dist(p, q)
    if length <= 1e-8:
        return None
    # 3 s * 1 m/s agrees with the chooser's progress horizon. Capping at
    # the current segment end prevents aiming across an unchecked corner.
    ahead = min(3.0, (1.0-u)*length)
    if ahead <= 1e-8:
        return None
    for travel in (ahead, ahead*0.5, ahead*0.25):
        fraction = min(1.0, u + travel/length)
        target = (p[0] + fraction*(q[0]-p[0]), p[1] + fraction*(q[1]-p[1]))
        if math.dist(xy, target) > 1e-8 and segment_clearance(xy, target) >= 0.5 - 1e-12:
            return target
    return None

def follow_cmd(pose, target):
    """The verified simple gate's nominal speed/angle rule, without object logic."""
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    error = math.atan2(math.sin(math.atan2(dy, dx)-yaw), math.cos(math.atan2(dy, dx)-yaw))
    v = 0.0 if abs(error) > 0.35 else min(0.3, 0.5*distance + 0.1)
    w = max(-0.4, min(0.4, 1.5*error))
    return v, w

def controller(t, pose, env):
    global _PLAN_POSE, _PLAN_CLEARANCE, _PATH, _REPLAN_PENDING, _PLAN_FAILED
    global _BLOCK_SINCE, _LAST_T, _NEXT_PLAN_T
    _PLAN_POSE = tuple(pose)
    if _LAST_T is not None and t < _LAST_T:
        _PATH, _REPLAN_PENDING, _PLAN_FAILED = None, True, True
        _BLOCK_SINCE, _NEXT_PLAN_T = None, t
    _LAST_T = t
    distance = math.hypot(pose[0]-GOAL[0], pose[1]-GOAL[1])
    yaw_error = math.atan2(math.sin(GOAL[2]-pose[2]), math.cos(GOAL[2]-pose[2]))
    goal_mode = distance <= 0.5
    goal_stop = goal_mode and abs(yaw_error) <= 0.15
    replan_now = (_REPLAN_PENDING or _PLAN_FAILED or _PATH is None) and t >= _NEXT_PLAN_T
    if goal_mode:
        v_follow = 0.0
        w_follow = 0.0 if goal_stop else max(-0.4, min(0.4, 1.5*yaw_error))
        target = GOAL[:2]
    else:
        if replan_now:
            candidate_path = plan_path()
            checked = None if candidate_path is None else check_path(candidate_path)
            accepted = (checked is not None and checked['first_bad_segment'] is None
                        and checked['min_gap_map'] >= 0.25 - 1e-12
                        and math.dist(candidate_path[0], pose[:2]) <= 1e-9
                        and math.dist(candidate_path[-1], GOAL[:2]) <= 1e-9)
            _PATH = candidate_path if accepted else None
            _PLAN_FAILED, _REPLAN_PENDING = not accepted, not accepted
            _NEXT_PLAN_T = t + 1.0 if not accepted else t
            _BLOCK_SINCE = None
        target = None if _PLAN_FAILED else lookahead_target(_PATH, pose)
        if target is None:
            _PATH, _PLAN_FAILED, _REPLAN_PENDING = None, True, True
            _NEXT_PLAN_T = max(_NEXT_PLAN_T, t + 1.0) if replan_now else _NEXT_PLAN_T
            v_follow, w_follow, target = 0.0, 0.0, GOAL[:2]
        else:
            v_follow, w_follow = follow_cmd(pose, target)
    v, w, info = choose_command(v_follow, w_follow, target)
    blocked = (info['chosen'] == 'stop' and v == 0.0 and w == 0.0
               and info['candidates']['follow']['t_hit_map'] is not None)
    if blocked:
        if _BLOCK_SINCE is None:
            _BLOCK_SINCE = t
        elif t - _BLOCK_SINCE > 10.0:
            _REPLAN_PENDING = True
            _PLAN_CLEARANCE = min(2.0, _PLAN_CLEARANCE + 0.1)
            _NEXT_PLAN_T = t
            _BLOCK_SINCE = None
    else:
        _BLOCK_SINCE = None
    done = goal_stop and v == 0.0 and w == 0.0
    return v, w, done
# ========================= [END EDIT REGION] ===============================
