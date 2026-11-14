"""Source fragment for static map geometry helpers. Contains no path search."""

MAP_TOOLS = r'''# ---- 정적 지도 격자 도구: 탐색은 모델이 구현합니다 ----
def _base_grid_parameters(origin, res):
    origin = tuple(float(v) for v in origin)
    res = float(res)
    if len(origin) != 2 or not all(math.isfinite(v) for v in (*origin, res)) or res <= 0.0:
        raise ValueError("grid origin must be finite (x,y) and resolution positive")
    return origin, res


def _base_grid_clearance(clearance_m):
    clearance_m = float(clearance_m)
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("grid clearance must be finite and nonnegative")
    # Same geometric floor as validate_plan/check_path: half width + margin.
    return _BASE_PATH_HALF_WIDTH + max(0.0, clearance_m - _BASE_PATH_HALF_WIDTH)


def occupancy_grid(clearance_m, res):
    """Return (blocked, origin, resolution), without searching for a path.

    blocked is a fresh bool ndarray indexed [iy, ix]; True means occupied.
    origin=(xmin,ymin) is the lower outer corner of cell (0,0), not its centre.
    xy_of gives cell centres. Each centre is tested at the exact clearance
    threshold used by validate_plan, including its 1e-12 m boundary tolerance.
    Free centres do not guarantee free connections: use segment_free on every
    neighbour edge, endpoint connector and simplified segment. Walls are included.
    The grid uses complete cells inside WALLS; no implicit endpoint snapping,
    neighbour search, route repair or A* is performed. Returned arrays are copies
    owned by the caller and do not change the static map or control state.
    """
    need = _base_grid_clearance(clearance_m)
    x0, x1, y0, y1 = WALLS
    origin, res = _base_grid_parameters((x0, y0), res)
    nx, ny = int(math.floor((x1-x0)/res)), int(math.floor((y1-y0)/res))
    if nx < 1 or ny < 1:
        raise ValueError("grid resolution leaves no complete cells inside WALLS")
    if nx*ny > 1000000:
        raise ValueError("grid resolution exceeds one million cells")
    xs = x0 + (np.arange(nx, dtype=np.float64)+0.5)*res
    ys = y0 + (np.arange(ny, dtype=np.float64)+0.5)*res
    # Bound temporary distance arrays even for a requested fine grid.
    blocked = np.empty((ny, nx), dtype=np.bool_)
    rows = max(1, 8192//nx)
    for first in range(0, ny, rows):
        distances = map_clearance(xs[None, :], ys[first:first+rows, None])
        blocked[first:first+rows, :] = distances < need - 1e-12
    return blocked, origin, res


def cell_of(xy, origin, res):
    """World point -> containing (ix,iy), with floor at cell boundaries.

    No clipping is performed. Check 0<=ix<grid.shape[1], 0<=iy<grid.shape[0]
    before indexing; negative numpy indices must not wrap into the map.
    """
    origin, res = _base_grid_parameters(origin, res)
    xy = tuple(float(v) for v in xy)
    if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
        raise ValueError("cell_of requires a finite world point (x,y)")
    return (int(math.floor((xy[0]-origin[0])/res)),
            int(math.floor((xy[1]-origin[1])/res)))


def xy_of(cell, origin, res):
    """Integer (ix,iy) -> centre world point (x,y); no clipping or search."""
    origin, res = _base_grid_parameters(origin, res)
    cell = tuple(float(v) for v in cell)
    if len(cell) != 2 or not all(math.isfinite(v) and v.is_integer() for v in cell):
        raise ValueError("xy_of requires integer cell indices (ix,iy)")
    return (origin[0]+(int(cell[0])+0.5)*res,
            origin[1]+(int(cell[1])+0.5)*res)


def segment_free(p, q, clearance_m):
    """Whether a segment passes validate_plan's sampled map geometry check.

    clearance_m is centre clearance, not body margin. This tests both ends and
    samples at most every 0.1 m via check_path; it does not check the mission's
    start/goal endpoint distances, robot heading or a swept rotating body.
    Invalid input raises ValueError, as check_path does. No control state changes.
    """
    need = _base_grid_clearance(clearance_m)
    checked = check_path([p, q], margin_m=need-_BASE_PATH_HALF_WIDTH)
    return (checked["first_bad_segment"] is None
            and checked["min_gap_map"] >= checked["margin_m"]-1e-12)


'''
