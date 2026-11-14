"""sim2d - a 2D pre-screening simulator for the Nova Carter warehouse arena.

Purpose: let layout designers try obstacle layouts against the L3 reference
avoider (the design reference - not the accident base, which is the
levels_zero2level empty controller) in a second, without booting Isaac Sim.  It is an APPROXIMATION of
levels/common/carter_env.py + the Isaac physics; use it to rank / reject
layouts, then confirm the chosen one in Isaac.

World
  * static: arena_map.json (0.25 m PhysX overlap sweep, z 0.05-0.55 m).  Every
    occupied cell is a tall obstacle (visible to the scan, physically blocking).
    For the SCAN the grid is upsampled to 0.05 m and "carved" with the free
    space that 327 x 61 real Isaac rays flew through in the recorded accident
    run (now distilled into shelving_carve.npz): the west shelving is porous at
    z=0.35 and the grid over-states surfaces by up to 0.25 m.  Held-out check:
    |range error| < 0.10 m on 84% of beams (29% without carving).  Carving
    only covers what was visible from the x~-6, y in [-1, 3.4] lane (shelf
    face y ~ -4..9, Forklift_01); elsewhere static geometry stays solid.
    Cells are tagged for the collision oracle:
        wall     - warehouse shell (walls, pillars, racks, pallets): NOT counted
        forklift_01 / forklift - cells intersecting those prims' bboxes: counted
        shelf    - cells intersecting /World/warehouse_extras bbox (west
                   shelving, wall column excluded): counted
    (the grid over-approximates real surfaces by up to one 0.25 m cell)
  * injected boxes: dicts {x, y, width (X), depth (Y), height}; axis-aligned,
    on the floor.  Always counted by the oracle.  Visible to the scan only if
    height >= scan z (0.35).  Physically blocking if height >= LOW_BOX_H
    (0.08 m, measured: a 0.05 m box is driven over while scraping, 0.10-0.30 m
    boxes stop the robot dead); lower boxes are traversable but every step
    spent on top of one counts as a contact.

Robot  (unicycle, 60 Hz, pose = chassis_link origin = rotation centre)
  * command path: 2-frame transport delay, then a rate limiter
    (2.05 m/s^2 linear, 2.6 rad/s^2 angular).  FITTED to the Isaac L3
    solution trajectory (levels/level3/results/solution_trajectory.csv):
    RMS error 0.008 m/s (v step) and 0.014 rad/s (w step).  This matches the
    DifferentialController node's maxAcceleration/maxAngularAcceleration
    rate limiter.  Still an approximation (no wheel slip, no tipping).
  * chassis footprint 0.708 x 0.477 m, NOT centred on the origin: it spans
    [-0.568, +0.140] m along the heading.  Evidence: in 6 Isaac runs the
    robot stopped against a box with its origin 0.14 m short of the box face,
    and the spawn-time robot bbox front edge is 0.141 m ahead of the origin.
    The rear therefore swings a 0.62 m radius when turning in place.
  * blocking: a step whose footprint would overlap a blocking obstacle is
    rejected (rotation-only, then translation-only fallbacks are tried), so
    "stuck against an invisible box" shows up as in Isaac.

Oracle: a step is a collision step when the footprint (inflated by 2 cm) touches
an oracle-counting obstacle, or a counted obstacle blocked the attempted move.
collision_steps is the analogue of the harness' collision_count (Isaac emits
~1 contact event per step while touching).

API
  run(start, goal, boxes=None, timeout=120, ...) -> dict
  path_exists(start, goal, boxes=None, clearance=0.6) -> dict | None
  (boxes=None with world=W given -> use W's boxes; pass world=World() once
   and reuse it to skip the ~0.1 s load)
  plot(result, boxes, path_png, ...)
  World(boxes).raycast_scan(pose, n_beams=61, fov_deg=180, z=0.35, max_range=6)
  make_l3_controller(goal) -> controller(t, pose, env) (verbatim L3 logic)

numpy only (matplotlib used by plot() when importable, else a built-in PNG
rasteriser).
"""

from __future__ import annotations

import heapq
import json
import math
import os
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA_JSON = os.path.join(HERE, "arena_map.json")  # bundled occupancy sweep

DT = 1.0 / 60.0
# --- robot model (see module docstring) ---
FOOT_FRONT = 0.140   # m ahead of the chassis_link origin
FOOT_REAR = -0.568   # m (0.708 m long)
FOOT_HALF_W = 0.2385  # m (0.477 m wide)
CMD_DELAY_FRAMES = 2
ACC_LIN = 2.05       # m/s^2
ACC_ANG = 2.6        # rad/s^2
SCAN_RES = 0.05      # m, scan-geometry raster (coarse grid upsampled)
# Where real Isaac rays at each scan height passed THROUGH the porous west
# shelving. Distilled once with carve_from_scans() from 327 poses x 61 beams
# recorded at z=0.35 and z=0.15 in the accident scene; only the geometry is
# kept, so this does not depend on which controller drove that run.
CARVE_DATA = os.path.join(HERE, "shelving_carve.npz")
LOW_BOX_H = 0.08     # boxes lower than this are driven over (still counted)
CONTACT_MARGIN = 0.02
SCAN_Z = 0.35
POS_TOL = 0.75
YAW_TOL = 0.26

TAG_FREE, TAG_WALL, TAG_FORKLIFT_01, TAG_FORKLIFT, TAG_SHELF = 0, 1, 2, 3, 4
TAG_NAMES = {TAG_WALL: "wall", TAG_FORKLIFT_01: "forklift_01",
             TAG_FORKLIFT: "forklift", TAG_SHELF: "shelf"}


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def _norm_box(b):
    return {"x": float(b["x"]), "y": float(b["y"]),
            "width": float(b.get("width", b.get("w"))),
            "depth": float(b.get("depth", b.get("d"))),
            "height": float(b.get("height", b.get("h")))}


# =============================================================================
# world
# =============================================================================
class World:
    def __init__(self, boxes=(), arena_json=ARENA_JSON, scan_z=SCAN_Z, carve=True):
        with open(arena_json) as f:
            m = json.load(f)
        self.x0, self.y0, self.step = m["x0"], m["y0"], m["step"]
        self.nx, self.ny = m["nx"], m["ny"]
        self.occ = np.array(m["grid"], dtype=bool)  # [j, i]
        assert self.occ.shape == (self.ny, self.nx)
        self.bboxes = m.get("bboxes", {})
        self.spawn = m.get("spawn")
        self.scan_z = scan_z
        # --- oracle tags
        tag = np.where(self.occ, TAG_WALL, TAG_FREE).astype(np.int8)
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        cx0 = self.x0 + ii * self.step
        cy0 = self.y0 + jj * self.step
        cx1, cy1 = cx0 + self.step, cy0 + self.step

        def inter(bb):
            return (cx1 > bb[0]) & (cx0 < bb[3]) & (cy1 > bb[1]) & (cy0 < bb[4])

        ex = self.bboxes.get("/World/warehouse_extras")
        if ex:
            # exclude the outer wall column (cell centre x < -10.25)
            tag[self.occ & inter(ex) & ((cx0 + cx1) / 2 > -10.25)] = TAG_SHELF
        for key, t in (("/World/warehouse_with_forklifts/Forklift", TAG_FORKLIFT),
                       ("/World/warehouse_with_forklifts/Forklift_01", TAG_FORKLIFT_01)):
            bb = self.bboxes.get(key)
            if bb:
                tag[self.occ & inter(bb)] = t
        self.tag = tag
        self.counts = tag >= TAG_FORKLIFT_01
        # occupied-cell centres (for SAT footprint tests)
        oj, oi = np.nonzero(self.occ)
        self._cell_cx = self.x0 + (oi + 0.5) * self.step
        self._cell_cy = self.y0 + (oj + 0.5) * self.step
        self._cell_tag = tag[oj, oi]
        # --- scan geometry: the coarse grid upsampled to SCAN_RES, optionally
        # "carved" with the free space real Isaac rays flew through
        r = int(round(self.step / SCAN_RES))
        self._fine = np.kron(self.occ, np.ones((r, r), dtype=bool))
        self._carve = _load_carve(self) if carve else {}
        self._scan_grids = {}
        self.set_boxes(boxes)

    # ------------------------------------------------------------ boxes
    def set_boxes(self, boxes):
        self.boxes = [_norm_box(b) for b in boxes]
        if self.boxes:
            B = np.array([[b["x"] - b["width"] / 2, b["y"] - b["depth"] / 2,
                           b["x"] + b["width"] / 2, b["y"] + b["depth"] / 2,
                           b["height"]] for b in self.boxes])
        else:
            B = np.zeros((0, 5))
        self._bx = B

    # ------------------------------------------------------------ queries
    def cell_of(self, x, y):
        return (int(math.floor((x - self.x0) / self.step)),
                int(math.floor((y - self.y0) / self.step)))

    def occupied_xy(self, x, y):
        i, j = self.cell_of(x, y)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return bool(self.occ[j, i])
        return False

    # ------------------------------------------------------------ scan
    def scan_grid(self, z):
        """Static geometry seen by a scan plane at height z (SCAN_RES grid)."""
        key = round(float(z), 3)
        g = self._scan_grids.get(key)
        if g is None:
            g = self._fine.copy()
            free = self._carve.get(key)
            if free is not None:
                g &= ~free
            self._scan_grids[key] = g
        return g

    def raycast_scan(self, pose, n_beams=61, fov_deg=180.0, z=None,
                     max_range=6.0, sample=0.02):
        """Re-implementation of Harness.raycast_scan: planar fan around the
        heading, rays start 0.5 m out, first hit, range from the robot centre,
        range == max_range means no hit.  Static cells are always visible;
        boxes only if height >= z.  A ray whose start point is already inside
        an obstacle reports 0.5 (PhysX initial-overlap hit)."""
        z = self.scan_z if z is None else z
        G = self.scan_grid(z)
        nyf, nxf = G.shape
        x, y, yaw = pose
        half = math.radians(fov_deg) / 2.0
        b = -half + np.arange(n_beams) * (2.0 * half / (n_beams - 1))
        a = yaw + b
        dx, dy = np.cos(a), np.sin(a)
        ox, oy = x + 0.5 * dx, y + 0.5 * dy
        L = max_range - 0.5
        s = np.arange(0.0, L + 1e-9, sample)
        px = ox[:, None] + dx[:, None] * s[None, :]
        py = oy[:, None] + dy[:, None] * s[None, :]
        ci = np.floor((px - self.x0) / SCAN_RES).astype(np.int64)
        cj = np.floor((py - self.y0) / SCAN_RES).astype(np.int64)
        inside = (ci >= 0) & (ci < nxf) & (cj >= 0) & (cj < nyf)
        hit = np.zeros_like(inside)
        hit[inside] = G[cj[inside], ci[inside]]
        anyhit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        d = np.where(anyhit, s[first], L)
        # refine the static hit to the exact cell boundary (slab test on the
        # hit cell) so the error is not quantised to `sample`
        for k in np.nonzero(anyhit & (first > 0))[0]:
            i, j = ci[k, first[k]], cj[k, first[k]]
            t = _ray_aabb(ox[k], oy[k], dx[k], dy[k],
                          self.x0 + i * SCAN_RES, self.y0 + j * SCAN_RES,
                          self.x0 + (i + 1) * SCAN_RES, self.y0 + (j + 1) * SCAN_RES)
            if t is not None and t <= d[k] + 1e-9:
                d[k] = max(t, s[first[k] - 1])
        # injected boxes (exact slab test)
        vis = self._bx[self._bx[:, 4] >= z] if len(self._bx) else self._bx
        if len(vis):
            tb = _ray_aabb_many(ox, oy, dx, dy, vis)  # (n_beams, n_boxes)
            d = np.minimum(d, np.min(tb, axis=1))
        d = np.minimum(d, L)
        return list(zip(b.tolist(), (0.5 + d).tolist()))

    # ------------------------------------------------------------ footprint
    def footprint_hits(self, pose, margin=0.0, front=FOOT_FRONT, rear=FOOT_REAR,
                       half_w=FOOT_HALF_W):
        """Obstacles overlapped by the chassis rectangle (inflated by margin).
        Returns (blocking_partners, counted_partners) as sets of labels:
        'box<i>', 'wall', 'forklift', 'forklift_01', 'shelf'."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        off = 0.5 * (front + rear)
        hl = 0.5 * (front - rear) + margin
        hw = half_w + margin
        cx, cy = x + c * off, y + s * off
        # |extent| of the rotated rect projected on world axes
        ex = abs(c) * hl + abs(s) * hw
        ey = abs(s) * hl + abs(c) * hw
        blocking, counted = set(), set()
        # --- static cells (AABBs) - SAT
        h = self.step / 2.0
        sel = (np.abs(self._cell_cx - cx) < ex + h) & (np.abs(self._cell_cy - cy) < ey + h)
        if sel.any():
            qx = self._cell_cx[sel] - cx
            qy = self._cell_cy[sel] - cy
            pu = np.abs(qx * c + qy * s)
            pv = np.abs(-qx * s + qy * c)
            ru = h * (abs(c) + abs(s))
            ov = (pu < hl + ru) & (pv < hw + ru)
            for t in np.unique(self._cell_tag[sel][ov]):
                lab = TAG_NAMES[int(t)]
                blocking.add(lab)
                if t >= TAG_FORKLIFT_01:
                    counted.add(lab)
        # --- boxes - SAT
        for k, bx in enumerate(self._bx):
            bcx, bcy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            bhx, bhy = (bx[2] - bx[0]) / 2, (bx[3] - bx[1]) / 2
            qx, qy = bcx - cx, bcy - cy
            if abs(qx) >= ex + bhx or abs(qy) >= ey + bhy:
                continue
            if abs(qx * c + qy * s) >= hl + bhx * abs(c) + bhy * abs(s):
                continue
            if abs(-qx * s + qy * c) >= hw + bhx * abs(s) + bhy * abs(c):
                continue
            lab = f"box{k}"
            counted.add(lab)
            if bx[4] >= LOW_BOX_H:
                blocking.add(lab)
        return blocking, counted


_CARVE_CACHE = {}


def carve_from_scans(world, samples, key, z, res=SCAN_RES, stop_short=0.03):
    """Carve mask (fine grid, True = remove from the scan geometry): every
    cell real Isaac rays crossed before their hit, minus the cells where some
    ray actually ended (hit evidence beats pass evidence).
    samples = [{'pose': (x,y,yaw), key: [61 ranges]}]."""
    nyf, nxf = world._fine.shape
    free = np.zeros((nyf, nxf), dtype=bool)
    hitc = np.zeros((nyf, nxf), dtype=bool)
    n = len(samples[0][key])
    bb = -math.pi / 2 + np.arange(n) * (math.pi / (n - 1))
    s = np.arange(0.5, 6.0 + 1e-9, 0.01)
    for smp in samples:
        x, y, yaw = smp["pose"]
        rng = np.asarray(smp[key], float)
        end = np.where(rng < 5.999, rng - stop_short, 6.0)
        a = yaw + bb
        px = x + np.cos(a)[:, None] * s[None, :]
        py = y + np.sin(a)[:, None] * s[None, :]
        m = s[None, :] <= end[:, None]
        ci = np.floor((px[m] - world.x0) / res).astype(np.int64)
        cj = np.floor((py[m] - world.y0) / res).astype(np.int64)
        ok = (ci >= 0) & (ci < nxf) & (cj >= 0) & (cj < nyf)
        free[cj[ok], ci[ok]] = True
        h = rng < 5.999
        hx = x + np.cos(a[h]) * (rng[h] + 0.01)
        hy = y + np.sin(a[h]) * (rng[h] + 0.01)
        hi = np.floor((hx - world.x0) / res).astype(np.int64)
        hj = np.floor((hy - world.y0) / res).astype(np.int64)
        ok = (hi >= 0) & (hi < nxf) & (hj >= 0) & (hj < nyf)
        hitc[hj[ok], hi[ok]] = True
    return free & ~hitc


def _load_carve(world, data=CARVE_DATA):
    """Carve masks per scan height, {z: bool mask on the fine grid}.

    To rebuild from a fresh scan log (dict with "samples", "z_blind",
    "z_seeing" as written by a run that records both fans), call
    carve_from_scans(world, samples, key, z) per height and np.savez_compressed
    the result with shape=world._fine.shape, keys "z<height>".
    """
    key = (data, tuple(world._fine.shape))
    if key in _CARVE_CACHE:
        return _CARVE_CACHE[key]
    out = {}
    if os.path.exists(data):
        npz = np.load(data)
        if npz["shape"].tolist() == list(world._fine.shape):
            out = {float(k[1:]): npz[k] for k in npz.files if k.startswith("z")}
    _CARVE_CACHE[key] = out
    return out


def _ray_aabb(ox, oy, dx, dy, x0, y0, x1, y1):
    tmin, tmax = -1e18, 1e18
    for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
    if tmax < max(tmin, 0.0):
        return None
    return max(tmin, 0.0)


def _ray_aabb_many(ox, oy, dx, dy, B):
    """rays (n,) vs boxes (m,5 [x0,y0,x1,y1,h]) -> entry distance (n,m), inf if miss."""
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)[:, None]
        idy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)[:, None]
        tx1 = (B[None, :, 0] - ox[:, None]) / idx
        tx2 = (B[None, :, 2] - ox[:, None]) / idx
        ty1 = (B[None, :, 1] - oy[:, None]) / idy
        ty2 = (B[None, :, 3] - oy[:, None]) / idy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    ok = tmax >= np.maximum(tmin, 0.0)
    return np.where(ok, np.maximum(tmin, 0.0), np.inf)


# =============================================================================
# the L3 reference avoider - levels/level3/solution_carter_run.py [EDIT REGION],
# logic verbatim, GOAL parameterised (constants overridable for experiments)
# =============================================================================
L3_PARAMS = dict(
    CRUISE_V=1.0, STOP_DIST=0.30, K_HEADING=1.8, W_MAX=1.5, BEARING_GATE=0.5,
    K_ALIGN=2.0, ALIGN_W_MAX=0.8, YAW_DONE=0.08,
    BLOCK_RANGE=2.2, SIDE_MARGIN=0.55, THREAT_DIST=1.4, FRONT_CONE=0.6, AVOID_V=0.5,
)


def make_l3_controller(goal, **overrides):
    P = dict(L3_PARAMS)
    unknown = set(overrides) - set(P)
    if unknown:
        raise ValueError(f"unknown controller params {unknown}")
    P.update(overrides)
    GOAL = tuple(goal)
    CRUISE_V, STOP_DIST, K_HEADING, W_MAX = P["CRUISE_V"], P["STOP_DIST"], P["K_HEADING"], P["W_MAX"]
    BEARING_GATE, K_ALIGN, ALIGN_W_MAX, YAW_DONE = P["BEARING_GATE"], P["K_ALIGN"], P["ALIGN_W_MAX"], P["YAW_DONE"]
    BLOCK_RANGE, SIDE_MARGIN, THREAT_DIST = P["BLOCK_RANGE"], P["SIDE_MARGIN"], P["THREAT_DIST"]
    FRONT_CONE, AVOID_V = P["FRONT_CONE"], P["AVOID_V"]

    def controller(t, pose, env):
        x, y, yaw = pose
        gx, gy, gyaw = GOAL
        d = math.hypot(gx - x, gy - y)

        if d <= STOP_DIST:
            # inside the goal disc: align to the commanded arrival yaw, finish
            yerr = _wrap(gyaw - yaw)
            if abs(yerr) > YAW_DONE:
                return 0.0, _clamp(K_ALIGN * yerr, ALIGN_W_MAX), False
            return 0.0, 0.0, True

        err = _wrap(math.atan2(gy - y, gx - x) - yaw)
        scan = env.raycast_scan()
        # hits at or beyond the goal distance cannot be in the way
        blocks = [(b - math.atan2(SIDE_MARGIN, r), b + math.atan2(SIDE_MARGIN, r))
                  for b, r in scan if r < min(BLOCK_RANGE, d)]

        def blocked(rb):
            return any(lo <= rb <= hi for lo, hi in blocks)

        front = [r for b, r in scan if abs(b) <= FRONT_CONE]
        threat = min(front) if front else 99.0

        if not blocked(err) and threat >= THREAT_DIST:
            # goal direction clear: L1 behaviour unchanged
            w = _clamp(K_HEADING * err, W_MAX)
            v = 0.0 if abs(err) > BEARING_GATE else min(CRUISE_V, 0.8 * d)
            return v, w, False

        # goal direction blocked: steer to the clear bearing nearest it
        cands = [b for b, r in scan if r >= BLOCK_RANGE and not blocked(b)]
        if not cands:
            return 0.0, W_MAX * 0.6, False  # boxed in: rotate in place and rescan
        best = min(cands, key=lambda b: abs(_wrap(b - err)))
        w = _clamp(K_HEADING * best, W_MAX)
        v = AVOID_V if threat < THREAT_DIST or abs(best) > BEARING_GATE else 0.8 * CRUISE_V
        return v, w, False

    controller.params = P
    controller.goal = GOAL
    return controller


class _EnvView:
    """The `env` handle a controller sees: env.raycast_scan(...) at the
    current ground-truth pose, same signature/defaults as the harness."""

    def __init__(self, world):
        self.world = world
        self.pose = (0.0, 0.0, 0.0)
        self.scans = 0

    def raycast_scan(self, n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0):
        self.scans += 1
        return self.world.raycast_scan(self.pose, n_beams, fov_deg, z, max_range)

    def gt_pose(self):
        return self.pose


# =============================================================================
# mission runner
# =============================================================================
def run(start, goal, boxes=None, timeout=120.0, controller=None, world=None,
        pos_tol=POS_TOL, yaw_tol=YAW_TOL, settle_s=1.0,
        abort_after_collision_s=None, delay_frames=CMD_DELAY_FRAMES,
        acc_lin=ACC_LIN, acc_ang=ACC_ANG, stuck_window_s=3.0, keep_traj=True,
        max_time_to_goal=None, startup_delay_s=0.0):
    """Run one mission.  start/goal = (x, y, yaw).  controller defaults to the
    L3 reference avoider aimed at `goal`.  Mirrors Harness.run_mission: loop at
    60 Hz until controller done / timeout (/ abort), then 1 s of zero command.

    abort_after_collision_s: the harness' economy abort (the Isaac runners use 5.0).
    max_time_to_goal: optional time-budget oracle (L3 uses 12 s).
    startup_delay_s: commands issued before this are dropped (Isaac's first
      /cmd_vel takes effect 0.0-0.25 s late run-to-run; default 0).

    Returns a dict:
      verdict_like       'pass' iff reached & yaw_ok & no collision
                         (& time_to_goal <= max_time_to_goal when given)
      start_overlap      obstacles the START footprint already overlaps
                         (should be empty - pick another start otherwise)
      reached            final position within pos_tol (harness reached_goal pos part)
      yaw_ok             final |yaw err| <= yaw_tol
      entered_goal_disc  ever within pos_tol; time_to_goal = first entry time
      collision          collision_steps > 0
      first_collision_t, collision_steps, partners {label: steps}
      blocked_steps      steps whose commanded motion was physically blocked
      stuck              not reached and moved < 0.10 m in the last stuck_window_s
      done, timed_out, aborted, final_pose, final_dist, min_goal_dist, sim_time
      traj               np.ndarray (N, 4) of t, x, y, yaw (every step)
    """
    if world is None:
        world = World(boxes or [])
    elif boxes is not None:
        world.set_boxes(boxes)      # boxes=None keeps the world's boxes
    ctrl = controller or make_l3_controller(goal)
    env = _EnvView(world)
    gx, gy, gyaw = goal
    x, y, yaw = (float(v) for v in start)
    start_overlap = sorted(set().union(*world.footprint_hits((x, y, yaw))))
    queue = deque([(0.0, 0.0)] * delay_frames)
    v_s = w_s = 0.0
    t = 0.0
    k = 0
    traj = []
    reached_t = None
    min_dist = float("inf")
    first_col = None
    col_steps = 0
    blocked_steps = 0
    partners = {}
    done = False
    timed_out = aborted = False
    n_steps_max = int(round(timeout / DT)) + 1

    def step_dyn(cmd):
        nonlocal x, y, yaw, v_s, w_s, col_steps, first_col, blocked_steps
        queue.append(cmd)
        vc, wc = queue.popleft()
        v_s += _clamp(vc - v_s, acc_lin * DT)
        w_s += _clamp(wc - w_s, acc_ang * DT)
        # unicycle, midpoint heading
        ym = yaw + 0.5 * w_s * DT
        cand = (x + v_s * DT * math.cos(ym), y + v_s * DT * math.sin(ym), _wrap(yaw + w_s * DT))
        blk, cnt_full = world.footprint_hits(cand)
        hit_counted = set()
        if blk:
            blocked_steps += 1
            hit_counted |= cnt_full & blk
            # fallbacks: rotate only, translate only, stay
            for alt in ((x, y, cand[2]), (cand[0], cand[1], yaw), (x, y, yaw)):
                if not world.footprint_hits(alt)[0] or alt == (x, y, yaw):
                    cand = alt
                    break
        x, y, yaw = cand
        _, cnt = world.footprint_hits((x, y, yaw), margin=CONTACT_MARGIN)
        hit_counted |= cnt
        if hit_counted:
            col_steps += 1
            if first_col is None:
                first_col = t
            for p in hit_counted:
                partners[p] = partners.get(p, 0) + 1

    while True:
        t = k * DT
        d = math.hypot(gx - x, gy - y)
        min_dist = min(min_dist, d)
        if reached_t is None and d <= pos_tol:
            reached_t = t
        if keep_traj:
            traj.append((t, x, y, yaw))
        if k >= n_steps_max - 1 or t >= timeout:
            timed_out = True
            break
        if done:
            break
        if (abort_after_collision_s is not None and first_col is not None
                and t - first_col >= abort_after_collision_s):
            aborted = True
            break
        env.pose = (x, y, yaw)
        v, w, done = ctrl(t, (x, y, yaw), env)
        if t < startup_delay_s:
            v, w = 0.0, 0.0
        step_dyn((float(v), float(w)))
        k += 1
    # settle: zero command for settle_s
    for _ in range(int(round(settle_s / DT))):
        t = k * DT
        step_dyn((0.0, 0.0))
        k += 1
        if keep_traj:
            traj.append((k * DT, x, y, yaw))

    traj = np.array(traj)
    final_d = math.hypot(gx - x, gy - y)
    yaw_err = abs(_wrap(yaw - gyaw))
    reached = final_d <= pos_tol
    yaw_ok = yaw_tol is None or yaw_err <= yaw_tol
    stuck = False
    if keep_traj and not reached and len(traj) > 1:
        tail = traj[traj[:, 0] >= traj[-1, 0] - stuck_window_s]
        stuck = float(np.hypot(*(tail[:, 1:3].max(0) - tail[:, 1:3].min(0)))) < 0.10
    ttg_ok = max_time_to_goal is None or (reached_t is not None and reached_t <= max_time_to_goal)
    return {
        "verdict_like": "pass" if (reached and yaw_ok and col_steps == 0 and ttg_ok) else "fail",
        "ttg_ok": ttg_ok,
        "start_overlap": start_overlap,
        "reached": reached,
        "yaw_ok": yaw_ok,
        "entered_goal_disc": reached_t is not None,
        "time_to_goal": None if reached_t is None else round(reached_t, 3),
        "collision": col_steps > 0,
        "first_collision_t": None if first_col is None else round(first_col, 3),
        "collision_steps": col_steps,
        "partners": partners,
        "blocked_steps": blocked_steps,
        "stuck": stuck,
        "done": bool(done),
        "timed_out": timed_out and not done,
        "aborted": aborted,
        "final_pose": (round(x, 4), round(y, 4), round(yaw, 4)),
        "final_dist": round(final_d, 4),
        "final_yaw_err": round(yaw_err, 4),
        "min_goal_dist": round(min_dist, 4),
        "sim_time": round(k * DT, 3),
        "start": tuple(start),
        "goal": tuple(goal),
        "boxes": [dict(b) for b in world.boxes],
        "traj": traj,
    }


# =============================================================================
# path existence (configuration-space check)
# =============================================================================
CS_RES = 0.0625  # m, quarter of the arena cell (static geometry is exact on it)
_STATIC_DIST = {}


def _surface_dist(obs, res, rmax_cells):
    """Exact distance (m) from each cell centre to the union of occupied cell
    SQUARES (separable, capped at rmax_cells).  numpy only."""
    ny, nx = obs.shape
    BIG = 10 ** 6
    idx = np.arange(ny)[:, None].repeat(nx, 1)
    last = np.maximum.accumulate(np.where(obs, idx, -BIG), axis=0)
    nxt = np.minimum.accumulate(np.where(obs, idx, 2 * BIG)[::-1], axis=0)[::-1]
    g = np.minimum(idx - last, nxt - idx).astype(np.float64)       # centre-centre, cells
    g = np.minimum(np.maximum(g - 0.5, 0.0), rmax_cells + 1.0)      # centre-surface
    g2 = g ** 2
    cap = float((rmax_cells + 1) ** 2)
    best = g2.copy()
    for dk in range(1, rmax_cells + 2):
        h2 = (dk - 0.5) ** 2
        sh = np.full_like(g2, cap)
        sh[:, dk:] = g2[:, :-dk]
        np.minimum(best, sh + h2, out=best)
        sh = np.full_like(g2, cap)
        sh[:, :-dk] = g2[:, dk:]
        np.minimum(best, sh + h2, out=best)
    return np.sqrt(best) * res


def _static_dist(world, res=CS_RES, rmax_m=3.0):
    key = (hash(world.occ.tobytes()), world.occ.shape, res)
    d = _STATIC_DIST.get(key)
    if d is None:
        r = int(round(world.step / res))
        obs = np.kron(world.occ, np.ones((r, r), dtype=bool))
        d = _surface_dist(obs, res, int(round(rmax_m / res)))
        _STATIC_DIST.clear()
        _STATIC_DIST[key] = d
    return d


def clearance_field(world, boxes=(), res=CS_RES):
    """(dist, xs, ys): distance (m) from every raster cell centre to the
    nearest obstacle surface - static cells exact, boxes (ALL heights) exact
    analytic.  0 inside obstacles.  Capped at 3 m."""
    d = _static_dist(world, res).copy()
    ny, nx = d.shape
    xs = world.x0 + (np.arange(nx) + 0.5) * res
    ys = world.y0 + (np.arange(ny) + 0.5) * res
    for b in (_norm_box(b) for b in boxes):
        ex = np.maximum(np.abs(xs - b["x"]) - b["width"] / 2, 0.0)
        ey = np.maximum(np.abs(ys - b["y"]) - b["depth"] / 2, 0.0)
        np.minimum(d, np.sqrt(ey[:, None] ** 2 + ex[None, :] ** 2), out=d)
    return d, xs, ys


_NB = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356), (-1, -1, 1.41421356))


def path_exists(start, goal, boxes=None, clearance=0.6, world=None,
                endpoint_radius=1.0, endpoint_clearance=0.35, bottleneck=True,
                res=CS_RES):
    """A* on a raster of the arena with every obstacle (walls, furniture, ALL
    boxes regardless of height) inflated by `clearance` = robot-centre
    keep-out (0.6 <=> every passage used is >= 1.2 m wide).  Within
    `endpoint_radius` of start/goal the keep-out relaxes to
    `endpoint_clearance`, so a start/goal tucked into a corner can leave it.
    A raster cell is free if its exact obstacle distance >= clearance - res/2
    (i.e. widths are judged to +-res = 6 cm).

    Returns None if no such path, else a dict:
      length            8-connected A* path length (m)
      path              [(x, y), ...] raster cell centres
      min_clearance     smallest obstacle distance along that path, outside the
                        endpoint discs (m)
      min_clearance_xy  where
      bottleneck_width  (bottleneck=True) width of the narrowest gap on the
                        WIDEST route (maximin clearance over all routes, outside
                        the endpoint discs) = 2 x maximin + res  (m, +-0.06).
                        Often sits at a corner approach (e.g. 2.8 m for an
                        empty arena corner-to-corner run)
      bottleneck_xy     where that gap is
      start_clearance, goal_clearance   obstacle distance at the endpoints (m)
    """
    if boxes is None:
        boxes = world.boxes if world is not None else []
    world = world or World(carve=False)
    dist, xs, ys = clearance_field(world, boxes, res)
    ny, nx = dist.shape
    N = nx * ny

    def cell(p):
        return (int(np.clip(np.floor((p[0] - world.x0) / res), 0, nx - 1)),
                int(np.clip(np.floor((p[1] - world.y0) / res), 0, ny - 1)))

    si, sj = cell(start)
    gi, gj = cell(goal)
    near = ((xs[None, :] - start[0]) ** 2 + (ys[:, None] - start[1]) ** 2 <= endpoint_radius ** 2) | \
           ((xs[None, :] - goal[0]) ** 2 + (ys[:, None] - goal[1]) ** 2 <= endpoint_radius ** 2)
    tol = 0.5 * res
    free = (dist >= clearance - tol) | (near & (dist >= endpoint_clearance))
    s_cl, g_cl = float(dist[sj, si]), float(dist[gj, gi])
    if not free[sj, si] or not free[gj, gi]:
        return None
    fr = free.ravel().tolist()
    S, Gc = sj * nx + si, gj * nx + gi
    nb = [(dj * nx + di, di, dj, c) for di, dj, c in _NB]
    # ---------------- A* (flat lists; no corner cutting)
    INF = float("inf")
    gcost = [INF] * N
    parent = [-1] * N
    gcost[S] = 0.0
    pq = [(math.hypot(gi - si, gj - sj), 0.0, S)]
    found = False
    while pq:
        _, gc, u = heapq.heappop(pq)
        if gc > gcost[u]:
            continue
        if u == Gc:
            found = True
            break
        uj, ui = divmod(u, nx)
        for off, di, dj, c in nb:
            ai, aj = ui + di, uj + dj
            if not (0 <= ai < nx and 0 <= aj < ny):
                continue
            v = u + off
            if not fr[v] or (di and dj and not (fr[u + di] and fr[u + dj * nx])):
                continue
            ng = gc + c
            if ng < gcost[v]:
                gcost[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + math.hypot(gi - ai, gj - aj), ng, v))
    if not found:
        return None
    path = []
    u = Gc
    while u >= 0:
        path.append(divmod(u, nx))
        if u == S:
            break
        u = parent[u]
    path.reverse()                                   # [(j, i), ...]
    pj = np.array([p[0] for p in path])
    pi = np.array([p[1] for p in path])
    on = dist[pj, pi]
    mid = ~near[pj, pi]
    if mid.any():
        k = int(np.argmin(np.where(mid, on, np.inf)))
        mcl, mxy = float(on[k]), (round(float(xs[pi[k]]), 3), round(float(ys[pj[k]]), 3))
    else:
        mcl, mxy = float("inf"), None
    out = {
        "length": round(float(gcost[Gc]) * res, 3),
        "path": [(round(float(xs[i]), 3), round(float(ys[j]), 3)) for j, i in path],
        "min_clearance": round(mcl, 3),
        "min_clearance_xy": mxy,
        "start_clearance": round(s_cl, 3),
        "goal_clearance": round(g_cl, 3),
    }
    if bottleneck:
        # widest route: maximise the minimum clearance (endpoint discs open)
        sc = np.where(near & (dist >= endpoint_clearance), np.inf, dist)
        sc[dist <= 0] = -1.0
        scl = sc.ravel().tolist()
        best = [-INF] * N
        arg = [-1] * N
        best[S] = scl[S]
        pq = [(-scl[S], S)]
        while pq:
            nbv, u = heapq.heappop(pq)
            nbv = -nbv
            if nbv < best[u]:
                continue
            if u == Gc:
                break
            uj, ui = divmod(u, nx)
            for off, di, dj, _ in nb:
                ai, aj = ui + di, uj + dj
                if not (0 <= ai < nx and 0 <= aj < ny):
                    continue
                v = u + off
                w = scl[v]
                if w <= 0:
                    continue
                m = nbv if nbv < w else w
                if m > best[v]:
                    best[v] = m
                    arg[v] = u
                    heapq.heappush(pq, (-m, v))
        bott = best[Gc]
        u, worst, bxy = Gc, INF, None
        while u >= 0:
            if scl[u] < worst:
                j, i = divmod(u, nx)
                worst, bxy = scl[u], (round(float(xs[i]), 3), round(float(ys[j]), 3))
            if u == S:
                break
            u = arg[u]
        # the true gap centre lies within res/2 of a raster centre -> + res
        out["bottleneck_width"] = round(2 * bott + res, 3) if bott != INF else INF
        out["bottleneck_xy"] = bxy
    return out


# =============================================================================
# plotting
# =============================================================================
def _foot_corners(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(FOOT_FRONT, FOOT_HALF_W), (FOOT_FRONT, -FOOT_HALF_W),
           (FOOT_REAR, -FOOT_HALF_W), (FOOT_REAR, FOOT_HALF_W)]
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in pts]


def plot(result, boxes=None, path_png="path.png", world=None, title=None,
         planned=None, show_scan_at_end=False):
    """Top-down PNG: grid (grey = shell, not counted; orange = forklifts /
    shelving, counted), boxes (red = tall, visible to the z=0.35 scan;
    magenta = low, invisible; label = index / height), start (+ footprint),
    goal (+ 0.75 m disc), trajectory, red X at the first collision, final
    footprint.  `planned` may be a path_exists() result to overlay its path.
    Uses matplotlib when importable, else a numpy-only raster PNG."""
    world = world or World()
    boxes = [_norm_box(b) for b in (boxes if boxes is not None else result.get("boxes", []))]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle
    except Exception:
        return _plot_raster(result, boxes, path_png, world, planned)
    fig, ax = plt.subplots(figsize=(7.5, 10))
    img = np.zeros(world.occ.shape + (3,))
    img[:] = 1.0
    img[world.tag == TAG_WALL] = (0.35, 0.35, 0.35)
    img[world.counts] = (0.95, 0.6, 0.2)
    ax.imshow(img, origin="lower", extent=(world.x0, world.x0 + world.nx * world.step,
                                           world.y0, world.y0 + world.ny * world.step),
              interpolation="nearest")
    for k, b in enumerate(boxes):
        low = b["height"] < SCAN_Z
        ax.add_patch(Rectangle((b["x"] - b["width"] / 2, b["y"] - b["depth"] / 2),
                               b["width"], b["depth"],
                               color="#d0217a" if low else "#c0392b", alpha=0.85))
        ax.text(b["x"], b["y"], f"{k}\n{b['height']:.2f}", ha="center", va="center",
                fontsize=6, color="white")
    if planned:
        p = np.array(planned["path"])
        ax.plot(p[:, 0], p[:, 1], "--", color="#2e86de", lw=1, label="clearance path")
    tr = result.get("traj")
    sx, sy, syaw = result["start"]
    gx, gy, gyaw = result["goal"]
    if tr is not None and len(tr):
        ax.plot(tr[:, 1], tr[:, 2], "-", color="#1b9e77", lw=1.5, label="trajectory")
        ft = result.get("first_collision_t")
        if ft is not None:
            k = int(np.argmin(np.abs(tr[:, 0] - ft)))
            ax.plot(tr[k, 1], tr[k, 2], "x", color="red", ms=10, mew=2,
                    label=f"first collision t={ft:.1f}s")
        ax.add_patch(Polygon(_foot_corners(tuple(tr[-1, 1:4])), closed=True,
                             fill=False, ec="black", lw=1))
    ax.add_patch(Polygon(_foot_corners((sx, sy, syaw)), closed=True, fill=False,
                         ec="#2e86de", lw=1))
    ax.plot(sx, sy, "o", color="#2e86de", label="start")
    ax.add_patch(Circle((gx, gy), POS_TOL, fill=False, ec="green", ls=":"))
    ax.plot(gx, gy, "*", color="green", ms=14, label="goal")
    ax.set_xlim(world.x0, world.x0 + world.nx * world.step)
    ax.set_ylim(world.y0 + 5, world.y0 + world.ny * world.step)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if title is None:
        title = (f"{result['verdict_like'].upper()}  reached={result['reached']}  "
                 f"ttg={result['time_to_goal']}  col_steps={result['collision_steps']} "
                 f"{sorted(result['partners'])}  stuck={result['stuck']}")
    ax.set_title(title, fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    return path_png


def _plot_raster(result, boxes, path_png, world, planned=None, ppm=20):
    """matplotlib-free fallback: numpy raster + minimal PNG writer."""
    import struct
    import zlib
    W = int(world.nx * world.step * ppm)
    H = int(world.ny * world.step * ppm)
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    def px(x, y):
        return (int((x - world.x0) * ppm), H - 1 - int((y - world.y0) * ppm))

    rep = np.kron(world.tag, np.ones((int(world.step * ppm),) * 2, dtype=np.int8))[::-1]
    rep = rep[:H, :W]
    img[:rep.shape[0], :rep.shape[1]][rep == TAG_WALL] = (90, 90, 90)
    img[:rep.shape[0], :rep.shape[1]][rep >= TAG_FORKLIFT_01] = (240, 150, 50)
    for b in boxes:
        x0, y0 = px(b["x"] - b["width"] / 2, b["y"] + b["depth"] / 2)
        x1, y1 = px(b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2)
        img[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = (208, 33, 122) if b["height"] < SCAN_Z else (192, 57, 43)

    def line(pts, col):
        for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
            n = max(abs(c - a), abs(d - b), 1)
            for s in np.linspace(0, 1, n + 1):
                u, v = int(round(a + (c - a) * s)), int(round(b + (d - b) * s))
                if 0 <= u < W and 0 <= v < H:
                    img[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = col

    if planned:
        line([px(*p) for p in planned["path"]], (46, 134, 222))
    tr = result.get("traj")
    if tr is not None and len(tr):
        line([px(r[1], r[2]) for r in tr[::3]], (27, 158, 119))
        c = _foot_corners(tuple(tr[-1, 1:4]))
        line([px(*p) for p in c + c[:1]], (0, 0, 0))
    for (x, y), col in ((result["start"][:2], (46, 134, 222)), (result["goal"][:2], (0, 160, 0))):
        u, v = px(x, y)
        img[max(v - 5, 0):v + 6, max(u - 5, 0):u + 6] = col
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(H))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path_png, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return path_png


def summary(res):
    """One-line text summary of a run() result."""
    return (f"{res['verdict_like']:4s} reached={res['reached']} ttg={res['time_to_goal']} "
            f"col_steps={res['collision_steps']} first_col={res['first_collision_t']} "
            f"partners={res['partners']} stuck={res['stuck']} final={res['final_pose']} "
            f"t={res['sim_time']}")


def sanity_check(verbose=True):
    """The L2/L3/accident reproductions (spawn (-6,-1,pi), goal (-6,5,pi/2))."""
    S, G = (-6.0, -1.0, math.pi), (-6.0, 5.0, 1.5708)
    cases = (
        ("L2 box, L3 controller (expect: reach, no collision)",
         [{"x": -6.2, "y": 2.0, "width": 0.8, "depth": 0.6, "height": 1.0}], {}),
        ("L3 box, L3 controller (Isaac: pass, ttg 8.28-8.30 s)",
         [{"x": -6.2, "y": 3.5, "width": 0.8, "depth": 0.6, "height": 1.0}], {"max_time_to_goal": 12.0}),
        ("accident 0.25 m box (Isaac: 360 contacts, first 5.88-5.90 s, stuck at y 3.36)",
         [{"x": -6.0, "y": 3.8, "width": 1.2, "depth": 0.6, "height": 0.25}], {"abort_after_collision_s": 5.0}),
    )
    out = []
    for name, bx, kw in cases:
        r = run(S, G, bx, **kw)
        out.append((name, r))
        if verbose:
            print(f"{name}\n    {summary(r)}")
    return out


if __name__ == "__main__":
    sanity_check()
