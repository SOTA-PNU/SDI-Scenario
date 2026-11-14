"""Prompt A-2-1 (REPORT.md 실험 15): scene facts measured in Isaac Sim, and the digests built on them.

Source: one Isaac probe of the A-2 arena (analysis/a21_scene_probe.py ->
results/analysis/a21_scene_probe.json), nothing hand-made:

  occupancy  PhysX overlap_box sweep of the walled interior, 0.05 m cells, z 0.03-0.60 m
             (the robot's height band), each occupied cell tagged with every prim PhysX
             reported there (goods on pallets give two); a scored prim labels the cell;
  robot      the collision outline measured with PhysX rays toward the chassis origin (the
             point whose pose the controller gets), as two boxes in the chassis frame:
             drive wheels forward -0.14..0.14, left -0.25..0.25; chassis and rear casters
             forward -0.59..0.13, left -0.18..0.18 (measured, rounded outward to 0.01);
  scoring    a chassis contact counts unless its partner is under the robot, GroundPlane
             or Warehouse_Empty_small_realtime (carter_env.Harness.excluded).

Gaps here are from the robot outline to the nearest occupied cell's edge, good to about
one cell (0.05 m). Used by agent_loop.py --a21: scene_obstacles, check_route and the
run_isaac_sim digest.
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBE = HERE / "results" / "analysis" / "a21_scene_probe.json"
CACHE = HERE / "results" / "analysis" / "a21_scene_edt.npz"
INF = 1e18

FOOTPRINT = (((-0.14, 0.14), (-0.25, 0.25)),      # drive wheels
             ((-0.59, 0.13), (-0.18, 0.18)))      # chassis and rear casters
TURN_SWEEP_R = 0.60
CHASSIS = "/World/Nova_Carter_ROS/chassis_link"
ROBOT = "/World/Nova_Carter_ROS"
WAREHOUSE = "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/"
SCAN_RAY_START = 0.5                              # carter_env.Harness.raycast_scan: ox = x + dx * 0.5


def short_name(path):
    """The name used everywhere in the outputs: arena_box_04, Forklift_01, SM_PaletteA_76, wall ..."""
    p = path
    if p.startswith(WAREHOUSE):
        n = p[len(WAREHOUSE):].split("/")[0]
        return "wall" if n.startswith("SM_WallA_") else n
    if p.startswith("/World/warehouse_with_forklifts/"):
        return p[len("/World/warehouse_with_forklifts/"):].split("/")[0]
    if p.startswith("/World/warehouse_extras/"):
        parts = p[len("/World/warehouse_extras/"):].split("/")
        return "warehouse_extras/" + "/".join(parts[:2])
    if p.startswith("/World/"):
        return p[len("/World/"):].split("/")[0]
    return p


def scored(path, excluded):
    return not any(path.startswith(e) for e in excluded)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _dt1d(f):
    """Felzenszwalb-Huttenlocher squared distance transform of one line, with argmin."""
    n = len(f)
    sites = [q for q in range(n) if f[q] < INF]
    if not sites:
        return [INF] * n, [-1] * n
    v = [0] * len(sites)
    z = [0.0] * (len(sites) + 1)
    k = 0
    v[0] = sites[0]
    z[0], z[1] = -INF, INF
    for q in sites[1:]:
        fq = f[q] + q * q
        while True:
            p = v[k]
            s = (fq - (f[p] + p * p)) / (2.0 * (q - p))
            if s <= z[k]:
                k -= 1
                continue
            break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    d, arg = [0.0] * n, [0] * n
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        p = v[k]
        d[q] = (q - p) * (q - p) + f[p]
        arg[q] = p
    return d, arg


def edt(occ):
    """Exact Euclidean distance (in cells) to the nearest occupied cell, and that cell."""
    ny, nx = occ.shape
    d1 = np.full((ny, nx), INF)
    a1 = np.full((ny, nx), -1, dtype=np.int32)
    for i in range(nx):
        d, a = _dt1d([0.0 if occ[j, i] else INF for j in range(ny)])
        d1[:, i] = d
        a1[:, i] = a
    dist = np.zeros((ny, nx))
    nj = np.zeros((ny, nx), dtype=np.int32)
    ni = np.zeros((ny, nx), dtype=np.int32)
    for j in range(ny):
        d, a = _dt1d(list(d1[j, :]))
        dist[j, :] = np.sqrt(d)
        ni[j, :] = a
        nj[j, :] = a1[j, a]
    return dist, nj, ni


class Scene:
    def __init__(self, probe=PROBE):
        self.probe_path = Path(probe)
        raw = self.probe_path.read_bytes()
        import hashlib
        self.sha16 = hashlib.sha256(raw).hexdigest()[:16]
        d = json.loads(raw)
        g = d["grid"]
        self.x0, self.y0, self.c = g["x0"], g["y0"], g["cell"]
        self.nx, self.ny = g["nx"], g["ny"]
        self.owners = g["owners"]
        self.excluded = d["excluded_prefixes"]
        self.own = np.array(g["rows"], dtype=np.int32)          # [j, i]: 0 free, k -> owners[k-1]
        self.occ = self.own > 0
        self.multi = {(j, i): [k - 1 for k in ks] for j, i, ks in g.get("multi", [])}
        self.start = d["start_pose"]
        self.static = d["static"]
        self.boxes = d["mission_boxes"]
        self._check_footprint(d["robot"]["outline_rays"])
        if CACHE.exists() and CACHE.stat().st_mtime >= self.probe_path.stat().st_mtime:
            z = np.load(CACHE)
            self.dist, self.nj, self.ni = z["dist"], z["nj"], z["ni"]
        else:
            self.dist, self.nj, self.ni = edt(self.occ)
            np.savez_compressed(CACHE, dist=self.dist, nj=self.nj, ni=self.ni)
        self.names = [short_name(o) for o in self.owners]
        self.is_scored = [scored(o, self.excluded) for o in self.owners]
        self._listing = None

    def cell_owners(self, j, i):
        """All owner indices of cell (j, i), scored first (as the probe sorted them)."""
        m = self.multi.get((j, i))
        if m is not None:
            return m
        k = int(self.own[j, i])
        return [k - 1] if k else []

    def pick(self, owners):
        """The label of a set of owners: a scored one when there is one."""
        sc = [o for o in owners if o >= 0 and self.is_scored[o]]
        if sc:
            return min(sc)
        rest = [o for o in owners if o >= 0]
        return min(rest) if rest else -1

    @staticmethod
    def _check_footprint(rays):
        """FOOTPRINT must cover the probe's PhysX ray hits (wheels / everything else)."""
        pts = [p for v in rays.values() for p in v]
        for (fx, fy), sel in ((FOOTPRINT[0], lambda p: p[2] in ("wheel_left", "wheel_right")),
                              (FOOTPRINT[1], lambda p: p[2] not in ("wheel_left", "wheel_right"))):
            for p in pts:
                if sel(p):
                    assert fx[0] - 1e-6 <= p[0] <= fx[1] + 1e-6 and fy[0] - 1e-6 <= p[1] <= fy[1] + 1e-6, p
        assert max(math.hypot(p[0], p[1]) for p in pts) <= TURN_SWEEP_R + 0.005

    # -------------------------------------------------------------- geometry
    def cell_of(self, x, y):
        return (np.floor((np.asarray(y) - self.y0) / self.c).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.c).astype(int))

    @staticmethod
    def outline_points(x, y, yaw, step=0.025, boxes=FOOTPRINT):
        """World points covering the boxes of the robot outline at pose (x, y, yaw)."""
        fxs, fys = [], []
        for (bx, by) in boxes:
            fx = np.linspace(bx[0], bx[1], int(math.ceil((bx[1] - bx[0]) / step - 1e-9)) + 1)
            fy = np.linspace(by[0], by[1], int(math.ceil((by[1] - by[0]) / step - 1e-9)) + 1)
            FX, FY = np.meshgrid(fx, fy)
            fxs.append(FX.ravel())
            fys.append(FY.ravel())
        FX, FY = np.concatenate(fxs), np.concatenate(fys)
        c, s = math.cos(yaw), math.sin(yaw)
        return x + FX * c - FY * s, y + FX * s + FY * c

    def clearance(self, poses, boxes=FOOTPRINT):
        """For each pose: (gap m, owner index or -1, (x, y) of the nearest occupied cell).

        gap is between the robot outline and the nearest occupied cell's edge; 0.0 means the
        outline overlaps an occupied cell. Outside the swept area counts as occupied.
        """
        out = []
        for (x, y, yaw) in poses:
            px, py = self.outline_points(x, y, yaw, boxes=boxes)
            j, i = self.cell_of(px, py)
            inside = (j >= 0) & (j < self.ny) & (i >= 0) & (i < self.nx)
            if not inside.all():
                out.append((0.0, -1, (round(float(px[~inside][0]), 2), round(float(py[~inside][0]), 2))))
                continue
            dd = self.dist[j, i]
            k = int(np.argmin(dd))
            if dd[k] <= 0:
                # overlap: label with a scored owner of any overlapped cell when there is one
                cells = {(int(a), int(b)) for a, b in zip(j[dd <= 0], i[dd <= 0])}
                owners = {o for (a, b) in cells for o in self.cell_owners(a, b)}
                lab = self.pick(owners)
                jj, ii = next(((a, b) for (a, b) in sorted(cells) if lab in self.cell_owners(a, b)),
                              (int(j[k]), int(i[k])))
                out.append((0.0, lab, (round(self.x0 + (ii + 0.5) * self.c, 2), round(self.y0 + (jj + 0.5) * self.c, 2))))
                continue
            jj, ii = int(self.nj[j[k], i[k]]), int(self.ni[j[k], i[k]])
            gap = max(0.0, (float(dd[k]) - 0.5) * self.c)
            out.append((round(gap, 3), self.pick(self.cell_owners(jj, ii)),
                        (round(self.x0 + (ii + 0.5) * self.c, 2), round(self.y0 + (jj + 0.5) * self.c, 2))))
        return out

    def overlap_owners(self, pose):
        """Every owner of the cells the outline overlaps at pose (-1 when it leaves the grid)."""
        px, py = self.outline_points(*pose)
        j, i = self.cell_of(px, py)
        inside = (j >= 0) & (j < self.ny) & (i >= 0) & (i < self.nx)
        out = set() if inside.all() else {-1}
        j, i = j[inside], i[inside]
        hit = self.occ[j, i]
        for a, b in {(int(a), int(b)) for a, b in zip(j[hit], i[hit])}:
            out.update(self.cell_owners(a, b))
        return out

    def near_objects(self, pose, limit_m):
        """{owner: (gap, (x, y))} for every object within limit_m of the outline at pose."""
        px, py = self.outline_points(*pose)
        j, i = self.cell_of(px, py)
        inside = (j >= 0) & (j < self.ny) & (i >= 0) & (i < self.nx)
        j, i = j[inside], i[inside]
        if not len(j):
            return {}
        pad = int(math.ceil((limit_m + 0.5 * self.c) / self.c)) + 1
        j0, j1 = max(0, int(j.min()) - pad), min(self.ny, int(j.max()) + pad + 1)
        i0, i1 = max(0, int(i.min()) - pad), min(self.nx, int(i.max()) + pad + 1)
        wj, wi = np.nonzero(self.occ[j0:j1, i0:i1])
        if not len(wj):
            return {}
        wj, wi = wj + j0, wi + i0
        oj, oi = np.array(sorted({(int(a), int(b)) for a, b in zip(j, i)})).T
        d = np.sqrt(np.min((wj[:, None] - oj[None, :]) ** 2 + (wi[:, None] - oi[None, :]) ** 2, axis=1))
        out = {}
        for dc, a, b in zip(d, wj, wi):
            g = 0.0 if dc <= 0 else max(0.0, (float(dc) - 0.5) * self.c)
            if g >= limit_m:
                continue
            for o in self.cell_owners(int(a), int(b)):
                if o not in out or g < out[o][0]:
                    out[o] = (round(g, 3), (round(self.x0 + (b + 0.5) * self.c, 2), round(self.y0 + (a + 0.5) * self.c, 2)))
        return out

    def label(self, own):
        if own < 0:
            return "outside_walls", 0
        return self.names[own], int(self.is_scored[own])

    def owner_near(self, x, y, r=0.10):
        """Name and scored of the nearest occupied cell within r of (x, y), else None."""
        j, i = self.cell_of(x, y)
        j, i = int(j), int(i)
        if not (0 <= j < self.ny and 0 <= i < self.nx):
            return ["outside_walls", 0]
        if self.dist[j, i] * self.c > r + self.c:
            return None
        return list(self.label(int(self.own[self.nj[j, i], self.ni[j, i]]) - 1))

    # -------------------------------------------------------------- obstacle list
    def components(self, merge_gap=0.10):
        """Occupied cells grouped by name; 8-connected pieces closer than merge_gap merged."""
        by_name = {}
        js, is_ = np.nonzero(self.occ)
        for a, b in zip(js.tolist(), is_.tolist()):
            for o in self.cell_owners(a, b):
                by_name.setdefault(self.names[o], set()).add((a, b))
        out = {}
        for n, cells in sorted(by_name.items()):
            seen, comps = set(), []
            for cell in list(cells):
                if cell in seen:
                    continue
                st, cj, ci = [cell], [], []
                seen.add(cell)
                while st:
                    a, b = st.pop()
                    cj.append(a)
                    ci.append(b)
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            nb = (a + da, b + db)
                            if nb in cells and nb not in seen:
                                seen.add(nb)
                                st.append(nb)
                comps.append([min(ci), max(ci) + 1, min(cj), max(cj) + 1])
            gapc = merge_gap / self.c
            merged = True
            while merged:
                merged = False
                for a in range(len(comps)):
                    for b in range(a + 1, len(comps)):
                        A, B = comps[a], comps[b]
                        if (A[0] - gapc <= B[1] and B[0] - gapc <= A[1]
                                and A[2] - gapc <= B[3] and B[2] - gapc <= A[3]):
                            comps[a] = [min(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), max(A[3], B[3])]
                            del comps[b]
                            merged = True
                            break
                    if merged:
                        break
            out[n] = [(round(self.x0 + r[0] * self.c, 2), round(self.x0 + r[1] * self.c, 2),
                       round(self.y0 + r[2] * self.c, 2), round(self.y0 + r[3] * self.c, 2)) for r in comps]
        return out

    def walls_inner_face(self):
        """Inner faces of the four walls from the wall cells, away from the corners."""
        W = np.zeros_like(self.occ)
        js_, is_ = np.nonzero(self.occ)
        for a, b in zip(js_.tolist(), is_.tolist()):
            if any(self.names[o] == "wall" for o in self.cell_owners(a, b)):
                W[a, b] = True
        js, is_ = np.nonzero(W)
        xs = self.x0 + is_ * self.c
        ys = self.y0 + js * self.c
        mid_y = (ys > -11.0) & (ys < 17.0)
        mid_x = (xs > -9.0) & (xs < 8.5)
        west = float((xs[mid_y & (xs < -9.5)] + self.c).max())
        east = float(xs[mid_y & (xs > 8.5)].min())
        south = float((ys[mid_x & (ys < -11.5)] + self.c).max())
        north = float(ys[mid_x & (ys > 17.0)].min())
        return {"x": [round(west, 2), round(east, 2)], "y": [round(south, 2), round(north, 2)]}

    def z_ranges(self):
        z = {}
        for s in self.static:
            n = short_name(s["path"])
            lo, hi = max(0.0, s["aabb"][2]), s["aabb"][5]
            if n in z:
                z[n] = (min(z[n][0], lo), max(z[n][1], hi))
            else:
                z[n] = (lo, hi)
        return z

    def listing(self):
        """scene_obstacles' full answer (cached; identical on every region-less call)."""
        if self._listing is not None:
            return self._listing
        comps = self.components()
        zr = self.z_ranges()
        sc = {}
        for n, s in zip(self.names, self.is_scored):
            sc[n] = int(s)
        rows = []
        for n, rects in comps.items():
            if n == "wall":
                continue
            lo, hi = zr.get(n, (0.0, 0.6))
            for (x0, x1, y0, y1) in rects:
                rows.append([n, sc[n], x0, x1, y0, y1, round(lo, 2), round(hi, 2)])
        rows.sort(key=lambda r: (r[4], r[2]))
        walls = self.walls_inner_face()
        # build-time checks (judge spec, REPORT.md 실험 15)
        for (px, py) in ((self.start[0], self.start[1]), (8.0, 16.5)):
            assert not any(r[2] <= px <= r[3] and r[4] <= py <= r[5] for r in rows), (px, py)
        for k, b in enumerate(self.boxes):
            r = [r for r in rows if r[0] == f"arena_box_{k:02d}"]
            assert len(r) == 1, (k, r)
            r = r[0]
            for got, want in ((r[2], b["x"] - b["width"] / 2), (r[3], b["x"] + b["width"] / 2),
                              (r[4], b["y"] - b["depth"] / 2), (r[5], b["y"] + b["depth"] / 2)):
                assert abs(got - want) <= 0.05 + 1e-6, (k, got, want)
        assert len(rows) <= 80, len(rows)
        have = {r[0] for r in rows if r[1]}
        for st in self.static:                 # every scored collider in the band shows up
            if st["scored"] and st["aabb"][2] < 0.60 and st["aabb"][5] > 0.03:
                assert short_name(st["path"]) in have, st["path"]
        self._listing = {
            "source": "Isaac Sim 4.5.0 PhysX, 0.05 m 칸, 바닥 위 0.03-0.60 m",
            "robot": {"pose_point": "pose 의 (x, y) = 두 구동 바퀴 축 가운데, 제자리 회전 중심",
                      "axes": "forward = yaw 방향, left = 왼쪽 90도",
                      "outline_m": [{"part": "구동 바퀴", "forward": list(FOOTPRINT[0][0]), "left": list(FOOTPRINT[0][1])},
                                    {"part": "차체와 뒤 캐스터", "forward": list(FOOTPRINT[1][0]),
                                     "left": list(FOOTPRINT[1][1])}],
                      "turn_sweep_radius_m": TURN_SWEEP_R, "height_m": [0, 0.5],
                      "scan": {"ray_start_m": SCAN_RAY_START, "distance_from": "pose 의 (x, y)"}},
            "walls_inner_face": walls,
            "scored": ("1 이면 PhysX 가 보고한 차체(chassis_link) 접촉이 collision_count 에 들어감. 0 은 창고 건물(벽, 기둥, "
                       "랙, 팔레트, 상자 통, 손수레)이라 세지 않지만 로봇을 막음"),
            "names": {"arena_box_NN": "/World/arena_box_NN",
                      "Forklift, Forklift_01": "/World/warehouse_with_forklifts/<name>/...",
                      "warehouse_extras/...": "/World/warehouse_extras/...",
                      "SM_..., KLT_...": "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/<name>/..."},
            "cols": ["name", "scored", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
            "obstacles": rows}
        n = len(json.dumps(self._listing, ensure_ascii=False, separators=(",", ":")))
        assert n <= 4400, n
        return self._listing


def obstacle_listing(scene, region=None):
    full = scene.listing()
    rows = full["obstacles"]
    if region is not None:
        x0, x1, y0, y1 = region
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        rows = [r for r in rows if r[2] <= x1 and r[3] >= x0 and r[4] <= y1 and r[5] >= y0]
    return {**{k: v for k, v in full.items() if k != "obstacles"}, "region": region,
            "count": len(rows), "obstacles": rows}


# ------------------------------------------------------------------ route check
def robot_part(pose, point):
    """front/middle/rear + left/centre/right of a world point, seen from the robot."""
    x, y, yaw = pose
    dx, dy = point[0] - x, point[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = dx * c + dy * s, -dx * s + dy * c
    fb = "front" if fx > 0.0 else ("rear" if fx < -0.40 else "middle")
    lr = "left" if fy > 0.08 else ("right" if fy < -0.08 else "centre")
    return f"{fb}-{lr}", [round(fx, 2), round(fy, 2)]


def _check_xy(x, y, what):
    if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 12 or abs(y) > 20:
        raise ValueError(f"{what} 가 범위(|x| <= 12, |y| <= 20)를 벗어났습니다")


def _spans(flags, scale):
    out, s0 = [], None
    for k, f in enumerate(flags + [False]):
        if f and s0 is None:
            s0 = k
        if not f and s0 is not None:
            out.append([round(s0 * scale, 2), round((k - 1) * scale, 2)])
            s0 = None
    return out


def check_route(scene, points, start_pose, goal_pose, step_m=0.05, turn_step_deg=5.0):
    """Sweep the robot outline along a waypoint polyline; facts only (no verdict, no fix)."""
    if not isinstance(points, list) or not 1 <= len(points) <= 60:
        raise ValueError("points 는 1-60 개의 [x, y] 목록입니다")
    pts = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError(f"점 {p!r} 는 [x, y] 가 아닙니다")
        x, y = float(p[0]), float(p[1])
        _check_xy(x, y, f"점 {p!r}")
        pts.append((x, y))
    sx, sy, syaw = (float(v) for v in start_pose)
    _check_xy(sx, sy, "start")
    if not math.isfinite(syaw):
        raise ValueError("start 의 yaw 가 숫자가 아닙니다")
    route = [(sx, sy)] + pts
    segs, overlaps, turns, turn_ov = [], [], [], []
    heading = syaw

    def objects(per_pose, scale):
        first = {}
        for k, owners in per_pose:
            for o in owners:
                name, sc = scene.label(o)
                if name not in first:
                    first[name] = (sc, round(k * scale, 2))
        objs = sorted(([nm, sc, f0] for nm, (sc, f0) in first.items()), key=lambda o: (-o[1], o[2]))
        return objs[:4]

    def turn_rec(at_label, at, h0, h1):
        d = wrap(h1 - h0)
        n = max(1, int(abs(math.degrees(d)) / turn_step_deg))
        poses = [(at[0], at[1], h0 + d * i / n) for i in range(n + 1)]
        res = scene.clearance(poses)
        k = min(range(len(res)), key=lambda i: (res[i][0], -int(res[i][1] >= 0 and scene.is_scored[res[i][1]])))
        gap, own, pt = res[k]
        name, sc = scene.label(own)
        zero = [(i, scene.overlap_owners(poses[i])) for i, r in enumerate(res) if r[0] <= 0.0]
        if zero:
            first = {}
            for i, owners in zero:
                for o in owners:
                    nm, s2 = scene.label(o)
                    if nm not in first:
                        first[nm] = (s2, round(poses[i][2], 2))
            objs = sorted(([nm, s2, y0] for nm, (s2, y0) in first.items()), key=lambda o: (-o[1], abs(o[2] - h0)))[:4]
            turn_ov.append({"at": at_label, "yaw": [round(poses[zero[0][0]][2], 2), round(poses[zero[-1][0]][2], 2)],
                            "objects": objs})
        return [at_label, round(h0, 2), round(h0 + d, 2), gap, name, sc, robot_part(poses[k], pt)[0]]

    for k in range(len(route) - 1):
        a, b = route[k], route[k + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6:
            continue
        hd = math.atan2(b[1] - a[1], b[0] - a[0])
        if abs(wrap(hd - heading)) > math.radians(1.0):
            turns.append(turn_rec("S" if k == 0 else k, a, heading, hd))
        heading = hd
        n = max(1, int(L / step_m))
        poses = [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, hd) for i in range(n + 1)]
        res = scene.clearance(poses)
        kk = min(range(len(res)), key=lambda i: (res[i][0], -int(res[i][1] >= 0 and scene.is_scored[res[i][1]])))
        gap, own, pt = res[kk]
        name, sc = scene.label(own)
        segs.append([k + 1, [round(a[0], 2), round(a[1], 2)], [round(b[0], 2), round(b[1], 2)], round(L, 2),
                     gap, name, sc, [round(poses[kk][0], 2), round(poses[kk][1], 2)], robot_part(poses[kk], pt)[0]])
        flags = [r[0] <= 0.0 for r in res]
        if any(flags):
            per_pose = [(i, scene.overlap_owners(poses[i])) for i, f in enumerate(flags) if f]
            overlaps.append({"seg": k + 1, "s": _spans(flags, L / n)[:3], "objects": objects(per_pose, L / n)})
    end = route[-1]
    out_end = math.hypot(end[0] - goal_pose[0], end[1] - goal_pose[1])
    if out_end <= 0.75:
        turns.append(turn_rec("G", end, heading, goal_pose[2]))
    gaps = [sg[4] for sg in segs] + [t[3] for t in turns]
    out = {"route_len": round(sum(sg[3] for sg in segs), 2), "min_gap": min(gaps) if gaps else None,
           "seg_cols": ["i", "from", "to", "len", "min_gap", "nearest", "scored", "at", "part"],
           "segments": segs, "overlaps": overlaps,
           "turn_cols": ["at", "from_yaw", "to_yaw", "min_gap", "nearest", "scored", "part"], "turns": turns,
           "turn_overlaps": turn_ov}
    if out_end > 0.75:
        out["end_to_goal_m"] = round(out_end, 2)
    return out


# ------------------------------------------------------------------ run digest
def _pose_at(rows, t):
    best = rows[0]
    for r in rows:
        if r[0] <= t + 1e-6:
            best = r
        else:
            break
    return best


def _r2(v):
    return None if v is None else round(v, 2)


def _meas_v(rows, t0, t1):
    """Mean forward speed measured from the poses between t0 and t1."""
    seg = [r for r in rows if t0 - 1e-6 <= r[0] <= t1 + 1e-6]
    if len(seg) < 2 or seg[-1][0] - seg[0][0] <= 0:
        return 0.0
    dist = 0.0
    for a, b in zip(seg, seg[1:]):
        dx, dy = b[1] - a[1], b[2] - a[2]
        dist += dx * math.cos(a[3]) + dy * math.sin(a[3])
    return round(dist / (seg[-1][0] - seg[0][0]), 2)


def _mean_cmd(rows, t0, t1):
    seg = [r for r in rows if t0 - 1e-6 <= r[0] <= t1 + 1e-6 and r[4] is not None]
    if not seg:
        return None
    return [round(sum(r[4] for r in seg) / len(seg), 2), round(sum(r[5] for r in seg) / len(seg), 2)]


def contact_episodes(scene, diag, rows, join_s=0.5):
    """Episodes per object name (collider pieces joined, gaps <= join_s bridged). The
    telemetry stops at the last controller call; scored reports the harness logged after
    it (last step, settle) are added from its own list."""
    eps = []
    for e in diag.get("contact_episodes") or []:
        touched, hp, t0, t1, n, pt, sep, rcol = e
        eps.append({"obj": short_name(touched), "t0": t0, "t1": t1, "n": n, "pt": pt, "sep": sep,
                    "scored": scored(hp, scene.excluded)})
    t_last = rows[-1][0] if rows else -1.0
    late = [c for c in (diag.get("harness_collisions") or []) if c[0] > t_last + 1e-6]
    for t, typ, partner in late:
        eps.append({"obj": short_name(partner), "t0": t, "t1": t, "n": 1, "pt": None, "sep": None, "scored": True})
    eps.sort(key=lambda e: e["t0"])
    merged = {}
    out = []
    for e in eps:
        m = merged.get(e["obj"])
        if m is not None and e["t0"] - m["t1"] <= join_s:
            m["t1"] = max(m["t1"], e["t1"])
            m["n"] += e["n"]
            m["scored"] = m["scored"] or e["scored"]
            if m["pt"] is None and e["pt"] is not None:
                m["pt"], m["sep"] = e["pt"], e["sep"]
            continue
        m = dict(e)
        merged[e["obj"]] = m
        out.append(m)
    return out


def contacts_out(eps, rows, k=4):
    out = []
    for e in eps[:k]:
        r = _pose_at(rows, e["t0"])
        pose = (r[1], r[2], r[3])
        d = {"obj": e["obj"], "scored": int(e["scored"]), "t": [_r2(e["t0"]), _r2(e["t1"])], "reports": e["n"],
             "pose": [_r2(v) for v in pose]}
        if e["pt"]:
            part, local = robot_part(pose, e["pt"])
            d.update(point=[_r2(e["pt"][0]), _r2(e["pt"][1])], on_robot=local, part=part)
        d["sep"] = e["sep"]
        d["cmd"] = _mean_cmd(rows, e["t0"] - 0.5, e["t0"])
        d["v_meas"] = _meas_v(rows, e["t0"] - 0.5, e["t0"])
        out.append(d)
    return out


def first_scored_t(diag, eps):
    hc = diag.get("harness_collisions") or []
    if hc:
        return hc[0][0]
    sc = [e["t0"] for e in eps if e["scored"]]
    return min(sc) if sc else None


def closest_approach(scene, rows, t_stop, every_s=0.1, limit_m=0.5, k=4):
    pick, last = [], -1e9
    for r in rows:
        if t_stop is not None and r[0] > t_stop:
            break
        if r[0] - last >= every_s - 1e-9:
            pick.append(r)
            last = r[0]
    if not pick:
        return []
    res = scene.clearance([(r[1], r[2], r[3]) for r in pick])
    best = {}
    for r, (gap0, own0, pt0) in zip(pick, res):
        if gap0 >= limit_m:
            continue                       # nothing can be nearer than the nearest
        pose = (r[1], r[2], r[3])
        if own0 < 0:
            cand = {"outside_walls": (gap0, 0, pt0)}
        else:
            cand = {}
            for o, (g, pt) in scene.near_objects(pose, limit_m).items():
                name, sc = scene.label(o)
                if name not in cand or g < cand[name][0]:
                    cand[name] = (g, sc, pt)
        for name, (g, sc, pt) in cand.items():
            if name not in best or g < best[name][0]:
                best[name] = (g, r, sc, pt)
    out = []
    for name, (gap, r, sc, pt) in sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][2]))[:k]:
        out.append({"obj": name, "scored": sc, "gap": round(gap, 2), "t": _r2(r[0]),
                    "pose": [_r2(r[1]), _r2(r[2]), _r2(r[3])], "part": robot_part((r[1], r[2], r[3]), pt)[0]})
    return out


def stall_spans(rows, radius_m=0.10, min_len_s=5.0):
    spans, k0, n = [], 0, len(rows)
    while k0 < n:
        a = rows[k0]
        k = k0
        while k + 1 < n and math.hypot(rows[k + 1][1] - a[1], rows[k + 1][2] - a[2]) < radius_m:
            k += 1
        if rows[k][0] - a[0] >= min_len_s:
            spans.append([k0, k])
            k0 = k + 1
        else:
            k0 += 1
    merged = []
    for s0, e in spans:
        if merged and rows[s0][0] - rows[merged[-1][1]][0] < 1.0:
            merged[-1][1] = e
        else:
            merged.append([s0, e])
    return merged


def stall_mode(seg):
    vs = [r[4] for r in seg if r[4] is not None]
    ws = [r[5] for r in seg if r[5] is not None]
    if not vs:
        return "unknown"
    n = len(vs)
    if sum(1 for v, w in zip(vs, ws) if abs(v) < 0.01 and abs(w) < 0.02) >= 0.9 * n:
        return "idle"
    net, total = 0.0, 0.0
    for a, b in zip(seg, seg[1:]):
        d = wrap(b[3] - a[3])
        net += d
        total += abs(d)
    flips = sum(1 for a, b in zip(ws, ws[1:]) if a * b < 0)
    # back and forth: >= 3 sign flips of w_cmd and at least 3x more turning than net turn
    if flips >= 3 and total >= 3.0 * max(abs(net), 0.1):
        return "oscillating"
    if abs(net) >= 0.5:
        return "turning"
    if sum(abs(v) for v in vs) / n >= 0.05:
        return "blocked"                  # drive command, robot stayed within 0.1 m for >= 5 s
    cmd_net = sum(a[5] * (b[0] - a[0]) for a, b in zip(seg, seg[1:]) if a[5] is not None)
    if abs(cmd_net) >= 1.0:
        return "blocked"                  # asked for >= 1 rad of turn, turned < 0.5 rad
    return "slow"


def stalls_out(scene, diag, rows, eps, k=3):
    spans = stall_spans(rows)
    out = []
    for s0, e in spans[:k]:
        seg = rows[s0:e + 1]
        r0, r1 = seg[0], seg[-1]
        vs = [r[4] for r in seg if r[4] is not None]
        ws = [r[5] for r in seg if r[5] is not None]
        sm = [(r[7], r[8], r) for r in seg if r[7] is not None]
        d = {"t": [_r2(r0[0]), _r2(r1[0])], "mode": stall_mode(seg), "at": [_r2(r1[1]), _r2(r1[2]), _r2(r1[3])],
             "moved_m": round(math.hypot(r1[1] - r0[1], r1[2] - r0[2]), 2),
             "yaw_range": [_r2(min(r[3] for r in seg)), _r2(max(r[3] for r in seg))],
             "v_cmd": [_r2(min(vs)), _r2(max(vs))] if vs else None,
             "w_cmd": [_r2(min(ws)), _r2(max(ws))] if ws else None,
             "v_meas": _meas_v(rows, r0[0], r1[0])}
        if sm:
            dmin, brg, r = min(sm, key=lambda q: q[0])
            hx = r[1] + dmin * math.cos(r[3] + brg)
            hy = r[2] + dmin * math.sin(r[3] + brg)
            own = scene.owner_near(hx, hy)
            d["scan_min"] = [_r2(dmin), _r2(brg), own[0] if own else None]
        gap, own, pt = scene.clearance([(r1[1], r1[2], r1[3])])[0]
        name, sc = scene.label(own)
        d["nearest"] = [name, sc, round(gap, 2), robot_part((r1[1], r1[2], r1[3]), pt)[0]]
        touching = sorted({(ep["obj"], int(ep["scored"])) for ep in eps if ep["t0"] <= r1[0] and ep["t1"] >= r0[0]})
        d["touching"] = [list(t) for t in touching][:3]
        st = dict(diag.get("state_initial") or {})
        for c in diag.get("state_changes") or []:
            if c[0] <= r0[0] + 1e-6:
                st[c[1]] = c[3]
        if st:
            d["state"] = dict(list(st.items())[:6])
        out.append(d)
    return out


def before_window(rows, t_mark, span_s=2.0, every_s=0.25, k=8):
    out, last = [], -1e9
    for r in rows:
        if t_mark - span_s - 1e-9 <= r[0] <= t_mark + 1e-9 and r[0] - last >= every_s - 1e-9:
            out.append([_r2(r[0]), _r2(r[1]), _r2(r[2]), _r2(r[3]), _r2(r[4]), _r2(r[5]), _r2(r[7])])
            last = r[0]
    return out[-k:]


def commands_out(diag, rows):
    vs = [r[4] for r in rows if r[4] is not None]
    ws = [r[5] for r in rows if r[5] is not None]
    out = {}
    if vs:
        n = len(vs)
        out["v_cmd"] = [_r2(min(vs)), _r2(max(vs)), _r2(sum(vs) / n)]
        out["w_cmd"] = [_r2(min(ws)), _r2(max(ws)), _r2(sum(ws) / n)]
        out["zero_frac"] = round(sum(1 for v, w in zip(vs, ws) if abs(v) < 1e-6 and abs(w) < 1e-6) / n, 2)
        out["first_motion_t"] = next((_r2(r[0]) for r in rows if r[4] is not None and (abs(r[4]) > 1e-6 or abs(r[5]) > 1e-6)), None)
        out["done_at"] = next((_r2(r[0]) for r in rows if r[6]), None)
    out["scan_calls"] = diag.get("scan_calls", 0)
    out["scan_args"] = [a for a, _ in (diag.get("scan_args") or [])][:3]
    return out


def goal_approach(rows, goal, metrics=None, pos_tol=0.75, near_m=3.0):
    gx, gy, gyaw = goal
    best, inside, disc = None, False, []
    for r in rows:
        d = math.hypot(gx - r[1], gy - r[2])
        if best is None or d < best[0]:
            best = (d, r)
        if d <= pos_tol and not inside:
            inside = True
            disc.append([_r2(r[0]), None])
        elif d > pos_tol and inside:
            inside = False
            disc[-1][1] = _r2(r[0])
    if best is None or best[0] > near_m:
        return None
    out = {"min": [round(best[0], 2), _r2(best[1][0])], "disc": disc[:3]}
    done = next((r for r in rows if r[6]), None)
    if done:
        out["done"] = [_r2(done[0]), round(math.hypot(gx - done[1], gy - done[2]), 2), round(abs(wrap(done[3] - gyaw)), 2)]
    if metrics and metrics.get("final_yaw_err_rad") is not None:
        out["final_yaw_err"] = round(metrics["final_yaw_err_rad"], 2)
    return out


def trajectory(rows, keep_t=(), n=80):
    """<= n rows [t, x, y, yaw, v_cmd, w_cmd], spaced evenly in activity (path length +
    0.25 m per rad turned); stationary spans keep only their ends; key times forced in."""
    if not rows:
        return []
    act = [0.0]
    for a, b in zip(rows, rows[1:]):
        dm = math.hypot(b[1] - a[1], b[2] - a[2])
        dy = abs(wrap(b[3] - a[3]))
        act.append(act[-1] + (dm if dm > 0.002 else 0.0) + 0.25 * (dy if dy > 0.003 else 0.0))
    forced = {0, len(rows) - 1}
    for t in keep_t:
        if t is not None:
            forced.add(min(range(len(rows)), key=lambda i: abs(rows[i][0] - t)))
    budget = max(2, n - len(forced))
    total = act[-1]
    idx = set(forced)
    if total > 1e-9:
        step, nxt = total / budget, 0.0
        for i, a in enumerate(act):
            if a >= nxt - 1e-12:
                idx.add(i)
                nxt = a + step
    pick = sorted(idx)
    if len(pick) > n:
        keep = sorted(forced)
        rest = [i for i in pick if i not in forced]
        stride = len(rest) / max(1, n - len(keep))
        pick = sorted(set(keep) | {rest[int(j * stride)] for j in range(max(0, n - len(keep)))})
    return [[_r2(rows[i][0]), _r2(rows[i][1]), _r2(rows[i][2]), _r2(rows[i][3]), _r2(rows[i][4]), _r2(rows[i][5])]
            for i in pick]


def digest(scene, diag, goal, metrics=None):
    """The run_isaac_sim telemetry fields, uncapped, plus the forced trajectory times."""
    rows = [tuple(r) for r in (diag.get("rows") or [])]
    out = {}
    if not rows:
        return out, []
    eps = contact_episodes(scene, diag, rows)
    t_first = first_scored_t(diag, eps)
    st = stalls_out(scene, diag, rows, eps)
    out["stalls"] = st
    out["contacts"] = contacts_out(eps, rows)
    if len(eps) > 4:
        out["contacts_omitted"] = len(eps) - 4
    if t_first is not None:
        out["before_contact"] = {"event": "scored_contact", "t": _r2(t_first),
                                 "cols": ["t", "x", "y", "yaw", "v_cmd", "w_cmd", "scan_min"],
                                 "rows": before_window(rows, t_first)}
    elif st:
        out["before_contact"] = {"event": "stall", "t": st[0]["t"][0],
                                 "cols": ["t", "x", "y", "yaw", "v_cmd", "w_cmd", "scan_min"],
                                 "rows": before_window(rows, st[0]["t"][0])}
    out["closest_approach"] = closest_approach(scene, rows, t_first)
    out["commands"] = commands_out(diag, rows)
    ga = goal_approach(rows, goal, metrics)
    if ga:
        out["goal_approach"] = ga
    keys = [t_first] + [e["t0"] for e in eps[:4]] + [s["t"][0] for s in st] + [s["t"][1] for s in st]
    if ga and ga["disc"]:
        keys.append(ga["disc"][0][0])
    keys.append(out["commands"].get("done_at"))
    return out, keys


def runtime_error_out(diag, region_start_line, region_end_line=None, runner=None):
    e = (diag or {}).get("controller_error")
    if not e:
        return None
    fr = e.get("frames") or []
    runner = runner or (diag or {}).get("runner")
    mine = [f for f in fr if f[0] == runner]
    inside = [f for f in mine if region_start_line and region_end_line and region_start_line < f[1] < region_end_line]
    last = inside[-1] if inside else (mine[-1] if mine else None)
    out = {"type": e.get("type"), "msg": e.get("msg"), "t": e.get("t"), "pose": e.get("pose"),
           "calls_before": e.get("calls_before")}
    if last:
        out["line"] = last[1]
        if region_start_line:
            out["region_line"] = last[1] - region_start_line + 1
        out["code"] = last[3]
    deepest = fr[-1] if fr else None
    if deepest and deepest[0] != runner:
        out["raised_in"] = {"file": deepest[0], "line": deepest[1], "function": deepest[2], "code": deepest[3]}
    return out


CAP = 8000
SHRINK = (("trajectory", 40), ("controller_output", 20), ("closest_approach", 2), ("contacts", 2),
          ("before_contact", 4), ("stalls", 2), ("trajectory", 20), ("controller_output", 6),
          ("closest_approach", 1), ("contacts", 1), ("stalls", 1))


def cap(out, limit=CAP, rebuild_traj=None):
    """Shrink the digest in a fixed order until it fits; note what was cut in 'trimmed'.
    rebuild_traj(n) re-samples the trajectory keeping its forced rows."""
    def size():
        return len(json.dumps({k: v for k, v in out.items() if k != "scene"}, ensure_ascii=False))
    trimmed = []
    for key, n in SHRINK:
        if size() <= limit:
            break
        v = out.get(key)
        if key == "before_contact" and isinstance(v, dict) and len(v.get("rows", [])) > n:
            trimmed.append(f"before_contact.rows {len(v['rows'])}->{n}")
            v["rows"] = v["rows"][-n:]
        elif key == "controller_output" and isinstance(v, list) and len(v) > n:
            trimmed.append(f"controller_output {len(v)}->{n}")
            out[key] = v[:n // 2] + v[-(n - n // 2):]
        elif key == "trajectory" and isinstance(v, list) and len(v) > n:
            trimmed.append(f"trajectory {len(v)}->{n}")
            if rebuild_traj is not None:
                out[key] = rebuild_traj(n)
            else:
                stride = len(v) / n
                out[key] = [v[int(i * stride)] for i in range(n - 1)] + [v[-1]]
        elif isinstance(v, list) and len(v) > n:
            trimmed.append(f"{key} {len(v)}->{n}")
            out[key] = v[:n]
    if trimmed:
        out["trimmed"] = trimmed
    return out
