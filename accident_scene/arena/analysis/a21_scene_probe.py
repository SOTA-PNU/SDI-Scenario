"""Scene facts for prompt A-2-1 (REPORT.md 실험 15): what Isaac Sim itself knows.

One Isaac boot of the A-2 arena (base_carter_run_a2.py MISSION, env v2), robot at
START, nothing driven. Read-only: the stage is queried, never edited (ArenaHarness
spawns the boxes and moves the robot exactly as every graded run does).

  [R] robot colliders: every prim under /World/Nova_Carter_ROS with a collision
      API, its rigid body (the nearest ancestor with RigidBodyAPI) and its bound
      in the chassis_link frame. Only chassis_link's shapes raise contact events
      (the harness puts PhysxContactReportAPI on chassis_link).
  [S] static colliders outside the robot: prim path, USD world AABB (all
      purposes), whether a contact with it is scored (the harness excludes the
      robot subtree, GroundPlane and Warehouse_Empty_small_realtime).
  [G] occupancy: a PhysX overlap_box sweep over the walled interior at CELL m,
      z band ZLO..ZHI m; every cell keeps all non-robot prims PhysX reports there
      (rows: a scored prim when there is one, else the first; multi: the rest).
  [Q] fidelity anchors: a few raycast spot checks printed for the log.

Output: results/analysis/a21_scene_probe.json (+ the console log via carter-run).

Run (serial, through the lock):
  CARTER_REPO=$HOME/SDI-Scenario accident_scene/arena/carter_run_locked.sh \\
      accident_scene/arena/analysis/a21_scene_probe.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import ArenaHarness  # noqa: E402
from carter_env import CHASSIS, ROBOT_ROOT  # noqa: E402

BASE = ARENA / "base_carter_run_a2.py"
_src = BASE.read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
OUT = ARENA / "results" / "analysis" / "a21_scene_probe.json"

CELL = 0.05
ZLO, ZHI = 0.03, 0.60              # floor clearance .. above the robot's top (0.554 m)
X0, X1, Y0, Y1 = -10.6, 9.9, -12.4, 18.4   # walled interior + a margin (walls ~-10.2/9.5, -12.0/18.0)
EXCLUDED = (ROBOT_ROOT, "/World/warehouse_with_forklifts/GroundPlane",
            "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime")

h = ArenaHarness(level="accident_arena", variant="a21_scene_probe",
                 out_dir=tempfile.mkdtemp(prefix="a21probe_"), **MISSION).boot()

import carb  # noqa: E402
import omni.physx  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

stage = h.stage
q = omni.physx.get_physx_scene_query_interface()
x_s, y_s, yaw_s = h.gt_pose()
print(f"[a21probe] chassis at ({x_s:+.4f}, {y_s:+.4f}, yaw {yaw_s:+.4f})", flush=True)

PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES, useExtentsHint=False)


def rng(r):
    lo, hi = r.GetMin(), r.GetMax()
    return [round(lo[0], 4), round(lo[1], 4), round(lo[2], 4), round(hi[0], 4), round(hi[1], 4), round(hi[2], 4)]


def body_of(prim):
    p = prim
    while p and p.IsValid() and str(p.GetPath()) != "/":
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(p.GetPath())
        p = p.GetParent()
    return None


def size_info(prim):
    t = prim.GetTypeName()
    d = {"type": str(t)}
    try:
        if t == "Cube":
            d["size"] = UsdGeom.Cube(prim).GetSizeAttr().Get()
        elif t == "Cylinder":
            c = UsdGeom.Cylinder(prim)
            d.update(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(), axis=str(c.GetAxisAttr().Get()))
        elif t == "Sphere":
            d["radius"] = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        elif t == "Capsule":
            c = UsdGeom.Capsule(prim)
            d.update(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(), axis=str(c.GetAxisAttr().Get()))
        elif t == "Mesh":
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            d["points"] = len(pts) if pts is not None else 0
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                d["approximation"] = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
    except Exception as e:  # noqa: BLE001
        d["err"] = repr(e)[:120]
    return d


# ------------------------------------------------------------------ [R] robot
chassis = stage.GetPrimAtPath(CHASSIS)
robot = []
for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_ROOT)):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    rel = bbc.ComputeRelativeBound(prim, chassis).ComputeAlignedRange()
    robot.append({"path": str(prim.GetPath()), "body": body_of(prim), "enabled": en,
                  "bound_in_chassis": rng(rel), "world": rng(bbc.ComputeWorldBound(prim).ComputeAlignedRange()),
                  **size_info(prim)})
print(f"\n===== [R] ROBOT COLLIDERS ({len(robot)}) =====", flush=True)
for r in robot:
    print(f"  {r['path']}  body={r['body']} enabled={r['enabled']} {r['type']} "
          f"chassis-frame={r['bound_in_chassis']}", flush=True)


def union(bounds):
    if not bounds:
        return None
    return [min(b[0] for b in bounds), min(b[1] for b in bounds), min(b[2] for b in bounds),
            max(b[3] for b in bounds), max(b[4] for b in bounds), max(b[5] for b in bounds)]


ch_shapes = [r for r in robot if r["body"] == CHASSIS and r["enabled"] is not False]
foot = {"chassis_shapes": union([r["bound_in_chassis"] for r in ch_shapes]),
        "all_shapes": union([r["bound_in_chassis"] for r in robot if r["enabled"] is not False])}
print(f"  chassis_link shapes union (chassis frame) = {foot['chassis_shapes']}", flush=True)
print(f"  all robot shapes union (chassis frame)    = {foot['all_shapes']}", flush=True)

# ------------------------------------------------------------------ [S] static colliders
static = []
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith(ROBOT_ROOT) or not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    if en is False:
        continue
    r = rng(bbc.ComputeWorldBound(prim).ComputeAlignedRange())
    if r[0] > 1e30:
        continue
    static.append({"path": p, "aabb": r, "scored": not any(p.startswith(e) for e in EXCLUDED),
                   **size_info(prim)})
inband = [s for s in static if s["aabb"][2] < ZHI and s["aabb"][5] > ZLO
          and s["aabb"][0] < X1 and s["aabb"][3] > X0 and s["aabb"][1] < Y1 and s["aabb"][4] > Y0]
print(f"\n===== [S] STATIC COLLIDERS: {len(static)} total, {len(inband)} in band "
      f"z {ZLO}-{ZHI} inside the interior =====", flush=True)
for s in sorted(inband, key=lambda s: s["path"])[:400]:
    print(f"  {'SCORED ' if s['scored'] else 'ignored'} {s['path']}  {s['type']}  aabb={s['aabb']}", flush=True)

# ------------------------------------------------------------------ [G] occupancy sweep
hz = (ZHI - ZLO) / 2.0
cz = (ZHI + ZLO) / 2.0
half = CELL / 2.0
nx = int(round((X1 - X0) / CELL))
ny = int(round((Y1 - Y0) / CELL))
owners, owner_ix = [], {}
grid = []                  # rows over y (j), columns over x (i): 0 free, k -> owners[k-1]
multi = []                 # [j, i, [k, ...]] for cells where PhysX reports more than one prim
t0 = time.time()
n_q = 0


def _scored(path):
    return not any(path.startswith(e) for e in EXCLUDED)


for j in range(ny):
    cy = Y0 + (j + 0.5) * CELL
    row = []
    for i in range(nx):
        cx = X0 + (i + 0.5) * CELL
        found = []

        def rep(hit, _f=found):
            path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
            if not path.startswith(ROBOT_ROOT) and path not in _f:
                _f.append(path)
            return True  # keep every prim in the cell (goods stand on pallets)

        q.overlap_box(carb.Float3(half, half, hz), carb.Float3(cx, cy, cz),
                      carb.Float4(0.0, 0.0, 0.0, 1.0), rep, False)
        n_q += 1
        if found:
            found.sort(key=lambda pth: (not _scored(pth), pth))   # a scored prim labels the cell
            ks = []
            for pth in found:
                k = owner_ix.get(pth)
                if k is None:
                    owners.append(pth)
                    k = owner_ix[pth] = len(owners)
                ks.append(k)
            row.append(ks[0])
            if len(ks) > 1:
                multi.append([j, i, ks])
        else:
            row.append(0)
    grid.append(row)
    if j % 100 == 0:
        print(f"  [G] row {j}/{ny}  {time.time() - t0:.1f}s  owners={len(owners)}", flush=True)
occ = sum(1 for r in grid for v in r if v)
print(f"\n===== [G] OCCUPANCY {nx}x{ny} cells @ {CELL} m, z {ZLO}-{ZHI}: {occ} occupied, "
      f"{len(owners)} owner prims, {n_q} overlap queries in {time.time() - t0:.1f}s =====", flush=True)
counts = {}
for r in grid:
    for v in r:
        if v:
            counts[owners[v - 1]] = counts.get(owners[v - 1], 0) + 1
for p, n in sorted(counts.items(), key=lambda kv: -kv[1])[:150]:
    print(f"  {n:6d}  {'SCORED ' if not any(p.startswith(e) for e in EXCLUDED) else 'ignored'} {p}", flush=True)

# ------------------------------------------------------------------ [Q] spot checks
print("\n===== [Q] SPOT CHECKS =====", flush=True)


def ray(x, y, z, dx, dy, max_d=8.0):
    hits = []

    def rep(hit):
        path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
        if not path.startswith(ROBOT_ROOT):
            hits.append((round(hit.distance, 4), path))
        return True

    q.raycast_all(carb.Float3(x, y, z), carb.Float3(dx, dy, 0.0), max_d, rep)
    return sorted(hits)[:3]


for (x, y, dx, dy, z) in ((-8.0, -6.77, 1.0, 0.0, 0.35), (-2.35, -4.5, 0.0, 1.0, 0.35),
                          (-7.85, -5.0, 0.0, 1.0, 0.30), (-7.85, -5.0, 0.0, 1.0, 0.10),
                          (0.0, 0.0, -1.0, 0.0, 0.30), (0.0, 0.0, 1.0, 0.0, 0.30),
                          (0.0, 0.0, 0.0, -1.0, 0.30), (0.0, 0.0, 0.0, 1.0, 0.30)):
    print(f"  ray from ({x:+.2f},{y:+.2f}, z {z}) dir ({dx:+.0f},{dy:+.0f}): {ray(x, y, z, dx, dy, 30.0)}", flush=True)

# ------------------------------------------------------------------ [F] footprint by PhysX rays
# USD bounds of the wheel/caster cylinders are wrong (extents of +-50..68 m), so the
# outline the scene really collides with is measured: rays from a 2 m ring toward the
# chassis origin, at several heights, keep the first robot shape they touch.
import math  # noqa: E402

print("\n===== [F] ROBOT OUTLINE (PhysX rays toward the chassis origin) =====", flush=True)
cyaw, syaw = math.cos(yaw_s), math.sin(yaw_s)
outline = {}
for z in (0.08, 0.12, 0.20, 0.30, 0.40, 0.47, 0.52):
    pts = []
    for k in range(180):
        th = math.radians(k * 2.0)
        ox, oy = x_s + 2.0 * math.cos(th), y_s + 2.0 * math.sin(th)
        dx, dy = -math.cos(th), -math.sin(th)
        hits = []

        def rep(hit, _h=hits):
            path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
            if path.startswith(ROBOT_ROOT):
                body = hit.rigid_body if isinstance(hit.rigid_body, str) else str(hit.rigid_body)
                _h.append((hit.distance, path, body))
            return True

        q.raycast_all(carb.Float3(ox, oy, z), carb.Float3(dx, dy, 0.0), 2.0, rep)
        if not hits:
            continue
        dist, path, body = min(hits)
        wx, wy = ox + dx * dist, oy + dy * dist
        fx = (wx - x_s) * cyaw + (wy - y_s) * syaw          # forward (chassis x)
        fy = -(wx - x_s) * syaw + (wy - y_s) * cyaw         # left (chassis y)
        pts.append([round(fx, 4), round(fy, 4), path.rsplit("/", 2)[-2] if "/" in path else path, body])
    outline[str(z)] = pts
    if pts:
        xs, ys = [p_[0] for p_ in pts], [p_[1] for p_ in pts]
        parts = sorted({p_[2] for p_ in pts})
        print(f"  z={z:.2f}: {len(pts)} hits, x {min(xs):+.3f}..{max(xs):+.3f}, y {min(ys):+.3f}..{max(ys):+.3f}, parts {parts}", flush=True)
    else:
        print(f"  z={z:.2f}: no robot shape", flush=True)
allp = [p_ for v in outline.values() for p_ in v]
chp = [p_ for p_ in allp if p_[3] == CHASSIS or "/chassis_link" in p_[3]]
if allp:
    print(f"  ALL heights: x {min(p_[0] for p_ in allp):+.3f}..{max(p_[0] for p_ in allp):+.3f}, "
          f"y {min(p_[1] for p_ in allp):+.3f}..{max(p_[1] for p_ in allp):+.3f}", flush=True)
if chp:
    print(f"  chassis_link only: x {min(p_[0] for p_ in chp):+.3f}..{max(p_[0] for p_ in chp):+.3f}, "
          f"y {min(p_[1] for p_ in chp):+.3f}..{max(p_[1] for p_ in chp):+.3f}, "
          f"max radius from origin {max(math.hypot(p_[0], p_[1]) for p_ in chp):.3f}", flush=True)
if allp:
    print(f"  max radius from origin (all shapes) {max(math.hypot(p_[0], p_[1]) for p_ in allp):.3f}", flush=True)

data = {"made": time.strftime("%F %T"), "base": BASE.name, "mission_boxes": MISSION["boxes"],
        "start_pose": [round(x_s, 4), round(y_s, 4), round(yaw_s, 4)],
        "excluded_prefixes": list(EXCLUDED),
        "robot": {"colliders": robot, "footprint_usd": foot, "outline_rays": outline},
        "static": inband, "static_total": len(static),
        "grid": {"x0": X0, "y0": Y0, "cell": CELL, "nx": nx, "ny": ny, "z_band": [ZLO, ZHI],
                 "owners": owners, "rows": grid, "multi": multi,
                 "label_rule": "rows hold a scored prim when the cell has one, else the first reported"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, separators=(",", ":")))
print(f"\n[a21probe] wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
h.close()
