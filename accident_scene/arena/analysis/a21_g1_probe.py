"""Gate G1 for prompt A-2-1 (REPORT.md 실험 15): the 0.05 m grid against live PhysX.

One Isaac boot of the A-2 arena (base_carter_run_a2.py MISSION), robot at START, nothing
driven, stage read-only:
  [O] at each pose of results/analysis/a21_g1_poses.json, PhysX overlap_box of each of the two
      robot-outline boxes (a21_scene.FOOTPRINT, z 0.03-0.60 m like the grid) grown by 0, 0.05
      and 0.10 m: which non-robot prims overlap. The analysis compares these gap bins with the
      grid's gap (analysis/a21_g1_check.py).
  [W] 8 raycasts at z 0.3 m toward the four walls: the inner face coordinates.
  [C] physxCollision contactOffset / restOffset on the robot colliders, the arena boxes and
      a few scene colliders (log only).
Output: results/analysis/a21_g1_probe.json

  CARTER_REPO=$HOME/SDI-Scenario accident_scene/arena/carter_run_locked.sh \\
      accident_scene/arena/analysis/a21_g1_probe.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import ArenaHarness  # noqa: E402
from carter_env import ROBOT_ROOT  # noqa: E402

BASE = ARENA / "base_carter_run_a2.py"
_src = BASE.read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
P = json.loads((ARENA / "results" / "analysis" / "a21_g1_poses.json").read_text())
OUT = ARENA / "results" / "analysis" / "a21_g1_probe.json"
h = ArenaHarness(level="accident_arena", variant="a21_g1_probe",
                 out_dir=tempfile.mkdtemp(prefix="a21g1_"), **MISSION).boot()

import carb  # noqa: E402
import omni.physx  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

q = omni.physx.get_physx_scene_query_interface()
zlo, zhi = P["z_band"]
cz, hz = (zlo + zhi) / 2.0, (zhi - zlo) / 2.0


def overlap(cx, cy, yaw, hx, hy):
    found = set()

    def rep(hit):
        path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
        if not path.startswith(ROBOT_ROOT):
            found.add(path)
        return True

    q.overlap_box(carb.Float3(hx, hy, hz), carb.Float3(cx, cy, cz),
                  carb.Float4(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)), rep, False)
    return sorted(found)


res = []
for (x, y, yaw) in P["poses"]:
    rec = {"pose": [x, y, yaw], "boxes": []}
    c, s = math.cos(yaw), math.sin(yaw)
    for (fx, fy) in P["footprint"]:
        mf, ml = (fx[0] + fx[1]) / 2.0, (fy[0] + fy[1]) / 2.0
        cx, cy = x + mf * c - ml * s, y + mf * s + ml * c
        hx, hy = (fx[1] - fx[0]) / 2.0, (fy[1] - fy[0]) / 2.0
        rec["boxes"].append({str(g): overlap(cx, cy, yaw, hx + g, hy + g) for g in (0.0, 0.05, 0.10)})
    res.append(rec)
print(f"[G1] {len(res)} poses queried", flush=True)


def ray(x, y, dx, dy, z=0.3, d=30.0):
    hits = []

    def rep(hit):
        path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
        if not path.startswith(ROBOT_ROOT):
            hits.append((hit.distance, path))
        return True

    q.raycast_all(carb.Float3(x, y, z), carb.Float3(dx, dy, 0.0), d, rep)
    hits.sort()
    return [round(hits[0][0], 4), hits[0][1]] if hits else None


rays = {}
for name, (x, y, dx, dy) in {"west_a": (-9.0, -6.0, -1, 0), "west_b": (-9.0, 14.0, -1, 0),
                             "east_a": (8.0, 0.0, 1, 0), "east_b": (8.0, 11.0, 1, 0),
                             "south_a": (-5.0, -11.0, 0, -1), "south_b": (4.0, -11.0, 0, -1),
                             "north_a": (-2.0, 17.0, 0, 1), "north_b": (3.0, 17.0, 0, 1)}.items():
    r = ray(x, y, dx, dy)
    face = None if r is None else round((x + dx * r[0]) if dx else (y + dy * r[0]), 3)
    rays[name] = {"from": [x, y], "dir": [dx, dy], "hit": r, "face": face}
    print(f"[G1] ray {name}: {r} -> face {face}", flush=True)

offs = {}
for prim in Usd.PrimRange(h.stage.GetPseudoRoot()):
    p = str(prim.GetPath())
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    if not (p.startswith(ROBOT_ROOT) or p.startswith("/World/arena_box_") or "Forklift" in p or "Rackshield" in p):
        continue
    api = PhysxSchema.PhysxCollisionAPI(prim)
    co = api.GetContactOffsetAttr().Get() if prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else None
    ro = api.GetRestOffsetAttr().Get() if prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else None
    offs[p] = {"contactOffset": co, "restOffset": ro}
print(f"[G1] offsets: {len(offs)} colliders; e.g. {list(offs.items())[:4]}", flush=True)
OUT.write_text(json.dumps({"poses": res, "rays": rays, "offsets": offs}, separators=(",", ":")))
print(f"[G1] wrote {OUT}", flush=True)
h.close()
