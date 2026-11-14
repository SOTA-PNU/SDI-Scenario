"""Does env.raycast_scan() report the CLOSEST hit? (hit-order check, read-only)

carter_env.Harness.raycast_scan() stops PhysX raycast_all() at the first
non-robot hit it is handed ("closest non-self hit is enough"). PhysX does not
promise that raycast_all reports touches in distance order. This probe boots
the arena with the given base's MISSION table and measures, beam by beam,
what the harness's rule returns against the true closest hit:

  [1] the single beam where a probe saw it: (2.5, 1.5, z 0.35) toward +y,
      box 04 in front of box 03 - all hits in the order PhysX reports them;
  [2] a sweep over free floor cells (0.5 m grid, arena interior, >= 0.6 m from
      any box, fixed-seed random heading): 61 beams x 180 deg exactly as
      raycast_scan builds them, counting beams whose first-reported hit is
      farther than the closest one, and which prims were swapped.

Run (serial; the guard counts this file as a benchmark process):
  CARTER_REPO=$HOME/SDI-Scenario carter-run \
      accident_scene/arena/analysis/scan_order_probe_scene.py <base.py> <label>
"""

import json
import math
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import ArenaHarness  # noqa: E402
from carter_env import ROBOT_ROOT  # noqa: E402

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ARENA / "base_carter_run.py"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "base"
_src = BASE.read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
print(f"[scanorder] {LABEL}: heights {[b['height'] for b in MISSION['boxes']]}", flush=True)

h = ArenaHarness(level="accident_arena", variant=f"scanorder_{LABEL}",
                 out_dir=tempfile.mkdtemp(prefix="scanorder_"), **MISSION).boot()

AMAP = json.loads((ARENA / "arena_map.json").read_text())


def free_cells(step=0.5, margin=0.6):
    x0, y0, s, grid = AMAP["x0"], AMAP["y0"], AMAP["step"], AMAP["grid"]
    out = []
    y = -12.2 + 0.6
    while y <= 17.9 - 0.6:
        x = -10.2 + 0.6
        while x <= 9.3 - 0.6:
            i, j = int((y - y0) / s), int((x - x0) / s)
            near_static = any(grid[ii][jj] for ii in range(i - 2, i + 3) for jj in range(j - 2, j + 3)
                              if 0 <= ii < len(grid) and 0 <= jj < len(grid[0]))
            near_box = any(abs(x - b["x"]) < b["width"] / 2 + margin and abs(y - b["y"]) < b["depth"] / 2 + margin
                           for b in MISSION["boxes"])
            if not near_static and not near_box:
                out.append((round(x, 2), round(y, 2)))
            x += step
        y += step
    return out


def all_hits(q, ox, oy, z, dx, dy, dist):
    import carb
    hits = []

    def rep(hh):
        path = hh.collision if isinstance(hh.collision, str) else str(hh.collision)
        hits.append((float(hh.distance), path))
        return True

    q.raycast_all(carb.Float3(ox, oy, z), carb.Float3(dx, dy, 0.0), dist, rep)
    return hits


def short(p):
    # NB: the warehouse shell lives under /World/warehouse_with_forklifts/ too, so
    # match the forklift prims themselves, not the substring "forklift"
    if p.startswith("/World/arena_box"):
        return p.rsplit("/", 1)[-1]
    if "/World/warehouse_extras" in p:
        return "shelf"
    if re.match(r"/World/warehouse_with_forklifts/Forklift", p):
        return "forklift"
    if "/Warehouse_Empty_small_realtime/" in p:
        return "shell"
    return "other"


state = {"done": False}


def controller(t, pose, env):
    if t >= 1.0 and not state["done"]:
        state["done"] = True
        import omni.physx
        q = omni.physx.get_physx_scene_query_interface()

        print("\n===== [1] ONE BEAM: (2.5, 1.5, 0.35) -> +y, 5.5 m, hits in PhysX report order =====",
              flush=True)
        for k, (d, p) in enumerate(all_hits(q, 2.5, 1.5, 0.35, 0.0, 1.0, 5.5)):
            print(f"  #{k}  d={d:.3f} m  {p}", flush=True)

        print("\n===== [2] SWEEP: first-reported vs closest non-robot hit, raycast_scan geometry =====",
              flush=True)
        rng = random.Random(7)
        cells = free_cells()
        beams = wrong = wrong_close = box_missed = 0
        pairs = Counter()
        errs = []
        for (x, y) in cells:
            yaw = rng.uniform(-math.pi, math.pi)
            for i in range(61):
                b = -math.pi / 2 + i * (math.pi / 60)
                a = yaw + b
                dx, dy = math.cos(a), math.sin(a)
                hits = [(d, p) for d, p in all_hits(q, x + 0.5 * dx, y + 0.5 * dy, 0.35, dx, dy, 5.5)
                        if not p.startswith(ROBOT_ROOT)]
                beams += 1
                if not hits:
                    continue
                first, near = hits[0], min(hits)
                if first[0] > near[0] + 0.01:
                    wrong += 1
                    errs.append(first[0] - near[0])
                    pairs[(short(near[1]), short(first[1]))] += 1
                    if near[0] + 0.5 <= 2.0:
                        wrong_close += 1
                    if near[1].startswith("/World/arena_box"):
                        box_missed += 1
        errs.sort()
        print(f"  poses {len(cells)}  beams {beams}  first-reported != closest: {wrong} "
              f"({100.0 * wrong / max(beams, 1):.2f} %)", flush=True)
        print(f"  of those: closest hit within 2.0 m of the robot {wrong_close}; "
              f"closest hit is an arena box {box_missed}", flush=True)
        if errs:
            print(f"  overstatement median {errs[len(errs) // 2]:.2f} m, max {errs[-1]:.2f} m", flush=True)
        print("  (closest prim -> prim the harness would report)  count:", flush=True)
        for (n, f), c in pairs.most_common(15):
            print(f"    {n:<14} -> {f:<14} {c}", flush=True)
    return 0.0, 0.0, t >= 1.5


h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
