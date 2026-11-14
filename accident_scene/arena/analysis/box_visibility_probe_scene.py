"""Box visibility probe: does each arena box reach the robot's sensing planes?

One Isaac boot with the MISSION table of the given base file (v1 or v2). The
robot is spawned at a vantage pose 1.2 m south of box 04 facing +y (probe
only - the arena itself is untouched), then:

  [A] spawned box heights as authored in the USD (xformOp:scale z);
  [B] for every box, a horizontal PhysX ray from 1.0 m outside one face at
      each sensing height - 0.15 (a lowered custom scan), 0.35 (env.raycast_scan
      default), the two RPLIDAR S2E mounts (~0.418) and the XT32 mount (~0.526,
      as a horizontal ring) - and the first non-robot prim it hits;
  [C] from the vantage, what the controller's env.raycast_scan() and the real
      RTX front 2D lidar (/front_2d_lidar/scan) report toward box 04
      (bearing 0) and box 05 (bearing -27 deg).

Run (serial; the guard counts this file as a benchmark process):
  CARTER_REPO=$HOME/SDI-Scenario carter-run \
      accident_scene/arena/analysis/box_visibility_probe_scene.py <base.py> <label>
"""

import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import ArenaHarness  # noqa: E402
from carter_env import CHASSIS, ROBOT_ROOT  # noqa: E402

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ARENA / "base_carter_run.py"
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE.parent.name
_src = BASE.read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
MISSION["start_xy_yaw"] = (2.5, 1.0, math.pi / 2.0)   # probe vantage, not the arena start
print(f"[boxprobe] {LABEL}: MISSION from {BASE}; heights "
      f"{[b['height'] for b in MISSION['boxes']]}", flush=True)

h = ArenaHarness(level="accident_arena", variant=f"boxprobe_{LABEL}",
                 out_dir=tempfile.mkdtemp(prefix="boxprobe_"), **MISSION).boot()

from pxr import Usd, UsdGeom  # noqa: E402

stage = h.stage


def world_xyz_yaw(path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    xa = m.TransformDir((1.0, 0.0, 0.0))
    return float(t[0]), float(t[1]), float(t[2]), math.atan2(xa[1], xa[0])


print("\n===== [A] SPAWNED BOX HEIGHTS (USD xformOp:scale z) =====", flush=True)
for i in range(len(MISSION["boxes"])):
    prim = stage.GetPrimAtPath(f"/World/arena_box_{i:02d}")
    sc = prim.GetAttribute("xformOp:scale").Get()
    print(f"  arena_box_{i:02d}: scale z = {float(sc[2]):.3f} m", flush=True)

FRONT_LIDAR = f"{ROBOT_ROOT}/chassis_link/front_RPLidar/RPLIDAR_S2E"
REAR_LIDAR = f"{ROBOT_ROOT}/chassis_link/rear_RPLidar/RPLIDAR_S2E"
XT32 = f"{ROBOT_ROOT}/chassis_link/XT_32/PandarXT_32_10hz"
cx, cy, cz, cyaw = world_xyz_yaw(CHASSIS)
fl = world_xyz_yaw(FRONT_LIDAR)
z_front, z_rear, z_xt = fl[2], world_xyz_yaw(REAR_LIDAR)[2], world_xyz_yaw(XT32)[2]
lidar_fwd = (fl[0] - cx) * math.cos(cyaw) + (fl[1] - cy) * math.sin(cyaw)
lidar_yaw_off = math.atan2(math.sin(fl[3] - cyaw), math.cos(fl[3] - cyaw))
PLANES = [("custom z=0.15", 0.15), ("raycast_scan z=0.35", 0.35),
          (f"front RPLIDAR z={z_front:.3f}", z_front), (f"rear RPLIDAR z={z_rear:.3f}", z_rear),
          (f"XT32 mount z={z_xt:.3f}", z_xt)]
print(f"\n[boxprobe] front RPLIDAR {lidar_fwd:+.3f} m ahead of the chassis origin, "
      f"yaw offset {math.degrees(lidar_yaw_off):+.1f} deg", flush=True)

latest = {}
try:
    from sensor_msgs.msg import LaserScan
    h.node.create_subscription(LaserScan, "/front_2d_lidar/scan",
                               lambda m: latest.__setitem__("scan", m), 10)
except Exception as e:  # noqa: BLE001
    print(f"[boxprobe] no LaserScan subscription: {e!r}", flush=True)


def first_hit(q, ox, oy, z, dx, dy, max_d=3.0):
    import carb
    best = {"d": None, "path": None}

    def rep(hh):
        path = hh.collision if isinstance(hh.collision, str) else str(hh.collision)
        if path.startswith(ROBOT_ROOT):
            return True
        if best["d"] is None or hh.distance < best["d"]:
            best["d"], best["path"] = float(hh.distance), path
        return True

    q.raycast_all(carb.Float3(ox, oy, z), carb.Float3(dx, dy, 0.0), max_d, rep)
    return best


def lidar_range(scan, bearing):
    """Median valid range within +-1 deg of a chassis-frame bearing, plus the lidar offset."""
    a = bearing - lidar_yaw_off
    rs = []
    for k, r in enumerate(scan.ranges):
        ang = scan.angle_min + k * scan.angle_increment
        if abs(math.atan2(math.sin(ang - a), math.cos(ang - a))) <= math.radians(1.0) \
                and scan.range_min < r < scan.range_max:
            rs.append(r)
    if not rs:
        return None
    rs.sort()
    return rs[len(rs) // 2] + lidar_fwd   # approx. distance from the chassis origin


state = {"done": False}


def controller(t, pose, env):
    if t >= 1.5 and not state["done"]:
        state["done"] = True
        import omni.physx
        q = omni.physx.get_physx_scene_query_interface()
        print("\n===== [B] FIRST HIT of a horizontal ray from 1.0 m outside each box =====", flush=True)
        print("  box  h     " + "  ".join(f"{n:<26}" for n, _ in PLANES), flush=True)
        for i, b in enumerate(MISSION["boxes"]):
            if i == 13:   # approach from the north (open side)
                ox, oy, dx, dy = b["x"], b["y"] + b["depth"] / 2 + 1.0, 0.0, -1.0
            else:
                ox, oy, dx, dy = b["x"], b["y"] - b["depth"] / 2 - 1.0, 0.0, 1.0
            cells = []
            for _, z in PLANES:
                hit = first_hit(q, ox, oy, z, dx, dy)
                own = hit["path"] is not None and hit["path"].startswith(f"/World/arena_box_{i:02d}")
                tag = "SEES BOX" if own else ("other:" + hit["path"].rsplit("/", 1)[-1][:12]
                                              if hit["path"] else "nothing<3m")
                cells.append(f"{tag:<12} {hit['d'] if hit['d'] is not None else float('nan'):5.2f}m")
            print(f"  {i:02d}  {b['height']:.2f}  " + "  ".join(f"{c:<26}" for c in cells), flush=True)

        print("\n===== [C] FROM THE VANTAGE (robot at (2.5, 1.0), facing +y) =====", flush=True)
        scan = env.raycast_scan()
        by = {round(math.degrees(bb)): d for bb, d in scan}
        rp = latest.get("scan")
        for name, deg, box_d in (("box 04 ahead", 0, 1.2), ("box 05 at -27 deg", -27, 3.14)):
            rl = lidar_range(rp, math.radians(deg)) if rp is not None else None
            print(f"  {name:<18} geometric distance to the box face ~{box_d:.2f} m | "
                  f"env.raycast_scan(z=0.35): {by.get(deg, float('nan')):.2f} m | "
                  f"RTX front 2D lidar: {('%.2f m' % rl) if rl is not None else 'no return'}",
                  flush=True)
    return 0.0, 0.0, t >= 2.5


h.run_mission(controller, abort_after_collision_s=5.0)
h.close()
