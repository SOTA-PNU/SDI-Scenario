"""Sensor inventory probe for the accident arena - read-only, one Isaac boot.

Answers:
  [1] which sensor prims the shipped Nova Carter carries in this scene and
      where they sit (world z at the arena start pose);
  [2] which ROS 2 bridge graph nodes publish them, and whether each sensor's
      render product is enabled;
  [3] which ROS 2 topics exist while a mission runs, who publishes and who
      subscribes - i.e. what the benchmark controller actually consumes;
  [4] the arena boxes against each horizontal sensing plane.

Boots exactly like a real arena run (ArenaHarness + base_carter_run.py's
MISSION table, same scene, same render products switched on). The controller
commands zero for 3 s; nothing is written outside a temporary directory.

Run (serial, the guard counts this file as a benchmark process):
  CARTER_REPO=$HOME/SDI-Scenario carter-run accident_scene/arena/analysis/sensor_probe_scene.py
"""

import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import SCAN_Z, ArenaHarness  # noqa: E402
from carter_env import ROBOT_ROOT  # noqa: E402

_src = (ARENA / "base_carter_run.py").read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})

h = ArenaHarness(level="accident_arena", variant="sensor_probe",
                 out_dir=tempfile.mkdtemp(prefix="sensor_probe_"), **MISSION).boot()

from pxr import Usd, UsdGeom  # noqa: E402

stage = h.stage
root = stage.GetPrimAtPath(ROBOT_ROOT)


def world_xyz(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


SENSOR_WORDS = ("lidar", "camera", "hawk", "owl", "imu", "stereo", "fisheye", "sensor")
print("\n===== [1] SENSOR PRIMS under", ROBOT_ROOT, "=====", flush=True)
types = {}
sensor_z = {}
for prim in Usd.PrimRange(root):
    tn = str(prim.GetTypeName())
    types[tn] = types.get(tn, 0) + 1
    name = prim.GetName().lower()
    is_sensor_type = tn in ("Camera", "OmniLidar", "IsaacImuSensor", "IsaacContactSensor") \
        or "Lidar" in tn or "Sensor" in tn
    if is_sensor_type or (prim.IsA(UsdGeom.Xformable) and any(w in name for w in SENSOR_WORDS)
                          and tn not in ("OmniGraph", "OmniGraphNode")):
        try:
            x, y, z = world_xyz(prim)
        except Exception:
            continue
        extra = ""
        for a in ("sensorModelPluginName", "sensorModelConfig", "omni:sensor:Core:modelName"):
            at = prim.GetAttribute(a)
            if at and at.HasAuthoredValue():
                extra += f" {a}={at.Get()!r}"
        print(f"  [{tn or '-'}] {prim.GetPath()}  z={z:.3f}{extra}", flush=True)
        if is_sensor_type:
            sensor_z[str(prim.GetPath())] = z
print("  prim type histogram:", dict(sorted(types.items(), key=lambda kv: -kv[1])), flush=True)

print("\n===== [2] ROS 2 / render-product GRAPH NODES under", ROBOT_ROOT, "=====", flush=True)
KEYS = ("inputs:topicName", "inputs:frameId", "inputs:enabled", "inputs:type",
        "inputs:cameraPrim", "inputs:lidarPrim", "inputs:renderProductPath")
for prim in Usd.PrimRange(root):
    if str(prim.GetTypeName()) != "OmniGraphNode":
        continue
    at = prim.GetAttribute("node:type")
    ntype = str(at.Get()) if at else "?"
    if not any(k in ntype for k in ("ROS2", "RenderProduct", "Odometry", "Imu", "Lidar", "Camera")):
        continue
    vals = []
    for k in KEYS:
        a = prim.GetAttribute(k)
        if a and a.HasAuthoredValue():
            vals.append(f"{k.split(':')[1]}={a.Get()!r}")
        r = prim.GetRelationship(k)
        if r and r.GetTargets():
            vals.append(f"{k.split(':')[1]}->{[str(t) for t in r.GetTargets()]}")
    print(f"  {ntype.split('.')[-1]:<34} {prim.GetPath()}  {' '.join(vals)}", flush=True)

# subscribe to the candidate sensor topics on the harness's own node, so the
# mission loop's spin_once() delivers them - shows what is published vs consumed
counts = {}
try:
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, LaserScan, PointCloud2

    def _cb(topic):
        def f(msg):
            c = counts.setdefault(topic, {"n": 0})
            c["n"] += 1
            if isinstance(msg, LaserScan) and "beams" not in c:
                rs = [r for r in msg.ranges if math.isfinite(r)]
                c.update(beams=len(msg.ranges), angle_min=round(msg.angle_min, 3),
                         angle_max=round(msg.angle_max, 3), range_max=round(msg.range_max, 2),
                         nearest=round(min(rs), 3) if rs else None, frame=msg.header.frame_id)
        return f

    for topic, typ in (("/front_2d_lidar/scan", LaserScan), ("/back_2d_lidar/scan", LaserScan),
                       ("/front_3d_lidar/lidar_points", PointCloud2), ("/chassis/odom", Odometry),
                       ("/odom", Odometry), ("/chassis/imu", Imu)):
        h.node.create_subscription(typ, topic, _cb(topic), 10)
except Exception as e:  # noqa: BLE001
    print(f"[probe] could not subscribe to sensor topics: {e!r}", flush=True)

state = {"listed": False}


def controller(t, pose, env):
    if t >= 2.0 and not state["listed"]:
        state["listed"] = True
        print("\n===== [3] ROS 2 TOPICS at t=2.0 s (publishers / subscribers) =====", flush=True)
        for name, tps in sorted(env.node.get_topic_names_and_types()):
            print(f"  {name:<44} pub={env.node.count_publishers(name)} "
                  f"sub={env.node.count_subscribers(name)}  {','.join(tps)}", flush=True)
        scan = env.raycast_scan()
        print(f"  env.raycast_scan(): {len(scan)} beams, nearest "
              f"{min(d for _, d in scan):.3f} m (the controller's only obstacle input)", flush=True)
    return 0.0, 0.0, t >= 3.0


h.run_mission(controller, abort_after_collision_s=5.0)

print("\n===== [3b] MESSAGES RECEIVED in ~4 s of sim (harness node subscriptions) =====", flush=True)
for topic in ("/front_2d_lidar/scan", "/back_2d_lidar/scan", "/front_3d_lidar/lidar_points",
              "/chassis/odom", "/odom", "/chassis/imu"):
    print(f"  {topic:<32} {counts.get(topic, {'n': 0})}", flush=True)

print("\n===== [4] ARENA BOXES vs HORIZONTAL SENSING PLANES =====", flush=True)
planes = {"env.raycast_scan default z": SCAN_Z}
for path, z in sensor_z.items():
    if "2d_lidar" in path.lower():
        planes[path.rsplit("/", 1)[-1] + " z"] = z
for k, z in planes.items():
    print(f"  plane {k} = {z:.3f} m", flush=True)
for i, b in enumerate(MISSION["boxes"]):
    seen = {k: ("hit" if b["height"] > z else "UNDER") for k, z in planes.items()}
    print(f"  box {i:02d} h={b['height']:.2f}  " + "  ".join(f"{k.split()[0]}:{v}" for k, v in seen.items()),
          flush=True)

h.close()
